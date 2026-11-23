"""
Model Trainer Module for IndoBERT Clickbait Detection System

Handles model initialisation, the training loop, checkpoint management,
batched prediction, and hyperparameter grid search for IndoBERT variants
(base, large, lite).
"""

import os
import gc
import json
import shutil
import time
import contextlib
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import config
from debug_logger import (
    dbg_dataset_created,
    dbg_tokenizer_samples,
    dbg_trainer_init,
    dbg_trainer_batch,
    dbg_trainer_epoch_end,
    dbg_trainer_predict_start,
    dbg_trainer_predict_batch,
    dbg_trainer_predict_done,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

class ClickbaitDataset(Dataset):
    """PyTorch Dataset: tokenises text strings and their binary labels."""

    def __init__(self, texts, labels, tokenizer, max_length=128, split_name=""):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        n_clickbait = sum(1 for l in labels if l == 1)
        dbg_dataset_created(split_name=split_name, n_samples=len(texts), n_clickbait=n_clickbait)
        dbg_tokenizer_samples(tokenizer=tokenizer, texts=texts, labels=labels, max_length=max_length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# MODEL ARCHITECTURE
# ---------------------------------------------------------------------------

class IndoBERTClassifier(nn.Module):
    """IndoBERT encoder + dropout + linear classification head."""

    def __init__(self, model_name="indobenchmark/indobert-base-p1", num_labels=2, dropout_rate=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        logger.info(f"IndoBERTClassifier: {model_name}  hidden={self.bert.config.hidden_size}")

    def forward(self, input_ids, attention_mask):
        # Use the [CLS] token (position 0) as the sentence representation.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_output))


# ---------------------------------------------------------------------------
# DRIVE COPY HELPER
# ---------------------------------------------------------------------------

def _copy_to_drive(src: str, dst: str, retries: int = 5, base_delay: float = 3.0) -> None:
    """
    Copy a fully-written local file to dst, with exponential back-off retries.

    train_pipeline.py calls this ONCE per domain after training completes to
    copy checkpoints from local staging (/content/...) to Google Drive.
    This keeps all torch.save() calls off the Drive FUSE mount.
    """
    last_err: Exception = RuntimeError("No attempts made")
    delay = base_delay
    for attempt in range(1, retries + 1):
        try:
            shutil.copy2(src, dst)
            return
        except OSError as exc:
            last_err = exc
            logger.warning(f"Drive copy attempt {attempt}/{retries} failed ({exc}). Retrying in {delay:.0f}s…")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Failed to copy to {dst} after {retries} attempts. Last error: {last_err}") from last_err


# ---------------------------------------------------------------------------
# TRAINER
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Manages training, evaluation, checkpoint save/load, and batched prediction
    for IndoBERT clickbait classifiers.

    One ModelTrainer instance corresponds to one specialist model for one domain.
    """

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        num_labels: int = 2,
        max_length: int = 128,
        device: Optional[str] = None,
        random_seed: int = 42,
        dropout_rate: float = 0.1,
        amp_mode: Optional[str] = None,
        num_workers: Optional[int] = None,
    ):
        self.model_name   = model_name
        self.num_labels   = num_labels
        self.max_length   = max_length
        self.random_seed  = random_seed
        self.dropout_rate = dropout_rate

        # None and "auto" both mean auto-detect.
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Fail fast with a clear message instead of a cryptic error at model.to().
        self._on_cuda = str(self.device).startswith("cuda")
        if self._on_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False. "
                "Check the GPU runtime/driver, or pass --device cpu."
            )

        # Performance settings (defaults come from config.system; see config.py).
        sys_cfg          = config.system
        self.amp_dtype   = self._resolve_amp_dtype(amp_mode or sys_cfg.AMP_MODE)
        self.scaler      = self._make_grad_scaler() if (
            self.amp_dtype is not None and self.amp_dtype == torch.float16) else None
        # Worker processes only help when feeding a GPU; keep CPU runs single-process.
        self.num_workers = (sys_cfg.NUM_WORKERS if num_workers is None else num_workers) \
            if self._on_cuda else 0
        self._configure_cuda_backends(sys_cfg.ALLOW_TF32)

        self._set_seed(random_seed)
        self.tokenizer        = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model            = None
        self.optimizer        = None
        self.scheduler        = None
        self.training_history = []

        logger.info(f"ModelTrainer: {model_name}  device={self.device}")
        dbg_trainer_init(model_name=model_name, device=str(self.device),
                         max_length=max_length, num_labels=num_labels, dropout_rate=dropout_rate)

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # A100 / mixed-precision helpers
    # ------------------------------------------------------------------

    def _resolve_amp_dtype(self, mode: str):
        """Map AMP_MODE -> autocast dtype (None = full fp32). bf16 needs no GradScaler."""
        mode = str(mode).lower()
        if mode not in ("auto", "bf16", "fp16", "off"):
            raise ValueError(f"amp_mode must be one of auto|bf16|fp16|off, got '{mode}'")
        if not self._on_cuda or mode == "off":
            return None
        if mode == "bf16":
            return torch.bfloat16
        if mode == "fp16":
            return torch.float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @staticmethod
    def _make_grad_scaler():
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda")
        return torch.cuda.amp.GradScaler()

    def _configure_cuda_backends(self, allow_tf32: bool) -> None:
        """Enable TF32 tensor cores for any fp32 matmul that remains outside autocast."""
        if not self._on_cuda:
            return
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def release_memory(self) -> None:
        """Drop model/optimizer references and return cached CUDA blocks to the driver.

        Call between domains / grid-search configs so VRAM does not accumulate.
        """
        self.model = None
        self.optimizer = None
        self.scheduler = None
        gc.collect()
        if self._on_cuda:
            torch.cuda.empty_cache()

    def initialize_model(self, dropout_rate: float = 0.1) -> None:
        self.dropout_rate = dropout_rate
        self.model = IndoBERTClassifier(self.model_name, self.num_labels, dropout_rate)
        self.model.to(self.device)
        logger.info(f"Model initialised on {self.device}")

    def prepare_data_loaders(self, train_df, val_df, batch_size=32):
        train_ds = ClickbaitDataset(train_df['text'].tolist(), train_df['label'].tolist(),
                                    self.tokenizer, self.max_length, split_name="train")
        val_ds   = ClickbaitDataset(val_df['text'].tolist(),   val_df['label'].tolist(),
                                    self.tokenizer, self.max_length, split_name="val")
        loader_kw = dict(
            num_workers=self.num_workers,
            pin_memory=self._on_cuda,
            persistent_workers=self.num_workers > 0,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kw)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kw)
        logger.info(f"DataLoaders: train={len(train_loader)} batches  val={len(val_loader)} batches")
        return train_loader, val_loader

    def configure_optimizer(self, learning_rate=2e-5, weight_decay=0.01,
                             num_training_steps=1000, num_warmup_steps=None) -> None:
        if self.model is None:
            raise RuntimeError("Call initialize_model() first.")
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if num_warmup_steps is None:
            num_warmup_steps = int(0.1 * num_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        logger.info(f"Optimizer: lr={learning_rate}  warmup_steps={num_warmup_steps}")

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_fn  = nn.CrossEntropyLoss()
        # Accumulate on-device: .item() every step forces a GPU->CPU sync that
        # stalls the pipeline. Read the totals once at the end of the epoch.
        total_loss = torch.zeros((), device=self.device)
        correct    = torch.zeros((), dtype=torch.long, device=self.device)
        total      = 0
        cache_every = config.system.EMPTY_CACHE_FREQUENCY

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            input_ids      = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels         = batch['labels'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits = self.model(input_ids, attention_mask)
            logits = logits.float()               # loss / argmax always in fp32
            loss   = loss_fn(logits, labels)

            if self.scaler is not None:           # fp16 only; bf16 needs no scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.scheduler.step()

            correct    += (torch.argmax(logits, dim=1) == labels).sum()
            total      += labels.size(0)
            total_loss += loss.detach()

            # dbg_trainer_batch only logs every 50th batch (and only in debug mode);
            # gate the call so the .cpu()/.item() syncs are not paid on every step.
            if batch_idx % 50 == 0:
                dbg_trainer_batch(batch_idx=batch_idx, input_ids_shape=tuple(input_ids.shape),
                                  labels=labels.cpu(), logits=logits.detach().cpu(), loss=loss.item())

            if cache_every and self._on_cuda and (batch_idx + 1) % cache_every == 0:
                torch.cuda.empty_cache()

        metrics = {'loss': total_loss.item() / len(train_loader), 'accuracy': correct.item() / total}
        dbg_trainer_epoch_end(epoch=epoch, phase="train", metrics=metrics)
        return metrics

    def evaluate(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = torch.zeros((), device=self.device)
        pred_chunks, label_chunks = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids      = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels         = batch['labels'].to(self.device, non_blocking=True)

                with self._autocast():
                    logits = self.model(input_ids, attention_mask)
                logits = logits.float()
                total_loss += loss_fn(logits, labels)
                pred_chunks.append(torch.argmax(logits, dim=1))
                label_chunks.append(labels)

        all_preds  = torch.cat(pred_chunks).cpu().numpy()
        all_labels = torch.cat(label_chunks).cpu().numpy()
        accuracy   = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, labels=[0, 1], average=None, zero_division=0
        )

        metrics = {
            'loss':        total_loss.item() / len(val_loader),
            'accuracy':    accuracy,
            'precision_0': precision[0], 'recall_0': recall[0], 'f1_0': f1[0],
            'precision_1': precision[1], 'recall_1': recall[1], 'f1_1': f1[1],
            'macro_f1':    float(np.mean(f1)),
        }
        dbg_trainer_epoch_end(epoch=0, phase="val", metrics=metrics)
        return metrics

    def train(
        self,
        train_df,
        val_df,
        num_epochs:               int   = 4,
        batch_size:               int   = 32,
        learning_rate:            float = 2e-5,
        weight_decay:             float = 0.01,
        dropout_rate:             float = 0.1,
        early_stopping_patience:  int   = 2,
        checkpoint_dir: Optional[str]   = None,
    ) -> Dict[str, List[float]]:
        """Run the complete training loop and return the history dict."""
        logger.info("Starting training")

        self.initialize_model(dropout_rate=dropout_rate)
        train_loader, val_loader = self.prepare_data_loaders(train_df, val_df, batch_size)

        num_training_steps = len(train_loader) * num_epochs
        self.configure_optimizer(learning_rate=learning_rate, weight_decay=weight_decay,
                                  num_training_steps=num_training_steps)

        best_macro_f1   = -1
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(f"Train  loss={train_metrics['loss']:.4f}  acc={train_metrics['accuracy']:.4f}")

            val_metrics = self.evaluate(val_loader)
            logger.info(
                f"Val    loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}  "
                f"f1_0={val_metrics['f1_0']:.4f}  f1_1={val_metrics['f1_1']:.4f}  "
                f"macro_f1={val_metrics['macro_f1']:.4f}"
            )

            self.training_history.append({
                'epoch':          epoch,
                'train_loss':     train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss':       val_metrics['loss'],
                'val_accuracy':   val_metrics['accuracy'],
                'val_precision_0': val_metrics['precision_0'],
                'val_recall_0':    val_metrics['recall_0'],
                'val_f1_0':        val_metrics['f1_0'],
                'val_precision_1': val_metrics['precision_1'],
                'val_recall_1':    val_metrics['recall_1'],
                'val_f1_1':        val_metrics['f1_1'],
                'val_macro_f1':    val_metrics['macro_f1'],
            })

            if val_metrics['macro_f1'] > best_macro_f1:
                best_macro_f1    = val_metrics['macro_f1']
                patience_counter = 0
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir, "best_model")
                    logger.info(f"Checkpoint saved  macro_f1={best_macro_f1:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience {patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info("Training complete")
        return self._format_history()

    def _format_history(self) -> Dict[str, List[float]]:
        """Reformat list-of-dicts history into dict-of-lists."""
        history: Dict[str, List] = {
            'epochs': [], 'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'val_precision_0': [], 'val_recall_0': [], 'val_f1_0': [],
            'val_precision_1': [], 'val_recall_1': [], 'val_f1_1': [],
            'val_macro_f1': [],
        }
        for entry in self.training_history:
            for key in history:
                source_key = 'epoch' if key == 'epochs' else key
                history[key].append(entry[source_key])
        return history

    def save_checkpoint(self, checkpoint_dir: str, checkpoint_name: str = "checkpoint") -> str:
        """
        Save model weights, optimizer, scheduler, and history to a local directory.

        Always write to a LOCAL path (e.g. /content/...). Copying to Drive is
        the responsibility of the caller (train_pipeline.py) via _copy_to_drive.

        The file is written to a temp name and atomically renamed, so a crash or
        preemption mid-write can never leave a truncated best_model.pt behind.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
        tmp_path = checkpoint_path + ".tmp"
        torch.save({
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history':     self.training_history,
            'model_name':           self.model_name,
            'config': {
                'num_labels':  self.num_labels,
                'max_length':  self.max_length,
                'random_seed': self.random_seed,
            },
        }, tmp_path)
        os.replace(tmp_path, checkpoint_path)
        logger.info(f"Checkpoint saved → {checkpoint_path}")

        history_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_history.json")
        try:
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as exc:
            logger.warning(f"Could not save history JSON: {exc}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # Load to CPU first: the file also holds AdamW state (~3x the weights), and
        # materialising that on the GPU would spike VRAM for every specialist loaded.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if self.model is None:
            self.initialize_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.get('optimizer_state_dict') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        logger.info(f"Checkpoint loaded ← {checkpoint_path}")

    def predict(self, texts: List[str], batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return (predictions, probabilities) for a list of text strings.

        batch_size=None -> config.system.INFERENCE_BATCH_SIZE on CUDA (no gradients,
        so far larger batches fit), 32 on CPU.
        """
        if self.model is None:
            raise RuntimeError("Model not initialised or loaded.")
        self.model.eval()
        if batch_size is None:
            batch_size = config.system.INFERENCE_BATCH_SIZE if self._on_cuda else 32

        dbg_trainer_predict_start(n_texts=len(texts), batch_size=batch_size)

        # Create a dataset with dummy labels (0); labels are ignored during inference.
        dataset = ClickbaitDataset(texts, [0] * len(texts), self.tokenizer,
                                   self.max_length, split_name="predict")
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=self.num_workers, pin_memory=self._on_cuda)

        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Predicting")):
                input_ids      = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                with self._autocast():
                    logits = self.model(input_ids, attention_mask)
                logits         = logits.float()
                probs          = torch.softmax(logits, dim=1)
                preds          = torch.argmax(logits, dim=1)

                dbg_trainer_predict_batch(batch_idx=batch_idx, logits=logits.cpu(),
                                          preds=preds.cpu(), probs=probs.cpu())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        preds_arr = np.array(all_preds)
        probs_arr = np.array(all_probs)
        from collections import Counter
        dbg_trainer_predict_done(n_preds=len(preds_arr), label_counts=dict(Counter(preds_arr.tolist())))
        return preds_arr, probs_arr


# ---------------------------------------------------------------------------
# HYPERPARAMETER SEARCH  (plain function — no class needed)
# ---------------------------------------------------------------------------

def run_hyperparameter_search(
    model_name:  str,
    param_grid:  Dict[str, List[Any]],
    train_df,
    val_df,
    num_epochs:  int = 4,
    max_length:  int = 128,
    device:      Optional[str] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Grid search over param_grid.  Trains a fresh ModelTrainer for every
    combination and returns the best params, best Macro-F1, and all results.

    Parameters
    ----------
    model_name  : HuggingFace model id
    param_grid  : e.g. {'learning_rate': [1e-5, 2e-5], 'batch_size': [16, 32]}
    train_df    : training DataFrame
    val_df      : validation DataFrame
    num_epochs  : epochs per configuration
    max_length  : token sequence length
    device      : 'cuda', 'cpu', or None (auto)
    random_seed : reproducibility seed

    Returns
    -------
    dict with keys: best_params, best_f1, all_results
    """
    keys         = list(param_grid.keys())
    combinations = list(product(*param_grid.values()))
    logger.info(f"Grid search: {len(combinations)} configurations")

    all_results  = []
    best_macro_f1 = 0.0
    best_params   = None

    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        logger.info(f"Config {idx}/{len(combinations)}: {params}")

        trainer = ModelTrainer(model_name=model_name, max_length=max_length,
                               device=device, random_seed=random_seed)
        history = trainer.train(train_df=train_df, val_df=val_df,
                                num_epochs=num_epochs, **params)
        run_best_f1 = max(history['val_macro_f1'])
        trainer.release_memory()   # 12 configs x 110M params would otherwise pile up in VRAM

        all_results.append({'params': params, 'best_macro_f1': run_best_f1, 'history': history})

        if run_best_f1 > best_macro_f1:
            best_macro_f1 = run_best_f1
            best_params   = params
            logger.info(f"New best Macro-F1: {best_macro_f1:.4f}")

    logger.info(f"Grid search complete. Best Macro-F1={best_macro_f1:.4f}  params={best_params}")
    return {'best_params': best_params, 'best_f1': best_macro_f1, 'all_results': all_results}


# ---------------------------------------------------------------------------
# BACKWARD COMPATIBILITY SHIM
# ---------------------------------------------------------------------------

class HyperparameterSearch:
    """
    Thin wrapper around run_hyperparameter_search() so train_pipeline.py
    does not need to change.
    """

    def __init__(self, model_name, param_grid, max_length=128, device=None, random_seed=42):
        self.model_name  = model_name
        self.param_grid  = param_grid
        self.max_length  = max_length
        self.device      = device
        self.random_seed = random_seed
        self.results     = []

    def search(self, train_df, val_df, num_epochs=4) -> Dict[str, Any]:
        result = run_hyperparameter_search(
            model_name=self.model_name,
            param_grid=self.param_grid,
            train_df=train_df,
            val_df=val_df,
            num_epochs=num_epochs,
            max_length=self.max_length,
            device=self.device,
            random_seed=self.random_seed,
        )
        self.results = result['all_results']
        return result

    def save_results(self, output_path: str) -> None:
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Search results saved → {output_path}")
