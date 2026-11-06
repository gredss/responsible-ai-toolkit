"""
Model Trainer Module for IndoBERT Clickbait Detection System

This module handles model initialization, training loop implementation,
hyperparameter grid search, checkpoint management, and evaluation utilities
for IndoBERT variants (base, large, lite).
"""

import os
import json
import shutil
import time
from itertools import product
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,BertTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

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


class ClickbaitDataset(Dataset):
    """
    PyTorch Dataset for clickbait detection.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        split_name: str = ""
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of text strings
            labels: List of binary labels (0 or 1)
            tokenizer: Tokenizer for text preprocessing
            max_length: Maximum sequence length
            split_name: Optional label for debug output (e.g. 'train', 'val')
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ── debug: dataset stats ───────────────────────────────────────────
        n_clickbait = sum(1 for l in labels if l == 1)
        dbg_dataset_created(
            split_name=split_name,
            n_samples=len(texts),
            n_clickbait=n_clickbait,
        )
        # ── debug: sample tokenisations ───────────────────────────────────
        dbg_tokenizer_samples(
            tokenizer=tokenizer,
            texts=texts,
            labels=labels,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class IndoBERTClassifier(nn.Module):
    """
    IndoBERT-based binary classifier for clickbait detection.
    """

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        num_labels: int = 2,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the classifier.

        Args:
            model_name: Name of the pretrained IndoBERT model
            num_labels: Number of output classes (2 for binary)
            dropout_rate: Dropout probability for regularization
        """
        super(IndoBERTClassifier, self).__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # Load pretrained IndoBERT
        self.bert = AutoModel.from_pretrained(model_name)

        # Get hidden size from model config
        self.hidden_size = self.bert.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        logger.info(f"Initialized {model_name} with hidden_size={self.hidden_size}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Logits [batch_size, num_labels]
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and classification layer
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def _copy_to_drive(src: str, dst: str, retries: int = 5, base_delay: float = 3.0) -> None:
    """
    Copy a fully-written local file to a destination path, with exponential
    back-off retries for transient OSError failures (e.g. Drive FUSE rate
    limits that return [Errno 95] Operation not supported).

    Design rationale
    ----------------
    src is always a local /tmp file that torch.save() has already closed and
    fully committed to local disk.  We therefore need only a reliable *copy*
    to the destination — no fsync, no kernel-buffer tricks, no magic-byte
    verification.  Those approaches are unreliable on network FUSE mounts and
    add latency without meaningful safety guarantees.

    The correct architectural pattern (enforced by train_pipeline.py) is:
      1. Train entirely with local checkpoint_dir (/content/...)
      2. Call _copy_to_drive once per domain after training completes
    This keeps Drive I/O off the hot training path entirely.

    Args:
        src: Local source path (fully written, e.g. /content/checkpoints/...)
        dst: Destination path (may be on Google Drive FUSE)
        retries: Maximum number of attempts (default 5)
        base_delay: Initial back-off delay in seconds; doubles each retry
    """
    last_err: Exception = RuntimeError("No attempts made")
    delay = base_delay
    for attempt in range(1, retries + 1):
        try:
            shutil.copy2(src, dst)
            return  # success
        except OSError as exc:
            last_err = exc
            logger.warning(
                f"Drive copy attempt {attempt}/{retries} failed "
                f"({exc}). Retrying in {delay:.0f}s…"
            )
            time.sleep(delay)
            delay *= 2  # exponential back-off
    raise RuntimeError(
        f"Failed to copy checkpoint to {dst} after {retries} attempts. "
        f"Last error: {last_err}"
    ) from last_err


class ModelTrainer:
    """
    Handles training, evaluation, and checkpoint management for IndoBERT models.
    """

    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        num_labels: int = 2,
        max_length: int = 128,
        device: Optional[str] = None,
        random_seed: int = 42,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the trainer.

        Args:
            model_name: Name of the pretrained IndoBERT model
            num_labels: Number of output classes
            max_length: Maximum sequence length
            device: Device to use ('cuda', 'cpu', or 'auto').
                    ``None`` and ``"auto"`` both trigger automatic detection.
            random_seed: Random seed for reproducibility
            dropout_rate: Default dropout rate used when the model is built
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.random_seed = random_seed
        self.dropout_rate = dropout_rate

        # Set device — None and "auto" both mean auto-detect
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Set random seeds
        self._set_seed(random_seed)

        # Initialize components
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case= True)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.training_history = []

        logger.info(f"ModelTrainer initialized for {model_name}")
        # ── debug: trainer init summary ────────────────────────────────────
        dbg_trainer_init(
            model_name=model_name,
            device=str(self.device),
            max_length=max_length,
            num_labels=num_labels,
            dropout_rate=self.dropout_rate,
        )

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def initialize_model(self, dropout_rate: float = 0.1) -> None:
        """
        Initialize the model.

        Args:
            dropout_rate: Dropout probability
        """
        self.dropout_rate = dropout_rate
        self.model = IndoBERTClassifier(
            model_name=self.model_name,
            num_labels=self.num_labels,
            dropout_rate=dropout_rate
        )
        self.model.to(self.device)
        logger.info("Model initialized and moved to device")
        # ── debug: actual dropout used ─────────────────────────────────────
        dbg_trainer_init(
            model_name=self.model_name,
            device=str(self.device),
            max_length=self.max_length,
            num_labels=self.num_labels,
            dropout_rate=dropout_rate,
        )

    def prepare_data_loaders(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare DataLoaders for training and validation.

        Args:
            train_df: Training DataFrame with 'text' and 'label' columns
            val_df: Validation DataFrame with 'text' and 'label' columns
            batch_size: Batch size for DataLoader

        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_dataset = ClickbaitDataset(
            texts=train_df['text'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            split_name="train",
        )

        val_dataset = ClickbaitDataset(
            texts=val_df['text'].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            split_name="val",
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        logger.info(f"DataLoaders prepared: train={len(train_loader)} batches, val={len(val_loader)} batches")

        return train_loader, val_loader

    def configure_optimizer(
        self,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_training_steps: int = 1000,
        num_warmup_steps: Optional[int] = None
    ) -> None:
        """
        Configure optimizer and learning rate scheduler.

        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            num_training_steps: Total number of training steps
            num_warmup_steps: Number of warmup steps (10% of total if None)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        # Configure optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Configure scheduler
        if num_warmup_steps is None:
            num_warmup_steps = int(0.1 * num_training_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        logger.info(f"Optimizer configured: lr={learning_rate}, warmup_steps={num_warmup_steps}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training DataLoader
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        loss_fn = nn.CrossEntropyLoss()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            logits = self.model(input_ids, attention_mask)

            # Calculate loss
            loss = loss_fn(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()
            self.scheduler.step()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct_predictions / total_samples
            })

            # ── debug: batch-level model inputs and outputs ────────────────
            dbg_trainer_batch(
                batch_idx=batch_idx,
                input_ids_shape=tuple(input_ids.shape),
                labels=labels.cpu(),
                logits=logits.detach().cpu(),
                loss=loss.item(),
            )

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples

        train_metrics = {'loss': avg_loss, 'accuracy': accuracy}
        # ── debug: epoch training summary ─────────────────────────────────
        dbg_trainer_epoch_end(epoch=epoch, phase="train", metrics=train_metrics)

        return train_metrics

    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation set.

        Args:
            val_loader: Validation DataLoader

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        loss_fn = nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels, all_predictions)

        precision, recall, f1, _ = precision_recall_fscore_support(
          all_labels,
          all_predictions,
          labels=[0, 1],
          average=None,
          zero_division=0
        )

        macro_f1 = np.mean(f1)

        avg_loss = total_loss / len(val_loader)

        val_metrics = {
          'loss': avg_loss,
          'accuracy': accuracy,

          # class 0
          'precision_0': precision[0],
          'recall_0': recall[0],
          'f1_0': f1[0],

          # class 1
          'precision_1': precision[1],
          'recall_1': recall[1],
          'f1_1': f1[1],

          # macro
          'macro_f1': macro_f1
        }
        # ── debug: validation epoch summary ───────────────────────────────
        dbg_trainer_epoch_end(epoch=0, phase="val", metrics=val_metrics)
        return val_metrics

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        dropout_rate: float = 0.1,
        early_stopping_patience: int = 2,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Complete training pipeline.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            dropout_rate: Dropout rate
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        logger.info("Starting training pipeline")

        # Initialize model
        self.initialize_model(dropout_rate=dropout_rate)

        # Prepare data loaders
        train_loader, val_loader = self.prepare_data_loaders(
            train_df, val_df, batch_size
        )

        # Configure optimizer
        num_training_steps = len(train_loader) * num_epochs
        self.configure_optimizer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_training_steps=num_training_steps
        )

        # Training loop
        best_macro_f1 = -1
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")

            # Evaluate
            val_metrics = self.evaluate(val_loader)
            logger.info(
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Accuracy: {val_metrics['accuracy']:.4f}, "
                f"F1_0: {val_metrics['f1_0']:.4f}, "
                f"F1_1: {val_metrics['f1_1']:.4f}, "
                f"Macro-F1: {val_metrics['macro_f1']:.4f}"
            )

            # Save history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],

                'val_precision_0': val_metrics['precision_0'],
                'val_recall_0': val_metrics['recall_0'],
                'val_f1_0': val_metrics['f1_0'],

                'val_precision_1': val_metrics['precision_1'],
                'val_recall_1': val_metrics['recall_1'],
                'val_f1_1': val_metrics['f1_1'],

                'val_macro_f1': val_metrics['macro_f1']
            })

            # Check for improvement.
            # best_macro_f1 starts at -1, so the first epoch (F1 >= 0) always
            # counts as an improvement and gets saved.
            if val_metrics['macro_f1'] > best_macro_f1:
                best_macro_f1 = val_metrics['macro_f1']
                patience_counter = 0

                # Save best model
                if checkpoint_dir:
                    # We save even if F1 is 0.0, so you have at least one checkpoint
                    self.save_checkpoint(checkpoint_dir, f"best_model")
                    logger.info(f"Model saved with Macro-F1: {best_macro_f1:.4f}")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        logger.info("Training completed")
        return self._format_history()

    def _format_history(self) -> Dict[str, List[float]]:
        """Format training history for easy access."""
        history = {
            'epochs': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],

            'val_precision_0': [],
            'val_recall_0': [],
            'val_f1_0': [],

            'val_precision_1': [],
            'val_recall_1': [],
            'val_f1_1': [],

            'val_macro_f1': []
        }

        for entry in self.training_history:
            history['epochs'].append(entry['epoch'])
            history['train_loss'].append(entry['train_loss'])
            history['train_accuracy'].append(entry['train_accuracy'])
            history['val_loss'].append(entry['val_loss'])
            history['val_accuracy'].append(entry['val_accuracy'])

            history['val_precision_0'].append(entry['val_precision_0'])
            history['val_recall_0'].append(entry['val_recall_0'])
            history['val_f1_0'].append(entry['val_f1_0'])

            history['val_precision_1'].append(entry['val_precision_1'])
            history['val_recall_1'].append(entry['val_recall_1'])
            history['val_f1_1'].append(entry['val_f1_1'])

            history['val_macro_f1'].append(entry['val_macro_f1'])

        return history

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        checkpoint_name: str = "checkpoint"
    ) -> str:
        """
        Save model checkpoint.

        checkpoint_dir must point to a local filesystem path (e.g. /content/...)
        so that torch.save() writes to fast local SSD.  Copying to Google Drive
        is the responsibility of the caller (train_pipeline.py) after training
        completes — never during the hot training loop.

        Args:
            checkpoint_dir: Local directory to save checkpoint
            checkpoint_name: Name for the checkpoint

        Returns:
            Path to saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_history': self.training_history,
            'model_name': self.model_name,
            'config': {
                'num_labels': self.num_labels,
                'max_length': self.max_length,
                'random_seed': self.random_seed
            }
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Save training history as JSON (non-critical; errors are logged, not raised)
        history_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_history.json")
        try:
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as hist_err:
            logger.warning(f"Could not save training history JSON: {hist_err}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()

        # Load state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if checkpoint.get('optimizer_state_dict') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.training_history = checkpoint.get('training_history', [])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for prediction

        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized or loaded")

        self.model.eval()

        # ── debug: predict call summary ────────────────────────────────────
        dbg_trainer_predict_start(n_texts=len(texts), batch_size=batch_size)

        # Create dummy labels for dataset
        dummy_labels = [0] * len(texts)
        dataset = ClickbaitDataset(
            texts=texts,
            labels=dummy_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            split_name="predict",
        )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Predicting")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                # ── debug: raw logits + predictions ───────────────────────
                dbg_trainer_predict_batch(
                    batch_idx=batch_idx,
                    logits=logits.cpu(),
                    preds=predictions.cpu(),
                    probs=probabilities.cpu(),
                )

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_preds_arr = np.array(all_predictions)
        all_probs_arr = np.array(all_probabilities)

        # ── debug: predict done ────────────────────────────────────────────
        from collections import Counter
        dbg_trainer_predict_done(
            n_preds=len(all_preds_arr),
            label_counts=dict(Counter(all_preds_arr.tolist())),
        )

        return all_preds_arr, all_probs_arr


class HyperparameterSearch:
    """
    Grid search for hyperparameter optimization.
    """

    def __init__(
        self,
        model_name: str,
        param_grid: Dict[str, List[Any]],
        max_length: int = 128,
        device: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize hyperparameter search.

        Args:
            model_name: Name of the model to tune
            param_grid: Dictionary of hyperparameters to search
            max_length: Maximum sequence length
            device: Device to use
            random_seed: Random seed
        """
        self.model_name = model_name
        self.param_grid = param_grid
        self.max_length = max_length
        self.device = device
        self.random_seed = random_seed
        self.results = []

        logger.info(f"HyperparameterSearch initialized for {model_name}")

    def search(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_epochs: int = 4
    ) -> Dict[str, Any]:
        """
        Perform grid search.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            num_epochs: Number of epochs per configuration

        Returns:
            Best hyperparameters and results
        """
        # Generate all combinations
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(product(*values))

        logger.info(f"Testing {len(combinations)} hyperparameter combinations")

        best_macro_f1 = 0
        best_params = None

        for idx, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            logger.info(f"Configuration {idx}/{len(combinations)}: {params}")

            # Train with current parameters
            trainer = ModelTrainer(
                model_name=self.model_name,
                max_length=self.max_length,
                device=self.device,
                random_seed=self.random_seed
            )

            history = trainer.train(
                train_df=train_df,
                val_df=val_df,
                num_epochs=num_epochs,
                **params
            )

            # Get best F1 from this run
            best_run_macro_f1  = max(history['val_macro_f1'])

            # Store results
            result = {
                'params': params,
                'best_macro_f1': best_run_macro_f1 ,
                'history': history
            }
            self.results.append(result)

            # Update best
            if best_run_macro_f1  > best_macro_f1:
                best_macro_f1 = best_run_macro_f1
                best_params = params
                logger.info(f"New best F1: {best_macro_f1:.4f}")

        logger.info(f"Grid search completed. Best F1: {best_macro_f1:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return {
            'best_params': best_params,
            'best_f1': best_macro_f1,
            'all_results': self.results
        }

    def save_results(self, output_path: str) -> None:
        """Save search results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Search results saved to {output_path}")
