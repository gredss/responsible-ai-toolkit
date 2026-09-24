"""
Debug Logger — Pipeline Observability Utility
=============================================

Central, configurable debug-logging facility for the IndoBERT Clickbait
Detection pipeline.  Every major processing stage in every module calls
helpers from this module to emit structured, human-readable debug output.

Usage
-----
Enable via environment variable (zero code changes required):

    DEBUG_PIPELINE=1 python train_pipeline.py --model base
    DEBUG_PIPELINE=1 python evaluate_pipeline.py --model base

Or enable programmatically before creating any pipeline object:

    from debug_logger import set_debug, dbg
    set_debug(True)

Or pass ``--debug`` on the CLI (wired in train_pipeline.py /
evaluate_pipeline.py).

When ``DEBUG_PIPELINE`` is not set (or is ``"0"``/``"false"``) and
``set_debug(False)`` was not called explicitly, **all debug helpers are
no-ops** — zero performance impact on normal runs.

Stage headers
-------------
Each stage uses a distinct bracket tag so output can be grepped:

    [DataManager]   [Tokenizer]    [ClickbaitDataset]
    [ModelTrainer]  [Evaluation]   [Perturbation]
    [Statistics]    [ErrorAnalysis]
"""

import os
import logging
import textwrap
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ── module-level toggle ────────────────────────────────────────────────────────

_DEBUG_ENABLED: bool = os.environ.get("DEBUG_PIPELINE", "0").lower() not in ("0", "false", "")

logger = logging.getLogger("debug_pipeline")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False   # don't duplicate into root logger


def set_debug(enabled: bool) -> None:
    """Enable or disable all pipeline debug output."""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = enabled


def _emit(msg: str) -> None:
    """Write *msg* to the debug logger when debug is enabled."""
    if _DEBUG_ENABLED:
        logger.debug(msg)


# ── formatting helpers ─────────────────────────────────────────────────────────

def _hdr(tag: str, title: str) -> str:
    return f"\n[{tag}] {'─'*4} {title} {'─'*(max(0, 60 - len(title)))}"


def _truncate(text: str, max_len: int = 120) -> str:
    return (text[:max_len] + "…") if len(text) > max_len else text


def _fmt_dict(d: Dict, indent: int = 4, precision: int = 4) -> str:
    pad = " " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, float):
            lines.append(f"{pad}{k}: {v:.{precision}f}")
        elif isinstance(v, (list, np.ndarray)) and hasattr(v, '__len__'):
            arr = np.array(v) if not isinstance(v, np.ndarray) else v
            if arr.ndim == 1 and len(arr) > 6:
                lines.append(f"{pad}{k}: [{arr[0]:.4g}, {arr[1]:.4g}, … {arr[-1]:.4g}]  (len={len(arr)})")
            else:
                lines.append(f"{pad}{k}: {v}")
        else:
            lines.append(f"{pad}{k}: {v}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# [DataManager] hooks
# ══════════════════════════════════════════════════════════════════════════════

def dbg_data_csv_loaded(domain: str, path: str, n_rows: int, label_counts: Dict) -> None:
    _emit(
        _hdr("DataManager", f"CSV loaded  →  domain={domain}") + "\n" +
        f"    file        : {path}\n"
        f"    rows loaded : {n_rows}\n"
        f"    labels      : {label_counts}"
    )


def dbg_data_combined(n_total: int, domains: List[str], label_counts: Dict) -> None:
    _emit(
        _hdr("DataManager", "All CSVs combined") + "\n" +
        f"    total samples : {n_total}\n"
        f"    domains       : {domains}\n"
        f"    label counts  : {label_counts}"
    )


def dbg_data_validation(domain: str, issues: Dict) -> None:
    _emit(
        _hdr("DataManager", f"Data quality check  →  {domain}") + "\n" +
        _fmt_dict(issues)
    )


def dbg_data_split(domain: str, n_train: int, n_val: int, n_test: int,
                   train_dist: Dict, val_dist: Dict, test_dist: Dict) -> None:
    _emit(
        _hdr("DataManager", f"Stratified split  →  {domain}") + "\n" +
        f"    train : {n_train:>5}   label dist: {train_dist}\n"
        f"    val   : {n_val:>5}   label dist: {val_dist}\n"
        f"    test  : {n_test:>5}   label dist: {test_dist}"
    )


def dbg_data_sample_texts(split_name: str, texts: List[str], labels: List[int],
                          n_samples: int = 3) -> None:
    _emit(_hdr("DataManager", f"Sample texts  ({split_name})"))
    for i in range(min(n_samples, len(texts))):
        label_str = "clickbait" if labels[i] == 1 else "non-clickbait"
        _emit(f"    [{i}] label={label_str}  len={len(texts[i])}")
        _emit(f"         {_truncate(texts[i])}")


# ══════════════════════════════════════════════════════════════════════════════
# [Tokenizer] hooks
# ══════════════════════════════════════════════════════════════════════════════

def dbg_tokenizer_samples(tokenizer: Any, texts: List[str], labels: List[int],
                          max_length: int, n_samples: int = 2) -> None:
    """Show token IDs, attention masks, and decoded tokens for a few examples."""
    if not _DEBUG_ENABLED:
        return
    _emit(_hdr("Tokenizer", f"Sample encodings  (max_length={max_length})"))
    for i in range(min(n_samples, len(texts))):
        enc = tokenizer(
            texts[i], max_length=max_length,
            padding="max_length", truncation=True,
            return_tensors="np"
        )
        ids       = enc["input_ids"][0]
        mask      = enc["attention_mask"][0]
        n_real    = int(mask.sum())
        decoded   = tokenizer.decode(ids[:n_real], skip_special_tokens=False)
        label_str = "clickbait" if labels[i] == 1 else "non-clickbait"
        _emit(
            f"  sample [{i}]  label={label_str}\n"
            f"    original   : {_truncate(texts[i])}\n"
            f"    input_ids  : {ids[:10].tolist()} … (total={len(ids)}, real={n_real})\n"
            f"    attn_mask  : {mask[:10].tolist()} … (n_pad={len(ids)-n_real})\n"
            f"    decoded    : {_truncate(decoded, 200)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# [ClickbaitDataset] hooks
# ══════════════════════════════════════════════════════════════════════════════

def dbg_dataset_created(split_name: str, n_samples: int, n_clickbait: int) -> None:
    _emit(
        _hdr("ClickbaitDataset", f"Dataset created  ({split_name})") + "\n" +
        f"    total     : {n_samples}\n"
        f"    clickbait : {n_clickbait}  ({n_clickbait/max(n_samples,1):.1%})\n"
        f"    non-cb    : {n_samples-n_clickbait}  ({(n_samples-n_clickbait)/max(n_samples,1):.1%})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# [ModelTrainer] hooks
# ══════════════════════════════════════════════════════════════════════════════

def dbg_trainer_init(model_name: str, device: str, max_length: int,
                     num_labels: int, dropout_rate: float) -> None:
    _emit(
        _hdr("ModelTrainer", "Initialised") + "\n" +
        f"    model      : {model_name}\n"
        f"    device     : {device}\n"
        f"    max_length : {max_length}\n"
        f"    num_labels : {num_labels}\n"
        f"    dropout    : {dropout_rate}"
    )


def dbg_trainer_batch(batch_idx: int, input_ids_shape: tuple, labels: Any,
                      logits: Any, loss: float, freq: int = 50) -> None:
    """Log model inputs / raw logits every *freq* batches."""
    if not _DEBUG_ENABLED or batch_idx % freq != 0:
        return
    import torch
    preds = torch.argmax(logits, dim=1) if hasattr(logits, 'dim') else np.argmax(logits, axis=1)
    _emit(
        _hdr("ModelTrainer", f"Batch {batch_idx}") + "\n" +
        f"    input_ids  shape : {input_ids_shape}\n"
        f"    labels           : {labels[:8].tolist()}  …\n"
        f"    logits  [0]      : {logits[0].tolist() if hasattr(logits[0],'tolist') else logits[0]}\n"
        f"    preds   [0:8]    : {preds[:8].tolist() if hasattr(preds[:8],'tolist') else list(preds[:8])}\n"
        f"    batch loss       : {loss:.6f}"
    )


def dbg_trainer_epoch_end(epoch: int, phase: str, metrics: Dict) -> None:
    _emit(
        _hdr("ModelTrainer", f"Epoch {epoch} — {phase} metrics") + "\n" +
        _fmt_dict(metrics)
    )


def dbg_trainer_predict_start(n_texts: int, batch_size: int) -> None:
    _emit(
        _hdr("ModelTrainer", "Predict") + "\n" +
        f"    texts      : {n_texts}\n"
        f"    batch_size : {batch_size}"
    )


def dbg_trainer_predict_batch(batch_idx: int, logits: Any, preds: Any,
                               probs: Any, freq: int = 50) -> None:
    if not _DEBUG_ENABLED or batch_idx % freq != 0:
        return
    _emit(
        _hdr("ModelTrainer", f"Predict  batch {batch_idx}") + "\n" +
        f"    logits[0] : {logits[0].tolist() if hasattr(logits[0],'tolist') else list(logits[0])}\n"
        f"    preds[0:4]: {preds[:4].tolist() if hasattr(preds[:4],'tolist') else list(preds[:4])}\n"
        f"    probs[0]  : {probs[0].tolist() if hasattr(probs[0],'tolist') else list(probs[0])}"
    )


def dbg_trainer_predict_done(n_preds: int, label_counts: Dict) -> None:
    _emit(
        _hdr("ModelTrainer", "Predict complete") + "\n" +
        f"    total preds   : {n_preds}\n"
        f"    label counts  : {label_counts}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# [Evaluation] hooks
# ══════════════════════════════════════════════════════════════════════════════

def dbg_eval_in_domain(domain: str, n_samples: int, metrics: Dict) -> None:
    _emit(
        _hdr("Evaluation", f"In-domain  →  {domain}") + "\n" +
        f"    samples : {n_samples}\n" +
        _fmt_dict({k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and k != 'confusion_matrix'})
    )


def dbg_eval_cross_domain(source: str, target: str, n_samples: int,
                           metrics: Dict) -> None:
    _emit(
        _hdr("Evaluation", f"Cross-domain  {source} → {target}") + "\n" +
        f"    test samples : {n_samples}\n" +
        _fmt_dict({k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and k != 'confusion_matrix'})
    )


def dbg_eval_confusion_matrix(context: str, cm: List[List[int]]) -> None:
    _emit(
        _hdr("Evaluation", f"Confusion matrix  ({context})") + "\n" +
        f"    [[TN={cm[0][0]:4d}  FP={cm[0][1]:4d}]\n"
        f"     [FN={cm[1][0]:4d}  TP={cm[1][1]:4d}]]"
    )


def dbg_eval_domain_shift(source: str, target: str, shift: Dict) -> None:
    _emit(
        _hdr("Evaluation", f"Domain shift  {source} → {target}") + "\n" +
        _fmt_dict({k: v for k, v in shift.items() if isinstance(v, (int, float))})
    )


# ══════════════════════════════════════════════════════════════════════════════
# [Perturbation] hooks
# ══════════════════════════════════════════════════════════════════════════════

def dbg_perturbation_samples(level: str, domain: str,
                              originals: List[str], perturbed: List[str],
                              n_samples: int = 3) -> None:
    _emit(_hdr("Perturbation", f"Sample pairs  level={level}  domain={domain}"))
    for i in range(min(n_samples, len(originals), len(perturbed))):
        orig_t = _truncate(originals[i])
        pert_t = _truncate(perturbed[i])
        _emit(
            f"  [{i}] original : {orig_t}\n"
            f"       perturbed: {pert_t}"
        )


def dbg_perturbation_stats(level: str, domain: str, n_texts: int,
                            char_change_mean: float, word_change_mean: float) -> None:
    _emit(
        _hdr("Perturbation", f"Stats  level={level}  domain={domain}") + "\n" +
        f"    texts              : {n_texts}\n"
        f"    mean char change   : {char_change_mean:.2%}\n"
        f"    mean word change   : {word_change_mean:.2%}"
    )


def dbg_perturbation_metrics(level: str, domain: str, metrics: Dict) -> None:
    _emit(
        _hdr("Perturbation", f"Metrics after perturbation  level={level}  domain={domain}") + "\n" +
        _fmt_dict({k: v for k, v in metrics.items()
                   if isinstance(v, (int, float)) and k != 'confusion_matrix'})
    )


def dbg_perturbation_robustness(level: str, domain: str, rob: Dict) -> None:
    _emit(
        _hdr("Perturbation", f"Robustness metrics  level={level}  domain={domain}") + "\n" +
        _fmt_dict({k: v for k, v in rob.items() if isinstance(v, (int, float))})
    )


# ══════════════════════════════════════════════════════════════════════════════
# [Statistics] hooks
# ══════════════════════════════════════════════════════════════════════════════

def dbg_stats_input(context: str, scores_a: np.ndarray, scores_b: np.ndarray,
                    label_a: str = "A", label_b: str = "B") -> None:
    _emit(
        _hdr("Statistics", f"Input scores  ({context})") + "\n" +
        f"    {label_a}  n={len(scores_a)}  mean={np.mean(scores_a):.4f}  "
        f"std={np.std(scores_a):.4f}  range=[{np.min(scores_a):.4f}, {np.max(scores_a):.4f}]\n"
        f"    {label_b}  n={len(scores_b)}  mean={np.mean(scores_b):.4f}  "
        f"std={np.std(scores_b):.4f}  range=[{np.min(scores_b):.4f}, {np.max(scores_b):.4f}]"
    )


def dbg_stats_bayesian(result: Dict) -> None:
    _emit(
        _hdr("Statistics", "Bayesian signed-rank test output") + "\n" +
        f"    n_samples        : {result.get('n_samples')}\n"
        f"    mean_difference  : {result.get('mean_difference', 0):.4f}\n"
        f"    p_value          : {result.get('p_value', 1):.4f}\n"
        f"    effect_size      : {result.get('effect_size', 0):.4f}\n"
        f"    credible_int_95  : {result.get('credible_interval_95', {})}\n"
        f"    rope_threshold   : ±{result.get('rope_threshold', 0):.4f}\n"
        f"    interpretation   : {result.get('interpretation')}\n"
        f"    conclusion       : {result.get('conclusion')}"
    )


def dbg_stats_rope(result: Dict) -> None:
    _emit(
        _hdr("Statistics", "ROPE analysis output") + "\n" +
        f"    mean_diff        : {result.get('mean_difference', 0):.4f}\n"
        f"    rope_bounds      : {result.get('rope_bounds', {})}\n"
        f"    prob_in_rope     : {result.get('probability_in_rope', 0):.4f}\n"
        f"    decision         : {result.get('decision')}\n"
        f"    interpretation   : {result.get('interpretation')}\n"
        f"    credible_ints    : {result.get('credible_intervals', {})}"
    )


def dbg_stats_bootstrap(result: Dict) -> None:
    _emit(
        _hdr("Statistics", "Bootstrap CI output") + "\n" +
        f"    observed         : {result.get('observed_statistic', 0):.4f}\n"
        f"    CI {int(result.get('confidence_level',0.95)*100)}%           : "
        f"{result.get('confidence_interval', {})}\n"
        f"    bootstrap_mean   : {result.get('bootstrap_mean', 0):.4f}\n"
        f"    bootstrap_std    : {result.get('bootstrap_std', 0):.4f}\n"
        f"    n_bootstrap      : {result.get('n_bootstrap')}"
    )


def dbg_stats_significance(test_name: str, result: Dict) -> None:
    _emit(
        _hdr("Statistics", f"{test_name} output") + "\n" +
        f"    statistic        : {result.get('statistic', 0):.4f}\n"
        f"    p_value          : {result.get('p_value', 1):.4f}\n"
        f"    is_significant   : {result.get('is_significant')}\n"
        f"    interpretation   : {result.get('interpretation')}"
    )


def dbg_stats_robustness_input(model_name: str, clean_scores: np.ndarray,
                                perturbed_dict: Dict) -> None:
    _emit(
        _hdr("Statistics", f"Robustness analysis input  model={model_name}") + "\n" +
        f"    clean  n={len(clean_scores)}  mean={np.mean(clean_scores):.4f}"
    )
    for level, scores in perturbed_dict.items():
        arr = np.array(scores)
        _emit(f"    {level:<8} n={len(arr)}  mean={np.mean(arr):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# [ErrorAnalysis] hooks
# ══════════════════════════════════════════════════════════════════════════════

def dbg_error_input(context: str, n_total: int, n_errors: int,
                    texts_sample: Optional[List[str]] = None) -> None:
    _emit(
        _hdr("ErrorAnalysis", f"Input  ({context})") + "\n" +
        f"    total samples : {n_total}\n"
        f"    errors        : {n_errors}  ({n_errors/max(n_total,1):.1%})"
    )
    if texts_sample and _DEBUG_ENABLED:
        _emit("    sample error texts:")
        for t in texts_sample[:2]:
            _emit(f"      • {_truncate(t)}")


def dbg_error_attribution(context: str, attribution: Dict) -> None:
    _emit(_hdr("ErrorAnalysis", f"Pattern attribution  ({context})"))
    ranked = sorted(attribution.items(), key=lambda x: x[1], reverse=True)
    for pattern, val in ranked:
        bar = "█" * max(0, round(abs(val) * 20))
        sign = "+" if val >= 0 else ""
        _emit(f"    {pattern:<28}  {sign}{val:>+.4f}  {bar}")


def dbg_error_summary(top_driver: Optional[str], ranked: List[str]) -> None:
    _emit(
        _hdr("ErrorAnalysis", "Global summary") + "\n" +
        f"    top error driver : {top_driver}\n"
        f"    ranked drivers   : {ranked}"
    )
