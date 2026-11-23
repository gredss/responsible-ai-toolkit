"""
Evaluation Engine for IndoBERT Clickbait Detection System

Covers metric calculation, in-domain evaluation, cross-domain evaluation (5×5
matrix), perturbation robustness testing, and results serialisation.

Public functions (module-level, no class needed):
    calculate_metrics               — accuracy, F1, MCC, ROC-AUC, confusion matrix
    calculate_robustness_metrics    — absolute and relative F1 drop
    calculate_domain_shift_metrics  — Source Drop (SD) and Target Drop (TD)
    calculate_prediction_flip       — flip counts and rates between clean / perturbed
    evaluate_in_domain              — score one specialist on its own test split
    evaluate_all_cross_domain_combinations — score all 5×5 source→target pairs
    evaluate_with_perturbation      — generate & score a fresh perturbation
    evaluate_existing_perturbation  — score an already-generated perturbed DataFrame

Orchestrator (class with real state):
    EvaluationEngine  — owns the dict of specialist trainers, the perturbation
                        engine, output paths, and runs the full 4-phase pipeline.
"""

import ast
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef,
)
import logging

from debug_logger import (
    dbg_eval_in_domain, dbg_eval_cross_domain, dbg_eval_confusion_matrix,
    dbg_eval_domain_shift, dbg_perturbation_metrics,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# One perturbation mechanism (synonym substitution), three intensity levels.
PERTURBATION_TYPES  = ['semantic']
PERTURBATION_LEVELS = ['low', 'medium', 'high']


# =============================================================================
# SECTION 1 — METRIC FUNCTIONS
# =============================================================================

def calculate_metrics(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    """
    Calculate accuracy, macro / per-class precision-recall-F1, MCC, and
    optionally ROC-AUC.  Also appends the confusion matrix.

    Parameters
    ----------
    y_true  : array of true binary labels (0 or 1)
    y_pred  : array of predicted labels
    y_proba : optional array of prediction probabilities (shape N×2 or N)

    Returns
    -------
    dict with keys:
        accuracy, macro_precision, macro_recall, macro_f1,
        precision_0/1, recall_0/1, f1_0/1, mcc, roc_auc, confusion_matrix
    """
    precision = precision_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    recall    = recall_score(   y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    f1        = f1_score(       y_true, y_pred, labels=[0, 1], average=None, zero_division=0)

    metrics = {
        'accuracy':        accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0),
        'macro_recall':    recall_score(   y_true, y_pred, labels=[0, 1], average='macro', zero_division=0),
        'macro_f1':        f1_score(       y_true, y_pred, labels=[0, 1], average='macro', zero_division=0),
        'precision_0': precision[0], 'recall_0': recall[0], 'f1_0': f1[0],
        'precision_1': precision[1], 'recall_1': recall[1], 'f1_1': f1[1],
        'mcc':     matthews_corrcoef(y_true, y_pred),
        'roc_auc': None,
    }

    if y_proba is not None:
        try:
            probs = y_proba[:, 1] if len(np.array(y_proba).shape) == 2 else y_proba
            metrics['roc_auc'] = roc_auc_score(y_true, probs)
        except Exception:
            metrics['roc_auc'] = None

    # labels=[0, 1] keeps the matrix 2x2 even when a (small, matched) subset holds one class only.
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return metrics


def calculate_robustness_metrics(clean_metrics: Dict, perturbed_metrics: Dict) -> Dict[str, float]:
    """
    Compute absolute and relative F1 / accuracy drop from clean to perturbed.

    Returns keys like macro_f1_drop, macro_f1_drop_pct for each of:
    accuracy, macro_precision, macro_recall, macro_f1.
    """
    robustness = {}
    for metric_name in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']:
        if metric_name not in clean_metrics or metric_name not in perturbed_metrics:
            continue
        clean_val    = clean_metrics[metric_name]
        perturbed_val = perturbed_metrics[metric_name]
        robustness[f'{metric_name}_drop'] = clean_val - perturbed_val
        robustness[f'{metric_name}_drop_pct'] = (
            (clean_val - perturbed_val) / clean_val * 100 if clean_val > 0 else 0.0
        )
    return robustness


def calculate_domain_shift_metrics(
    source_metrics:          Dict[str, float],
    target_metrics:          Dict[str, float],
    in_domain_target_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute Source Drop (SD) and Target Drop (TD) for each of:
    accuracy, macro_precision, macro_recall, macro_f1.

    SD = how much the source specialist loses when tested on the target domain.
    TD = how much the target-domain specialist loses vs the cross-domain model.
    """
    shift_metrics = {}
    for metric_name in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']:
        # Source Drop
        if metric_name in source_metrics and metric_name in target_metrics:
            source_val      = source_metrics[metric_name]
            cross_val       = target_metrics[metric_name]
            shift_metrics[f'sd_{metric_name}']     = source_val - cross_val
            shift_metrics[f'sd_{metric_name}_pct'] = (
                (source_val - cross_val) / source_val * 100 if source_val > 0 else 0.0
            )
        # Target Drop
        if metric_name in in_domain_target_metrics and metric_name in target_metrics:
            in_domain_val = in_domain_target_metrics[metric_name]
            cross_val     = target_metrics[metric_name]
            shift_metrics[f'td_{metric_name}']     = in_domain_val - cross_val
            shift_metrics[f'td_{metric_name}_pct'] = (
                (in_domain_val - cross_val) / in_domain_val * 100 if in_domain_val > 0 else 0.0
            )
    return shift_metrics


def calculate_prediction_flip(
    clean_predictions,
    perturbed_predictions,
    true_labels,
    successful_perturbations=None,
) -> Dict[str, Any]:
    """
    Count and rate prediction flips between clean and perturbed predictions.

    A flip occurs when clean_prediction != perturbed_prediction.
    Reports:
      - overall flip count / rate
      - 0→1 and 1→0 flip direction counts
      - correct→incorrect and incorrect→correct transition rates
      - flip rate among successfully perturbed samples (if supplied)
    """
    clean_preds     = np.asarray(clean_predictions)
    perturbed_preds = np.asarray(perturbed_predictions)
    true_labels     = np.asarray(true_labels)

    if not (len(clean_preds) == len(perturbed_preds) == len(true_labels)):
        raise ValueError("Prediction and label arrays must have the same length.")

    n = len(clean_preds)

    flip_mask        = clean_preds != perturbed_preds
    flip_count       = int(flip_mask.sum())
    flip_rate        = flip_count / n if n > 0 else 0.0
    flip_0_to_1      = int(((clean_preds == 0) & (perturbed_preds == 1)).sum())
    flip_1_to_0      = int(((clean_preds == 1) & (perturbed_preds == 0)).sum())

    clean_correct    = clean_preds == true_labels
    perturbed_correct = perturbed_preds == true_labels
    correct_to_incorrect = int((clean_correct & ~perturbed_correct).sum())
    incorrect_to_correct = int((~clean_correct & perturbed_correct).sum())
    n_clean_correct      = int(clean_correct.sum())
    n_clean_incorrect    = int((~clean_correct).sum())

    # Conditional rates
    cti_rate = correct_to_incorrect / n if n > 0 else 0.0
    itc_rate = incorrect_to_correct / n if n > 0 else 0.0
    cti_rate_among_correct   = correct_to_incorrect / n_clean_correct   if n_clean_correct   > 0 else 0.0
    itc_rate_among_incorrect = incorrect_to_correct / n_clean_incorrect if n_clean_incorrect > 0 else 0.0

    # Rates among successfully perturbed samples only
    successful_count           = None
    flip_count_successful_only = None
    flip_rate_successful_only  = None
    successful_rate            = None
    if successful_perturbations is not None:
        succ_mask = np.asarray(successful_perturbations, dtype=bool)
        if len(succ_mask) != n:
            raise ValueError("successful_perturbations must have the same length as predictions.")
        successful_count           = int(succ_mask.sum())
        flip_count_successful_only = int((flip_mask & succ_mask).sum())
        flip_rate_successful_only  = flip_count_successful_only / successful_count if successful_count > 0 else 0.0
        successful_rate            = successful_count / n if n > 0 else 0.0

    return {
        "total_samples":                       n,
        "successful_perturbations":            successful_count,
        "successful_perturbation_rate":        successful_rate,
        "successful_perturbation_rate_pct":    successful_rate * 100 if successful_rate is not None else None,
        "flip_count":                          flip_count,
        "flip_rate":                           flip_rate,
        "flip_rate_pct":                       flip_rate * 100,
        "flip_0_to_1_count":                   flip_0_to_1,
        "flip_1_to_0_count":                   flip_1_to_0,
        "flip_0_to_1_rate":                    flip_0_to_1 / n if n > 0 else 0.0,
        "flip_1_to_0_rate":                    flip_1_to_0 / n if n > 0 else 0.0,
        "flip_count_successful_only":          flip_count_successful_only,
        "flip_rate_successful_only":           flip_rate_successful_only,
        "flip_rate_successful_only_pct":       flip_rate_successful_only * 100 if flip_rate_successful_only is not None else None,
        "flips": {
            "correct_to_incorrect":                            correct_to_incorrect,
            "incorrect_to_correct":                            incorrect_to_correct,
            "correct_to_incorrect_rate":                       cti_rate,
            "correct_to_incorrect_rate_pct":                   cti_rate * 100,
            "incorrect_to_correct_rate":                       itc_rate,
            "incorrect_to_correct_rate_pct":                   itc_rate * 100,
            "clean_correct_count":                             n_clean_correct,
            "clean_incorrect_count":                           n_clean_incorrect,
            "correct_to_incorrect_rate_among_correct":         cti_rate_among_correct,
            "correct_to_incorrect_rate_among_correct_pct":     cti_rate_among_correct * 100,
            "incorrect_to_correct_rate_among_incorrect":       itc_rate_among_incorrect,
            "incorrect_to_correct_rate_among_incorrect_pct":   itc_rate_among_incorrect * 100,
        },
    }


# =============================================================================
# SECTION 2 — EVALUATION FUNCTIONS
# =============================================================================

def evaluate_in_domain(trainer, test_df: pd.DataFrame, domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate a trained ModelTrainer on its own domain's test data.

    Parameters
    ----------
    trainer  : ModelTrainer with a loaded checkpoint
    test_df  : DataFrame with 'text' and 'label' columns
    domain   : domain name for logging (optional)

    Returns a dict with keys: domain, num_samples, metrics, predictions,
    probabilities, true_labels.
    """
    domain_str = f" ({domain})" if domain else ""
    logger.info(f"In-domain evaluation{domain_str}: {len(test_df)} samples")

    y_true         = test_df['label'].values
    y_pred, y_proba = trainer.predict(test_df['text'].tolist())
    metrics        = calculate_metrics(y_true, y_pred, y_proba)

    logger.info(f"In-domain Macro-F1={metrics['macro_f1']:.4f}  F1_0={metrics['f1_0']:.4f}  F1_1={metrics['f1_1']:.4f}")
    dbg_eval_in_domain(domain=domain or "?", n_samples=len(test_df), metrics=metrics)
    if metrics.get('confusion_matrix'):
        dbg_eval_confusion_matrix(context=f"in-domain/{domain}", cm=metrics['confusion_matrix'])

    return {
        'domain':        domain,
        'num_samples':   len(test_df),
        'metrics':       metrics,
        'predictions':   y_pred.tolist(),
        'probabilities': y_proba.tolist(),
        'true_labels':   y_true.tolist(),
    }


def evaluate_all_cross_domain_combinations(
    model_trainers: Dict[str, Any],
    test_data:      Dict[str, pd.DataFrame],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Evaluate all source→target domain pairs (5×5 matrix).

    For each source domain we pick the specialist trainer for that source and
    run it on the test data of every target domain.  Source domains with no
    loaded checkpoint are skipped with a warning.

    Returns a dict keyed by (source_domain, target_domain) tuples.
    """
    domains = list(test_data.keys())
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for source_domain in domains:
        if source_domain not in model_trainers:
            logger.warning(
                f"Skipping cross-domain for source '{source_domain}': no specialist loaded."
            )
            continue

        trainer = model_trainers[source_domain]

        for target_domain in domains:
            if target_domain not in test_data:
                continue
            logger.info(f"Cross-domain: {source_domain} → {target_domain}")

            target_df       = test_data[target_domain]
            y_true          = target_df['label'].values
            y_pred, y_proba = trainer.predict(target_df['text'].tolist())
            metrics         = calculate_metrics(y_true, y_pred, y_proba)

            dbg_eval_cross_domain(source=source_domain, target=target_domain,
                                  n_samples=len(target_df), metrics=metrics)
            if metrics.get('confusion_matrix'):
                dbg_eval_confusion_matrix(
                    context=f"cross-domain/{source_domain}->{target_domain}",
                    cm=metrics['confusion_matrix'],
                )

            results[(source_domain, target_domain)] = {
                'source_domain': source_domain,
                'target_domain': target_domain,
                'num_samples':   len(target_df),
                'metrics':       metrics,
                'predictions':   y_pred.tolist(),
                'probabilities': y_proba.tolist(),
                'true_labels':   y_true.tolist(),
            }

    return results


def evaluate_with_perturbation(
    trainer,
    perturbation_engine,
    test_df:           pd.DataFrame,
    perturbation_level: str,
    domain:            Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a fresh perturbation, run the model, and return metrics.

    This path is used when perturbed data has not been pre-generated.
    Failed perturbations are retained in the DataFrame and counted separately.
    """
    domain_str = f" ({domain})" if domain else ""
    logger.info(f"Evaluating {perturbation_level}-level perturbation{domain_str}")

    perturbed_df    = perturbation_engine.apply_to_dataframe(test_df.copy(), text_column='text', level=perturbation_level)
    y_true          = perturbed_df['label'].values
    y_pred, y_proba = trainer.predict(perturbed_df['text'].tolist())
    metrics         = calculate_metrics(y_true, y_pred, y_proba)

    dbg_perturbation_metrics(level=perturbation_level, domain=domain or "?", metrics=metrics)
    logger.info(f"{perturbation_level.capitalize()} Macro-F1={metrics['macro_f1']:.4f}")

    return {
        'domain':          domain,
        'perturbation_level': perturbation_level,
        'num_samples':     len(perturbed_df),
        'metrics':         metrics,
        'predictions':     y_pred.tolist(),
        'probabilities':   y_proba.tolist(),
        'true_labels':     y_true.tolist(),
    }


def evaluate_existing_perturbation(
    trainer,
    perturbed_df:      pd.DataFrame,
    clean_results:     Dict[str, Any],
    perturbation_level: str,
    domain:            Optional[str] = None,
) -> Dict[str, Any]:
    """
    Score an already-generated perturbed DataFrame (no new perturbation is applied).

    Also computes robustness drop vs clean_results, word-replacement cosine
    statistics, and prediction-flip counts.

    Parameters
    ----------
    trainer           : ModelTrainer with a loaded checkpoint
    perturbed_df      : output of PerturbationEngine.apply_to_dataframe()
    clean_results     : dict returned by evaluate_in_domain() on the same subset
    perturbation_level : 'low', 'medium', or 'high'
    domain            : domain name for logging

    Returns a dict with: domain, perturbation_level, num_samples, metrics,
    robustness_metrics, prediction_flip, perturbation_statistics, predictions,
    probabilities, true_labels.
    """
    logger.info(f"Evaluating existing {perturbation_level}-level perturbation ({domain})")

    y_true          = perturbed_df['label'].values
    y_pred, y_proba = trainer.predict(perturbed_df['text'].tolist())
    metrics         = calculate_metrics(y_true, y_pred, y_proba)
    robustness      = calculate_robustness_metrics(clean_results['metrics'], metrics)

    if 'perturbation_success' not in perturbed_df.columns:
        raise ValueError("evaluate_existing_perturbation: 'perturbation_success' column is missing.")

    prediction_flip = calculate_prediction_flip(
        clean_predictions=clean_results['predictions'],
        perturbed_predictions=y_pred.tolist(),
        true_labels=y_true.tolist(),
        successful_perturbations=perturbed_df['perturbation_success'].tolist(),
    )

    # Collect per-word cosine similarity statistics from the 'replacements' column.
    # The column is saved to CSV, so its values may come back as strings.
    similarities = []
    if 'replacements' in perturbed_df.columns:
        for replacements in perturbed_df['replacements']:
            if isinstance(replacements, str):
                try:
                    replacements = ast.literal_eval(replacements)
                except (ValueError, SyntaxError):
                    continue
            if not isinstance(replacements, list):
                continue
            for replacement in replacements:
                if not isinstance(replacement, dict):
                    continue
                similarity = replacement.get('word_cosine_similarity')
                if similarity is None:
                    continue
                try:
                    similarities.append(float(similarity))
                except (TypeError, ValueError):
                    continue

    if similarities:
        similarity_statistics = {
            'count':  len(similarities),
            'mean':   float(np.mean(similarities)),
            'median': float(np.median(similarities)),
            'std':    float(np.std(similarities)),
            'min':    float(np.min(similarities)),
            'max':    float(np.max(similarities)),
        }
    else:
        similarity_statistics = {'count': 0, 'mean': None, 'median': None, 'std': None, 'min': None, 'max': None}

    perturbation_stats = {
        'mean_word_change_rate': float(perturbed_df['actual_ratio_all_words'].mean())
            if 'actual_ratio_all_words' in perturbed_df.columns else None,
        'mean_similarity_score': float(perturbed_df['similarity_score_mean'].dropna().mean())
            if 'similarity_score_mean' in perturbed_df.columns and perturbed_df['similarity_score_mean'].notna().any() else None,
        'min_similarity_score':  float(perturbed_df['similarity_score_min'].dropna().min())
            if 'similarity_score_min' in perturbed_df.columns and perturbed_df['similarity_score_min'].notna().any() else None,
        'max_similarity_score':  float(perturbed_df['similarity_score_max'].dropna().max())
            if 'similarity_score_max' in perturbed_df.columns and perturbed_df['similarity_score_max'].notna().any() else None,
        'semantic_similarity': similarity_statistics,
    }

    return {
        'domain':                  domain,
        'perturbation_level':      perturbation_level,
        'num_samples':             len(perturbed_df),
        'perturbation_statistics': perturbation_stats,
        'metrics':                 metrics,
        'robustness_metrics':      robustness,
        'prediction_flip':         prediction_flip,
        'predictions':             y_pred.tolist(),
        'probabilities':           y_proba.tolist(),
        'true_labels':             y_true.tolist(),
    }


# =============================================================================
# SECTION 3 — ORCHESTRATOR
# =============================================================================

class EvaluationEngine:
    """
    Orchestrates the full 4-phase evaluation pipeline.

    State owned by this class:
        model_trainers          — {domain: ModelTrainer} for every loaded specialist
        perturbation_engine     — PerturbationEngine instance
        output_dir              — where complete_evaluation.json is saved
        perturbation_output_dir — where per-domain perturbed CSVs are saved

    Phases run by run_complete_evaluation():
        1. In-domain evaluation (5 domains)
        2. Cross-domain evaluation (5×5 matrix) + domain-shift metrics
        3. Generate all perturbations once (saves CSVs)
        4. In-domain perturbation evaluation (common-subset, apple-to-apple)
        5. Cross-domain × perturbation interaction (same common-subset logic)
    """

    def __init__(
        self,
        model_trainers:          Dict[str, Any],
        perturbation_engine,
        output_dir:              str           = "evaluation_results",
        perturbation_output_dir: Optional[str] = None,
    ):
        self.model_trainers          = model_trainers
        self.perturbation_engine     = perturbation_engine
        self.output_dir              = output_dir
        self.perturbation_output_dir = perturbation_output_dir

        os.makedirs(output_dir, exist_ok=True)
        if perturbation_output_dir is not None:
            os.makedirs(perturbation_output_dir, exist_ok=True)

        logger.info(f"EvaluationEngine ready. Results → {output_dir}")
        if perturbation_output_dir:
            logger.info(f"Perturbation data → {perturbation_output_dir}")

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _successful_ids(perturbed_df: pd.DataFrame) -> set:
        """IDs of rows where perturbation_success == True."""
        if 'id' not in perturbed_df.columns:
            raise ValueError("'id' column missing from perturbed DataFrame.")
        if 'perturbation_success' not in perturbed_df.columns:
            raise ValueError("'perturbation_success' column missing from perturbed DataFrame.")
        successful = perturbed_df['perturbation_success'].astype(bool)
        return set(perturbed_df.loc[successful, 'id'].tolist())

    @staticmethod
    def _mean_achieved_intensity(perturbed_df: pd.DataFrame) -> Optional[float]:
        """Mean word-change ratio over successfully perturbed rows only."""
        successful = perturbed_df['perturbation_success'].astype(bool)
        rows = perturbed_df.loc[successful]
        if len(rows) == 0:
            return None
        if 'actual_ratio_all_words' in rows.columns:
            return float(rows['actual_ratio_all_words'].astype(float).mean())
        return None

    # ------------------------------------------------------------------
    # PHASE 3 — GENERATE PERTURBATIONS
    # ------------------------------------------------------------------

    def generate_all_perturbations(
        self,
        test_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Apply the perturbation engine to every domain × level combination once.

        Returns perturbed_test_data[domain]['semantic'][level] = perturbed_df.
        Each perturbed DataFrame is also saved to
            <perturbation_output_dir>/<domain>/semantic/<level>.csv
        """
        logger.info("Generating all perturbations ONCE")
        perturbed_test_data: Dict[str, Dict[str, Any]] = {}

        for domain, test_df in test_data.items():
            logger.info(f"  Perturbing domain: {domain}")
            perturbed_test_data[domain] = {}

            domain_output_dir = os.path.join(self.perturbation_output_dir, str(domain).strip().lower())
            os.makedirs(domain_output_dir, exist_ok=True)

            for ptype in PERTURBATION_TYPES:
                perturbed_test_data[domain][ptype] = {}

                for level in PERTURBATION_LEVELS:
                    logger.info(f"    {domain} - {ptype} - {level}")
                    perturbed_df = self.perturbation_engine.apply_to_dataframe(
                        test_df.copy(), text_column='text', level=level
                    )

                    # Validate that the engine produced the three success-gate columns.
                    required_cols = ['is_same_as_original', 'similarity_in_range', 'perturbation_in_range']
                    missing = [c for c in required_cols if c not in perturbed_df.columns]
                    if missing:
                        raise ValueError(f"Perturbation validation columns missing: {missing}")

                    # A perturbation succeeds only when all three gates hold:
                    #   1. The headline actually changed.
                    #   2. Every replacement's cosine is in [0.80, 0.95].
                    #   3. The word-change count hit the requested target (10/20/30%).
                    perturbed_df['perturbation_success'] = (
                        (~perturbed_df['is_same_as_original'].astype(bool))
                        & perturbed_df['similarity_in_range'].astype(bool)
                        & perturbed_df['perturbation_in_range'].astype(bool)
                    )
                    perturbed_df['perturbation_type'] = ptype

                    print(f"\nperturbation_success ({ptype} - {level}):")
                    print(perturbed_df['perturbation_success'].value_counts(dropna=False))

                    perturbed_test_data[domain][ptype][level] = perturbed_df

                    type_output_dir = os.path.join(domain_output_dir, ptype)
                    os.makedirs(type_output_dir, exist_ok=True)
                    output_path = os.path.join(type_output_dir, f"{level}.csv")
                    perturbed_df.to_csv(output_path, index=False)
                    logger.info(f"    Saved: {output_path}")

        logger.info("All perturbations generated and saved.")
        return perturbed_test_data

    # ------------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------------

    def run_complete_evaluation(
        self,
        test_data: Dict[str, pd.DataFrame],
        include_perturbations: bool = True,
    ):
        """
        Run all four evaluation phases and return a nested results dict.

        Result structure:
            results['in_domain'][domain]                           — phase 1
            results['cross_domain'][(source, target)]             — phase 2
            results['perturbation'][domain][ptype]['clean']        — phase 4 baseline
            results['perturbation'][domain][ptype][level]          — phase 4 perturbed
            results['cross_domain_perturbation'][key]['clean']     — phase 5 baseline
            results['cross_domain_perturbation'][key][ptype][level]— phase 5 perturbed
        """
        logger.info("Starting complete evaluation pipeline")

        results = {
            'timestamp': datetime.now().isoformat(),
            'domains':   list(test_data.keys()),
            'in_domain': {}, 'cross_domain': {},
            'perturbation': {}, 'cross_domain_perturbation': {},
        }

        # ── Phase 1: In-domain ────────────────────────────────────────────
        logger.info("Phase 1: In-domain evaluation")
        for domain, test_df in test_data.items():
            if domain in self.model_trainers:
                results['in_domain'][domain] = evaluate_in_domain(
                    self.model_trainers[domain], test_df, domain
                )

        # ── Phase 2: Cross-domain + domain-shift ──────────────────────────
        logger.info("Phase 2: Cross-domain evaluation")
        results['cross_domain'] = evaluate_all_cross_domain_combinations(
            self.model_trainers, test_data
        )
        logger.info("Phase 2.1: Domain-shift metrics")
        in_domain_data = results['in_domain']
        for (source, target), cross_res in results['cross_domain'].items():
            if source == target:
                continue
            if source in in_domain_data and target in in_domain_data:
                shift = calculate_domain_shift_metrics(
                    in_domain_data[source]['metrics'],
                    cross_res['metrics'],
                    in_domain_data[target]['metrics'],
                )
                cross_res['domain_shift'] = shift
                dbg_eval_domain_shift(source=source, target=target, shift=shift)
            else:
                logger.warning(f"Missing in-domain data for shift: {source} → {target}")

        # ── Phase 3: Generate perturbations ───────────────────────────────
        perturbed_test_data = None
        if include_perturbations:
            perturbed_test_data = self.generate_all_perturbations(test_data)

        # ── Phase 4: In-domain perturbation evaluation ────────────────────
        # For each domain, only the samples that were successfully perturbed at
        # ALL three levels are scored — the common-subset (apple-to-apple) design.
        if include_perturbations:
            logger.info("Phase 4: In-domain perturbation evaluation")
            import pattern_analysis  # lazy import

            for domain in test_data:
                if domain not in self.model_trainers:
                    continue

                trainer        = self.model_trainers[domain]
                domain_test_df = test_data[domain]
                domain_perturbation: Dict[str, Any] = {}

                for ptype in PERTURBATION_TYPES:
                    domain_perturbation[ptype] = {}
                    level_frames = {
                        level: perturbed_test_data[domain][ptype][level]
                        for level in PERTURBATION_LEVELS
                    }

                    # Log per-level success rates.
                    for level in PERTURBATION_LEVELS:
                        df = level_frames[level]
                        success_ids   = self._successful_ids(df)
                        success_rate  = len(success_ids) / len(df) if len(df) else 0.0
                        mean_intensity = self._mean_achieved_intensity(df)
                        logger.info(
                            "[%s | %s | %s] success: %d/%d (%.2f%%)  mean_intensity=%s",
                            domain, ptype, level,
                            len(success_ids), len(df), success_rate * 100,
                            f"{mean_intensity:.4f}" if mean_intensity is not None else "n/a",
                        )

                    # Warn about duplicate IDs (diagnostic only).
                    for level, count in pattern_analysis.count_duplicate_ids_per_level(level_frames).items():
                        if count > 0:
                            logger.warning("[%s | %s | %s] %d duplicate id(s) detected.",
                                           domain, ptype, level, count)

                    # Keep only IDs that succeeded at all three levels.
                    common_ids = pattern_analysis.align_common_successful_ids(
                        level_frames, drop_duplicate_ids=False, sort_result=False
                    )
                    logger.info("[%s | %s] common successful IDs: %d", domain, ptype, len(common_ids))

                    # Subset clean baseline — same IDs as the perturbed levels.
                    subset_clean_df      = domain_test_df[domain_test_df['id'].isin(common_ids)].copy()
                    subset_clean_results = evaluate_in_domain(trainer, subset_clean_df, domain)
                    subset_clean_results['common_ids_count'] = len(common_ids)

                    domain_perturbation[ptype]['clean'] = subset_clean_results
                    if 'clean' not in domain_perturbation:
                        domain_perturbation['clean'] = subset_clean_results

                    print(f"\n[{domain} | {ptype}] Common subset sample counts:")
                    print(f"    Clean  : {len(subset_clean_df)}")
                    for level in PERTURBATION_LEVELS:
                        n = int(level_frames[level]['id'].isin(common_ids).sum())
                        print(f"    {level.capitalize():<7}: {n}")

                    # Score each perturbed level against the same clean baseline.
                    for level in PERTURBATION_LEVELS:
                        filtered_df = level_frames[level][level_frames[level]['id'].isin(common_ids)].copy()
                        level_results = evaluate_existing_perturbation(
                            trainer, filtered_df, subset_clean_results, level, domain
                        )
                        level_results['common_ids_count'] = len(common_ids)
                        domain_perturbation[ptype][level] = level_results

                results['perturbation'][domain] = domain_perturbation

        # ── Phase 5: Cross-domain × perturbation ──────────────────────────
        if include_perturbations:
            cross_pert_results = self.run_cross_domain_with_perturbations(
                test_data, perturbed_test_data
            )
            results['cross_domain_perturbation'] = cross_pert_results['cross_domain_perturbation']

        logger.info("Complete evaluation pipeline finished")
        self.save_results(results)
        return results

    # ------------------------------------------------------------------
    # PHASE 5 — CROSS-DOMAIN × PERTURBATION
    # ------------------------------------------------------------------

    def run_cross_domain_with_perturbations(
        self,
        test_data:          Dict[str, pd.DataFrame],
        perturbed_test_data: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Score every source→target pair at Clean / Low / Medium / High using the
        same common-subset alignment as phase 4.  The common IDs are computed
        once per (target_domain, ptype) so they are not recomputed 5× per target.
        """
        import pattern_analysis  # lazy import

        logger.info("Cross-domain perturbation evaluation (common-subset)")

        results = {'timestamp': datetime.now().isoformat(), 'cross_domain_perturbation': {}}
        domains = list(test_data.keys())

        # Pre-compute common IDs per (target_domain, ptype).
        common_ids_cache: Dict[str, Dict[str, Any]] = {}
        for target_domain in domains:
            if target_domain not in test_data:
                continue
            common_ids_cache[target_domain] = {}
            for ptype in PERTURBATION_TYPES:
                level_frames = {
                    level: perturbed_test_data[target_domain][ptype][level]
                    for level in PERTURBATION_LEVELS
                }
                common_ids = pattern_analysis.align_common_successful_ids(
                    level_frames, drop_duplicate_ids=False, sort_result=False
                )
                common_ids_cache[target_domain][ptype] = common_ids
                logger.info("[cross | target=%s | %s] common subset: %d",
                            target_domain, ptype, len(common_ids))

        for source_domain in domains:
            if source_domain not in self.model_trainers:
                continue
            trainer = self.model_trainers[source_domain]

            for target_domain in domains:
                if target_domain not in test_data:
                    continue

                key = f"{source_domain}_to_{target_domain}"
                results['cross_domain_perturbation'][key] = {}

                for ptype in PERTURBATION_TYPES:
                    results['cross_domain_perturbation'][key][ptype] = {}
                    common_ids = common_ids_cache[target_domain][ptype]

                    # Subset clean baseline.
                    target_df      = test_data[target_domain]
                    subset_tgt_df  = target_df[target_df['id'].isin(common_ids)].copy()
                    y_true_clean   = subset_tgt_df['label'].values
                    y_pred_clean, y_proba_clean = trainer.predict(subset_tgt_df['text'].tolist())
                    clean_metrics  = calculate_metrics(y_true_clean, y_pred_clean, y_proba_clean)
                    clean_preds    = y_pred_clean.tolist()

                    subset_clean_result = {
                        'metrics':          clean_metrics,
                        'predictions':      clean_preds,
                        'probabilities':    y_proba_clean.tolist(),
                        'true_labels':      y_true_clean.tolist(),
                        'common_ids_count': len(common_ids),
                    }
                    results['cross_domain_perturbation'][key]['clean']          = subset_clean_result
                    results['cross_domain_perturbation'][key][ptype]['clean']   = subset_clean_result

                    print(f"\n[{source_domain}→{target_domain} | {ptype}] Common subset:")
                    print(f"    Clean  : {len(subset_tgt_df)}")

                    # Score each perturbed level.
                    for level in PERTURBATION_LEVELS:
                        perturbed_df_full = perturbed_test_data[target_domain][ptype][level]
                        perturbed_df      = perturbed_df_full[perturbed_df_full['id'].isin(common_ids)].copy()
                        print(f"    {level.capitalize():<7}: {len(perturbed_df)}")

                        # Collect similarity stats from DataFrame attrs when available.
                        raw_sims = perturbed_df_full.attrs.get('replacement_similarities', [])
                        if raw_sims:
                            sim_stats = {
                                'count': len(raw_sims), 'mean': float(np.mean(raw_sims)),
                                'median': float(np.median(raw_sims)), 'std': float(np.std(raw_sims)),
                                'min': float(np.min(raw_sims)), 'max': float(np.max(raw_sims)),
                            }
                        else:
                            sim_stats = {'count': 0, 'mean': None, 'median': None,
                                         'std': None, 'min': None, 'max': None}

                        perturbation_statistics = {
                            **perturbed_df_full.attrs.get('perturbation_stats', {}),
                            'semantic_similarity': sim_stats,
                        }

                        y_true          = perturbed_df['label'].values
                        y_pred, y_proba = trainer.predict(perturbed_df['text'].tolist())
                        perturbed_metrics = calculate_metrics(y_true, y_pred, y_proba)
                        robustness        = calculate_robustness_metrics(clean_metrics, perturbed_metrics)

                        if 'perturbation_success' not in perturbed_df.columns:
                            raise ValueError(
                                f"'perturbation_success' missing for {target_domain}-{ptype}-{level}"
                            )

                        prediction_flip = calculate_prediction_flip(
                            clean_predictions=clean_preds,
                            perturbed_predictions=y_pred.tolist(),
                            true_labels=y_true.tolist(),
                            successful_perturbations=perturbed_df['perturbation_success'].tolist(),
                        )

                        results['cross_domain_perturbation'][key][ptype][level] = {
                            'metrics':                 perturbed_metrics,
                            'robustness_metrics':      robustness,
                            'prediction_flip':         prediction_flip,
                            'perturbation_statistics': perturbation_statistics,
                            'predictions':             y_pred.tolist(),
                            'probabilities':           y_proba.tolist(),
                            'true_labels':             y_true.tolist(),
                            'common_ids_count':        len(common_ids),
                        }
                        logger.info("%s→%s (%s-%s): Macro-F1=%.4f  n=%d",
                                    source_domain, target_domain, ptype, level,
                                    perturbed_metrics['macro_f1'], len(perturbed_df))

        logger.info("Cross-domain perturbation evaluation complete.")
        return results

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def save_results(self, results: Dict[str, Any], filename: str = 'evaluation_results.json') -> str:
        """Serialise results to JSON, converting tuple keys and numpy types."""
        from utils import make_json_serializable
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(make_json_serializable(results), f, indent=2)
        logger.info(f"Results saved → {output_path}")
        return output_path
