"""
Pattern-Based Robustness Analysis for IndoBERT Clickbait Detection
==================================================================

This module analyses how an *already-trained* IndoBERT specialist behaves when
the same headline is perturbed by SYNONYM replacement at three intensity
levels - Low, Medium, High. It does NOT train or modify the model.

The flow is deliberately linear and procedural so each step can be explained
one at a time:

    1.  Read the per-domain Low / Medium / High *semantic* (synonym)
        perturbation CSVs written by ``EvaluationEngine.generate_all_perturbations``.
        We keep four columns only: id, label, perturbed_text, perturbation_success.
    2.  REPORT A - per-level perturbation validity: for each level independently,
        how many samples succeeded vs failed. Failures are reported, not hidden.
    3.  Align by the original headline ``id``: keep only the ids that were
        SUCCESSFULLY perturbed at ALL THREE levels (an intersection of the
        per-level successful-id sets). Matching is always by id, never by row
        position.
    4.  Classify each matched id into exactly one of five exact-string-equality
        patterns (L=M=H, M=H!=L, L=M!=H, L=H!=M, L!=M!=H).
    5.  REPORT B - per-pattern distribution and, when a predictor is supplied,
        the model's Accuracy / Macro-F1 per (domain, pattern, level).

Design notes
------------
*   The only part that needs the model is the prediction step, which accepts a
    ``predict_fn`` callable, so the analysis logic can be tested without a GPU
    or checkpoint.
*   Exact string equality is used for pattern assignment. Different strings do
    NOT necessarily carry different meaning or monotonically-increasing
    semantic change.
*   Nothing here overwrites the source datasets, checkpoints, or existing
    results.

Label mapping: 0 = non-clickbait, 1 = clickbait.
"""

from __future__ import annotations

import os
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# The three perturbation intensity levels, always processed in this order.
LEVELS: Tuple[str, str, str] = ("low", "medium", "high")

# The five exact-string-equality patterns.
PATTERN_LMH_EQUAL = "L=M=H"          # all three identical
PATTERN_MH_EQ_L_DIFF = "M=H!=L"      # medium == high, low differs
PATTERN_LM_EQ_H_DIFF = "L=M!=H"      # low == medium, high differs
PATTERN_LH_EQ_M_DIFF = "L=H!=M"      # low == high, medium differs
PATTERN_ALL_DIFF = "L!=M!=H"         # all three pairwise different

PATTERNS: Tuple[str, ...] = (
    PATTERN_LMH_EQUAL,
    PATTERN_MH_EQ_L_DIFF,
    PATTERN_LM_EQ_H_DIFF,
    PATTERN_LH_EQ_M_DIFF,
    PATTERN_ALL_DIFF,
)

# Canonical domain -> raw-file mapping (mirrors config.DataConfig.DOMAIN_FILES).
DOMAIN_FILES: Dict[str, str] = {
    "Technology": "technology.csv",
    "Politics": "politic.csv",
    "Health": "health.csv",
    "Sport": "sport.csv",
    "Education": "education.csv",
}


# ---------------------------------------------------------------------------
# PATTERN CLASSIFICATION  (exact string equality across the three levels)
# ---------------------------------------------------------------------------

def classify_pattern(low_text: str, medium_text: str, high_text: str) -> str:
    """
    Assign exactly one of the five exact-string-equality patterns.

    Pairwise equality is computed on the exact strings. For the all-different
    pattern we explicitly verify low != medium, low != high and medium != high.
    """
    low_equals_medium = (low_text == medium_text)
    low_equals_high = (low_text == high_text)
    medium_equals_high = (medium_text == high_text)

    # All three identical.
    if low_equals_medium and medium_equals_high:
        return PATTERN_LMH_EQUAL

    # Medium and High identical, Low differs.
    if medium_equals_high and not low_equals_medium:
        return PATTERN_MH_EQ_L_DIFF

    # Low and Medium identical, High differs.
    if low_equals_medium and not medium_equals_high:
        return PATTERN_LM_EQ_H_DIFF

    # Low and High identical, Medium differs.
    if low_equals_high and not low_equals_medium:
        return PATTERN_LH_EQ_M_DIFF

    # None of the pairs are equal: all three strings are pairwise different.
    assert not low_equals_medium
    assert not low_equals_high
    assert not medium_equals_high
    return PATTERN_ALL_DIFF



# ---------------------------------------------------------------------------
# METRIC HELPERS  (accuracy / macro-F1 for one group of predictions)
# ---------------------------------------------------------------------------

# A predict function maps a list of texts -> list of int labels (0/1), in the
# SAME order as the input. This indirection keeps the analysis logic
# independent of torch so it can be tested with a simple stub.
PredictFn = Callable[[List[str]], Sequence[int]]


def compute_macro_f1(true_labels: Sequence[int],
                     predicted_labels: Sequence[int]) -> float:
    """Macro-F1 with labels=[0, 1] and zero_division=0 (sklearn)."""
    from sklearn.metrics import f1_score
    macro_f1_value = f1_score(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        average="macro",
        zero_division=0,
    )
    return float(macro_f1_value)


def compute_accuracy(true_labels: Sequence[int],
                     predicted_labels: Sequence[int]) -> float:
    """Plain accuracy (sklearn). Returns NaN for an empty group."""
    from sklearn.metrics import accuracy_score
    if len(true_labels) == 0:
        return float("nan")
    accuracy_value = accuracy_score(true_labels, predicted_labels)
    return float(accuracy_value)



# ---------------------------------------------------------------------------
# PATH RESOLUTION  (maps the generate_all_perturbations layout to level paths)
# ---------------------------------------------------------------------------

def resolve_semantic_level_paths(
    results_root: str,
    domain: str,
    ptype: str = "semantic",
) -> Dict[str, str]:
    """
    Build the {level: path} mapping for the SYNONYM (semantic) perturbation
    CSVs produced by ``EvaluationEngine.generate_all_perturbations``.

    Layout: ``<results_root>/<domain-lowercased>/<ptype>/<level>.csv``.

    ``results_root`` is typically ``/content/drive/MyDrive/thesis/results/<model>``
    on Colab, but any directory following the same layout works.
    """
    domain_folder_name = str(domain).strip().lower()
    domain_directory = os.path.join(results_root, domain_folder_name, ptype)

    level_paths: Dict[str, str] = {}
    for level in LEVELS:
        level_paths[level] = os.path.join(domain_directory, f"{level}.csv")
    return level_paths


def make_checkpoint_predict_fn(
    checkpoint_dir: str,
    model_name: str,
    domain: str,
    full_model_name: str = "indobenchmark/indobert-base-p1",
    max_length: int = 128,
    device: Optional[str] = None,
    random_seed: int = 42,
    batch_size: int = 32,
) -> PredictFn:
    """
    Build a ``predict_fn`` backed by the EXISTING trained specialist checkpoint.

    This reuses the project's ``ModelTrainer`` unchanged - it loads
    ``<checkpoint_dir>/<model_name>/<Domain.capitalize()>/best_model.pt`` and
    returns a callable ``texts -> predicted int labels`` using the exact
    tokenizer / preprocessing / inference path already in the codebase.

    The model is loaded ONCE and captured in the closure, so repeated calls
    (one per level) do not reload weights. No training occurs.

    Import of ``model_trainer`` is deferred so that the pure analysis functions
    in this module can be used without torch installed.
    """
    from model_trainer import ModelTrainer  # deferred import (needs torch)

    capitalised_domain = str(domain).capitalize()
    checkpoint_path = os.path.join(
        checkpoint_dir, model_name, capitalised_domain, "best_model.pt"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found for {model_name}/{domain}: {checkpoint_path}"
        )

    trainer = ModelTrainer(
        model_name=full_model_name,
        max_length=max_length,
        device=device,
        random_seed=random_seed,
    )
    trainer.load_checkpoint(checkpoint_path)
    logger.info("Loaded checkpoint for predict_fn: %s", checkpoint_path)

    def predict_texts(texts: List[str]) -> List[int]:
        if len(texts) == 0:
            return []

        predicted_labels, _probabilities = trainer.predict(
            texts, batch_size=batch_size
        )

        predicted_labels_flat = np.asarray(predicted_labels).ravel().tolist()
        predicted_labels_as_int = []
        for predicted_label in predicted_labels_flat:
            predicted_labels_as_int.append(int(predicted_label))
        return predicted_labels_as_int

    return predict_texts


# ---------------------------------------------------------------------------
# SIMPLE ENGINE -> ANALYSIS HANDOFF  (one plain DataFrame per level)
# ---------------------------------------------------------------------------
#
# The perturbation engine already saves everything we need to a plain CSV at
#     <results_root>/<domain>/semantic/<level>.csv
# with explicit column names. For this analysis we only read four of them:
#
#     id                    - original headline id (stable sample identifier)
#     label                 - ground-truth label (0 or 1)
#     perturbed_text        - the perturbed headline for this level
#     perturbation_success  - True only when the row was changed AND its
#                             word-cosine is in [0.80, 0.95] AND it hit its
#                             token target (words_changed == target_count)
#
# That is the entire handoff: pandas reads the CSV, we keep those four columns,
# and every later step works on a standard DataFrame. No wrapper classes, no
# column-guessing, no intermediate "tidy" objects.

SUCCESS_ANALYSIS_COLUMNS = ["id", "label", "perturbed_text", "perturbation_success"]


def read_level_csv(path: str) -> pd.DataFrame:
    """
    Read ONE perturbed-level CSV and return a plain DataFrame holding just the
    four columns the success-based analysis needs.

    Failed perturbations are KEPT here (Report A needs them); they are only
    dropped later, for Report B, by the id-alignment step.
    """
    frame = pd.read_csv(path)

    # Expected columns are known and fixed; stop at the first one that's absent.
    for column_name in SUCCESS_ANALYSIS_COLUMNS:
        if column_name not in frame.columns:
            raise KeyError(
                f"{path}: missing expected column '{column_name}'. "
                f"Available columns: {list(frame.columns)}"
            )

    slim_frame = frame[SUCCESS_ANALYSIS_COLUMNS].copy()
    # Make the success flag a plain True/False column (treat missing as False).
    slim_frame["perturbation_success"] = (
        slim_frame["perturbation_success"].fillna(False).astype(bool)
    )
    return slim_frame


# ---------------------------------------------------------------------------
# STANDARDIZED "COMMON ID" ALIGNMENT  (one method for the whole project)
# ---------------------------------------------------------------------------
#
# This is the single, defensible way the project aligns samples across the
# Low / Medium / High perturbation levels. Every module (this analysis AND the
# evaluation engine) calls the SAME function so the methodology is identical
# everywhere and can be explained in one place during the defence.
#
# The rule, stated plainly:
#   1. At each level, a sample "counts" only if it was SUCCESSFULLY perturbed
#      (perturbation_success == True).
#   2. We then keep only the ids that succeeded at ALL THREE levels
#      (an intersection of the per-level successful-id sets).
#   3. Matching is always by the original headline ``id`` - never by row
#      position - so the same sample is compared across levels.
#
# Two behaviours are deliberately OPTIONAL and OFF by default, so this function
# reproduces the evaluation engine's existing results exactly:
#   * drop_duplicate_ids  - if True, an id that appears more than once inside a
#                           single level is removed entirely (we cannot know
#                           which duplicate row is correct). The engine does
#                           NOT do this today, so it defaults to False.
#   * sort_result         - if True, return a deterministically ordered list
#                           (sorted by str). The engine uses the ids as a set
#                           (membership only), so ordering does not affect its
#                           numbers; it defaults to False here and the engine
#                           ignores order anyway.


def successful_ids_for_level(level_frame: pd.DataFrame) -> set:
    """
    Return the SET of ids that were successfully perturbed in ONE level frame.

    This mirrors the evaluation engine's original ``_successful_ids`` exactly:
    take the rows whose ``perturbation_success`` is True and collect their
    ``id`` values into a set. No duplicate handling, no ordering - identical to
    the engine's current behaviour.
    """
    if "id" not in level_frame.columns:
        raise ValueError("Cannot align by id: 'id' column is missing.")
    if "perturbation_success" not in level_frame.columns:
        raise ValueError(
            "Cannot align by success: 'perturbation_success' column is missing."
        )

    success_flag = level_frame["perturbation_success"].astype(bool)
    successful_rows = level_frame.loc[success_flag]
    return set(successful_rows["id"].tolist())


def count_duplicate_ids_per_level(
    level_frame_by_level: Dict[str, pd.DataFrame],
) -> Dict[str, int]:
    """
    Report (do NOT drop) how many ids are duplicated within each level.

    This is purely diagnostic: it lets us SEE whether duplicate ids exist and
    how many, so their impact can be assessed separately before deciding to
    change any behaviour. It never alters the data.

    Returns a mapping ``{level: number_of_ids_that_appear_more_than_once}``.
    """
    duplicate_count_by_level: Dict[str, int] = {}
    for level in LEVELS:
        level_frame = level_frame_by_level[level]
        # value_counts() gives how many times each id appears; an id is a
        # duplicate when that count is greater than 1.
        id_value_counts = level_frame["id"].value_counts()
        number_of_duplicated_ids = int((id_value_counts > 1).sum())
        duplicate_count_by_level[level] = number_of_duplicated_ids
    return duplicate_count_by_level


def align_common_successful_ids(
    level_frame_by_level: Dict[str, pd.DataFrame],
    drop_duplicate_ids: bool = False,
    sort_result: bool = False,
):
    """
    The standardized "common id" alignment used across the whole project.

    Steps (explicit, so each can be explained):
      1. For every level, collect the set of successfully-perturbed ids.
      2. Optionally drop ids that are duplicated within a level (OFF by default
         to preserve the evaluation engine's existing results).
      3. Intersect the per-level sets so only ids successful at ALL THREE
         levels remain.
      4. Return either a set (default, engine-compatible) or, if requested, a
         deterministically sorted list.

    Parameters
    ----------
    level_frame_by_level:
        Mapping ``{"low": df, "medium": df, "high": df}``. Each df must have an
        ``id`` column and a ``perturbation_success`` column.
    drop_duplicate_ids:
        If True, remove any id that appears more than once inside a single
        level before intersecting. Default False (engine-preserving).
    sort_result:
        If True, return a list ordered by ``sorted(..., key=str)``. Default
        False, which returns a plain set (what the engine uses).
    """
    successful_ids_by_level: Dict[str, set] = {}
    for level in LEVELS:
        level_frame = level_frame_by_level[level]

        if drop_duplicate_ids:
            # Remove every copy of any id that is duplicated within this level.
            deduplicated_frame = level_frame.drop_duplicates(
                subset="id", keep=False
            )
            successful_ids_by_level[level] = successful_ids_for_level(
                deduplicated_frame
            )
        else:
            successful_ids_by_level[level] = successful_ids_for_level(level_frame)

    # Intersect across the three levels with an explicit loop.
    common_successful_ids = successful_ids_by_level[LEVELS[0]]
    for level in LEVELS[1:]:
        common_successful_ids = common_successful_ids & successful_ids_by_level[level]

    if sort_result:
        return sorted(common_successful_ids, key=str)
    return common_successful_ids


# ---------------------------------------------------------------------------
# REPORT A - PERTURBATION VALIDITY  (independent per level, failures kept)
# ---------------------------------------------------------------------------

def print_report_a_validity(validity_table: pd.DataFrame) -> None:
    """
    Print REPORT A directly in the notebook: per-level perturbation validity,
    with success rates AND failure counts. Failures are shown, never hidden.
    """
    print("=" * 78)
    print("REPORT A - PERTURBATION VALIDITY (independent per severity level)")
    print("=" * 78)
    print(
        "A sample is VALID at a level when it was changed, its word-cosine is "
        "in [0.80, 0.95], and it hit its token target (10/20/30%)."
    )
    print(
        "Failures are reported here, NOT discarded. They are only excluded from "
        "Report B's matched subset."
    )
    print("")

    if validity_table.empty:
        print("   (no validity data available)")
        print("")
        return

    domains = sorted(validity_table["domain"].unique().tolist())
    for domain in domains:
        print(f"[{domain}]")
        domain_rows = validity_table[validity_table["domain"] == domain]
        for level in LEVELS:
            level_rows = domain_rows[domain_rows["level"] == level]
            if level_rows.empty:
                continue
            level_row = level_rows.iloc[0]

            number_of_successful_rows = int(level_row["n_success"])
            number_of_total_rows = int(level_row["n_total"])
            number_of_failed_rows = int(level_row["n_fail"])
            success_rate_percentage = float(level_row["success_rate"]) * 100.0

            print(
                f"   {level.capitalize():<7} "
                f"success = {number_of_successful_rows:>4}/{number_of_total_rows:<4} "
                f"({success_rate_percentage:5.1f}%)   "
                f"failed = {number_of_failed_rows:>4}"
            )
        print("")


# ---------------------------------------------------------------------------
# REPORT B - PATTERN DISTRIBUTION & EVALUATION  (matched successful subset)
# ---------------------------------------------------------------------------

def build_success_pattern_distribution(classified_by_domain: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Per-(domain, pattern) distribution over the MATCHED SUCCESSFUL subset.

    Every one of the five patterns is listed for every domain, even at zero, so
    the table shape is stable. There is no Failed category here on purpose.

    Columns: domain, pattern, n, n_label0, n_label1, pct_of_matched,
             denom_matched
    """
    distribution_rows: List[dict] = []
    domains = sorted(classified_by_domain.keys())
    for domain in domains:
        classified_table = classified_by_domain[domain]
        number_of_matched_samples = int(len(classified_table))

        for pattern in PATTERNS:
            if classified_table.empty:
                pattern_rows = classified_table
            else:
                pattern_rows = classified_table[classified_table["pattern"] == pattern]

            count = int(len(pattern_rows))
            if count > 0:
                count_label0 = int((pattern_rows["label"] == 0).sum())
                count_label1 = int((pattern_rows["label"] == 1).sum())
            else:
                count_label0 = 0
                count_label1 = 0

            if number_of_matched_samples > 0:
                percentage_of_matched = 100.0 * count / number_of_matched_samples
            else:
                percentage_of_matched = np.nan

            distribution_rows.append({
                "domain": domain,
                "pattern": pattern,
                "n": count,
                "n_label0": count_label0,
                "n_label1": count_label1,
                "pct_of_matched": percentage_of_matched,
                "denom_matched": number_of_matched_samples,
            })

    return pd.DataFrame(distribution_rows)


def evaluate_success_patterns(
    classified_by_domain: Dict[str, pd.DataFrame],
    predict_fn_factory: Optional[Callable[[str], PredictFn]],
) -> pd.DataFrame:
    """
    Compute Accuracy and Macro-F1 per (domain, pattern, level) on the MATCHED
    SUCCESSFUL subset only.

    For each domain we build one predict_fn (bound to that domain's specialist)
    and score the Low / Medium / High perturbed text of every matched sample.
    The ground-truth label is identical across levels for a given id.

    Returns columns:
        domain, pattern, level, n, n_label0, n_label1, accuracy, macro_f1
    If ``predict_fn_factory`` is None, an empty frame is returned (Report B then
    shows distribution only).
    """
    metric_columns = [
        "domain", "pattern", "level", "n",
        "n_label0", "n_label1", "accuracy", "macro_f1",
    ]
    if predict_fn_factory is None:
        return pd.DataFrame(columns=metric_columns)

    metric_rows: List[dict] = []
    domains = sorted(classified_by_domain.keys())
    for domain in domains:
        classified_table = classified_by_domain[domain]
        if classified_table.empty:
            continue

        # Build this domain's predictor once, then reuse it for all three levels.
        predict_fn = predict_fn_factory(domain)

        # Predict every level's text for the whole matched subset of this domain.
        predictions_by_level: Dict[str, List[int]] = {}
        for level in LEVELS:
            text_column_name = f"text_{level}"
            texts_for_level = classified_table[text_column_name].astype(str).tolist()
            predictions_for_level = list(predict_fn(texts_for_level))
            if len(predictions_for_level) != len(texts_for_level):
                raise RuntimeError(
                    f"predict_fn returned {len(predictions_for_level)} predictions "
                    f"for {len(texts_for_level)} texts at level '{level}' "
                    f"(domain={domain})"
                )
            predictions_by_level[level] = predictions_for_level

        # Attach predictions back to the table so we can slice per pattern.
        scored_table = classified_table.copy().reset_index(drop=True)
        for level in LEVELS:
            scored_table[f"pred_{level}"] = np.asarray(
                predictions_by_level[level], dtype=int
            )

        # Now compute metrics per pattern and per level.
        for pattern in PATTERNS:
            pattern_group = scored_table[scored_table["pattern"] == pattern]
            group_size = int(len(pattern_group))
            true_labels = pattern_group["label"].astype(int).tolist()
            count_label0 = int((pattern_group["label"] == 0).sum())
            count_label1 = int((pattern_group["label"] == 1).sum())

            for level in LEVELS:
                if group_size == 0:
                    accuracy_value = float("nan")
                    macro_f1_value = float("nan")
                else:
                    predicted_labels = pattern_group[f"pred_{level}"].astype(int).tolist()
                    accuracy_value = compute_accuracy(true_labels, predicted_labels)
                    macro_f1_value = compute_macro_f1(true_labels, predicted_labels)

                metric_rows.append({
                    "domain": domain,
                    "pattern": pattern,
                    "level": level,
                    "n": group_size,
                    "n_label0": count_label0,
                    "n_label1": count_label1,
                    "accuracy": accuracy_value,
                    "macro_f1": macro_f1_value,
                })

    return pd.DataFrame(metric_rows)


def print_report_b_patterns(
    success_distribution: pd.DataFrame,
    success_metrics: pd.DataFrame,
) -> None:
    """
    Print REPORT B directly in the notebook: pattern distribution and model
    metrics, computed STRICTLY on the matched successful subset of ids.
    """
    print("=" * 78)
    print("REPORT B - PATTERN DISTRIBUTION & EVALUATION (matched successful subset)")
    print("=" * 78)
    print(
        "Only ids successfully perturbed at ALL THREE levels are used here, "
        "aligned by the original headline id."
    )
    print("")

    if success_distribution.empty:
        print("   (no matched successful samples -> nothing to report)")
        print("")
        return

    # --- B.1 distribution -------------------------------------------------
    print("B.1  Text-equality pattern distribution (per domain)")
    domains = sorted(success_distribution["domain"].unique().tolist())
    for domain in domains:
        domain_distribution = success_distribution[
            success_distribution["domain"] == domain
        ]
        if len(domain_distribution) > 0:
            denominator_matched = int(domain_distribution["denom_matched"].iloc[0])
        else:
            denominator_matched = 0
        print(f"   [{domain}] matched successful samples = {denominator_matched}")

        for _, distribution_row in domain_distribution.iterrows():
            percentage_of_matched = distribution_row["pct_of_matched"]
            if pd.notna(percentage_of_matched):
                percentage_text = f"{percentage_of_matched:5.1f}%"
            else:
                percentage_text = "   NA"
            print(
                f"       {distribution_row['pattern']:<9} "
                f"n={int(distribution_row['n']):>4}  ({percentage_text})  "
                f"label0={int(distribution_row['n_label0'])}, "
                f"label1={int(distribution_row['n_label1'])}"
            )
        print("")

    # --- B.2 metrics ------------------------------------------------------
    print("B.2  Model metrics on the matched subset (Accuracy / Macro-F1)")
    if success_metrics.empty:
        print(
            "   (model evaluation was skipped -> distribution only. "
            "Provide a predict_fn_factory to populate metrics.)"
        )
        print("")
        return

    metric_domains = sorted(success_metrics["domain"].unique().tolist())
    for domain in metric_domains:
        print(f"   [{domain}]")
        domain_metrics = success_metrics[success_metrics["domain"] == domain]
        for pattern in PATTERNS:
            pattern_metrics_rows = domain_metrics[domain_metrics["pattern"] == pattern]
            if pattern_metrics_rows.empty:
                continue

            group_size = int(pattern_metrics_rows["n"].iloc[0])
            if group_size == 0:
                # Skip patterns that were not observed in this domain.
                continue

            accuracy_parts = []
            macro_f1_parts = []
            for level in LEVELS:
                level_rows = pattern_metrics_rows[pattern_metrics_rows["level"] == level]
                level_initial = level[:1].upper()
                if level_rows.empty:
                    accuracy_parts.append(f"{level_initial}=NA")
                    macro_f1_parts.append(f"{level_initial}=NA")
                    continue

                accuracy_value = float(level_rows["accuracy"].iloc[0])
                macro_f1_value = float(level_rows["macro_f1"].iloc[0])
                accuracy_parts.append(f"{level_initial}={accuracy_value:.3f}")
                macro_f1_parts.append(f"{level_initial}={macro_f1_value:.3f}")

            print(
                f"       {pattern:<9} n={group_size:>4}  "
                f"Acc[{' '.join(accuracy_parts)}]  "
                f"MacroF1[{' '.join(macro_f1_parts)}]"
            )
        print("")


# ---------------------------------------------------------------------------
# ORCHESTRATOR - ONE LINEAR FLOW FOR THE NOTEBOOK
# ---------------------------------------------------------------------------

def run_success_based_pattern_analysis(
    results_root: str,
    predict_fn_factory: Optional[Callable[[str], PredictFn]] = None,
    domains: Optional[Sequence[str]] = None,
    ptype: str = "semantic",
    print_reports: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    End-to-end, success-based pattern analysis designed to be called from a
    notebook cell. It flows linearly and prints the two required reports:

        Load data  ->  Report A (validity)  ->  success-align ids
                    ->  assign patterns  ->  evaluate model  ->  Report B

    Parameters
    ----------
    results_root:
        Directory containing ``<domain>/<ptype>/<level>.csv`` produced by
        ``EvaluationEngine.generate_all_perturbations``.
    predict_fn_factory:
        Callable ``domain -> predict_fn``. If None, Report B shows distribution
        only (no model metrics). Use :func:`make_checkpoint_predict_fn` bound
        per domain, or reuse already-loaded specialist trainers.
    domains:
        Which domains to analyse. Defaults to every canonical domain that has a
        ``<domain>/<ptype>`` directory on disk.
    ptype:
        Perturbation family folder name. Always ``"semantic"`` (synonym). No
        typo logic exists in this pipeline.
    output_dir:
        If given, the three result tables are written there as CSV files
        (``validity_by_level.csv``, ``pattern_distribution.csv``,
        ``pattern_metrics.csv``) so a notebook can later read and plot them.
        If ``None`` (default), nothing is written.

    Returns
    -------
    dict with every computed table so the notebook can inspect or save them:
        validity_table, success_distribution, success_metrics,
        classified_by_domain, common_ids_by_domain
    """
    # --- decide which domains to process ---------------------------------
    if domains is None:
        discovered_domains = []
        for canonical_domain in DOMAIN_FILES:
            domain_folder_name = canonical_domain.lower()
            candidate_directory = os.path.join(results_root, domain_folder_name, ptype)
            if os.path.isdir(candidate_directory):
                discovered_domains.append(canonical_domain)
        domains = discovered_domains
        if len(domains) == 0:
            raise FileNotFoundError(
                f"No <domain>/{ptype} directories found under {results_root}."
            )
    logger.info("Success-based analysis domains: %s", list(domains))

    validity_rows: List[dict] = []
    classified_by_domain: Dict[str, pd.DataFrame] = {}
    common_ids_by_domain: Dict[str, List] = {}

    # Process one domain at a time. Everything below works on plain DataFrames
    # read straight from the engine's CSVs - no wrapper objects in between.
    for domain in domains:
        level_paths = resolve_semantic_level_paths(results_root, domain, ptype)

        # Read the three level CSVs into plain DataFrames.
        frame_by_level: Dict[str, pd.DataFrame] = {}
        for level in LEVELS:
            frame_by_level[level] = read_level_csv(level_paths[level])

        # -- Report A input: count validity per level (failures are kept) --
        # For each level we simply count how many rows succeeded vs failed,
        # using the engine's own perturbation_success flag.
        for level in LEVELS:
            level_frame = frame_by_level[level]
            success_flag = level_frame["perturbation_success"]

            number_of_total_rows = int(len(level_frame))
            number_of_successful_rows = int(success_flag.sum())
            number_of_failed_rows = number_of_total_rows - number_of_successful_rows
            if number_of_total_rows > 0:
                success_rate = number_of_successful_rows / number_of_total_rows
            else:
                success_rate = 0.0

            validity_rows.append({
                "domain": domain,
                "level": level,
                "n_total": number_of_total_rows,
                "n_success": number_of_successful_rows,
                "n_fail": number_of_failed_rows,
                "success_rate": success_rate,
            })

        # -- Report B input: success-based id alignment --------------------
        # Use the SINGLE standardized alignment helper. For this analysis we
        # turn ON duplicate dropping and deterministic ordering, so one id
        # cannot corrupt the aligned subset and runs are reproducible.
        ordered_common_ids = align_common_successful_ids(
            frame_by_level,
            drop_duplicate_ids=True,
            sort_result=True,
        )
        common_ids_by_domain[domain] = ordered_common_ids

        # Index each level by id so we can line up the same id across levels.
        indexed_by_level: Dict[str, pd.DataFrame] = {}
        for level in LEVELS:
            deduplicated = frame_by_level[level].drop_duplicates(
                subset="id", keep=False
            )
            indexed_by_level[level] = deduplicated.set_index("id")

        # Build one aligned+classified row per common id.
        classified_rows: List[dict] = []
        for sample_id in ordered_common_ids:
            low_row = indexed_by_level["low"].loc[sample_id]
            medium_row = indexed_by_level["medium"].loc[sample_id]
            high_row = indexed_by_level["high"].loc[sample_id]

            low_text = str(low_row["perturbed_text"])
            medium_text = str(medium_row["perturbed_text"])
            high_text = str(high_row["perturbed_text"])

            # Exact-string pattern assignment (unchanged methodology).
            pattern = classify_pattern(low_text, medium_text, high_text)

            classified_rows.append({
                "domain": domain,
                "id": sample_id,
                "label": int(low_row["label"]),
                "text_low": low_text,
                "text_medium": medium_text,
                "text_high": high_text,
                "pattern": pattern,
            })

        classified_columns = [
            "domain", "id", "label",
            "text_low", "text_medium", "text_high", "pattern",
        ]
        classified_by_domain[domain] = pd.DataFrame(
            classified_rows, columns=classified_columns
        )

        logger.info(
            "[%s] matched successful ids across low/medium/high: %d",
            domain, len(ordered_common_ids),
        )

    validity_table = pd.DataFrame(
        validity_rows,
        columns=["domain", "level", "n_total", "n_success", "n_fail", "success_rate"],
    )

    # --- distribution + model metrics (Report B input) --------------------
    success_distribution = build_success_pattern_distribution(classified_by_domain)
    success_metrics = evaluate_success_patterns(
        classified_by_domain, predict_fn_factory
    )

    # --- print the two reports directly in the notebook -------------------
    if print_reports:
        print_report_a_validity(validity_table)
        print_report_b_patterns(success_distribution, success_metrics)

    # --- optionally save the three result tables as CSV -------------------
    # The notebook's visualization cells read exactly these three files, so the
    # evaluation run can produce them and the notebook only has to plot them.
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        validity_table.to_csv(
            os.path.join(output_dir, "validity_by_level.csv"), index=False
        )
        success_distribution.to_csv(
            os.path.join(output_dir, "pattern_distribution.csv"), index=False
        )
        success_metrics.to_csv(
            os.path.join(output_dir, "pattern_metrics.csv"), index=False
        )
        logger.info("Saved pattern-analysis tables to: %s", output_dir)

    return {
        "validity_table": validity_table,
        "success_distribution": success_distribution,
        "success_metrics": success_metrics,
        "classified_by_domain": classified_by_domain,
        "common_ids_by_domain": common_ids_by_domain,
    }


# ---------------------------------------------------------------------------
# CONSOLIDATED ENTRY POINT  (checkpoint wiring + one linear run)
# ---------------------------------------------------------------------------
#
# This folds in what used to live in the separate ``run_pattern_analysis.py``
# runner: it maps a short model key to its HuggingFace id, builds a per-domain
# predictor from the EXISTING trained checkpoints, and runs the success-based
# analysis end to end. No training happens; no argparse/CLI is needed - call
# ``main(...)`` directly from a notebook cell.

# Map the short model key to the HuggingFace model id the checkpoint was
# fine-tuned from (mirrors evaluate_pipeline.py model_mapping).
SHORT_NAME_TO_HUGGINGFACE_MODEL = {
    "base": "indobenchmark/indobert-base-p1",
    "large": "indobenchmark/indobert-large-p1",
    "lite": "indobenchmark/indobert-lite-base-p1",
}


def build_checkpoint_predict_fn_factory(
    checkpoint_dir: str,
    model_name: str,
    full_model_name: str,
    max_length: int = 128,
    device: Optional[str] = None,
    random_seed: int = 42,
    batch_size: int = 32,
):
    """
    Return a ``domain -> predict_fn`` factory bound to the trained checkpoints.

    The analysis calls this factory once per domain. For each domain it builds
    a predictor that loads that domain's specialist checkpoint and returns
    predicted labels. No training happens here.
    """
    def predict_fn_factory_for_domain(domain: str):
        return make_checkpoint_predict_fn(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            domain=domain,
            full_model_name=full_model_name,
            max_length=max_length,
            device=device,
            random_seed=random_seed,
            batch_size=batch_size,
        )

    return predict_fn_factory_for_domain


def main(
    results_root: str,
    checkpoint_dir: Optional[str] = None,
    model_name: str = "base",
    domains: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 128,
    random_seed: int = 42,
    skip_model: bool = False,
    print_reports: bool = True,
) -> Dict[str, object]:
    """
    Consolidated, notebook-friendly entry point for the success-based pattern
    analysis. One linear call:

        Load data -> Report A (validity) -> align common ids
                  -> assign patterns -> evaluate model -> Report B

    Parameters
    ----------
    results_root:
        Directory containing ``<domain>/semantic/<level>.csv``.
    checkpoint_dir:
        Root checkpoint directory. Required unless ``skip_model=True``. Each
        specialist is expected at
        ``<checkpoint_dir>/<model_name>/<Domain.capitalize()>/best_model.pt``.
    model_name:
        Short model key ("base" / "large" / "lite"); anything else is treated
        as a full HuggingFace id.
    skip_model:
        If True, run distribution-only (Report B shows no model metrics).
    """
    # Resolve the full HuggingFace model id (fall back to the string as given).
    if model_name in SHORT_NAME_TO_HUGGINGFACE_MODEL:
        full_model_name = SHORT_NAME_TO_HUGGINGFACE_MODEL[model_name]
    else:
        full_model_name = model_name

    # Decide whether we will score the model or only compute the distribution.
    if skip_model:
        predict_fn_factory = None
    else:
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir is required unless skip_model=True")
        predict_fn_factory = build_checkpoint_predict_fn_factory(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            full_model_name=full_model_name,
            max_length=max_length,
            device=device,
            random_seed=random_seed,
            batch_size=batch_size,
        )

    results = run_success_based_pattern_analysis(
        results_root=results_root,
        predict_fn_factory=predict_fn_factory,
        domains=domains,
        ptype="semantic",
        print_reports=print_reports,
    )
    return results
