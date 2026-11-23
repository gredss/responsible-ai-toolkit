"""
Pattern-Based Robustness Analysis for IndoBERT Clickbait Detection
==================================================================

Analyses how a trained IndoBERT specialist behaves when the same headline is
perturbed by SYNONYM replacement at three intensity levels (Low / Medium / High).
Does NOT train or modify the model.

Linear flow
-----------
1.  Read the per-domain Low / Medium / High semantic perturbation CSVs written
    by EvaluationEngine.generate_all_perturbations.  Keep four columns only:
    id, label, perturbed_text, perturbation_success.
2.  REPORT A — per-level perturbation validity: how many samples succeeded vs
    failed at each level independently.  Failures are reported, never hidden.
3.  Align by original headline id: keep only ids that were SUCCESSFULLY
    perturbed at ALL THREE levels (intersection of per-level success sets).
    Matching is always by id, never by row position.
4.  Classify each matched id into one of five exact-string-equality patterns
    (L=M=H, M=H!=L, L=M!=H, L=H!=M, L!=M!=H).
5.  REPORT B — per-pattern distribution and, when a predictor is supplied,
    model Accuracy / Macro-F1 per (domain, pattern, level).

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

LEVELS: Tuple[str, str, str] = ("low", "medium", "high")

PATTERN_LMH_EQUAL    = "L=M=H"
PATTERN_MH_EQ_L_DIFF = "M=H!=L"
PATTERN_LM_EQ_H_DIFF = "L=M!=H"
PATTERN_LH_EQ_M_DIFF = "L=H!=M"
PATTERN_ALL_DIFF      = "L!=M!=H"

PATTERNS: Tuple[str, ...] = (
    PATTERN_LMH_EQUAL,
    PATTERN_MH_EQ_L_DIFF,
    PATTERN_LM_EQ_H_DIFF,
    PATTERN_LH_EQ_M_DIFF,
    PATTERN_ALL_DIFF,
)

DOMAIN_FILES: Dict[str, str] = {
    "Technology": "technology.csv",
    "Politics":   "politic.csv",
    "Health":     "health.csv",
    "Sport":      "sport.csv",
    "Education":  "education.csv",
}

SETTING_OVERALL      = "overall_matched"
SETTING_ALL_DIFFERENT = "all_different"
SETTINGS: Tuple[str, str] = (SETTING_OVERALL, SETTING_ALL_DIFFERENT)

# Validation targets only — never used to filter or force results.
EXPECTED_SAMPLE_SIZES: Dict[str, Dict[str, int]] = {
    SETTING_OVERALL: {
        "Education": 88, "Health": 77, "Politics": 83, "Sport": 51, "Technology": 62,
    },
    SETTING_ALL_DIFFERENT: {
        "Education": 78, "Health": 70, "Politics": 75, "Sport": 40, "Technology": 49,
    },
}

# Columns that this analysis reads from every perturbed-level CSV.
SUCCESS_ANALYSIS_COLUMNS = ["id", "label", "perturbed_text", "perturbation_success"]

# Type alias: a predict function maps a list of texts to predicted int labels.
PredictFn = Callable[[List[str]], Sequence[int]]

# Maps short model keys to HuggingFace model ids.
SHORT_NAME_TO_HUGGINGFACE_MODEL = {
    "base":  "indobenchmark/indobert-base-p1",
    "large": "indobenchmark/indobert-large-p1",
    "lite":  "indobenchmark/indobert-lite-base-p1",
}


# ---------------------------------------------------------------------------
# PATTERN CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_pattern(low_text: str, medium_text: str, high_text: str) -> str:
    """
    Assign exactly one of the five exact-string-equality patterns to a triple
    of perturbed texts for the same headline id.
    """
    lm = (low_text == medium_text)
    lh = (low_text == high_text)
    mh = (medium_text == high_text)

    if lm and mh:    return PATTERN_LMH_EQUAL
    if mh and not lm: return PATTERN_MH_EQ_L_DIFF
    if lm and not mh: return PATTERN_LM_EQ_H_DIFF
    if lh and not lm: return PATTERN_LH_EQ_M_DIFF

    # All three strings are pairwise different.
    assert not lm and not lh and not mh
    return PATTERN_ALL_DIFF


# ---------------------------------------------------------------------------
# METRIC HELPERS
# ---------------------------------------------------------------------------

def compute_macro_f1(true_labels: Sequence[int], predicted_labels: Sequence[int]) -> float:
    """Macro-F1 with labels=[0, 1] and zero_division=0."""
    from sklearn.metrics import f1_score
    return float(f1_score(true_labels, predicted_labels, labels=[0, 1],
                          average="macro", zero_division=0))


def compute_accuracy(true_labels: Sequence[int], predicted_labels: Sequence[int]) -> float:
    """Plain accuracy. Returns NaN for an empty group."""
    from sklearn.metrics import accuracy_score
    if len(true_labels) == 0:
        return float("nan")
    return float(accuracy_score(true_labels, predicted_labels))


# ---------------------------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------------------------

def resolve_semantic_level_paths(results_root: str, domain: str, ptype: str = "semantic") -> Dict[str, str]:
    """
    Build the {level: path} mapping for synonym perturbation CSVs produced by
    EvaluationEngine.generate_all_perturbations.

    Layout: <results_root>/<domain-lowercased>/<ptype>/<level>.csv
    """
    domain_dir = os.path.join(results_root, str(domain).strip().lower(), ptype)
    return {level: os.path.join(domain_dir, f"{level}.csv") for level in LEVELS}


def make_checkpoint_predict_fn(
    checkpoint_dir:  str,
    model_name:      str,
    domain:          str,
    full_model_name: str           = "indobenchmark/indobert-base-p1",
    max_length:      int           = 128,
    device:          Optional[str] = None,
    random_seed:     int           = 42,
    batch_size:      int           = 32,
) -> PredictFn:
    """
    Build a predict_fn backed by an EXISTING trained specialist checkpoint.
    The model is loaded once and captured in the closure; no training happens.

    Import of model_trainer is deferred so that the pure analysis functions
    can be used without torch installed.
    """
    from model_trainer import ModelTrainer  # deferred import (needs torch)

    checkpoint_path = os.path.join(checkpoint_dir, model_name, str(domain).capitalize(), "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found for {model_name}/{domain}: {checkpoint_path}")

    trainer = ModelTrainer(model_name=full_model_name, max_length=max_length,
                           device=device, random_seed=random_seed)
    trainer.load_checkpoint(checkpoint_path)
    logger.info("Loaded checkpoint for predict_fn: %s", checkpoint_path)

    def predict_texts(texts: List[str]) -> List[int]:
        if not texts:
            return []
        predicted_labels, _ = trainer.predict(texts, batch_size=batch_size)
        return [int(label) for label in predicted_labels]

    return predict_texts


# ---------------------------------------------------------------------------
# CSV READING
# ---------------------------------------------------------------------------

def read_level_csv(path: str) -> pd.DataFrame:
    """
    Read one perturbed-level CSV and return a slim DataFrame with the four
    columns the success-based analysis needs.

    Failed perturbations are KEPT (Report A needs them); they are only excluded
    from Report B by the id-alignment step.
    """
    frame = pd.read_csv(path)
    for col in SUCCESS_ANALYSIS_COLUMNS:
        if col not in frame.columns:
            raise KeyError(f"{path}: missing column '{col}'. Available: {list(frame.columns)}")
    slim = frame[SUCCESS_ANALYSIS_COLUMNS].copy()
    slim["perturbation_success"] = slim["perturbation_success"].astype(bool)
    return slim


# ---------------------------------------------------------------------------
# ID ALIGNMENT  (single standardised method used across the whole project)
# ---------------------------------------------------------------------------

def successful_ids_for_level(level_frame: pd.DataFrame) -> set:
    """
    Return the set of ids that were successfully perturbed in one level frame.
    Mirrors EvaluationEngine._successful_ids exactly (no duplicate handling).
    """
    if "id" not in level_frame.columns:
        raise ValueError("'id' column is missing.")
    if "perturbation_success" not in level_frame.columns:
        raise ValueError("'perturbation_success' column is missing.")
    success_flag = level_frame["perturbation_success"].astype(bool)
    return set(level_frame.loc[success_flag, "id"].tolist())


def count_duplicate_ids_per_level(level_frame_by_level: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """
    Report (do NOT drop) how many ids are duplicated within each level.
    Purely diagnostic; never alters data.
    Returns {level: number_of_ids_appearing_more_than_once}.
    """
    result: Dict[str, int] = {}
    for level in LEVELS:
        counts = level_frame_by_level[level]["id"].value_counts()
        result[level] = int((counts > 1).sum())
    return result


def align_common_successful_ids(
    level_frame_by_level: Dict[str, pd.DataFrame],
    drop_duplicate_ids:   bool = False,
    sort_result:          bool = False,
):
    """
    Standardised common-id alignment used by both this module and the
    evaluation engine.

    Steps:
      1. Collect the set of successfully-perturbed ids for every level.
      2. Optionally deduplicate within each level (OFF by default to preserve
         the evaluation engine's existing results).
      3. Intersect the three sets so only ids successful at ALL THREE levels remain.
      4. Return a set (default) or a sorted list (if sort_result=True).

    Parameters
    ----------
    level_frame_by_level : {"low": df, "medium": df, "high": df}
    drop_duplicate_ids   : if True, keep only the first row per id per level
                           before intersecting (default False)
    sort_result          : if True, return sorted(ids) instead of a set
    """
    successful_by_level: Dict[str, set] = {}
    for level in LEVELS:
        frame = level_frame_by_level[level]
        if drop_duplicate_ids:
            frame = frame.drop_duplicates(subset="id", keep="first")
        successful_by_level[level] = successful_ids_for_level(frame)

    # Intersect across all three levels.
    common = successful_by_level[LEVELS[0]]
    for level in LEVELS[1:]:
        common = common & successful_by_level[level]

    if sort_result:
        return sorted(common, key=str)
    return common


# ---------------------------------------------------------------------------
# REPORT A — PERTURBATION VALIDITY
# ---------------------------------------------------------------------------

def print_report_a_validity(validity_table: pd.DataFrame) -> None:
    """Print Report A: per-level perturbation validity with failure counts."""
    print("=" * 78)
    print("REPORT A - PERTURBATION VALIDITY (independent per severity level)")
    print("=" * 78)
    print("A sample is VALID when it was changed, its word-cosine is in [0.80, 0.95],")
    print("and it hit its token target (10/20/30%). Failures are reported, NOT discarded.")
    print("")

    if validity_table.empty:
        print("   (no validity data available)")
        print("")
        return

    for domain in sorted(validity_table["domain"].unique()):
        print(f"[{domain}]")
        domain_rows = validity_table[validity_table["domain"] == domain]
        for level in LEVELS:
            level_rows = domain_rows[domain_rows["level"] == level]
            if level_rows.empty:
                continue
            row = level_rows.iloc[0]
            n_succ  = int(row["n_success"])
            n_total = int(row["n_total"])
            n_fail  = int(row["n_fail"])
            pct     = float(row["success_rate"]) * 100.0
            print(f"   {level.capitalize():<7} success = {n_succ:>4}/{n_total:<4} ({pct:5.1f}%)   failed = {n_fail:>4}")
        print("")


# ---------------------------------------------------------------------------
# REPORT B — PATTERN DISTRIBUTION & EVALUATION
# ---------------------------------------------------------------------------

def build_success_pattern_distribution(classified_by_domain: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Per-(domain, pattern) distribution over the matched successful subset.
    All five patterns are listed for every domain, even at zero count.

    Columns: domain, pattern, n, n_label0, n_label1, pct_of_matched, denom_matched
    """
    rows: List[dict] = []
    for domain in sorted(classified_by_domain.keys()):
        classified = classified_by_domain[domain]
        n_matched  = int(len(classified))

        for pattern in PATTERNS:
            group = classified[classified["pattern"] == pattern] if not classified.empty else classified
            count = int(len(group))
            rows.append({
                "domain":         domain,
                "pattern":        pattern,
                "n":              count,
                "n_label0":       int((group["label"] == 0).sum()) if count > 0 else 0,
                "n_label1":       int((group["label"] == 1).sum()) if count > 0 else 0,
                "pct_of_matched": 100.0 * count / n_matched if n_matched > 0 else np.nan,
                "denom_matched":  n_matched,
            })

    return pd.DataFrame(rows)


def score_matched_tables(
    classified_by_domain: Dict[str, pd.DataFrame],
    predict_fn_factory:   Callable[[str], PredictFn],
) -> Dict[str, pd.DataFrame]:
    """
    Run the model once per (domain, level) over the matched subset and attach
    prediction columns (pred_clean, pred_low, pred_medium, pred_high) to a
    copy of each classified table.

    Both Report B (per-pattern) and Report C (evaluation settings) slice these
    same scored tables, so every metric is derived from identical predictions.
    Domains with an empty matched table are omitted from the result.
    """
    scored: Dict[str, pd.DataFrame] = {}
    for domain in sorted(classified_by_domain.keys()):
        classified = classified_by_domain[domain]
        if classified.empty:
            continue

        predict_fn   = predict_fn_factory(domain)
        has_clean    = "text_original" in classified.columns
        levels_score = (["clean"] if has_clean else []) + list(LEVELS)

        scored_table = classified.copy().reset_index(drop=True)
        for level in levels_score:
            col_name = "text_original" if level == "clean" else f"text_{level}"
            texts    = scored_table[col_name].astype(str).tolist()
            preds    = list(predict_fn(texts))
            if len(preds) != len(texts):
                raise RuntimeError(
                    f"predict_fn returned {len(preds)} predictions for {len(texts)} texts "
                    f"at level '{level}' (domain={domain})"
                )
            scored_table[f"pred_{level}"] = np.asarray(preds, dtype=int)

        scored[domain] = scored_table
    return scored


def evaluate_success_patterns(
    classified_by_domain: Dict[str, pd.DataFrame],
    predict_fn_factory:   Optional[Callable[[str], PredictFn]],
    scored_by_domain:     Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Compute Accuracy and Macro-F1 per (domain, pattern, level) on the matched
    successful subset.

    Returns columns: domain, pattern, level, n, n_label0, n_label1,
    accuracy, macro_f1.  Returns an empty frame when predict_fn_factory is None.
    """
    metric_cols = ["domain", "pattern", "level", "n", "n_label0", "n_label1", "accuracy", "macro_f1"]
    if predict_fn_factory is None:
        return pd.DataFrame(columns=metric_cols)

    if scored_by_domain is None:
        scored_by_domain = score_matched_tables(classified_by_domain, predict_fn_factory)

    rows: List[dict] = []
    for domain in sorted(scored_by_domain.keys()):
        scored       = scored_by_domain[domain]
        levels_score = [lv for lv in ("clean",) + LEVELS if f"pred_{lv}" in scored.columns]

        for pattern in PATTERNS:
            group      = scored[scored["pattern"] == pattern]
            n          = int(len(group))
            true_labels = group["label"].astype(int).tolist()
            n_label0   = int((group["label"] == 0).sum())
            n_label1   = int((group["label"] == 1).sum())

            for level in levels_score:
                if n == 0:
                    acc, f1 = float("nan"), float("nan")
                else:
                    preds = group[f"pred_{level}"].astype(int).tolist()
                    acc   = compute_accuracy(true_labels, preds)
                    f1    = compute_macro_f1(true_labels, preds)
                rows.append({
                    "domain": domain, "pattern": pattern, "level": level,
                    "n": n, "n_label0": n_label0, "n_label1": n_label1,
                    "accuracy": acc, "macro_f1": f1,
                })

    return pd.DataFrame(rows)


def evaluate_evaluation_settings(scored_by_domain: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute Accuracy and Macro-F1 for the two evaluation settings per domain
    and per level, from already-scored matched tables.

    Settings
    --------
    SETTING_OVERALL       — all ids in the matched subset.
    SETTING_ALL_DIFFERENT — only ids whose L/M/H texts are pairwise different.

    Returns columns: domain, setting, level, n, n_label0, n_label1,
    accuracy, macro_f1.
    """
    setting_filters = {
        SETTING_OVERALL:       lambda t: t,
        SETTING_ALL_DIFFERENT: lambda t: t[t["pattern"] == PATTERN_ALL_DIFF],
    }
    rows: List[dict] = []
    for domain in sorted(scored_by_domain.keys()):
        scored       = scored_by_domain[domain]
        levels_score = [lv for lv in ("clean",) + LEVELS if f"pred_{lv}" in scored.columns]

        for setting, row_filter in setting_filters.items():
            subset      = row_filter(scored)
            n           = int(len(subset))
            true_labels = subset["label"].astype(int).tolist()
            n_label0    = int((subset["label"] == 0).sum())
            n_label1    = int((subset["label"] == 1).sum())

            for level in levels_score:
                if n == 0:
                    acc, f1 = float("nan"), float("nan")
                else:
                    preds = subset[f"pred_{level}"].astype(int).tolist()
                    acc   = compute_accuracy(true_labels, preds)
                    f1    = compute_macro_f1(true_labels, preds)
                rows.append({
                    "domain": domain, "setting": setting, "level": level,
                    "n": n, "n_label0": n_label0, "n_label1": n_label1,
                    "accuracy": acc, "macro_f1": f1,
                })

    return pd.DataFrame(rows, columns=["domain", "setting", "level", "n",
                                        "n_label0", "n_label1", "accuracy", "macro_f1"])


def check_expected_sample_sizes(classified_by_domain: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare observed setting sizes against EXPECTED_SAMPLE_SIZES.
    Discrepancies are logged as warnings; data is never altered.

    Returns columns: domain, setting, expected_n, observed_n, matches.
    """
    rows: List[dict] = []
    for domain in sorted(classified_by_domain.keys()):
        classified = classified_by_domain[domain]
        observed = {
            SETTING_OVERALL:       int(len(classified)),
            SETTING_ALL_DIFFERENT: int((classified["pattern"] == PATTERN_ALL_DIFF).sum())
                                   if not classified.empty else 0,
        }
        for setting, observed_n in observed.items():
            expected_n = EXPECTED_SAMPLE_SIZES.get(setting, {}).get(domain)
            if expected_n is None:
                matches, exp_val = None, np.nan
            else:
                matches  = (observed_n == expected_n)
                exp_val  = expected_n
                if not matches:
                    logger.warning("[%s] %s: observed=%d expected=%d",
                                   domain, setting, observed_n, expected_n)
            rows.append({"domain": domain, "setting": setting,
                         "expected_n": exp_val, "observed_n": observed_n, "matches": matches})

    return pd.DataFrame(rows, columns=["domain", "setting", "expected_n", "observed_n", "matches"])


# ---------------------------------------------------------------------------
# REPORT PRINTERS
# ---------------------------------------------------------------------------

def print_report_b_patterns(
    success_distribution: pd.DataFrame,
    success_metrics:      pd.DataFrame,
) -> None:
    """Print Report B: pattern distribution and model metrics on the matched subset."""
    print("=" * 78)
    print("REPORT B - PATTERN DISTRIBUTION & EVALUATION (matched successful subset)")
    print("=" * 78)
    print("Only ids successfully perturbed at ALL THREE levels, aligned by headline id.")
    print("")

    if success_distribution.empty:
        print("   (no matched successful samples)")
        print("")
        return

    # B.1 — distribution
    print("B.1  Text-equality pattern distribution (per domain)")
    for domain in sorted(success_distribution["domain"].unique()):
        dom_dist = success_distribution[success_distribution["domain"] == domain]
        n_matched = int(dom_dist["denom_matched"].iloc[0]) if len(dom_dist) > 0 else 0
        print(f"   [{domain}] matched = {n_matched}")
        for _, row in dom_dist.iterrows():
            pct = f"{row['pct_of_matched']:5.1f}%" if pd.notna(row["pct_of_matched"]) else "   NA"
            print(f"       {row['pattern']:<9} n={int(row['n']):>4}  ({pct})  "
                  f"label0={int(row['n_label0'])}, label1={int(row['n_label1'])}")
        print("")

    # B.2 — metrics
    print("B.2  Model metrics on the matched subset (Accuracy / Macro-F1)")
    if success_metrics.empty:
        print("   (model evaluation skipped — provide a predict_fn_factory)")
        print("")
        return

    level_initials = {"clean": "C", "low": "L", "medium": "M", "high": "H"}
    for domain in sorted(success_metrics["domain"].unique()):
        print(f"   [{domain}]")
        dom_met = success_metrics[success_metrics["domain"] == domain]
        avail_levels = [lv for lv in ("clean",) + LEVELS if lv in dom_met["level"].values]

        for pattern in PATTERNS:
            pat_rows = dom_met[dom_met["pattern"] == pattern]
            if pat_rows.empty or int(pat_rows["n"].iloc[0]) == 0:
                continue
            n = int(pat_rows["n"].iloc[0])
            acc_parts, f1_parts = [], []
            for level in avail_levels:
                lv_rows = pat_rows[pat_rows["level"] == level]
                init = level_initials.get(level, level[:1].upper())
                if lv_rows.empty:
                    acc_parts.append(f"{init}=NA")
                    f1_parts.append(f"{init}=NA")
                else:
                    acc_parts.append(f"{init}={float(lv_rows['accuracy'].iloc[0]):.3f}")
                    f1_parts.append(f"{init}={float(lv_rows['macro_f1'].iloc[0]):.3f}")
            print(f"       {pattern:<9} n={n:>4}  Acc[{' '.join(acc_parts)}]  MacroF1[{' '.join(f1_parts)}]")
        print("")


def print_report_c_settings(
    setting_metrics:   pd.DataFrame,
    sample_size_check: pd.DataFrame,
) -> None:
    """Print Report C: Macro-F1 per level for each evaluation setting."""
    print("=" * 78)
    print("REPORT C - EVALUATION SETTINGS (Macro-F1 per level, in-domain)")
    print("=" * 78)
    print("Clean / Low / Medium / High use exactly the same ids within each setting.")
    print("")

    if setting_metrics.empty:
        print("   (model evaluation skipped)")
        print("")
        return

    level_initials = {"clean": "C", "low": "L", "medium": "M", "high": "H"}
    for domain in sorted(setting_metrics["domain"].unique()):
        print(f"[{domain}]")
        dom_rows = setting_metrics[setting_metrics["domain"] == domain]
        for setting in SETTINGS:
            set_rows = dom_rows[dom_rows["setting"] == setting]
            if set_rows.empty:
                continue
            observed_n    = int(set_rows["n"].iloc[0])
            expected_text = "n/a"
            status_text   = ""
            if not sample_size_check.empty:
                chk = sample_size_check[
                    (sample_size_check["domain"] == domain) &
                    (sample_size_check["setting"] == setting)
                ]
                if not chk.empty and pd.notna(chk["expected_n"].iloc[0]):
                    expected_text = str(int(chk["expected_n"].iloc[0]))
                    status_text   = "OK" if bool(chk["matches"].iloc[0]) else "MISMATCH"

            parts = []
            for level in ("clean",) + LEVELS:
                lv_rows = set_rows[set_rows["level"] == level]
                if lv_rows.empty:
                    continue
                parts.append(f"{level_initials[level]}={float(lv_rows['macro_f1'].iloc[0]):.3f}")
            print(f"   {setting:<16} n={observed_n:>4} (expected {expected_text:>4}) "
                  f"{status_text:<8} MacroF1[{' '.join(parts)}]")
        print("")


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_success_based_pattern_analysis(
    results_root:       str,
    predict_fn_factory: Optional[Callable[[str], PredictFn]] = None,
    domains:            Optional[Sequence[str]]               = None,
    ptype:              str                                   = "semantic",
    print_reports:      bool                                  = True,
    output_dir:         Optional[str]                         = None,
    test_data_by_domain: Optional[Dict[str, "pd.DataFrame"]] = None,
) -> Dict[str, object]:
    """
    End-to-end success-based pattern analysis designed to be called from a
    notebook cell.

    Flow:
        Load data → Report A (validity) → align common ids
                  → classify patterns → evaluate model → Report B → Report C

    Parameters
    ----------
    results_root        : directory with <domain>/<ptype>/<level>.csv files
    predict_fn_factory  : callable domain → predict_fn; None = distribution only
    domains             : which domains to analyse; defaults to all present on disk
    ptype               : perturbation family folder name (always "semantic")
    print_reports       : whether to print Reports A / B / C
    output_dir          : if given, save result tables as CSV files
    test_data_by_domain : optional original (unperturbed) test DataFrames keyed by
                          domain name; enables apple-to-apple clean scoring

    Returns
    -------
    dict: validity_table, success_distribution, success_metrics,
          setting_metrics, sample_size_check, classified_by_domain,
          common_ids_by_domain
    """
    # Decide which domains to process.
    if domains is None:
        domains = [d for d in DOMAIN_FILES
                   if os.path.isdir(os.path.join(results_root, d.lower(), ptype))]
        if not domains:
            raise FileNotFoundError(f"No <domain>/{ptype} directories found under {results_root}.")
    logger.info("Pattern analysis domains: %s", list(domains))

    validity_rows:        List[dict]              = []
    classified_by_domain: Dict[str, pd.DataFrame] = {}
    common_ids_by_domain: Dict[str, List]         = {}

    for domain in domains:
        level_paths = resolve_semantic_level_paths(results_root, domain, ptype)

        # Read the three level CSVs.
        frame_by_level: Dict[str, pd.DataFrame] = {}
        for level in LEVELS:
            frame_by_level[level] = read_level_csv(level_paths[level])

        # Collect validity counts for Report A.
        for level in LEVELS:
            success_flag = frame_by_level[level]["perturbation_success"]
            n_total      = int(len(frame_by_level[level]))
            n_success    = int(success_flag.sum())
            validity_rows.append({
                "domain":       domain,
                "level":        level,
                "n_total":      n_total,
                "n_success":    n_success,
                "n_fail":       n_total - n_success,
                "success_rate": n_success / n_total if n_total > 0 else 0.0,
            })

        # Align common successful ids (drop duplicates and sort for reproducibility).
        ordered_common_ids = align_common_successful_ids(
            frame_by_level, drop_duplicate_ids=True, sort_result=True
        )
        common_ids_by_domain[domain] = ordered_common_ids

        # Build a deduplicated index per level for fast id-based lookup.
        indexed: Dict[str, pd.DataFrame] = {}
        for level in LEVELS:
            indexed[level] = (
                frame_by_level[level]
                .drop_duplicates(subset="id", keep="first")
                .set_index("id")
            )

        # Build one classified row per common id.
        classified_rows: List[dict] = []
        for sample_id in ordered_common_ids:
            low_row    = indexed["low"].loc[sample_id]
            medium_row = indexed["medium"].loc[sample_id]
            high_row   = indexed["high"].loc[sample_id]

            low_text    = str(low_row["perturbed_text"])
            medium_text = str(medium_row["perturbed_text"])
            high_text   = str(high_row["perturbed_text"])

            classified_rows.append({
                "domain":      domain,
                "id":          sample_id,
                "label":       int(low_row["label"]),
                "text_low":    low_text,
                "text_medium": medium_text,
                "text_high":   high_text,
                "pattern":     classify_pattern(low_text, medium_text, high_text),
            })

        classified_by_domain[domain] = pd.DataFrame(
            classified_rows,
            columns=["domain", "id", "label", "text_low", "text_medium", "text_high", "pattern"],
        )

        # Attach original clean text for apple-to-apple comparison when available.
        if test_data_by_domain is not None:
            original_df = test_data_by_domain.get(domain)
            if original_df is None:
                original_df = test_data_by_domain.get(domain.capitalize())
            if original_df is not None and "id" in original_df.columns and "text" in original_df.columns:
                id_to_text = original_df.set_index("id")["text"].to_dict()
                classified_by_domain[domain]["text_original"] = (
                    classified_by_domain[domain]["id"].map(id_to_text)
                )
                missing = classified_by_domain[domain]["text_original"].isna().sum()
                if missing > 0:
                    logger.warning("[%s] %d matched id(s) had no original text in test_data_by_domain.",
                                   domain, missing)
            else:
                logger.warning("[%s] test_data_by_domain provided but domain key / columns not found.",
                               domain)

        logger.info("[%s] common ids across low/medium/high: %d", domain, len(ordered_common_ids))

    validity_table = pd.DataFrame(
        validity_rows,
        columns=["domain", "level", "n_total", "n_success", "n_fail", "success_rate"],
    )

    # Distribution + model metrics (Reports B and C share the scored tables).
    success_distribution = build_success_pattern_distribution(classified_by_domain)

    if predict_fn_factory is not None:
        scored_by_domain = score_matched_tables(classified_by_domain, predict_fn_factory)
    else:
        scored_by_domain = {}

    success_metrics  = evaluate_success_patterns(classified_by_domain, predict_fn_factory,
                                                  scored_by_domain=scored_by_domain)
    setting_metrics  = evaluate_evaluation_settings(scored_by_domain)
    sample_size_check = check_expected_sample_sizes(classified_by_domain)

    if print_reports:
        print_report_a_validity(validity_table)
        print_report_b_patterns(success_distribution, success_metrics)
        print_report_c_settings(setting_metrics, sample_size_check)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        validity_table.to_csv(      os.path.join(output_dir, "validity_by_level.csv"),    index=False)
        success_distribution.to_csv(os.path.join(output_dir, "pattern_distribution.csv"), index=False)
        success_metrics.to_csv(     os.path.join(output_dir, "pattern_metrics.csv"),      index=False)
        setting_metrics.to_csv(     os.path.join(output_dir, "setting_metrics.csv"),      index=False)
        sample_size_check.to_csv(   os.path.join(output_dir, "sample_size_check.csv"),    index=False)
        logger.info("Saved pattern-analysis tables → %s", output_dir)

    return {
        "validity_table":        validity_table,
        "success_distribution":  success_distribution,
        "success_metrics":       success_metrics,
        "setting_metrics":       setting_metrics,
        "sample_size_check":     sample_size_check,
        "classified_by_domain":  classified_by_domain,
        "common_ids_by_domain":  common_ids_by_domain,
    }


# ---------------------------------------------------------------------------
# CONVENIENCE WRAPPERS
# ---------------------------------------------------------------------------

def build_checkpoint_predict_fn_factory(
    checkpoint_dir:  str,
    model_name:      str,
    full_model_name: str,
    max_length:      int           = 128,
    device:          Optional[str] = None,
    random_seed:     int           = 42,
    batch_size:      int           = 32,
):
    """
    Return a domain → predict_fn factory bound to trained specialist checkpoints.
    Called once; each domain triggers make_checkpoint_predict_fn for that domain.
    """
    def predict_fn_factory_for_domain(domain: str) -> PredictFn:
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
    results_root:        str,
    checkpoint_dir:      Optional[str]                        = None,
    model_name:          str                                  = "base",
    domains:             Optional[Sequence[str]]              = None,
    device:              Optional[str]                        = None,
    batch_size:          int                                  = 32,
    max_length:          int                                  = 128,
    random_seed:         int                                  = 42,
    skip_model:          bool                                 = False,
    print_reports:       bool                                 = True,
    test_data_by_domain: Optional[Dict[str, "pd.DataFrame"]] = None,
) -> Dict[str, object]:
    """
    Notebook-friendly entry point for the success-based pattern analysis.

    Parameters
    ----------
    results_root   : directory with <domain>/semantic/<level>.csv files
    checkpoint_dir : required unless skip_model=True; specialists at
                     <checkpoint_dir>/<model_name>/<Domain>/best_model.pt
    model_name     : "base", "large", "lite", or a full HuggingFace id
    skip_model     : if True, run distribution-only (no model metrics)
    """
    full_model_name = SHORT_NAME_TO_HUGGINGFACE_MODEL.get(model_name, model_name)

    if skip_model:
        predict_fn_factory = None
    else:
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir is required unless skip_model=True.")
        predict_fn_factory = build_checkpoint_predict_fn_factory(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            full_model_name=full_model_name,
            max_length=max_length,
            device=device,
            random_seed=random_seed,
            batch_size=batch_size,
        )

    return run_success_based_pattern_analysis(
        results_root=results_root,
        predict_fn_factory=predict_fn_factory,
        domains=domains,
        ptype="semantic",
        print_reports=print_reports,
        test_data_by_domain=test_data_by_domain,
    )
