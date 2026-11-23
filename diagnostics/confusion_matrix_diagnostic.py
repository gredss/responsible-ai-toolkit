"""
Confusion Matrix Diagnostic — follow-up for Bayesian PPC outliers
IndoBERT Clickbait Detection Robustness Study

PURPOSE
-------
run_posterior_predictive_check() lists the (model, source, target, condition)
cells whose observed Macro-F1 falls outside the model's own 95 % posterior
predictive interval (``ppc['outside_95_rows']``).  This module looks INSIDE
those cells: it rebuilds the classification result behind each aggregate
Macro-F1 and reports the confusion matrix, per-class metrics and the
misclassified headlines.

WHAT THIS MODULE DOES NOT DO
----------------------------
  * It does not touch the Bayesian model, prior, likelihood, random effects,
    contrasts or the PPC.  It only READS ``outside_95_rows``.
  * It does not train, re-split, re-perturb or use train / validation data.
  * It never overwrites existing evaluation files.  It writes only four fixed
    file names (+ a plots/ folder) into its own output folder and refuses to
    replace them unless overwrite=True.
  * No file in src/ is modified; src/ modules are imported and reused
    (evaluation_engine.calculate_metrics, pattern_analysis.align_common_
    successful_ids / read_level_csv, statistical_analyzer.apply_boundary_
    correction, model_trainer.ModelTrainer for optional re-inference).

WHERE THE CLASSIFICATIONS COME FROM
-----------------------------------
complete_evaluation.json already stores ``predictions`` / ``true_labels`` /
``probabilities`` for every cell, in the order of the rows that were scored,
but WITHOUT ids or texts.  The ids are therefore rebuilt from the saved test
split (``<splits_dir>/<domain>/test.csv``) and the saved perturbed CSVs
(``<results_root>/<model>/<domain>/semantic/<level>.csv``) with the project's
own ``pattern_analysis.align_common_successful_ids``.  Two row layouts are
tried and the first whose length AND labels equal the stored arrays is used:

    common_subset  rows successful at all three levels (current evaluation code)
    full_test_set  every test row (older runs stored cross-domain cells this way)

If neither layout reproduces the stored arrays the group is INVALID.

Optional re-inference (``predictor_factory`` / --reinfer): the existing
``best_model.pt`` of the SOURCE domain is loaded and ModelTrainer.predict is
run on the SAME rows (same ids, same text the model saw originally).  The
prediction arrays are compared element by element with the stored ones.
Equal Macro-F1 is never taken as proof of identical predictions.

VALIDATION STATUS (per group)
-----------------------------
  VALID        every check passed; interpretation is written.
  INVALID      a check failed (see ``validation_notes``).  Metrics are shown for
               transparency, but NO interpretation is made and no misclassified
               examples are exported.
  UNAVAILABLE  a required component is missing (file, stored block, ...).

Checks: stored vs recomputed Macro-F1 (tol 1e-6); sample count vs stored
count; labels and ids vs the saved dataset; PPC ``observed`` vs recomputed F1
(raw or boundary-corrected, tol 1e-4 because outside_95_rows is rounded to
4 decimals); with re-inference also the prediction arrays.

LABELS
------
0 = non-clickbait, 1 = clickbait.  Positive class for TP/FP/FN/TN is 1.
    TN = cm[0,0]   FP = cm[0,1]   FN = cm[1,0]   TP = cm[1,1]   (labels=[0,1])

PUBLIC API
----------
  identify_outside_groups(ppc_source)               -> pd.DataFrame
  compute_confusion_metrics(y_true, y_pred)         -> dict
  run_confusion_matrix_diagnostic(ppc_source, ...)  -> dict   (main entry point)
  run_all_tests()                                   synthetic, no torch / PyMC

CLI
---
  python diagnostics/confusion_matrix_diagnostic.py --ppc-json statistical_analysis.json \\
      --results-root <results> --splits-dir <outputs/data_splits> [--reinfer --checkpoint-dir ...]
  python diagnostics/confusion_matrix_diagnostic.py --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# src/ is read-only and stays untouched; we only import from it.
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pattern_analysis as _pa                                  # noqa: E402
from evaluation_engine import calculate_metrics                 # noqa: E402
from statistical_analyzer import apply_boundary_correction      # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

GROUP_KEYS = ["model", "source_domain", "target_domain", "perturbation_condition"]
LABELS = [0, 1]
CLASS_NAMES = {0: "non-clickbait", 1: "clickbait"}

DEFAULT_OUTPUT_DIR = os.path.join("statistical_analysis", "confusion_matrix_diagnostic")
OUTPUT_FILES = {
    "summary": "confusion_matrix_summary.csv",
    "classes": "classification_details.csv",
    "errors":  "misclassified_examples.csv",
    "report":  "diagnostic_report.txt",
}
PLOTS_DIRNAME = "plots"

F1_TOL  = 1e-6          # stored vs recomputed Macro-F1
PPC_TOL = 1e-4          # outside_95_rows is rounded to 4 decimals
DEFAULT_N_CELLS = 300   # 3 models x 5 sources x 5 targets x 4 conditions (boundary correction N)

VALID, INVALID, UNAVAILABLE = "VALID", "INVALID", "UNAVAILABLE"

# predictor_factory(model, source_domain) -> predict(texts) -> (preds, probs)
PredictorFactory = Callable[[str, str], Callable[[List[str]], Tuple[Any, Any]]]


# ============================================================================
# SECTION 1 — IDENTIFY GROUPS FROM outside_95_rows
# ============================================================================

def _extract_rows(ppc_source: Any) -> pd.DataFrame:
    """Accept a ppc dict, a full analysis dict, a list of rows, a DataFrame or a JSON path."""
    if isinstance(ppc_source, (str, os.PathLike)):
        with open(ppc_source, "r", encoding="utf-8") as fh:
            ppc_source = json.load(fh)

    if isinstance(ppc_source, pd.DataFrame):
        return ppc_source.copy()
    if isinstance(ppc_source, dict):
        if "outside_95_rows" in ppc_source:
            rows = ppc_source["outside_95_rows"]
        elif isinstance(ppc_source.get("ppc"), dict) and "outside_95_rows" in ppc_source["ppc"]:
            rows = ppc_source["ppc"]["outside_95_rows"]
        else:
            raise ValueError("No 'outside_95_rows' key in the PPC input "
                             "(expected ppc['outside_95_rows'] or analysis['ppc']['outside_95_rows']).")
    elif isinstance(ppc_source, list):
        rows = ppc_source
    else:
        raise TypeError(f"Unsupported PPC input type: {type(ppc_source)!r}")
    return pd.DataFrame(rows)


def identify_outside_groups(ppc_source: Any) -> pd.DataFrame:
    """
    One row per distinct (model, source, target, condition) outside its 95 %
    posterior predictive interval.  Names are read from the data; nothing is
    hard-coded.  ``direction`` is 'below' (observed < lo95), 'above'
    (observed > hi95) or 'on_edge' (only possible through 4-decimal rounding).
    Empty input gives an empty DataFrame with the same columns.
    """
    value_cols = ["observed", "lo95", "hi95"]
    out_cols = GROUP_KEYS + value_cols + ["direction"]

    df = _extract_rows(ppc_source)
    if df.empty:
        return pd.DataFrame(columns=out_cols)

    missing = [c for c in GROUP_KEYS + value_cols if c not in df.columns]
    if missing:
        raise ValueError(f"outside_95_rows is missing columns: {missing}")

    df = df[GROUP_KEYS + value_cols].copy()
    for key in GROUP_KEYS:
        df[key] = df[key].astype(str)
    n_dup = int(df.duplicated(GROUP_KEYS).sum())
    if n_dup:
        logger.warning("[cm-diag] %d duplicate group key(s) in outside_95_rows; keeping first.", n_dup)
    df = df.drop_duplicates(GROUP_KEYS, keep="first")

    def _direction(row) -> str:
        if row["observed"] < row["lo95"]:
            return "below"
        if row["observed"] > row["hi95"]:
            return "above"
        return "on_edge"

    df["direction"] = df.apply(_direction, axis=1)
    return df.sort_values(GROUP_KEYS).reset_index(drop=True)[out_cols]


# ============================================================================
# SECTION 2 — LOAD STORED RESULTS AND DATASET (read-only)
# ============================================================================

def load_eval_results(results_root: str, models: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Load ``<results_root>/<model>/complete_evaluation.json`` for each model.
    Returns (all_results, missing_paths).  Files are only read."""
    all_results: Dict[str, Any] = {}
    missing: List[str] = []
    for model in models:
        path = os.path.join(results_root, model, "complete_evaluation.json")
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path, "r", encoding="utf-8") as fh:
            all_results[model] = json.load(fh)
    return all_results, missing


def select_stored_block(all_results: Dict[str, Any], model: str, source: str,
                        target: str, cond: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Find the stored evaluation block behind one observation.  Routing mirrors
    statistical_analyzer.assemble_observations(): diagonal cells come from
    results['perturbation'], everything else (and a missing diagonal) from
    results['cross_domain_perturbation'].  Returns (block | None, route | reason).
    """
    res = all_results.get(model)
    if res is None:
        return None, f"results for model '{model}' were not loaded"

    if source == target:
        dblock = (res.get("perturbation") or {}).get(source) or {}
        block = dblock.get("clean") if cond == "clean" else (dblock.get("semantic") or {}).get(cond)
        if block:
            return block, f"perturbation/{source}/{cond}"

    key = f"{source}_to_{target}"
    cblock = (res.get("cross_domain_perturbation") or {}).get(key) or {}
    block = cblock.get("clean") if cond == "clean" else (cblock.get("semantic") or {}).get(cond)
    if block:
        return block, f"cross_domain_perturbation/{key}/{cond}"
    return None, f"no stored block for {source}->{target} / {cond} in model '{model}'"


class _DataStore:
    """Lazy, cached, read-only access to the saved test splits and perturbed CSVs."""

    def __init__(self, splits_dir: str, perturbed_root: str):
        self.splits_dir = splits_dir
        self.perturbed_root = perturbed_root
        self._test: Dict[str, pd.DataFrame] = {}
        self._levels: Dict[Tuple[str, str], Dict[str, pd.DataFrame]] = {}

    def test_df(self, domain: str) -> pd.DataFrame:
        if domain not in self._test:
            path = os.path.join(self.splits_dir, domain.lower(), "test.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"test split not found: {path}")
            df = pd.read_csv(path)
            missing = [c for c in ("id", "text", "label") if c not in df.columns]
            if missing:
                raise KeyError(f"{path}: missing columns {missing}")
            self._test[domain] = df
        return self._test[domain]

    def level_frames(self, model: str, domain: str) -> Dict[str, pd.DataFrame]:
        key = (model, domain)

        if key not in self._levels:
            paths = {
                level: os.path.join(
                    self.perturbed_root,
                    model,
                    "perturbation_results",
                    domain.lower(),
                    "semantic",
                    f"{level}.csv"
                )
                for level in ("low", "medium", "high")
            }

            frames = {}

            for level, path in paths.items():
                print(f"[DEBUG] Looking for {level}: {path}")
                print(f"[DEBUG] Exists: {os.path.exists(path)}")

                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"perturbed CSV not found: {path}"
                    )

                frames[level] = _pa.read_level_csv(path)

            self._levels[key] = frames

        return self._levels[key]



def reconstruct_candidates(store: _DataStore, model: str, target: str,
                           cond: str) -> List[Tuple[str, pd.DataFrame]]:
    """
    Candidate row layouts for one cell; each frame has columns
    id, text (what the model saw), original_text, label.
    """
    test = store.test_df(target)
    frames = store.level_frames(model, target)
    common = _pa.align_common_successful_ids(frames, drop_duplicate_ids=False, sort_result=False)

    if cond == "clean":
        base = pd.DataFrame({"id": test["id"], "text": test["text"],
                             "original_text": test["text"], "label": test["label"]})
    else:
        if cond not in frames:
            raise KeyError(f"unknown perturbation level '{cond}'")
        lf = frames[cond]
        original = test.drop_duplicates("id").set_index("id")["text"]
        base = pd.DataFrame({"id": lf["id"], "text": lf["perturbed_text"],
                             "original_text": lf["id"].map(original), "label": lf["label"]})

    return [
        ("common_subset", base[base["id"].isin(common)].reset_index(drop=True)),
        ("full_test_set", base.reset_index(drop=True)),
    ]


def pick_alignment(candidates: List[Tuple[str, pd.DataFrame]],
                   stored_labels: List[int]) -> Tuple[Optional[str], Optional[pd.DataFrame], List[str]]:
    """First layout whose length AND label sequence equal the stored true_labels."""
    tried: List[str] = []
    for name, frame in candidates:
        ok = (len(frame) == len(stored_labels)
              and frame["label"].astype(int).tolist() == [int(v) for v in stored_labels])
        tried.append(f"{name}: n={len(frame)}{'' if ok else ' (no match)'}")
        if ok:
            return name, frame, tried
    return None, None, tried


def check_dataset_consistency(test_df: pd.DataFrame, frame: pd.DataFrame) -> List[str]:
    """Ids must exist in the saved test split and carry the same label there."""
    problems: List[str] = []
    ids_unknown = sorted(set(frame["id"]) - set(test_df["id"]))
    if ids_unknown:
        problems.append(f"{len(ids_unknown)} id(s) not in the test split (e.g. {ids_unknown[:3]})")
    if not test_df["id"].is_unique:
        problems.append("test split has duplicate ids; id-level matching is ambiguous")
        return problems
    ref = test_df.set_index("id")["label"]
    known = frame[frame["id"].isin(ref.index)]
    n_bad = int((known["label"].astype(int).values != ref.loc[known["id"]].astype(int).values).sum())
    if n_bad:
        problems.append(f"{n_bad} label(s) differ between the perturbed CSV and the test split")
    return problems


# ============================================================================
# SECTION 3 — METRICS
# ============================================================================

def compute_confusion_metrics(y_true, y_pred) -> Dict[str, Any]:
    """
    Binary confusion matrix and metrics, label order [0, 1], positive class 1.

    Matrix layout (rows = actual, columns = predicted):
                    pred 0   pred 1
        actual 0      TN       FP
        actual 1      FN       TP

    Per-class and macro scores come from evaluation_engine.calculate_metrics
    (the same function that produced the stored Macro-F1), so the numbers are
    defined identically to the original evaluation.  Only the matrix itself is
    computed here, with labels=[0, 1], so a single-class group still gives 2x2.
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred differ in length: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("empty group: no samples")
    bad = (set(np.unique(y_true)) | set(np.unique(y_pred))) - set(LABELS)
    if bad:
        raise ValueError(f"labels outside {LABELS}: {sorted(bad)}")

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # undefined MCC on single-class groups
        base = calculate_metrics(y_true, y_pred)

    n = int(y_true.size)
    n0, n1 = int((y_true == 0).sum()), int((y_true == 1).sum())
    p0, p1 = int((y_pred == 0).sum()), int((y_pred == 1).sum())
    return {
        "n": n,
        "n_actual_0": n0, "n_actual_1": n1,
        "pct_actual_0": n0 / n, "pct_actual_1": n1 / n,
        "n_pred_0": p0, "n_pred_1": p1,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "confusion_matrix": cm.tolist(),
        "precision_0": float(base["precision_0"]), "recall_0": float(base["recall_0"]), "f1_0": float(base["f1_0"]),
        "precision_1": float(base["precision_1"]), "recall_1": float(base["recall_1"]), "f1_1": float(base["f1_1"]),
        "macro_precision": float(base["macro_precision"]),
        "macro_recall": float(base["macro_recall"]),
        "macro_f1": float(base["macro_f1"]),
        "accuracy": float(base["accuracy"]),
    }


# ============================================================================
# SECTION 4 — VALIDATION AND PER-GROUP DIAGNOSIS
# ============================================================================

def _ppc_match(observed: float, f1: float, n_cells: int) -> Tuple[Optional[str], float]:
    """
    Does the PPC ``observed`` correspond to this F1?  The PPC compares
    boundary-corrected values, and the correction is applied to ALL rows only
    when some cell is exactly 0 or 1 (sv_n_cells).  Raw and corrected forms are
    both accepted.  Returns (which_form | None, smallest_abs_difference).
    """
    raw_diff = abs(observed - f1)
    corrected = float(apply_boundary_correction(np.array([f1]), n=n_cells)[0])
    cor_diff = abs(observed - corrected)
    if raw_diff <= PPC_TOL:
        return "raw", raw_diff
    if cor_diff <= PPC_TOL:
        return "boundary_corrected", cor_diff
    return None, min(raw_diff, cor_diff)


def diagnose_group(
    group: Dict[str, Any],
    all_results: Dict[str, Any],
    store: Optional[_DataStore],
    predictor_factory: Optional[PredictorFactory] = None,
    n_cells: int = DEFAULT_N_CELLS,
) -> Dict[str, Any]:
    """
    Rebuild and validate the classification behind ONE outside-95 cell.

    Returns a dict with: status, notes, metrics (or None), rows (frame with
    id/text/original_text/label/pred/prob_clickbait or None), and provenance
    fields.  Never raises for missing data; the reason goes into ``notes``.
    """
    model, src, tgt, cond = (group[k] for k in GROUP_KEYS)
    out: Dict[str, Any] = {
        "status": UNAVAILABLE, "notes": [], "metrics": None, "rows": None,
        "stored_block": None, "alignment": None, "ppc_form": None,
        "stored_macro_f1": None, "reinference": "not_run",
    }
    notes: List[str] = out["notes"]

    block, route = select_stored_block(all_results, model, src, tgt, cond)
    if block is None:
        notes.append(f"stored result missing: {route}")
        return out
    out["stored_block"] = route

    for field in ("predictions", "true_labels"):
        if field not in block:
            notes.append(f"stored block lacks '{field}'")
            return out
    stored_pred = [int(v) for v in block["predictions"]]
    stored_true = [int(v) for v in block["true_labels"]]
    probs = block.get("probabilities")
    stored_f1 = (block.get("metrics") or {}).get("macro_f1")
    out["stored_macro_f1"] = None if stored_f1 is None else float(stored_f1)

    if len(stored_pred) != len(stored_true):
        out["status"] = INVALID
        notes.append(f"stored predictions ({len(stored_pred)}) and true_labels ({len(stored_true)}) differ in length")
        return out

    # ---- metrics from the stored arrays (needs no dataset) ----------------
    try:
        metrics = compute_confusion_metrics(stored_true, stored_pred)
    except ValueError as exc:
        out["status"] = INVALID
        notes.append(f"metrics could not be computed: {exc}")
        return out
    out["metrics"] = metrics
    failures: List[str] = []
    unavailable = False        # a required component is missing (not a failed check)

    # ---- check: recomputed vs stored Macro-F1 -----------------------------
    if stored_f1 is None:
        failures.append("stored Macro-F1 missing; cannot compare")
    elif abs(metrics["macro_f1"] - float(stored_f1)) > F1_TOL:
        failures.append(f"Macro-F1 mismatch: recomputed {metrics['macro_f1']:.6f} vs stored {float(stored_f1):.6f}")

    # ---- check: PPC observed vs recomputed --------------------------------
    form, diff = _ppc_match(float(group["observed"]), metrics["macro_f1"], n_cells)
    out["ppc_form"] = form
    if form is None:
        failures.append(f"PPC observed {float(group['observed']):.4f} does not match recomputed Macro-F1 "
                        f"{metrics['macro_f1']:.4f} (raw or boundary-corrected; |diff|={diff:.4f})")

    # ---- check: sample count / stored count -------------------------------
    cic = block.get("common_ids_count")
    if cic is not None and int(cic) != len(stored_pred):
        failures.append(f"stored common_ids_count={cic} but {len(stored_pred)} predictions stored")

    # ---- rebuild ids + texts from the saved dataset -----------------------
    rows = None
    if store is None:
        notes.append("dataset not provided; ids/texts not reconstructed (no misclassified examples)")
        unavailable = True
    else:
        try:
            candidates = reconstruct_candidates(store, model, tgt, cond)
        except (FileNotFoundError, KeyError) as exc:
            notes.append(f"dataset component missing: {exc}")
            candidates = None
            unavailable = True
        if candidates is not None:
            name, frame, tried = pick_alignment(candidates, stored_true)
            if frame is None:
                failures.append("ids cannot be matched to the stored arrays (" + "; ".join(tried) + ")")
            else:
                out["alignment"] = name
                problems = check_dataset_consistency(store.test_df(tgt), frame)
                failures.extend(problems)
                rows = frame.copy()
                rows["pred"] = stored_pred
                if probs is not None and len(probs) == len(stored_pred):
                    rows["prob_clickbait"] = [float(p[1]) if hasattr(p, "__len__") else float(p) for p in probs]
                else:
                    rows["prob_clickbait"] = np.nan

    # ---- optional independent re-inference --------------------------------
    if predictor_factory is not None and rows is not None:
        try:
            predict = predictor_factory(model, src)
            new_pred, _ = predict(rows["text"].tolist())
            new_pred = np.asarray(new_pred, dtype=int)
            if len(new_pred) != len(rows):
                failures.append(f"re-inference returned {len(new_pred)} predictions for {len(rows)} rows")
                out["reinference"] = "failed"
            else:
                n_diff = int((new_pred != np.asarray(stored_pred)).sum())
                re_f1 = compute_confusion_metrics(stored_true, new_pred)["macro_f1"]
                out["reinference"] = "identical" if n_diff == 0 else "different"
                out["reinference_n_different"] = n_diff
                out["reinference_macro_f1"] = re_f1
                if n_diff:
                    failures.append(f"re-inference differs from stored predictions in {n_diff}/{len(rows)} rows "
                                    f"(re-inferred Macro-F1 {re_f1:.6f}; stored {metrics['macro_f1']:.6f})")
        except Exception as exc:  # checkpoint/torch missing etc. -> report, never substitute
            out["reinference"] = "unavailable"
            notes.append(f"re-inference not possible: {exc}")
            unavailable = True
    elif predictor_factory is None:
        notes.append("validation level: stored predictions only (no independent re-inference)")

    out["rows"] = rows
    if failures:                       # a failed check outranks a missing component
        out["status"] = INVALID
        notes.extend(failures)
    elif unavailable or rows is None:
        out["status"] = UNAVAILABLE
    else:
        out["status"] = VALID
    return out


# ============================================================================
# SECTION 5 — RE-INFERENCE FACTORY (optional; needs torch + existing checkpoints)
# ============================================================================

def make_checkpoint_predictor_factory(checkpoint_dir: str, device: Optional[str] = None,
                                      random_seed: int = 42, max_length: int = 128) -> PredictorFactory:
    """
    Predictor factory backed by EXISTING checkpoints (no training).  Layout as in
    EvaluationPipeline.load_models: <checkpoint_dir>/<model>/<Source>/best_model.pt.
    Only the most recent trainer is kept in memory (groups are processed sorted
    by model/source, so a checkpoint is loaded once).
    """
    cache: Dict[str, Any] = {}

    def factory(model: str, source_domain: str):
        key = f"{model}/{source_domain}"
        if key not in cache:
            if model not in _pa.SHORT_NAME_TO_HUGGINGFACE_MODEL:
                raise KeyError(f"unknown model variant '{model}'")
            path = os.path.join(checkpoint_dir, model, str(source_domain).capitalize(), "best_model.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"checkpoint not found: {path}")
            from model_trainer import ModelTrainer          # deferred: needs torch
            trainer = ModelTrainer(model_name=_pa.SHORT_NAME_TO_HUGGINGFACE_MODEL[model],
                                   max_length=max_length, device=device, random_seed=random_seed)
            trainer.load_checkpoint(path)
            cache.clear()
            cache[key] = trainer
        return cache[key].predict

    return factory


# ============================================================================
# SECTION 6 — OUTPUT TABLES
# ============================================================================

SUMMARY_COLUMNS = GROUP_KEYS + [
    "validation_status", "observed_f1_ppc", "lo95", "hi95", "direction",
    "stored_macro_f1", "n", "n_actual_0", "n_actual_1", "pct_actual_1",
    "TN", "FP", "FN", "TP", "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "ppc_value_form", "stored_block", "row_alignment", "reinference", "validation_notes",
]
CLASS_COLUMNS = GROUP_KEYS + ["validation_status", "class", "class_name",
                              "precision", "recall", "f1", "support"]
ERROR_COLUMNS = GROUP_KEYS + ["validation_status", "id", "text", "original_text",
                              "true_label", "pred_label", "error_type",
                              "prob_clickbait", "confidence_in_prediction"]


def build_output_tables(groups: pd.DataFrame, diagnoses: List[Dict[str, Any]]
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows, class_rows, error_frames = [], [], []
    for (_, g), d in zip(groups.iterrows(), diagnoses):
        keys = {k: g[k] for k in GROUP_KEYS}
        m = d["metrics"]
        srow = {**keys, "validation_status": d["status"],
                "observed_f1_ppc": g["observed"], "lo95": g["lo95"], "hi95": g["hi95"],
                "direction": g["direction"], "stored_macro_f1": d["stored_macro_f1"],
                "ppc_value_form": d["ppc_form"], "stored_block": d["stored_block"],
                "row_alignment": d["alignment"], "reinference": d["reinference"],
                "validation_notes": " | ".join(d["notes"])}
        if m:
            for c in ("n", "n_actual_0", "n_actual_1", "pct_actual_1", "TN", "FP", "FN", "TP",
                      "accuracy", "macro_precision", "macro_recall", "macro_f1"):
                srow[c] = m[c]
            for cls in LABELS:
                class_rows.append({**keys, "validation_status": d["status"], "class": cls,
                                   "class_name": CLASS_NAMES[cls],
                                   "precision": m[f"precision_{cls}"], "recall": m[f"recall_{cls}"],
                                   "f1": m[f"f1_{cls}"], "support": m[f"n_actual_{cls}"]})
        summary_rows.append(srow)

        # Misclassified headlines are exported only for VALID groups.
        if d["status"] == VALID and d["rows"] is not None:
            r = d["rows"]
            wrong = r[r["label"].astype(int) != r["pred"].astype(int)].copy()
            if len(wrong):
                wrong["true_label"] = wrong["label"].astype(int)
                wrong["pred_label"] = wrong["pred"].astype(int)
                wrong["error_type"] = np.where(wrong["pred_label"] == 1, "FP", "FN")
                wrong["confidence_in_prediction"] = np.where(
                    wrong["pred_label"] == 1, wrong["prob_clickbait"], 1.0 - wrong["prob_clickbait"])
                for k in GROUP_KEYS:
                    wrong[k] = keys[k]
                wrong["validation_status"] = d["status"]
                error_frames.append(wrong.sort_values("confidence_in_prediction", ascending=False)[ERROR_COLUMNS])

    summary = pd.DataFrame(summary_rows).reindex(columns=SUMMARY_COLUMNS)
    classes = pd.DataFrame(class_rows).reindex(columns=CLASS_COLUMNS)
    errors = (pd.concat(error_frames, ignore_index=True) if error_frames
              else pd.DataFrame(columns=ERROR_COLUMNS))
    return summary, classes, errors


# ============================================================================
# SECTION 7 — PLOTS
# ============================================================================

def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(s))


def _draw_cm(ax, d: Dict[str, Any], g: Dict[str, Any], small: bool = False) -> None:
    m = d["metrics"]
    cm = np.asarray(m["confusion_matrix"])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, cm.max()))
    names = [[f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"], [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, names[i][j], ha="center", va="center", fontsize=9 if small else 12,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 non-click", "1 click"], fontsize=7 if small else 9)
    ax.set_yticklabels(["0 non-click", "1 click"], fontsize=7 if small else 9)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
    ax.set_title(f"{g['model']} {g['source_domain']}->{g['target_domain']} [{g['perturbation_condition']}]\n"
                 f"n={m['n']}  Macro-F1={m['macro_f1']:.3f}  ({d['status']})",
                 fontsize=7 if small else 9)


def save_plots(groups: pd.DataFrame, diagnoses: List[Dict[str, Any]], plots_dir: str) -> List[str]:
    """One PNG per group with metrics + a combined grid.  Returns written paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("[cm-diag] matplotlib unavailable; plots skipped.")
        return []

    items = [(g, d) for (_, g), d in zip(groups.iterrows(), diagnoses) if d["metrics"]]
    if not items:
        return []
    os.makedirs(plots_dir, exist_ok=True)
    written: List[str] = []

    for g, d in items:
        fig, ax = plt.subplots(figsize=(3.8, 3.6))
        _draw_cm(ax, d, g)
        fig.tight_layout()
        path = os.path.join(plots_dir, "cm_" + "_".join(_safe(g[k]) for k in GROUP_KEYS) + ".png")
        fig.savefig(path, dpi=150); plt.close(fig); written.append(path)

    ncols = min(4, len(items)); nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (g, d) in zip(axes.ravel(), items):
        ax.axis("on"); _draw_cm(ax, d, g, small=True)
    fig.tight_layout()
    path = os.path.join(plots_dir, "confusion_matrices_all_groups.png")
    fig.savefig(path, dpi=150); plt.close(fig); written.append(path)
    return written


# ============================================================================
# SECTION 8 — REPORT
# ============================================================================

def _interpret(g: Dict[str, Any], d: Dict[str, Any]) -> List[str]:
    """Descriptive reading of ONE validated group.  Called only when status == VALID."""
    m = d["metrics"]; lines: List[str] = []
    side = {"below": "BELOW", "above": "ABOVE"}.get(g["direction"], "on the edge of")
    lines.append(f"Observed Macro-F1 {float(g['observed']):.4f} is {side} the 95% predictive interval "
                 f"[{float(g['lo95']):.4f}, {float(g['hi95']):.4f}].")
    errors = m["FP"] + m["FN"]
    if errors == 0:
        lines.append("No misclassifications: every headline is classified correctly.")
    else:
        if m["FP"] > m["FN"]:
            dom = "false positives dominate (non-clickbait predicted as clickbait)"
        elif m["FN"] > m["FP"]:
            dom = "false negatives dominate (clickbait predicted as non-clickbait)"
        else:
            dom = "false positives and false negatives are equal"
        lines.append(f"{errors} error(s) out of {m['n']}: FP={m['FP']}, FN={m['FN']}; {dom}.")
    r0, r1 = m["recall_0"], m["recall_1"]
    if m["n_actual_0"] and m["n_actual_1"] and abs(r0 - r1) >= 0.05:
        weak = "non-clickbait (0)" if r0 < r1 else "clickbait (1)"
        lines.append(f"Recall differs by class (class 0 = {r0:.3f}, class 1 = {r1:.3f}); "
                     f"the weaker class is {weak}.")
    minor = 1 if m["n_actual_1"] <= m["n_actual_0"] else 0
    n_minor = m[f"n_actual_{minor}"]
    tail = (f"in the smaller class (class {minor}, n={n_minor}) one error moves that class's recall by "
            f"{100 / n_minor:.1f} points." if n_minor else
            f"class {minor} has no actual samples here, so its recall is undefined (reported as 0).")
    lines.append(f"Actual labels: {m['n_actual_0']} class 0 / {m['n_actual_1']} class 1 "
                 f"({100 * m['pct_actual_1']:.1f}% clickbait). One error moves accuracy by {100 / m['n']:.2f} points; "
                 + tail)
    if d["alignment"] == "full_test_set" and "cross_domain" in (d["stored_block"] or ""):
        lines.append("Stored cell covers the full test split (not the common-successful subset).")
    return lines


def build_report(groups: pd.DataFrame, diagnoses: List[Dict[str, Any]], summary: pd.DataFrame,
                 context: Dict[str, Any]) -> str:
    L: List[str] = []
    bar = "=" * 80
    L += [bar, "CONFUSION MATRIX DIAGNOSTIC — PPC observations outside the 95% interval",
          "IndoBERT Clickbait Robustness Study", bar, "",
          "Labels: 0 = non-clickbait, 1 = clickbait; positive class = 1.",
          "Matrix rows = actual, columns = predicted:  [[TN, FP], [FN, TP]]  (label order [0, 1]).", ""]

    status_counts = pd.Series([d["status"] for d in diagnoses]).value_counts().to_dict()
    L += ["SCOPE", "-" * 40,
          f"  Groups outside the 95% interval : {len(groups)}",
          f"  Status counts                   : {status_counts}",
          f"  Validation level                : "
          f"{'independent re-inference + stored predictions' if context['reinfer'] else 'stored predictions only (no independent re-inference)'}",
          "  Source of predictions           : complete_evaluation.json (same checkpoints, test IDs, texts and",
          "                                    perturbations as the original evaluation; nothing retrained,",
          "                                    re-split or re-perturbed; train/validation data not used).",
          "  Not changed                     : Bayesian model, likelihood, priors, random effects, contrasts, PPC.", ""]

    L += ["VALIDATION RULES", "-" * 40,
          "  VALID        recomputed Macro-F1 == stored Macro-F1 (1e-6); sample count and ids match the dataset;",
          "               labels match; PPC 'observed' == recomputed F1 (raw or boundary-corrected, 1e-4);",
          "               with re-inference also every prediction identical.",
          "  INVALID      a check failed -> metrics shown for transparency, NO interpretation, no examples.",
          "  UNAVAILABLE  a needed component is missing -> reported below, nothing substituted.",
          "  Note: equal Macro-F1 does not prove identical predictions; that is only established by the",
          "        element-wise comparison of a re-inference run.", ""]

    L += ["GROUPS", "-" * 40]
    for (_, g), d in zip(groups.iterrows(), diagnoses):
        L.append(f"[{d['status']}] {g['model']} | {g['source_domain']} -> {g['target_domain']} | "
                 f"{g['perturbation_condition']}")
        m = d["metrics"]
        if m:
            L.append(f"    n={m['n']}  actual 0/1 = {m['n_actual_0']}/{m['n_actual_1']}   "
                     f"TN={m['TN']} FP={m['FP']} FN={m['FN']} TP={m['TP']}")
            L.append(f"    class0 P/R/F1 = {m['precision_0']:.3f}/{m['recall_0']:.3f}/{m['f1_0']:.3f}   "
                     f"class1 P/R/F1 = {m['precision_1']:.3f}/{m['recall_1']:.3f}/{m['f1_1']:.3f}")
            L.append(f"    macro P/R/F1 = {m['macro_precision']:.3f}/{m['macro_recall']:.3f}/{m['macro_f1']:.3f}   "
                     f"accuracy = {m['accuracy']:.3f}")
        for note in d["notes"]:
            L.append(f"    note: {note}")
        if d["status"] == VALID:
            for line in _interpret(g, d):
                L.append(f"    -> {line}")
        else:
            L.append("    -> NOT INTERPRETED (status is not VALID).")
        L.append("")

    valid = [(g, d) for (_, g), d in zip(groups.iterrows(), diagnoses) if d["status"] == VALID]
    L += ["OVERALL READING (VALID groups only)", "-" * 40]
    if not valid:
        L.append("  No VALID group; nothing can be concluded from the confusion matrices.")
    else:
        fp = sum(d["metrics"]["FP"] for _, d in valid); fn = sum(d["metrics"]["FN"] for _, d in valid)
        below = sum(1 for g, _ in valid if g["direction"] == "below")
        above = sum(1 for g, _ in valid if g["direction"] == "above")
        L.append(f"  {len(valid)} valid group(s): {below} below / {above} above the interval; "
                 f"pooled FP={fp}, FN={fn}.")
        L.append("  A cell outside the interval means the Beta model's prediction for that cell was off; the")
        L.append("  confusion matrix shows WHAT the classifier did there (error type, class balance, small n).")
        L.append("  It does not by itself prove WHY the Bayesian model missed the cell.")
    L.append("")

    missing = sorted({n for d in diagnoses if d["status"] == UNAVAILABLE for n in d["notes"]
                      if "validation level" not in n})
    if context.get("missing_results"):
        missing = [f"missing results file: {p}" for p in context["missing_results"]] + missing
    L += ["MISSING COMPONENTS", "-" * 40]
    L += [f"  - {m}" for m in missing] if missing else ["  none"]
    L += ["", "OUTPUT FILES", "-" * 40]
    L += [f"  {name}" for name in OUTPUT_FILES.values()]
    L += [f"  {PLOTS_DIRNAME}/ ({context.get('n_plots', 0)} PNG file(s))", ""]
    return "\n".join(L)


# ============================================================================
# SECTION 9 — MAIN ENTRY POINT
# ============================================================================

def run_confusion_matrix_diagnostic(
    ppc_source: Any,
    results_root: Optional[str] = None,
    splits_dir: Optional[str] = None,
    perturbed_root: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    all_results: Optional[Dict[str, Any]] = None,
    predictor_factory: Optional[PredictorFactory] = None,
    overwrite: bool = False,
    make_plots: bool = True,
    n_cells: int = DEFAULT_N_CELLS,
) -> Dict[str, Any]:
    """
    Diagnose every observation outside the 95 % posterior predictive interval.

    Parameters
    ----------
    ppc_source        : ``run_posterior_predictive_check()`` result (or the full analysis
                        dict, a list of outside_95_rows, or a path to statistical_analysis.json)
    results_root      : folder holding ``<model>/complete_evaluation.json`` and the perturbed
                        CSVs ``<model>/<domain>/semantic/<level>.csv`` (read-only)
    splits_dir        : folder holding ``<domain>/test.csv`` (outputs/data_splits; read-only)
    perturbed_root    : defaults to results_root
    output_dir        : NEW folder for the four result files + plots/
    all_results       : optional {model: complete_evaluation dict}; skips reading from disk
    predictor_factory : optional re-inference (see make_checkpoint_predictor_factory)
    overwrite         : allow replacing this module's own output files (never any other file)
    n_cells           : N of the Smithson-Verkuilen correction used by the PPC (default 300)

    Returns a dict: status ('no_groups' | 'completed'), groups, summary, classes,
    errors, diagnoses, output_dir, files.
    """
    groups = identify_outside_groups(ppc_source)
    if groups.empty:
        msg = "No observations outside the 95% posterior predictive interval - there are no groups to inspect."
        print(msg)
        return {"status": "no_groups", "message": msg, "groups": groups, "files": {}}

    print(f"{len(groups)} group(s) outside the 95% interval:")
    print(groups.to_string(index=False))

    targets = {name: os.path.join(output_dir, fname) for name, fname in OUTPUT_FILES.items()}
    clash = [p for p in targets.values() if os.path.exists(p)]
    if clash and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file(s): {clash}. "
                              "Use a new output_dir or pass overwrite=True.")

    missing_results: List[str] = []
    if all_results is None:
        if results_root is None:
            all_results = {}
            missing_results.append("results_root not provided (complete_evaluation.json unavailable)")
        else:
            all_results, missing_results = load_eval_results(results_root, sorted(groups["model"].unique()))
    store = (_DataStore(splits_dir, perturbed_root or results_root)
             if splits_dir and (perturbed_root or results_root) else None)

    diagnoses = [diagnose_group(g.to_dict(), all_results, store, predictor_factory, n_cells)
                 for _, g in groups.iterrows()]
    for (_, g), d in zip(groups.iterrows(), diagnoses):
        logger.info("[cm-diag] %s %s %s->%s %s: %s", d["status"], g["model"], g["source_domain"],
                    g["target_domain"], g["perturbation_condition"], "; ".join(d["notes"][-2:]))

    summary, classes, errors = build_output_tables(groups, diagnoses)

    os.makedirs(output_dir, exist_ok=True)
    plots: List[str] = save_plots(groups, diagnoses, os.path.join(output_dir, PLOTS_DIRNAME)) if make_plots else []
    summary.to_csv(targets["summary"], index=False)
    classes.to_csv(targets["classes"], index=False)
    errors.to_csv(targets["errors"], index=False)
    report = build_report(groups, diagnoses, summary, {
        "reinfer": predictor_factory is not None, "missing_results": missing_results, "n_plots": len(plots)})
    with open(targets["report"], "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"Diagnostic written to: {output_dir}")

    return {"status": "completed", "groups": groups, "summary": summary, "classes": classes,
            "errors": errors, "diagnoses": diagnoses, "output_dir": output_dir,
            "files": {**targets, "plots": plots}, "report": report}


# ============================================================================
# SECTION 10 — SELF-TESTS (synthetic data; no torch, no PyMC, no real files)
# ============================================================================

def _build_fixture(root: str, models=("mX", "mY"), domains=("DomA", "DomB"), n: int = 20, seed: int = 0):
    """
    Create a tiny, fully synthetic evaluation tree with made-up model / domain
    names (proves nothing is hard-coded):
        <root>/splits/<domain>/test.csv
        <root>/results/<model>/perturbation_results/<domain-lower>/semantic/{low,medium,high}.csv
        <root>/results/<model>/complete_evaluation.json
    In-domain perturbation cells use the common-successful subset (current
    layout); cross-domain cells store the FULL test split (as the real run does).
    """
    rng = np.random.default_rng(seed)
    splits = os.path.join(root, "splits"); results = os.path.join(root, "results")
    tests: Dict[str, pd.DataFrame] = {}
    for di, dom in enumerate(domains):
        ids = np.arange(1000 * (di + 1), 1000 * (di + 1) + n)
        labels = (np.arange(n) % 3 == 0).astype(int)          # ~1/3 class 1
        df = pd.DataFrame({"id": ids, "text": [f"headline {i} {dom}" for i in ids], "label": labels})
        os.makedirs(os.path.join(splits, dom.lower()), exist_ok=True)
        df.to_csv(os.path.join(splits, dom.lower(), "test.csv"), index=False)
        tests[dom] = df

    def block(y_true, flip_idx):
        y_pred = np.array(y_true).copy(); y_pred[list(flip_idx)] ^= 1
        p1 = np.where(y_pred == 1, 0.9, 0.1)
        return {"metrics": calculate_metrics(np.array(y_true), y_pred),
                "predictions": y_pred.tolist(), "true_labels": list(map(int, y_true)),
                "probabilities": np.stack([1 - p1, p1], axis=1).tolist()}

    for model in models:
        res = {"perturbation": {}, "cross_domain_perturbation": {}}
        common: Dict[str, np.ndarray] = {}
        for dom in domains:
            df = tests[dom]; frames = {}
            for li, level in enumerate(("low", "medium", "high")):
                ok = rng.random(n) > (0.1 + 0.15 * li)
                frames[level] = pd.DataFrame({
                    "id": df["id"], "label": df["label"],
                    "perturbed_text": [t + f" ~{level}" for t in df["text"]],
                    "perturbation_success": ok})
                
                d = os.path.join(results, model, "perturbation_results", dom.lower(), "semantic")
                os.makedirs(d, exist_ok=True)

                frames[level].to_csv(os.path.join(d, f"{level}.csv"), index=False)

            ids_ok = _pa.align_common_successful_ids(frames)
            common[dom] = df["id"].isin(ids_ok).to_numpy()

        for src in domains:
            for tgt in domains:
                key = f"{src}_to_{tgt}"; y = tests[tgt]["label"].to_numpy(); cm_mask = common[tgt]
                cell = {"semantic": {}}
                cell["clean"] = block(y, rng.choice(n, 3, replace=False))
                for level in ("low", "medium", "high"):
                    cell["semantic"][level] = block(y, rng.choice(n, 4, replace=False))
                res["cross_domain_perturbation"][key] = cell
                if src == tgt:
                    sub = y[cm_mask]; sem = {}
                    for level in ("low", "medium", "high"):
                        b = block(sub, rng.choice(len(sub), 2, replace=False)); b["common_ids_count"] = int(cm_mask.sum())
                        sem[level] = b
                    cb = block(sub, rng.choice(len(sub), 1)); cb["common_ids_count"] = int(cm_mask.sum())
                    res["perturbation"][src] = {"clean": cb, "semantic": sem}
        os.makedirs(os.path.join(results, model), exist_ok=True)
        with open(os.path.join(results, model, "complete_evaluation.json"), "w") as fh:
            json.dump(res, fh)
    return splits, results


def _ppc_row(model, src, tgt, cond, f1, lo=0.0, hi=0.0):
    return {"model": model, "source_domain": src, "target_domain": tgt, "perturbation_condition": cond,
            "observed": round(f1, 4), "lo95": lo, "hi95": hi}


def run_all_tests() -> None:
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="cmdiag_")
    try:
        splits, results = _build_fixture(tmp)
        with open(os.path.join(results, "mX", "complete_evaluation.json")) as fh:
            mx = json.load(fh)
        f1 = lambda blk: blk["metrics"]["macro_f1"]  # noqa: E731
        diag = mx["perturbation"]["DomA"]["semantic"]["medium"]            # common-subset layout
        crossb = mx["cross_domain_perturbation"]["DomA_to_DomB"]["semantic"]["low"]   # full layout
        clean_b = mx["perturbation"]["DomB"]["clean"]

        # --- 1. empty input -> clean message, no files ------------------------
        out = run_confusion_matrix_diagnostic({"outside_95_rows": []}, output_dir=os.path.join(tmp, "o_empty"))
        assert out["status"] == "no_groups" and not os.path.exists(os.path.join(tmp, "o_empty"))

        # --- 2. identify groups: dedupe + direction + names from data ---------
        a, b = f1(diag), f1(crossb)
        rows = [_ppc_row("mX", "DomA", "DomA", "medium", a, a - 0.2, a - 0.1),   # above its interval
                _ppc_row("mX", "DomA", "DomA", "medium", a, a - 0.2, a - 0.1),   # duplicate key
                _ppc_row("mX", "DomA", "DomB", "low", b, b + 0.1, b + 0.2)]      # below its interval
        g = identify_outside_groups({"ppc": {"outside_95_rows": rows}})
        assert len(g) == 2 and sorted(g["direction"]) == ["above", "below"]

        # --- 3. metrics: hand-computed matrix, single-class group stays 2x2 ---
        m = compute_confusion_metrics([0, 0, 0, 1, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0])
        assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (3, 1, 1, 3) and m["confusion_matrix"] == [[3, 1], [1, 3]]
        assert abs(m["precision_1"] - 0.75) < 1e-12 and abs(m["recall_0"] - 0.75) < 1e-12
        s = compute_confusion_metrics([1, 1, 1], [1, 1, 0])
        assert s["confusion_matrix"] == [[0, 0], [1, 2]] and s["n_actual_0"] == 0
        for bad in (lambda: compute_confusion_metrics([], []), lambda: compute_confusion_metrics([0, 1], [0]),
                    lambda: compute_confusion_metrics([0, 2], [0, 1])):
            try:
                bad(); raise AssertionError("expected ValueError")
            except ValueError:
                pass

        # single-class group: interpretation text must not divide by zero
        sc = compute_confusion_metrics([0, 0, 0, 0], [0, 0, 1, 0])
        txt = " ".join(_interpret({"direction": "above", "observed": 0.5, "lo95": 0.1, "hi95": 0.2},
                                  {"metrics": sc, "alignment": None, "stored_block": ""}))
        assert "undefined" in txt

        # --- 4. full run, VALID groups (stored-predictions level) -------------
        rows = [_ppc_row("mX", "DomA", "DomA", "medium", f1(diag)),
                _ppc_row("mX", "DomA", "DomB", "low", f1(crossb)),
                _ppc_row("mX", "DomB", "DomB", "clean", f1(clean_b))]
        o1 = os.path.join(tmp, "o1")
        r = run_confusion_matrix_diagnostic(rows, results_root=results, splits_dir=splits, output_dir=o1)
        assert r["status"] == "completed"
        assert [d["status"] for d in r["diagnoses"]] == [VALID] * 3, [d["notes"] for d in r["diagnoses"]]
        for fname in OUTPUT_FILES.values():
            assert os.path.exists(os.path.join(o1, fname)), fname
        assert len(r["files"]["plots"]) == 4
        summ = pd.read_csv(os.path.join(o1, OUTPUT_FILES["summary"]))
        assert len(summ) == 3 and (summ["TN"] + summ["FP"] + summ["FN"] + summ["TP"] == summ["n"]).all()
        by = summ.set_index(["source_domain", "target_domain", "perturbation_condition"])
        assert by.loc[("DomA", "DomA", "medium"), "row_alignment"] == "common_subset"
        assert by.loc[("DomA", "DomB", "low"), "row_alignment"] == "full_test_set"
        assert by.loc[("DomA", "DomB", "low"), "n"] == 20 and by.loc[("DomA", "DomA", "medium"), "n"] < 20
        cls = pd.read_csv(os.path.join(o1, OUTPUT_FILES["classes"])); assert len(cls) == 6
        errs = pd.read_csv(os.path.join(o1, OUTPUT_FILES["errors"]))
        n_err = int((summ["FP"] + summ["FN"]).sum()); assert len(errs) == n_err and n_err > 0
        assert set(errs["error_type"]) <= {"FP", "FN"} and errs["text"].notna().all()
        pert = errs[errs["perturbation_condition"] == "low"]
        assert pert["text"].str.endswith("~low").all() and not pert["original_text"].str.contains("~").any()
        assert "[VALID]" in r["report"] and "NOT INTERPRETED" not in r["report"]

        # --- 5. overwrite guard ------------------------------------------------
        try:
            run_confusion_matrix_diagnostic(rows, results_root=results, splits_dir=splits, output_dir=o1)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        run_confusion_matrix_diagnostic(rows, results_root=results, splits_dir=splits, output_dir=o1, overwrite=True)

        # --- 6. INVALID: stored Macro-F1 tampered; PPC value mismatch ---------
        bad_all = json.loads(json.dumps({"mX": mx}))
        bad_all["mX"]["perturbation"]["DomA"]["semantic"]["medium"]["metrics"]["macro_f1"] += 0.01
        st = _DataStore(splits, results)
        d = diagnose_group(_ppc_row("mX", "DomA", "DomA", "medium", f1(diag)), bad_all, st)
        assert d["status"] == INVALID and any("Macro-F1 mismatch" in n for n in d["notes"])
        d = diagnose_group(_ppc_row("mX", "DomA", "DomA", "medium", f1(diag) - 0.05), {"mX": mx}, st)
        assert d["status"] == INVALID and any("PPC observed" in n for n in d["notes"])
        o2 = os.path.join(tmp, "o2")
        r2 = run_confusion_matrix_diagnostic([_ppc_row("mX", "DomA", "DomA", "medium", f1(diag) - 0.05)],
                                             all_results={"mX": mx}, results_root=results, splits_dir=splits, output_dir=o2)
        assert len(pd.read_csv(os.path.join(o2, OUTPUT_FILES["errors"]))) == 0
        assert "NOT INTERPRETED" in r2["report"] and "[VALID]" not in r2["report"]

        # boundary-corrected PPC value is accepted (N = 300 cells)
        corr = float(apply_boundary_correction(np.array([f1(diag)]), n=300)[0])
        d = diagnose_group(_ppc_row("mX", "DomA", "DomA", "medium", corr), {"mX": mx}, st)
        assert d["status"] == VALID and d["ppc_form"] == "boundary_corrected"

        # ids not alignable (labels reordered) -> INVALID
        scr = json.loads(json.dumps({"mX": mx}))
        t = scr["mX"]["cross_domain_perturbation"]["DomA_to_DomB"]["semantic"]["low"]
        t["true_labels"] = t["true_labels"][::-1]
        d = diagnose_group(_ppc_row("mX", "DomA", "DomB", "low", f1(crossb)), scr, st)
        assert d["status"] == INVALID and any("ids cannot be matched" in n for n in d["notes"])

        # --- 7. UNAVAILABLE: model not loaded / dataset missing ---------------
        d = diagnose_group(_ppc_row("mZ", "DomA", "DomA", "low", 0.5), {"mX": mx}, st)
        assert d["status"] == UNAVAILABLE and d["metrics"] is None
        d = diagnose_group(_ppc_row("mX", "DomA", "DomA", "medium", f1(diag)), {"mX": mx}, None)
        assert d["status"] == UNAVAILABLE and d["metrics"] is not None
        d = diagnose_group(_ppc_row("mX", "DomA", "DomA", "medium", f1(diag)), {"mX": mx},
                           _DataStore(os.path.join(tmp, "nowhere"), results))
        assert d["status"] == UNAVAILABLE and any("dataset component missing" in n for n in d["notes"])

        # --- 8. re-inference: identical OK; different preds INVALID even when Macro-F1 is equal
        blk = crossb; y_t = np.array(blk["true_labels"]); y_p = np.array(blk["predictions"])
        same = lambda model, src: (lambda texts: (y_p.copy(), None))                      # noqa: E731
        d = diagnose_group(_ppc_row("mX", "DomA", "DomB", "low", f1(crossb)), {"mX": mx}, st, same)
        assert d["status"] == VALID and d["reinference"] == "identical"
        # swap a wrong->right and a right->wrong prediction of the SAME class: same Macro-F1, different predictions
        wrong1 = np.where((y_t == 1) & (y_p == 0))[0]; right1 = np.where((y_t == 1) & (y_p == 1))[0]
        wrong0 = np.where((y_t == 0) & (y_p == 1))[0]; right0 = np.where((y_t == 0) & (y_p == 0))[0]
        y_alt = y_p.copy()
        if len(wrong1) and len(right1):
            y_alt[wrong1[0]], y_alt[right1[0]] = 1, 0           # moves the miss to another headline
        else:
            y_alt[wrong0[0]], y_alt[right0[0]] = 0, 1
        assert (y_alt != y_p).sum() == 2
        assert abs(compute_confusion_metrics(y_t, y_alt)["macro_f1"] - f1(crossb)) < 1e-12   # equal F1 ...
        d = diagnose_group(_ppc_row("mX", "DomA", "DomB", "low", f1(crossb)), {"mX": mx}, st,
                           lambda m_, s_: (lambda texts: (y_alt.copy(), None)))
        assert d["status"] == INVALID and d["reinference"] == "different"               # ... still INVALID
        def _boom(m_, s_): raise FileNotFoundError("checkpoint not found: x")
        d = diagnose_group(_ppc_row("mX", "DomA", "DomB", "low", f1(crossb)), {"mX": mx}, st, _boom)
        assert d["status"] == UNAVAILABLE and d["reinference"] == "unavailable"

        # --- 9. file-system guard: existing evaluation files untouched --------
        before = os.path.getmtime(os.path.join(results, "mX", "complete_evaluation.json"))
        run_confusion_matrix_diagnostic(rows, results_root=results, splits_dir=splits,
                                        output_dir=os.path.join(tmp, "o3"))
        assert os.path.getmtime(os.path.join(results, "mX", "complete_evaluation.json")) == before
        print("All confusion-matrix diagnostic tests passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# SECTION 11 — CLI
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Confusion matrix diagnostic for PPC outside-95% observations.")
    p.add_argument("--self-test", action="store_true", help="run synthetic tests and exit")
    p.add_argument("--ppc-json", help="statistical_analysis.json (must contain ppc.outside_95_rows)")
    p.add_argument("--results-root", help="folder with <model>/complete_evaluation.json and perturbed CSVs")
    p.add_argument("--perturbed-root", help="folder with <model>/perturbation_results/<domain-lower>/semantic/<level>.csv (default: results root)")
    p.add_argument("--splits-dir", help="folder with <domain>/test.csv (outputs/data_splits)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--reinfer", action="store_true", help="independent re-inference from existing checkpoints")
    p.add_argument("--checkpoint-dir", help="folder with <model>/<Domain>/best_model.pt (needed for --reinfer)")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true", help="replace this tool's own output files")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if a.self_test:
        run_all_tests(); return 0
    if not (a.ppc_json and a.results_root and a.splits_dir):
        p.error("--ppc-json, --results-root and --splits-dir are required (or use --self-test)")
    factory = None
    if a.reinfer:
        if not a.checkpoint_dir:
            p.error("--reinfer needs --checkpoint-dir")
        factory = make_checkpoint_predictor_factory(a.checkpoint_dir, device=a.device)
    res = run_confusion_matrix_diagnostic(
        a.ppc_json, results_root=a.results_root, splits_dir=a.splits_dir, perturbed_root=a.perturbed_root,
        output_dir=a.output_dir, predictor_factory=factory, overwrite=a.overwrite, make_plots=not a.no_plots)
    return 0 if res["status"] in ("completed", "no_groups") else 1


if __name__ == "__main__":
    sys.exit(main())
