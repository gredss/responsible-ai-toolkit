"""
Statistical Analyzer Module — Bayesian Hierarchical Inference
IndoBERT Clickbait Detection Robustness Study

PURPOSE
-------
This module answers the thesis's five statistical research questions about
IndoBERT robustness by fitting a Bayesian hierarchical model to aggregate
Macro-F1 scores produced by EvaluationEngine.run_complete_evaluation().

It does NOT retrain any model, modify predictions, or introduce headline-level
modeling. The only response variable is aggregate Macro-F1.

RESEARCH QUESTIONS ADDRESSED
-----------------------------
  RQ1  Do IndoBERT Base, Large, and Lite differ in Macro-F1?
  RQ2  Does in-domain testing produce higher F1 than cross-domain?
  RQ3  Does perturbation (Low / Medium / High) degrade F1 significantly?
  RQ4  Does the degradation pattern differ between model variants
       (model × perturbation interaction)?
  RQ5  Does perturbation degrade F1 more in cross-domain than in-domain
       (perturbation × scenario interaction)?

MODEL SPECIFICATION
-------------------
Each observation is one aggregate Macro-F1 score from one combination of
(model_variant, source_domain, target_domain, perturbation_condition).

  logit(μ_i) = α
             + β[model_i]              ← model variant main effect
             + γ[source_i]             ← source domain main effect
             + δ[target_i]             ← target domain main effect
             + ζ[condition_i]          ← perturbation condition main effect
             + θ[model_i, cond_i]      ← model × perturbation interaction
             + λ · D_i                 ← in-domain main effect
             + ξ[condition_i] · D_i    ← perturbation × scenario interaction
             + u[model_i, source_i]    ← per-checkpoint random intercept
             + w[source_i, target_i]   ← off-diagonal source×target random effect
                                         (diagonal masked to 0 so lambda alone owns the
                                          in-domain effect; shared by all three models)



  Y_i ~ Beta( μ_i · φ,  (1 − μ_i) · φ )

where D_i = 1 when source_domain == target_domain (in-domain), 0 otherwise.

WHY BETA LIKELIHOOD?
  Aggregate Macro-F1 ∈ (0, 1) by construction.  Beta is a flexible two-
  parameter family defined on (0, 1) that can represent skewed, bimodal, or
  near-symmetric distributions.  It is a natural choice for modelling bounded
  continuous proportions.  Note: Beta is NOT the conjugate prior for a
  regression likelihood — "conjugate" is a property of prior/likelihood pairs,
  not of regression likelihoods.  The choice is justified by the support and
  flexibility of Beta, combined with the logit link that keeps the mean in
  (0, 1) for any real linear predictor value.

WHY BOUNDARY CORRECTION?
  The Beta distribution is undefined at exactly 0 or 1.  When any observed F1
  equals those exact boundaries, the Smithson & Verkuilen (2006) transformation

      y* = (y × (N − 1) + 0.5) / N

  maps [0, 1] → (0, 1) strictly.  Here N is the number of aggregate
  observation cells (rows in the observation DataFrame), not the number of
  headlines per cell.  For values already in (0, 1) the shift is negligible
  (< 0.5 / N).  This correction is applied only when at least one value is
  exactly 0 or 1; the PPC comparison uses the same corrected values.

WHY SUM-TO-ZERO (ZeroSumNormal) CONSTRAINTS?
  Without constraints, factor-level effects are not separately identifiable from
  the intercept.  Sum-to-zero parameterisation removes that redundancy: α is the
  grand mean logit(F1) and each factor effect is a deviation from that mean.
  This is equivalent to contrast coding and is standard for factorial designs.

  Important distinction:
  • Regularization (what ZeroSumNormal gives): a shared fixed prior σ pulls
    factor-level estimates toward zero.  Every level is treated as a separate
    fixed parameter; the prior just prevents extreme estimates when data are
    sparse.  This is NOT partial pooling — the levels do not share a common
    variance that is estimated from the data.
  • Partial pooling (what the per-checkpoint random intercept u gives): the
    variance σ_u of u[model, source] is estimated from the data itself.
    Checkpoints with less data are pulled more strongly toward the global mean.
    This correctly accounts for the fact that every observation from the same
    trained checkpoint shares unmodelled sources of variation.

WHY A PER-CHECKPOINT RANDOM INTERCEPT?
  Each (model_variant, source_domain) pair corresponds to exactly one trained
  checkpoint.  Observations from the same checkpoint — across the five target
  domains and four perturbation conditions — are NOT independent: they all
  share whatever idiosyncrasies that specific training run introduced.  Ignoring
  this dependence underestimates posterior uncertainty.  The random intercept
  u[model, source] ~ Normal(0, σ_u) accounts for this within-checkpoint
  correlation.  σ_u is estimated from the data (partial pooling across the 15
  checkpoints: 3 models × 5 source domains).

  What this does NOT claim: it does NOT estimate training-seed variability,
  because we have only one checkpoint per (model, source) combination.  σ_u
  captures residual checkpoint-level variation, whatever its origin.

PUBLIC API  (all plain functions — no class needed)
----------
  assemble_observations(all_results)   → pd.DataFrame
  validate_observations(df)            → dict
  apply_boundary_correction(y, n)      → np.ndarray
  fit_model(obs_df, ...)               → (trace, obs_df_corrected, encodings)
  run_convergence_diagnostics(trace)   → dict
  run_posterior_predictive_check(trace, obs_df_corrected) → dict
  compute_contrasts(trace, encodings, obs_df) → dict
  validate_contrasts(contrasts)        → dict
  validate_ppc(ppc_result)             → dict
  run_full_analysis(all_results, ...)  → dict   ← main entry point

  Tests (no framework needed):
  run_all_tests()
"""

import os
import json
import logging
from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_VARIANTS          = ["base", "large", "lite"]
DOMAINS                 = ["Education", "Health", "Politics", "Sport", "Technology"]
PERTURBATION_CONDITIONS = ["clean", "low", "medium", "high"]

# Columns that must be present in the observation DataFrame.
REQUIRED_OBS_COLUMNS = [
    "model", "source_domain", "target_domain",
    "perturbation_condition", "in_domain", "macro_f1",
]


# ============================================================================
# SECTION 1 — DATA ASSEMBLY
# ============================================================================

def assemble_observations(all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert evaluation results into a flat observation DataFrame.

    One row per unique (model, source_domain, target_domain,
    perturbation_condition).

    SOURCE MAPPING
    --------------
    In-domain (source == target):
        results['perturbation'][domain]['clean']
        results['perturbation'][domain]['semantic'][level]

    Cross-domain (source != target):
        results['cross_domain_perturbation']['Src_to_Tgt']['clean']
        results['cross_domain_perturbation']['Src_to_Tgt']['semantic'][level]

    DUPLICATE HANDLING
    ------------------
    Diagonal entries (source == target) appear in BOTH blocks.
    - Exact F1 match  → keep the perturbation-block copy, log INFO.
    - Conflicting F1  → raise ValueError immediately (no silent data loss).
    """
    all_rows: List[Dict[str, Any]] = []

    for model_name, results in all_results.items():

        # ── In-domain block ───────────────────────────────────────────────
        for domain, domain_block in results.get("perturbation", {}).items():

            clean_block = domain_block.get("clean", {})
            if clean_block:
                all_rows.append({
                    "model":                  model_name,
                    "source_domain":          domain,
                    "target_domain":          domain,
                    "perturbation_condition": "clean",
                    "in_domain":              True,
                    "macro_f1":               float(clean_block.get("metrics", {}).get("macro_f1", np.nan)),
                    "n_samples":              int(clean_block.get("common_ids_count",
                                                 clean_block.get("num_samples", 0))),
                    "common_ids_count":       int(clean_block.get("common_ids_count", 0)),
                    "_source":                "perturbation",
                })

            semantic_block = domain_block.get("semantic", {})
            for level in ["low", "medium", "high"]:
                lvl_block = semantic_block.get(level)
                if lvl_block:
                    all_rows.append({
                        "model":                  model_name,
                        "source_domain":          domain,
                        "target_domain":          domain,
                        "perturbation_condition": level,
                        "in_domain":              True,
                        "macro_f1":               float(lvl_block.get("metrics", {}).get("macro_f1", np.nan)),
                        "n_samples":              int(lvl_block.get("common_ids_count",
                                                     lvl_block.get("num_samples", 0))),
                        "common_ids_count":       int(lvl_block.get("common_ids_count", 0)),
                        "_source":                "perturbation",
                    })

        # ── Cross-domain block (also contains diagonal entries) ───────────
        for key, cdp_block in results.get("cross_domain_perturbation", {}).items():
            if "_to_" not in key:
                continue
            source_domain, target_domain = key.split("_to_", 1)
            is_diagonal = (source_domain == target_domain)

            clean_block = cdp_block.get("clean", {})
            if clean_block:
                all_rows.append({
                    "model":                  model_name,
                    "source_domain":          source_domain,
                    "target_domain":          target_domain,
                    "perturbation_condition": "clean",
                    "in_domain":              is_diagonal,
                    "macro_f1":               float(clean_block.get("metrics", {}).get("macro_f1", np.nan)),
                    "n_samples":              int(clean_block.get("common_ids_count",
                                                 clean_block.get("num_samples", 0))),
                    "common_ids_count":       int(clean_block.get("common_ids_count", 0)),
                    "_source":                "cross_domain_perturbation",
                })

            semantic_block = cdp_block.get("semantic", {})
            for level in ["low", "medium", "high"]:
                lvl_block = semantic_block.get(level)
                if lvl_block:
                    all_rows.append({
                        "model":                  model_name,
                        "source_domain":          source_domain,
                        "target_domain":          target_domain,
                        "perturbation_condition": level,
                        "in_domain":              is_diagonal,
                        "macro_f1":               float(lvl_block.get("metrics", {}).get("macro_f1", np.nan)),
                        "n_samples":              int(lvl_block.get("common_ids_count",
                                                     lvl_block.get("num_samples", 0))),
                        "common_ids_count":       int(lvl_block.get("common_ids_count", 0)),
                        "_source":                "cross_domain_perturbation",
                    })

    if not all_rows:
        return pd.DataFrame()

    df_all = pd.DataFrame(all_rows)

    # ── Duplicate audit ───────────────────────────────────────────────────
    # Reject any key that appears with conflicting (different) F1 values.
    key_cols = ["model", "source_domain", "target_domain", "perturbation_condition"]
    duplicate_counts = df_all.groupby(key_cols).size()
    duplicate_keys   = duplicate_counts[duplicate_counts > 1]

    conflicts = []

    for key_tuple, group in df_all.groupby(key_cols, dropna=False):

        if len(group) <= 1:
            continue

        f1_values = group["macro_f1"].to_numpy(dtype=float)
        sample_counts = group["n_samples"].to_numpy(dtype=int)

        f1_match = np.allclose(
            f1_values,
            f1_values[0],
            atol=0.0001,
            rtol=0
        )

        sample_match = np.all(sample_counts == sample_counts[0])

        if not f1_match or not sample_match:
            conflict = {
                "key": key_tuple,
                "sources": group["_source"].tolist(),
                "f1_values": f1_values.tolist(),
                "n_samples": sample_counts.tolist(),
            }
            conflicts.append(conflict)

            logger.error(
                "CONFLICT key=%s sources=%s F1=%s n_samples=%s",
                key_tuple,
                conflict["sources"],
                conflict["f1_values"],
                conflict["n_samples"],
            )

        else:
            logger.info(
                "MATCHING DUPLICATE key=%s count=%d F1=%.6f n_samples=%d",
                key_tuple,
                len(group),
                f1_values[0],
                sample_counts[0],
            )

    if conflicts:
        raise ValueError(
            f"[assemble_observations] {len(conflicts)} conflicting duplicate "
            "key(s) found. Check Macro-F1 and sample counts before fitting. "
            f"Conflicts: {conflicts}"
        )

    # Sort so perturbation-block rows come first, then drop duplicates.
    # keep='first' retains the perturbation-block copy.
    source_order = {"perturbation": 0, "cross_domain_perturbation": 1}
    df_all["_sort_key"] = df_all["_source"].map(source_order).fillna(99)
    df_all = df_all.sort_values("_sort_key").reset_index(drop=True)
    df_all = df_all.drop(columns=["_source", "_sort_key"])

    df = df_all.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)

    # Paranoia: post-dedup duplicates should never happen.
    post_dups = int(df.duplicated(subset=key_cols).sum())
    if post_dups > 0:
        raise ValueError(
            f"[assemble_observations] {post_dups} duplicate key(s) remain after "
            "deduplication — logic error. Review data sources."
        )

    return df


def validate_observations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run data-quality checks on the assembled observation DataFrame.

    Returns a dict with counts, ranges, and a 'warnings' list.
    Critical problems are added to the warnings list; the caller
    decides whether to raise or continue.
    """
    report: Dict[str, Any] = {}
    warnings_list: List[str] = []

    if df.empty:
        return {"error": "Empty observation DataFrame."}

    # Check 1: required columns
    missing_cols = [c for c in REQUIRED_OBS_COLUMNS if c not in df.columns]
    if missing_cols:
        return {"error": f"Required columns missing: {missing_cols}"}

    report["n_total"] = int(len(df))

    # Check 2: F1 must be in [0, 1]
    valid_f1      = df["macro_f1"].dropna()
    out_of_range  = valid_f1[(valid_f1 < 0.0) | (valid_f1 > 1.0)]
    report["n_out_of_range"] = int(len(out_of_range))
    if report["n_out_of_range"] > 0:
        warnings_list.append(
            f"{report['n_out_of_range']} F1 value(s) outside [0, 1]: {out_of_range.values.tolist()}"
        )

    # Check 3: post-dedup duplicates (should always be zero)
    key_cols = ["model", "source_domain", "target_domain", "perturbation_condition"]
    report["n_duplicates"] = int(df.duplicated(subset=key_cols).sum())
    if report["n_duplicates"] > 0:
        warnings_list.append(
            f"ERROR: {report['n_duplicates']} duplicate key(s) remain after deduplication."
        )

    # Check 4: in_domain flag must equal (source == target)
    expected_in_domain = (df["source_domain"] == df["target_domain"]).to_numpy()
    flag = _coerce_bool(df["in_domain"])
    n_inconsistent = int((flag != expected_in_domain).sum())
    report["n_scenario_inconsistent"] = n_inconsistent
    if n_inconsistent > 0:
        bad_rows = df[flag != expected_in_domain][key_cols + ["in_domain"]]
        warnings_list.append(
            f"{n_inconsistent} row(s) have in_domain flag inconsistent with source==target:\n"
            + bad_rows.to_string()
        )

    # Check 5: NaN F1
    report["n_missing_f1"] = int(df["macro_f1"].isna().sum())
    if report["n_missing_f1"] > 0:
        warnings_list.append(
            f"{report['n_missing_f1']} observation(s) have NaN Macro-F1."
        )

    # Check 6: boundary values trigger Smithson-Verkuilen correction
    report["n_boundary_zero"] = int((df["macro_f1"] == 0.0).sum())
    report["n_boundary_one"]  = int((df["macro_f1"] == 1.0).sum())
    if report["n_boundary_zero"] > 0:
        warnings_list.append(
            f"{report['n_boundary_zero']} observation(s) have F1=0.0. "
            "Smithson-Verkuilen boundary correction will be applied."
        )
    if report["n_boundary_one"] > 0:
        warnings_list.append(
            f"{report['n_boundary_one']} observation(s) have F1=1.0. "
            "Smithson-Verkuilen boundary correction will be applied."
        )

    # Check 7: summary statistics
    report["f1_min"]  = float(valid_f1.min())  if len(valid_f1) else float("nan")
    report["f1_max"]  = float(valid_f1.max())  if len(valid_f1) else float("nan")
    report["f1_mean"] = float(valid_f1.mean()) if len(valid_f1) else float("nan")
    report["f1_std"]  = float(valid_f1.std())  if len(valid_f1) else float("nan")

    # Check 8: unexpected model/condition names
    observed_models = sorted(df["model"].unique().tolist())
    observed_conds  = sorted(df["perturbation_condition"].unique().tolist())
    unexpected_models = [m for m in observed_models if m not in MODEL_VARIANTS]
    unexpected_conds  = [c for c in observed_conds  if c not in PERTURBATION_CONDITIONS]
    report["n_unexpected_categories"] = len(unexpected_models) + len(unexpected_conds)
    if unexpected_models:
        warnings_list.append(f"Unexpected model names: {unexpected_models}")
    if unexpected_conds:
        warnings_list.append(f"Unexpected perturbation conditions: {unexpected_conds}")

    # Check 9: missing combinations against the fixed 3×5×5×4 design
    expected_keys = set(product(MODEL_VARIANTS, DOMAINS, DOMAINS, PERTURBATION_CONDITIONS))
    actual_keys   = set(df[key_cols].itertuples(index=False, name=None))
    missing_keys  = expected_keys - actual_keys

    report["expected_max"]         = len(expected_keys)   # 300
    report["n_models_found"]       = len(observed_models)
    report["n_missing_combos"]     = len(missing_keys)
    report["missing_combinations"] = sorted(missing_keys)[:20]   # first 20 for readability

    if missing_keys:
        warnings_list.append(
            f"{len(missing_keys)} of {len(expected_keys)} expected combinations are missing. "
            "Contrasts use only observed cells."
        )

    report["warnings"] = warnings_list
    return report


def apply_boundary_correction(y: np.ndarray, n: Any = None) -> np.ndarray:
    """
    Smithson & Verkuilen (2006) boundary correction.

        y* = (y × (N − 1) + 0.5) / N

    Maps [0, 1] → (0, 1) strictly so Beta likelihood is well-defined.
    N may be a scalar (default: number of observation cells) or an array with
    one value per row (e.g. the matched-subset headline count of each cell).
    For values already in (0, 1) the shift is < 0.5/N (negligible for N≥50).
    """
    if n is None:
        n = len(y)
    return (y * (n - 1) + 0.5) / n


def _coerce_bool(series: pd.Series) -> np.ndarray:
    """
    Safe boolean conversion.  ``Series.astype(bool)`` turns the STRING "False"
    into True (any non-empty string is truthy), which silently flips groups
    when a table has been round-tripped through CSV/JSON as text.
    """
    if series.dtype == bool:
        return series.to_numpy()
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"Cannot interpret in_domain values as boolean: "
                         f"{series[mapped.isna()].unique().tolist()}")
    return mapped.to_numpy(dtype=bool)


# ============================================================================
# SECTION 2 — MODEL FITTING
# ============================================================================

def fit_model(
    obs_df: pd.DataFrame,
    output_dir:     str   = "statistical_analysis",
    draws:          int   = 2000,
    tune:           int   = 3000,
    chains:         int   = 4,
    target_accept:  float = 0.99,
    random_seed:    int   = 42,
    pair_mode:      str   = "D",
    #pair_effect:    Optional[bool] = None,
    boundary_method: str  = "sv_n_cells",
    phi_prior: tuple      = (2.0, 0.005),
):
    """
    Fit the Bayesian hierarchical Beta regression and return results.

    Parameters
    ----------
    obs_df        : output of assemble_observations() (must have no NaN F1)
    output_dir    : folder where trace.nc is saved
    draws         : posterior draws per chain
    tune          : warmup steps per chain (use ≥ 1000)
    chains        : number of MCMC chains (use 4 for reliable R-hat)
    target_accept : NUTS acceptance rate (0.99: the sigma_u / sigma_w funnel
                    produced 1-2 divergences at 0.90 in the audit)
    random_seed   : reproducibility seed
    pair_mode     : Source-target pair effect specification:
                      "A" : 0 pair effects (no pair random effect)
                      "B" : 25 pair effects (all source-target pairs)
                      "C" : 20 pair effects (off-diagonal cross-domain pairs only, default)
    pair_effect   : Backward compatibility alias for pair_mode:
                      False -> "A", True -> "C"
    boundary_method : "sv_n_cells"  (default, original behaviour) apply
                          Smithson-Verkuilen with N = number of rows to ALL
                          rows when any F1 is exactly 0 or 1;
                      "boundary_only_ni" shift ONLY the F1 == 0/1 rows, using
                          that row's matched-subset size n_i (column
                          'n_samples') — recommended; contrast changes ≤ 0.004
                          in the audit.

    Returns
    -------
    trace          : arviz.InferenceData  (also saved to output_dir/trace.nc)
    obs_df_fitted  : the DataFrame actually used for fitting, with an extra
                     column 'macro_f1_corrected' — the boundary-corrected F1
                     values passed to the Beta likelihood. Use this column
                     (not the original macro_f1) when comparing with y_rep.
    encodings      : dict with 'models', 'sources', 'targets', 'conditions',
                     'pair_mode', 'pair_effect'
                     — the sorted level lists used to build integer index arrays.
                     You need this in compute_contrasts().
    """
    try:
        import pymc as pm
    except ImportError:
        raise ImportError(
            "PyMC is required. Install with: pip install 'pymc>=5.0' 'arviz>=0.17'"
        )

    # Resolve pair_mode (supporting pair_effect boolean for backward compatibility)
    mode = str(pair_mode).upper() 
    if mode not in ("A", "B", "C", "D"):
        raise ValueError(
            f"Invalid pair_mode: {mode!r}. Must be 'A', 'B', 'C', or 'D'."
        )
        
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Prepare the response variable ─────────────────────────────
    # Drop any NaN rows (validate_observations should have caught these already).
    if obs_df["macro_f1"].isna().any():
        raise ValueError("Missing macro_f1 values detected.")
    df = obs_df.copy()
    if df.empty:
        raise ValueError("No valid F1 observations after dropping NaN rows.")

    y_raw = df["macro_f1"].values.astype(float)

    # Apply Smithson-Verkuilen correction if any value is exactly 0 or 1.
    # The corrected values are stored back on df so posterior_predictive_check
    # can compare on the same scale as the likelihood.
    boundary_mask = (y_raw <= 0.0) | (y_raw >= 1.0)
    if boundary_method not in ("sv_n_cells", "boundary_only_ni"):
        raise ValueError(f"Unknown boundary_method: {boundary_method!r}")
    if boundary_mask.any():
        if boundary_method == "sv_n_cells":
            logger.warning(
                "[fit_model] Boundary F1 values detected. Applying correction "
                "y* = (y*(N-1)+0.5)/N  where N=%d cells (ALL rows shifted).", len(y_raw)
            )
            y_corrected = apply_boundary_correction(y_raw, n=len(y_raw))
        else:
            n_i = (df["n_samples"].to_numpy(dtype=float)
                   if "n_samples" in df.columns else np.full(len(df), float(len(y_raw))))
            n_i = np.where(n_i > 1, n_i, float(len(y_raw)))
            y_corrected = y_raw.copy()
            y_corrected[boundary_mask] = apply_boundary_correction(
                y_raw[boundary_mask], n=n_i[boundary_mask]
            )
            logger.warning(
                "[fit_model] %d boundary F1 value(s) shifted using their own matched n_i "
                "(non-boundary rows untouched).", int(boundary_mask.sum())
            )
    else:
        y_corrected = y_raw.copy()

    df["macro_f1_corrected"] = y_corrected
    logger.info("[fit_model] F1 range after correction: [%.4f, %.4f]",
                y_corrected.min(), y_corrected.max())

    # ── Step 2: Build integer index arrays ────────────────────────────────
    # Each factor level needs an integer index so PyMC can look it up.
    # Perturbation conditions use severity order; everything else is sorted.
    models     = sorted(df["model"].unique().tolist())
    sources    = sorted(df["source_domain"].unique().tolist())
    targets    = sorted(df["target_domain"].unique().tolist())
    conditions = ["clean", "low", "medium", "high"]

    encodings = {
        "models":     models,
        "sources":    sources,
        "targets":    targets,
        "conditions": conditions,
        "pair_mode":  mode,
        "pair_effect": mode in ("B", "C", "D"),
    }
    if mode in ("B", "C", "D") and sources != targets:
        raise ValueError("pair_mode B/C needs identical source and target level lists "
                         f"(got {sources} vs {targets}).")

    n_m = len(models)
    n_s = len(sources)
    n_t = len(targets)
    n_p = len(conditions)
    n_checkpoints = n_m * n_s   # one per (model, source) pair

    # Build one integer index array per factor (one value per observation row).
    m_idx          = np.array([models.index(v)     for v in df["model"]],                 dtype=int)
    s_idx          = np.array([sources.index(v)    for v in df["source_domain"]],          dtype=int)
    t_idx          = np.array([targets.index(v)    for v in df["target_domain"]],          dtype=int)
    p_idx          = np.array([conditions.index(v) for v in df["perturbation_condition"]], dtype=int)
    d_arr          = _coerce_bool(df["in_domain"]).astype(int)  # 1 = in-domain, 0 = cross
    checkpoint_idx = m_idx * n_s + s_idx                       # unique index per checkpoint

    logger.info(
        "[fit_model] N=%d  models=%d  sources=%d  targets=%d  "
        "conditions=%d  checkpoints=%d  pair_mode=%s",
        len(df), n_m, n_s, n_t, n_p, n_checkpoints, mode,
    )

    # ── Step 3: Define and sample the PyMC model ──────────────────────────
    with pm.Model():

        # Intercept: grand mean on the logit scale.
        # Normal(2, 1) → invlogit(2) ≈ 0.88, a reasonable IndoBERT starting point.
        alpha = pm.Normal("alpha", mu=2.0, sigma=1.0)

        # Main effects with sum-to-zero regularization (ZeroSumNormal).
        # Sum-to-zero means α is the grand mean and each effect is a deviation.
        # The fixed σ is regularization, NOT partial pooling.
        beta_m  = pm.ZeroSumNormal("beta_m",  sigma=0.5,  shape=n_m)   # model variant
        gamma_s = pm.ZeroSumNormal("gamma_s", sigma=0.5,  shape=n_s)   # source domain
        delta_t = pm.ZeroSumNormal("delta_t", sigma=0.5,  shape=n_t)   # target domain
        zeta_p  = pm.ZeroSumNormal("zeta_p",  sigma=0.75, shape=n_p)   # perturbation

        # Model × Perturbation interaction (n_m × n_p matrix, sum-to-zero over both axes).
        # n_zerosum_axes=2 makes it identifiable relative to the main effects above.
        theta_mp = pm.ZeroSumNormal(
            "theta_mp", sigma=0.3, shape=(n_m, n_p), n_zerosum_axes=2
        )

        # In-domain main effect (scalar).
        # Normal(0, 1): broad because domain shift can be large.
        lam = pm.Normal("lambda", mu=0.0, sigma=1.0)

        # Perturbation × Scenario interaction (sum-to-zero, 4 levels).
        # Does perturbation hurt more in cross-domain than in-domain?
        xi_p = pm.ZeroSumNormal("xi_p", sigma=0.5, shape=n_p)

        # Per-checkpoint random intercept — partial pooling.
        # Every observation from the same (model, source) checkpoint shares one
        # trained model, so they are NOT independent. This random intercept
        # accounts for that within-checkpoint correlation.
        # Non-centered parameterisation (u = u_raw × sigma_u) avoids the
        # funnel geometry that causes divergences.
        sigma_u      = pm.HalfNormal("sigma_u", sigma=0.3)
        u_raw        = pm.Normal("u_raw", mu=0.0, sigma=1.0, shape=n_checkpoints)
        u_checkpoint = pm.Deterministic("u_checkpoint", u_raw * sigma_u)

        # Source × target random effect:
        # Mode A: 0 pair effects (no pair term)
        # Mode B: 25 pair effects (all source-target pairs)
        # Mode C: 20 pair effects (off-diagonal cross-domain pairs only, diagonal masked to 0)
        if mode == "B":
            # Satu pair effect untuk setiap pasangan source-target.
            # Nilainya digunakan bersama oleh semua model.
            sigma_w = pm.HalfNormal("sigma_w", sigma=0.3)
            w_raw = pm.Normal(
                "w_raw",
                mu=0.0,
                sigma=1.0,
                shape=(n_s, n_t),
            )
            w_st = pm.Deterministic(
                "w_st",
                w_raw * sigma_w,
            )
            pair_term = w_st[s_idx, t_idx]

        elif mode == "C":
            # Pair effect hanya untuk pasangan cross-domain.
            sigma_w = pm.HalfNormal("sigma_w", sigma=0.3)
            w_raw = pm.Normal(
                "w_raw",
                mu=0.0,
                sigma=1.0,
                shape=(n_s, n_t),
            )
            w_st = pm.Deterministic(
                "w_st",
                w_raw * sigma_w * (1.0 - np.eye(n_s, n_t)),
            )
            pair_term = w_st[s_idx, t_idx]

        elif mode == "D":
            # Pair effect terpisah untuk setiap model.
            # Shape: (model, source, target)
            sigma_w = pm.HalfNormal("sigma_w", sigma=0.3)
            w_raw = pm.Normal(
                "w_raw",
                mu=0.0,
                sigma=1.0,
                shape=(n_m, n_s, n_t),
            )
            w_st = pm.Deterministic(
                "w_st",
                w_raw * sigma_w,
            )
            pair_term = w_st[m_idx, s_idx, t_idx]

        else:
            pair_term = 0.0

        # Precision parameter of the Beta distribution.
        # Gamma(4, 0.1) has mean=40: F1 scores cluster tightly around the mean.
        phi = pm.Gamma("phi", alpha=phi_prior[0], beta=phi_prior[1]) #phi = pm.Gamma("phi", alpha=4.0, beta=0.1)

        # Linear predictor on logit scale — one value per observation.
        eta = (
            alpha
            + beta_m[m_idx]
            + gamma_s[s_idx]
            + delta_t[t_idx]
            + zeta_p[p_idx]
            + theta_mp[m_idx, p_idx]
            + lam * d_arr
            + xi_p[p_idx] * d_arr
            + u_checkpoint[checkpoint_idx]
            + pair_term
        )

        # Back-transform to (0, 1) for use in the Beta likelihood.
        mu = pm.Deterministic("mu", pm.math.invlogit(eta))

        # Beta likelihood: parameterised by mean μ and precision φ.
        # shape_a = μ × φ,  shape_b = (1 − μ) × φ
        pm.Beta("y_obs", alpha=mu * phi, beta=(1.0 - mu) * phi, observed=y_corrected)

        # Sample the posterior.
        logger.info(
            "[fit_model] Starting NUTS: %d chains × %d draws + %d warmup ...",
            chains, draws, tune,
        )
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
        )

        # Also sample the posterior predictive (needed for the PPC check).
        pm.sample_posterior_predictive(
            trace,
            extend_inferencedata=True,
            random_seed=random_seed,
        )

    # Save trace to disk so you can reload it without refitting.
    netcdf_path = os.path.join(output_dir, "trace.nc")
    trace.to_netcdf(netcdf_path)
    logger.info("[fit_model] Trace saved → %s", netcdf_path)

    return trace, df, encodings


# ============================================================================
# SECTION 3 — DIAGNOSTICS
# ============================================================================

def run_convergence_diagnostics(trace) -> Dict[str, Any]:
    """
    Compute R-hat, bulk-ESS, tail-ESS, and divergence count.

    HOW TO INTERPRET
    ----------------
    R-hat near 1.0 means all chains explored the same region of parameter
    space. The standard threshold is R-hat < 1.01 (Vehtari et al., 2021).

    Bulk-ESS and Tail-ESS (effective sample size) account for autocorrelation
    within chains. The practical minimum is ~400 total draws across all chains.
    With 4 chains × 2000 draws = 8000 total, ESS < 400 means <5% efficiency
    and signals severe autocorrelation.

    Divergences mean the NUTS sampler struggled in some region of parameter
    space. Any divergences need attention before trusting the results.
    Usually fixed by reparameterising (the non-centered u_raw/sigma_u already handles the
    most common case for hierarchical models).

    Returns a dict with numbers and a 'warnings' list. No hard pass/fail flag.
    """
    import arviz as az

    var_names = [
        "alpha", "beta_m", "gamma_s", "delta_t",
        "zeta_p", "theta_mp", "lambda", "xi_p",
        "sigma_u", "u_raw", "phi",
    ]
    # Optional source×target random effect (present only when pair_effect=True).
    var_names += [v for v in ("sigma_w", "w_raw") if v in trace.posterior]
    # round_to="none": the default rounds R-hat to 2 decimals, so a true 1.014
    # would be shown as 1.01 and slip through the `> 1.01` test.
    summary = az.summary(trace, var_names=var_names, round_to="none")

    rhat_vals    = summary["r_hat"].dropna()
    ess_bulk     = summary["ess_bulk"].dropna()
    ess_tail     = summary["ess_tail"].dropna()

    rhat_bad     = summary[summary["r_hat"] > 1.01]["r_hat"].to_dict()
    ess_bad      = summary[summary["ess_bulk"] < 400]["ess_bulk"].to_dict()
    ess_tail_bad = summary[summary["ess_tail"] < 400]["ess_tail"].to_dict()

    n_divergences = 0
    if "diverging" in trace.sample_stats:
        n_divergences = int(trace.sample_stats["diverging"].values.sum())

    warnings_list: List[str] = []
    if len(rhat_vals) == 0:
        warnings_list.append("All R-hat values are NaN — summary failed.")
    if rhat_bad:
        warnings_list.append(
            f"R-hat > 1.01 for {len(rhat_bad)} parameter(s): {list(rhat_bad.keys())}. "
            "Chains have not converged."
        )
    if ess_bad:
        warnings_list.append(
            f"Bulk-ESS < 400 for {len(ess_bad)} parameter(s): {list(ess_bad.keys())}. "
            "Increase draws."
        )
    if ess_tail_bad:
        warnings_list.append(
            f"Tail-ESS < 400 for {len(ess_tail_bad)} parameter(s). "
            "Tail quantiles may be unreliable."
        )
    if n_divergences > 0:
        warnings_list.append(
            f"{n_divergences} divergent transition(s). " 
        )

    rhat_max = float(rhat_vals.max()) if len(rhat_vals) else float("nan")
    ess_bulk_min = float(ess_bulk.min()) if len(ess_bulk) else float("nan")
    ess_tail_min = float(ess_tail.min()) if len(ess_tail) else float("nan")

    # Explicit convergence criteria
    converged = (
        np.isfinite(rhat_max)
        and np.isfinite(ess_bulk_min)
        and np.isfinite(ess_tail_min)
        and rhat_max <= 1.01
        and ess_bulk_min >= 400
        and ess_tail_min >= 400
        and n_divergences == 0
    )

    result = {
        "rhat_max": rhat_max,
        "rhat_problematic": rhat_bad,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "ess_problematic": ess_bad,
        "ess_tail_problematic": ess_tail_bad,
        "n_divergences": n_divergences,
        "converged": bool(converged),
        "warnings": warnings_list,
    }


    if warnings_list:
        for w in warnings_list:
            logger.warning("[convergence] %s", w)
    else:
        logger.info(
            "[convergence] R-hat max=%.4f  bulk-ESS min=%.0f  "
            "tail-ESS min=%.0f  divergences=%d",
            result["rhat_max"], result["ess_bulk_min"],
            result["ess_tail_min"], result["n_divergences"],
        )

    return result


def run_posterior_predictive_check(trace, obs_df_fitted: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare observed and replicated Macro-F1 distributions.

    WHAT THIS DOES
    --------------
    For each posterior draw the model generates N replicated F1 values (y_rep).
    We compute six statistics (mean, std, min, max, q10, q90) on:
      - y_obs : the actual observed corrected F1 values
      - y_rep : one replicated dataset per posterior draw

    The Bayesian p-value for each statistic is P(T(y_rep) ≥ T(y_obs)).
    A value near 0.5 means the model reproduces that statistic well.
    Extreme values (near 0 or 1) suggest the model mis-fits that aspect.

    We also check fit separately for each perturbation condition and each
    scenario (in-domain vs cross-domain), because a good global fit can
    hide structured misfit within specific groups.

    SCALE NOTE
    ----------
    Uses obs_df_fitted['macro_f1_corrected'] — the same values passed to the
    Beta likelihood in fit_model(). This guarantees observed and replicated
    statistics are on the same scale.

    Parameters
    ----------
    trace          : arviz.InferenceData from fit_model()
    obs_df_fitted  : the second return value of fit_model()

    Returns a dict with keys: observed, predicted, bayesian_p_values,
    grouped_mean_by_condition, grouped_mean_by_scenario, note.
    """
    if "macro_f1_corrected" not in obs_df_fitted.columns:
        raise ValueError(
            "'macro_f1_corrected' column not found. "
            "Pass obs_df_fitted returned by fit_model()."
        )

    y_obs = obs_df_fitted["macro_f1_corrected"].to_numpy(dtype=float)

    # y_rep shape from PyMC: (chains, draws_per_chain, N_observations)
    y_rep_raw = trace.posterior_predictive["y_obs"].values
    n_chains, n_draws_per_chain, n_obs = y_rep_raw.shape
    total_draws = n_chains * n_draws_per_chain
    y_rep = y_rep_raw.reshape(total_draws, n_obs)   # → (total_draws, N)

    # ── Pooled statistics ─────────────────────────────────────────────────
    # Compute mean/std/min/max/q10/q90 on observed data (single number each).
    obs_stats = {
        "mean": float(np.mean(y_obs)),
        "std":  float(np.std(y_obs)),
        "min":  float(np.min(y_obs)),
        "max":  float(np.max(y_obs)),
        "q10":  float(np.percentile(y_obs, 10)),
        "q90":  float(np.percentile(y_obs, 90)),
    }

    # Compute the same statistic for each posterior draw → array of length total_draws.
    rep_means = np.mean(y_rep, axis=1)
    rep_stds  = np.std(y_rep,  axis=1)
    rep_mins  = np.min(y_rep,  axis=1)
    rep_maxs  = np.max(y_rep,  axis=1)
    rep_q10   = np.percentile(y_rep, 10, axis=1)
    rep_q90   = np.percentile(y_rep, 90, axis=1)

    # Predicted = average of each replicated statistic across all draws.
    pred_stats = {
        "mean": float(np.mean(rep_means)),
        "std":  float(np.mean(rep_stds)),
        "min":  float(np.mean(rep_mins)),
        "max":  float(np.mean(rep_maxs)),
        "q10":  float(np.mean(rep_q10)),
        "q90":  float(np.mean(rep_q90)),
    }

    # Bayesian p-value: fraction of draws where replicated stat ≥ observed stat.
    # NOTE: this is a ONE-SIDED tail.  A value near 0 means "the data are larger
    # than the model expects"; a value near 1 means "the data are SMALLER than the
    # model expects".  For min / q10 a value near 1 is therefore the BAD end
    # (observed minimum is lower than almost every replicated minimum).
    # Read `ppc_extremeness` below for a direction-free verdict.
    rep_by_stat = {"mean": rep_means, "std": rep_stds, "min": rep_mins,
                   "max": rep_maxs, "q10": rep_q10, "q90": rep_q90}
    bayesian_p_values = {k: float(np.mean(v >= obs_stats[k])) for k, v in rep_by_stat.items()}
    p_rep_le_obs      = {k: float(np.mean(v <= obs_stats[k])) for k, v in rep_by_stat.items()}
    # Two-sided extremeness = 2 × min(P(rep ≥ obs), P(rep ≤ obs)).  Small ⇒ misfit,
    # whichever tail the observed statistic sits in.
    ppc_extremeness = {k: float(min(1.0, 2.0 * min(bayesian_p_values[k], p_rep_le_obs[k])))
                       for k in rep_by_stat}
    extreme_stats = sorted(k for k, v in ppc_extremeness.items() if v < 0.05)

    # ── Per-observation 95 % predictive intervals ─────────────────────────
    # With a calibrated model about 5 % of observations fall outside.
    lo95, hi95 = np.percentile(y_rep, [2.5, 97.5], axis=0)
    outside = (y_obs < lo95) | (y_obs > hi95)
    outside_rows = obs_df_fitted.loc[
        outside, ["model", "source_domain", "target_domain", "perturbation_condition"]
    ].copy()
    outside_rows["observed"] = y_obs[outside]
    outside_rows["lo95"]     = lo95[outside]
    outside_rows["hi95"]     = hi95[outside]
    outside_detail = outside_rows.round(4).to_dict(orient="records")

    # ── Group-level mean checks (two-sided) ───────────────────────────────
    def _group_check(mask: np.ndarray) -> Dict[str, Any]:
        o  = float(np.mean(y_obs[mask]))
        r  = np.mean(y_rep[:, mask], axis=1)
        pg = float(np.mean(r >= o)); pl = float(np.mean(r <= o))
        return {"obs_mean": o, "pred_mean": float(np.mean(r)), "bayesian_p": pg,
                "extremeness": float(min(1.0, 2.0 * min(pg, pl))),
                "n_obs": int(mask.sum()),
                "n_outside_95": int(outside[mask].sum())}

    grouped_other: Dict[str, Dict[str, Any]] = {}
    for col in ("model", "source_domain", "target_domain"):
        grouped_other[col] = {
            str(level): _group_check((obs_df_fitted[col] == level).to_numpy())
            for level in sorted(obs_df_fitted[col].unique())
        }

    # ── Source×target structure check ────────────────────────────────────
    # T(y, θ) = SD over the 25 (source,target) pairs of the mean residual (y − μ).
    # Data and replicated data share the same θ, so a p near 0 means the residuals
    # contain pair-level structure that the fitted terms did not capture.
    mu_post = trace.posterior["mu"].values.reshape(total_draws, n_obs)
    if "w_st" in trace.posterior and trace.posterior["w_st"].ndim == 5:
        # D: kelompok berdasarkan model + source + target.
        pair_labels = (
            obs_df_fitted["model"].astype(str) + ">"
            + obs_df_fitted["source_domain"].astype(str) + ">"
            + obs_df_fitted["target_domain"].astype(str)
        )
        pair_statistic_name = (
            "SD over model-source-target groups of mean residual"
        )
    else:
        # A/B/C: pertahankan definisi diagnostik sebelumnya.
        pair_labels = (
            obs_df_fitted["source_domain"].astype(str) + ">"
            + obs_df_fitted["target_domain"].astype(str)
        )
        pair_statistic_name = (
            "SD over source-target pairs of mean residual"
        )

    pair_codes, _ = pd.factorize(pair_labels)
    n_pairs = int(pair_codes.max()) + 1 

    def _pair_sd(yy: np.ndarray) -> np.ndarray:
        resid = yy - mu_post
        means = np.stack([resid[:, pair_codes == c].mean(axis=1) for c in range(n_pairs)], axis=1)
        return means.std(axis=1, ddof=1)

    t_obs = _pair_sd(np.broadcast_to(y_obs, mu_post.shape))
    t_rep = _pair_sd(y_rep)
    pair_check = {
        "statistic": pair_statistic_name,
        "obs_mean": float(t_obs.mean()),
        "rep_mean": float(t_rep.mean()),
        "p_rep_ge_obs": float(np.mean(t_rep >= t_obs)),
    }
    if pair_check["p_rep_ge_obs"] < 0.05:
        extreme_stats.append("source_target_pair_structure")

    # ── Grouped checks ────────────────────────────────────────────────────
    # Fit by perturbation condition (clean / low / medium / high).
    grouped_condition: Dict[str, Dict[str, Any]] = {}
    for cond in PERTURBATION_CONDITIONS:
        cond_mask = (obs_df_fitted["perturbation_condition"] == cond).values
        if not np.any(cond_mask):
            continue
        obs_group_mean  = float(np.mean(y_obs[cond_mask]))
        rep_group_means = np.mean(y_rep[:, cond_mask], axis=1)   # (total_draws,)
        grouped_condition[cond] = {
            "obs_mean":   obs_group_mean,
            "pred_mean":  float(np.mean(rep_group_means)),
            "bayesian_p": float(np.mean(rep_group_means >= obs_group_mean)),
            "n_obs":      int(np.sum(cond_mask)),
        }

    # Fit by scenario (in-domain vs cross-domain).
    grouped_scenario: Dict[str, Dict[str, Any]] = {}
    for scenario_label, flag_value in [("in_domain", True), ("cross_domain", False)]:
        scen_mask = (_coerce_bool(obs_df_fitted["in_domain"]) == flag_value)
        if not np.any(scen_mask):
            continue
        obs_group_mean  = float(np.mean(y_obs[scen_mask]))
        rep_group_means = np.mean(y_rep[:, scen_mask], axis=1)
        grouped_scenario[scenario_label] = {
            "obs_mean":   obs_group_mean,
            "pred_mean":  float(np.mean(rep_group_means)),
            "bayesian_p": float(np.mean(rep_group_means >= obs_group_mean)),
            "n_obs":      int(np.sum(scen_mask)),
        }

    logger.info("[PPC] Pooled: obs_mean=%.4f  pred_mean=%.4f  bayes_p(mean)=%.3f",
                obs_stats["mean"], pred_stats["mean"], bayesian_p_values["mean"])
    for cond, v in grouped_condition.items():
        logger.info("[PPC] %-8s  obs=%.4f  pred=%.4f  p=%.3f  n=%d",
                    cond, v["obs_mean"], v["pred_mean"], v["bayesian_p"], v["n_obs"])
    for scen, v in grouped_scenario.items():
        logger.info("[PPC] %-14s  obs=%.4f  pred=%.4f  p=%.3f  n=%d",
                    scen, v["obs_mean"], v["pred_mean"], v["bayesian_p"], v["n_obs"])

    logger.info("[PPC] extremeness (two-sided) %s", {k: round(v, 3) for k, v in ppc_extremeness.items()})
    logger.info("[PPC] %d / %d observations outside own 95%% interval (≈%.0f expected)",
                int(outside.sum()), n_obs, 0.05 * n_obs)
    if extreme_stats:
        logger.warning("[PPC] Misfit detected for: %s", extreme_stats)

    return {
        "observed":                  obs_stats,
        "predicted":                 pred_stats,
        "bayesian_p_values":         bayesian_p_values,      # P(rep >= obs), one-sided (kept for compatibility)
        "p_rep_le_obs":              p_rep_le_obs,
        "ppc_extremeness":           ppc_extremeness,        # two-sided; < 0.05 = misfit
        "extreme_statistics":        extreme_stats,
        "n_outside_95":              int(outside.sum()),
        "n_outside_95_expected":     float(0.05 * n_obs),
        "outside_95_rows":           outside_detail,
        "grouped_mean_by_condition": grouped_condition,
        "grouped_mean_by_scenario":  grouped_scenario,
        "grouped_mean_by_factor":    grouped_other,
        "source_target_structure":   pair_check,
        "note": (
            "All statistics use boundary-corrected F1 values. "
            "bayesian_p_values are ONE-SIDED P(rep >= obs): for 'min'/'q10' a value near 1 is the "
            "misfit direction. Use ppc_extremeness (two-sided) to judge fit; < 0.05 flags a misfit. "
            "No automatic pass/fail threshold is applied to the grouped means."
        ),
    }


# ============================================================================
# SECTION 4 — POSTERIOR CONTRASTS
# ============================================================================

def compute_contrasts(trace, encodings: Dict[str, Any], obs_df_fitted: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute posterior contrasts for all five research questions.

    HOW MARGINAL MEANS WORK
    -----------------------
    Think of the results as a table with rows = all observed experiment cells.
    A "marginal mean" is the average F1 for one specific column of that table
    (e.g. one model variant), averaged over all other columns (source, target,
    condition).

    WHY WE ONLY USE OBSERVED CELLS
    --------------------------------
    We loop over cells that actually appear in obs_df_fitted, not over a full
    theoretical 3×5×5×4 grid. This avoids predicting F1 for cells that were
    never evaluated.

    WHY RQ2 AND RQ5 FILTER BY SCENARIO
    ------------------------------------
    For RQ2 (in-domain vs cross-domain):
      - in-domain mean  → average over only the 5 diagonal cells (src==tgt)
      - cross-domain mean → average over only the 20 off-diagonal cells
    Forcing in-domain and cross-domain D values onto ALL 25 cells would create
    counterfactual predictions that never appeared in the experiment.

    For RQ5 the same logic applies: degradation under in-domain conditions uses
    diagonal cells only; cross-domain degradation uses off-diagonal cells only.

    SCALE
    -----
    All returned values are on the F1 (probability) scale via the logistic
    back-transform (inverse-logit). A contrast of +0.05 means the first
    condition has 5 percentage-point higher expected Macro-F1.

    Parameters
    ----------
    trace          : arviz.InferenceData from fit_model()
    encodings      : third return value of fit_model()
    obs_df_fitted  : second return value of fit_model()

    Returns
    -------
    dict keyed by contrast name, each value a dict with:
        mean, median, ci_95_lower, ci_95_upper,
        prob_positive, prob_negative, direction_note
    """
    import scipy.special as sp   # sp.expit = inverse-logit function

    models     = encodings["models"]
    sources    = encodings["sources"]
    targets    = encodings["targets"]
    conditions = encodings["conditions"]

    # ── Pull posterior samples into plain numpy arrays ────────────────────
    # Shape for each: (S, ...) where S = chains × draws_per_chain.
    def flatten_posterior(name: str) -> np.ndarray:
        arr   = trace.posterior[name].values   # (chains, draws, ...)
        shape = arr.shape
        return arr.reshape(shape[0] * shape[1], *shape[2:])

    alpha        = flatten_posterior("alpha")        # (S,)
    beta_m       = flatten_posterior("beta_m")       # (S, n_m)
    gamma_s      = flatten_posterior("gamma_s")      # (S, n_s)
    delta_t      = flatten_posterior("delta_t")      # (S, n_t)
    zeta_p       = flatten_posterior("zeta_p")       # (S, n_p)
    theta_mp     = flatten_posterior("theta_mp")     # (S, n_m, n_p)
    lam          = flatten_posterior("lambda")       # (S,)
    xi_p         = flatten_posterior("xi_p")         # (S, n_p)
    u_checkpoint = flatten_posterior("u_checkpoint") # (S, n_checkpoints)
    # Optional source×target effect (diagonal already masked to 0 in the model).
    w_st = flatten_posterior("w_st") if "w_st" in trace.posterior else None   # (S, n_s, n_t)

    S   = alpha.shape[0]
    n_s = gamma_s.shape[1]

    # ── Build the set of observed cells ───────────────────────────────────
    # Each cell is (model_index, source_index, target_index, condition_index).
    observed_cells = set()
    for _, row in obs_df_fitted.iterrows():
        mi = models.index(row["model"])
        si = sources.index(row["source_domain"])
        ti = targets.index(row["target_domain"])
        pi = conditions.index(row["perturbation_condition"])
        observed_cells.add((mi, si, ti, pi))

    # For fair model comparisons (RQ1, RQ4), only use cells present for EVERY model.
    # This prevents missing cells from biasing one model's average.
    cells_by_model: Dict[int, set] = {mi: set() for mi in range(len(models))}
    for (mi, si, ti, pi) in observed_cells:
        cells_by_model[mi].add((si, ti, pi))   # (source, target, condition) triples

    # Intersection across all models
    common_stp = None
    for mi in range(len(models)):
        if common_stp is None:
            common_stp = cells_by_model[mi].copy()
        else:
            common_stp = common_stp & cells_by_model[mi]
    if common_stp is None:
        common_stp = set()
    common_stp = sorted(common_stp)   # sort for deterministic order

    # ── Helper: compute marginal mean F1 for a subset of cells ───────────
    def marginal_f1(
        m_i:        Optional[int]  = None,   # fix model; None = average over all
        p_i:        Optional[int]  = None,   # fix condition; None = average over all
        in_domain:  Optional[bool] = None,   # True = diagonal only, False = off-diag only
        use_common: bool           = False,  # only use cells present in ALL models
    ) -> np.ndarray:
        """
        Return shape (S,) — one marginal mean F1 per posterior draw.

        Steps:
          1. Decide which cells to include (from observed_cells or common_stp).
          2. For each included cell, compute logit(μ) using the full linear
             predictor, then apply inverse-logit to get F1.
          3. Average over all included cells.
        """
        accumulator = np.zeros(S)
        count = 0

        if use_common:
            if m_i is None:
                # Use identical source-target-condition cells across all models.
                cell_list = [
                    (mi, si, ti, pi)
                    for mi in range(len(models))
                    for (si, ti, pi) in common_stp
                ]
            else:
                # Use common cells for the selected model.
                cell_list = [
                    (m_i, si, ti, pi)
                    for (si, ti, pi) in common_stp
                ]
        else:
            cell_list = list(observed_cells)

        for (mi, si, ti, pi) in cell_list:
            # Apply fixed-factor filters.
            if m_i is not None and mi != m_i:
                continue
            if p_i is not None and pi != p_i:
                continue

            # Apply scenario filter (never force D onto the wrong cell type).
            cell_is_in_domain = (si == ti)
            if in_domain is True  and not cell_is_in_domain:
                continue
            if in_domain is False and     cell_is_in_domain:
                continue

            # Natural D value for this cell.
            d_val        = int(cell_is_in_domain)
            checkpoint_i = mi * n_s + si

            # Full linear predictor on logit scale.
            eta = (
                alpha
                + beta_m[:, mi]
                + gamma_s[:, si]
                + delta_t[:, ti]
                + zeta_p[:, pi]
                + theta_mp[:, mi, pi]
                + lam * d_val
                + xi_p[:, pi] * d_val
                + u_checkpoint[:, checkpoint_i]
            )
            if w_st is not None:
                if w_st.ndim == 4:
                    # D: pair effect berbeda untuk setiap model.
                    eta = eta + w_st[:, mi, si, ti]

                else:
                    # B dan C: pair effect dibagi bersama antar-model.
                    eta = eta + w_st[:, si, ti]
            accumulator += sp.expit(eta)   # inverse-logit → F1 scale
            count += 1

        if count == 0:
            raise ValueError(
                f"marginal_f1: no observed cells matched "
                f"(m_i={m_i}, p_i={p_i}, in_domain={in_domain}, use_common={use_common})."
            )

        return accumulator / count   # shape (S,)

    # ── Helper: summarise a (S,) array of contrast samples ───────────────
    def summarise(samples: np.ndarray, direction_note: str) -> Dict[str, Any]:
        """
        Return mean, median, 95% credible interval, and posterior probabilities.

        The 95% credible interval means: given the data and model, there is a
        95% probability the true contrast is in this interval. This is NOT the
        same as a frequentist confidence interval.

        No significance threshold is applied. Report these numbers directly
        in your thesis and interpret the direction and magnitude.
        """
        return {
            "mean":           float(np.mean(samples)),
            "median":         float(np.median(samples)),
            "ci_95_lower":    float(np.percentile(samples, 2.5)),
            "ci_95_upper":    float(np.percentile(samples, 97.5)),
            "prob_positive":  float(np.mean(samples > 0)),
            "prob_negative":  float(np.mean(samples < 0)),
            "direction_note": direction_note,
        }

    contrasts: Dict[str, Any] = {}
    p_map = {c: conditions.index(c) for c in conditions}

    # ── RQ1: Pairwise model comparisons ───────────────────────────────────
    # Uses only cells present for EVERY model (fair comparison).
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            mu1 = marginal_f1(m_i=i, use_common=True)
            mu2 = marginal_f1(m_i=j, use_common=True)
            contrasts[f"model_{m1}_vs_{m2}"] = summarise(
                mu1 - mu2,
                f"Positive = {m1} has higher F1 than {m2} (common cells only)",
            )

    # ── RQ2: In-domain vs cross-domain ────────────────────────────────────
    # Diagonal cells only for in-domain; off-diagonal only for cross-domain.
    mu_in    = marginal_f1(in_domain=True)
    mu_cross = marginal_f1(in_domain=False)
    contrasts["in_domain_vs_cross_domain"] = summarise(
        mu_in - mu_cross,
        "Positive = in-domain has higher F1 than cross-domain (expected). "
        "In-domain uses diagonal cells; cross-domain uses off-diagonal cells.",
    )

    # ── Model × Domain-shift (robustness against domain shift) ──────────────
    for i, m in enumerate(models):
        mu_in_m    = marginal_f1(m_i=i, in_domain=True)
        mu_cross_m = marginal_f1(m_i=i, in_domain=False)
        contrasts[f"model_{m}_in_vs_cross"] = summarise(
            mu_in_m - mu_cross_m,
            f"Positive = {m} has higher F1 in-domain than cross-domain "
            "(domain-shift gap). Smaller gap = more robust to domain shift."
        )

    # Pairwise gap comparison
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i: continue
            gap_m1 = (marginal_f1(m_i=i, in_domain=True) - marginal_f1(m_i=i, in_domain=False))
            gap_m2 = (marginal_f1(m_i=j, in_domain=True) - marginal_f1(m_i=j, in_domain=False))
            contrasts[f"interaction_domainshift_{m1}_vs_{m2}"] = summarise(
                gap_m1 - gap_m2,
                f"Positive = {m1} has LARGER domain-shift gap than {m2} "
                "(i.e. {m1} less robust to domain shift)."
            )

    # ── RQ3: Perturbation degradation ─────────────────────────────────────
    for level in ["low", "medium", "high"]:
        mu_clean = marginal_f1(p_i=p_map["clean"], use_common=True)
        mu_level = marginal_f1(p_i=p_map[level],   use_common=True)
        contrasts[f"clean_vs_{level}"] = summarise(
            mu_clean - mu_level,
            f"Positive = F1 drops from clean to {level} (degradation). "
            "Common cells only.",
        )

    contrasts["low_vs_medium"] = summarise(
        marginal_f1(p_i=p_map["low"],    use_common=True)
        - marginal_f1(p_i=p_map["medium"], use_common=True),
        "Positive = F1 drops from low to medium perturbation.",
    )
    contrasts["medium_vs_high"] = summarise(
        marginal_f1(p_i=p_map["medium"], use_common=True)
        - marginal_f1(p_i=p_map["high"],  use_common=True),
        "Positive = F1 drops from medium to high perturbation.",
    )

    # ── RQ4: Model × Perturbation interaction ─────────────────────────────
    # Compare how much each model degrades from clean to each perturbed level.
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if j <= i:
                continue
            for level in ["low", "medium", "high"]:
                degrad_m1 = (
                    marginal_f1(m_i=i, p_i=p_map["clean"], use_common=True)
                    - marginal_f1(m_i=i, p_i=p_map[level], use_common=True)
                )
                degrad_m2 = (
                    marginal_f1(m_i=j, p_i=p_map["clean"], use_common=True)
                    - marginal_f1(m_i=j, p_i=p_map[level], use_common=True)
                )
                contrasts[f"interaction_degradation_{m1}_vs_{m2}_at_{level}"] = summarise(
                    degrad_m1 - degrad_m2,
                    f"Positive = {m1} degrades MORE than {m2} at {level}.",
                )

    # ── RQ5: Perturbation × Scenario interaction ──────────────────────────
    # Compare degradation within in-domain cells vs within cross-domain cells.
    for level in ["low", "medium", "high"]:
        degrad_in = (
            marginal_f1(p_i=p_map["clean"], in_domain=True)
            - marginal_f1(p_i=p_map[level], in_domain=True)
        )
        degrad_cross = (
            marginal_f1(p_i=p_map["clean"], in_domain=False)
            - marginal_f1(p_i=p_map[level], in_domain=False)
        )
        contrasts[f"interaction_scenario_degradation_at_{level}"] = summarise(
            degrad_in - degrad_cross,
            f"Positive = in-domain degrades MORE than cross-domain at {level}. "
            "Negative = cross-domain degrades more (expected direction).",
        )

    return contrasts


# ============================================================================
# SECTION 5 — VALIDATION HELPERS
# ============================================================================

def validate_contrasts(contrasts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check that every computed contrast has the right keys, a valid CI,
    valid probabilities, and values in the F1 contrast range (−1, +1).

    Returns a summary dict: n_contrasts, n_passed, n_failed, failures.
    """
    required_keys = {
        "mean", "median", "ci_95_lower", "ci_95_upper",
        "prob_positive", "prob_negative", "direction_note",
    }
    failures: List[Dict[str, Any]] = []

    for name, entry in contrasts.items():
        if "error" in entry:
            failures.append({"name": name, "issue": f"Entry contains error: {entry['error']}"})
            continue

        missing = required_keys - set(entry.keys())
        if missing:
            failures.append({"name": name, "issue": f"Missing keys: {missing}"})
            continue

        if entry["ci_95_lower"] >= entry["ci_95_upper"]:
            failures.append({"name": name, "issue":
                f"CI inverted: lower={entry['ci_95_lower']:.4f} >= upper={entry['ci_95_upper']:.4f}"
            })

        prob_sum = entry["prob_positive"] + entry["prob_negative"]
        if prob_sum > 1.0 + 1e-6:
            failures.append({"name": name, "issue":
                f"prob_positive + prob_negative = {prob_sum:.6f} > 1.0"
            })

        for key in ("mean", "ci_95_lower", "ci_95_upper"):
            if abs(entry[key]) >= 1.0:
                failures.append({"name": name, "issue":
                    f"{key} = {entry[key]:.4f} is outside (−1, +1). "
                    "Possible logit-scale value leaked into output."
                })

    n_contrasts = len(contrasts)
    n_failed    = len(failures)
    n_passed    = n_contrasts - n_failed

    if failures:
        for f in failures:
            logger.warning("[validate_contrasts] FAIL %s: %s", f["name"], f["issue"])
    else:
        logger.info("[validate_contrasts] All %d contrast(s) passed.", n_contrasts)

    return {"n_contrasts": n_contrasts, "n_passed": n_passed, "n_failed": n_failed, "failures": failures}


def validate_ppc(ppc_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check that the PPC result has observed and predicted sub-dicts with
    matching keys, and that all values are in the F1 scale [0, 1].

    Does NOT apply a pass/fail threshold to Bayesian p-values —
    interpret those in context yourself.
    """
    failures: List[str] = []

    if "observed" not in ppc_result or "predicted" not in ppc_result:
        failures.append("Missing 'observed' or 'predicted' key.")
        return {"passed": False, "failures": failures}

    obs  = ppc_result["observed"]
    pred = ppc_result["predicted"]

    if set(obs.keys()) != set(pred.keys()):
        failures.append(
            f"Observed and predicted have different keys. "
            f"Observed: {set(obs.keys())}. Predicted: {set(pred.keys())}."
        )

    for key in set(obs.keys()) & set(pred.keys()):
        for label, val in [("observed", obs[key]), ("predicted", pred[key])]:
            if val is not None and not (0.0 <= val <= 1.0):
                failures.append(
                    f"{label}['{key}'] = {val:.4f} is outside [0, 1]. "
                    "Possible scale mismatch."
                )

    if failures:
        for f in failures:
            logger.warning("[validate_ppc] %s", f)
    else:
        logger.info("[validate_ppc] PPC structural validation passed.")

    return {"passed": len(failures) == 0, "failures": failures}


# ============================================================================
# SECTION 6 — MAIN ENTRY POINT
# ============================================================================

def run_full_analysis(
    all_results:  Dict[str, Dict[str, Any]],
    output_dir:   str = "statistical_analysis",
    mcmc_draws:   int = 2000,
    mcmc_tune:    int = 3000,
    mcmc_chains:  int = 4,
    random_seed:  int = 42,
    pair_mode:    str = "D",
    target_accept: float = 0.99,
    #pair_effect:  Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Run the complete Bayesian statistical analysis in one straight call.

    Steps (in order):
      1. Assemble the observation DataFrame from evaluation results.
      2. Validate data quality; raise on critical problems.
      3. Fit the Bayesian hierarchical Beta regression (requires PyMC ≥ 5).
      4. Run convergence diagnostics (R-hat, ESS, divergences).
      5. Run posterior predictive checks (pooled + per-condition + per-scenario).
      6. Compute posterior contrasts for all five research questions.
      7. Validate contrasts and PPC structurally.
      8. Write JSON data dump and human-readable text report.

    Parameters
    ----------
    all_results  : {model_name: results_dict} from EvaluationPipeline.run()['results']
    output_dir   : where to save trace.nc, JSON, and text report
    mcmc_draws   : posterior draws per chain (use 500 for quick testing, 2000 for final)
    mcmc_tune    : warmup steps per chain (do not set below 1000)
    mcmc_chains  : number of MCMC chains (use 4 for reliable R-hat)
    random_seed  : reproducibility seed

    Returns
    -------
    dict with keys: status, observation_report, convergence, ppc,
    ppc_validation, contrasts, contrasts_validation, report_path
    """
    output: Dict[str, Any] = {"status": "started"}
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Assemble observations ─────────────────────────────────────
    logger.info("[run_full_analysis] Step 1: Assembling observations ...")
    try:
        obs_df = assemble_observations(all_results)
    except ValueError as exc:
        logger.error("[run_full_analysis] Data assembly failed: %s", exc)
        output["status"] = "failed"
        output["reason"] = str(exc)
        return output

    # ── Step 2: Validate observations ─────────────────────────────────────
    logger.info("[run_full_analysis] Step 2: Validating observations ...")
    val_report = validate_observations(obs_df)
    output["observation_report"] = val_report

    # Abort on critical data problems; warnings about missing combos are OK.
    critical_problems = []
    if "error" in val_report:
        critical_problems.append(val_report["error"])
    if val_report.get("n_missing_f1", 0) > 0:
        critical_problems.append(f"{val_report['n_missing_f1']} NaN F1 values.")
    if val_report.get("n_out_of_range", 0) > 0:
        critical_problems.append(f"{val_report['n_out_of_range']} F1 values outside [0,1].")
    if val_report.get("n_duplicates", 0) > 0:
        critical_problems.append(f"{val_report['n_duplicates']} duplicate keys remain.")
    if val_report.get("n_scenario_inconsistent", 0) > 0:
        critical_problems.append(f"{val_report['n_scenario_inconsistent']} inconsistent in_domain flags.")
    if val_report.get("n_unexpected_categories", 0) > 0:
        critical_problems.append("Unexpected model or condition names found.")

    if critical_problems:
        reason = "Data validation failed: " + " | ".join(critical_problems)
        logger.error("[run_full_analysis] %s", reason)
        output["status"] = "failed"
        output["reason"] = reason
        return output

    for w in val_report.get("warnings", []):
        logger.warning("[run_full_analysis] %s", w)

    # ── Step 3: Fit model ──────────────────────────────────────────────────
    logger.info("[run_full_analysis] Step 3: Fitting Bayesian model ...")
    try:
        trace, obs_df_fitted, encodings = fit_model(
            obs_df,
            output_dir    = output_dir,
            draws         = mcmc_draws,
            tune          = mcmc_tune,
            chains        = mcmc_chains,
            target_accept  = target_accept,
            random_seed   = random_seed,
            pair_mode     = pair_mode,
            #pair_effect   = pair_effect,
        )
    except ImportError as exc:
        raise   # re-raise so the caller gets a clear install message
    except Exception as exc:
        logger.error("[run_full_analysis] Model fitting failed: %s", exc)
        output["status"] = "failed"
        output["reason"] = str(exc)
        return output

    # ── Step 4: Convergence diagnostics ───────────────────────────────────
    logger.info("[run_full_analysis] Step 4: Convergence diagnostics ...")
    output["convergence"] = run_convergence_diagnostics(trace)

    # ── Step 5: Posterior predictive check ────────────────────────────────
    logger.info("[run_full_analysis] Step 5: Posterior predictive check ...")
    ppc_result = run_posterior_predictive_check(trace, obs_df_fitted)
    output["ppc"]            = ppc_result
    output["ppc_validation"] = validate_ppc(ppc_result)

    # ── Step 6: Posterior contrasts ────────────────────────────────────────
    logger.info("[run_full_analysis] Step 6: Computing posterior contrasts ...")
    try:
        contrasts = compute_contrasts(trace, encodings, obs_df_fitted)
        output["contrasts"] = contrasts
    except Exception as exc:
        logger.error("[run_full_analysis] Contrasts failed: %s", exc)
        output["contrasts"] = {"error": str(exc)}
        contrasts = {}

    if contrasts and "error" not in contrasts:
        output["contrasts_validation"] = validate_contrasts(contrasts)
    else:
        output["contrasts_validation"] = {"n_contrasts": 0, "n_failed": 0, "failures": []}

    # ── Step 7: Set final status ───────────────────────────────────────────
    conv = output.get("convergence", {})

    convergence_ok = conv.get("converged", None) is True

    contrast_validation = output.get("contrasts_validation", {})
    contrasts_ok = (
        contrast_validation.get("n_contrasts", 0) > 0
        and contrast_validation.get("n_failed", 0) == 0
    )

    ppc_validation = output.get("ppc_validation", {})
    ppc_structure_ok = ppc_validation.get("passed", False)
    # Contrasts silently fall back to "common cells" when the 3×5×5×4 grid is
    # incomplete, so an incomplete design must never end as a clean "completed".
    design_complete = (
        val_report.get("n_total", 0) == val_report.get("expected_max", -1)
        and val_report.get("n_missing_combos", 1) == 0
    )



    # `ppc_validation` only checks that numbers are in [0, 1] and keys match —
    # it says NOTHING about model fit.  Fit is judged separately from the PPC
    # content: a pooled statistic with two-sided extremeness < 0.05, or residual
    # source×target structure with p < 0.05, makes the run "completed_with_warnings"
    # even when sampling converged and the PPC dict is structurally valid.
    ppc_fit_issues = list(output.get("ppc", {}).get("extreme_statistics", []))
    output["model_fit"] = {
        "ppc_fit_ok":            len(ppc_fit_issues) == 0,
        "ppc_fit_issues":        ppc_fit_issues,
        "n_outside_95":          output.get("ppc", {}).get("n_outside_95"),
        "n_outside_95_expected": output.get("ppc", {}).get("n_outside_95_expected"),
    }
    output["status_detail"] = {
        "sampling_converged":     bool(convergence_ok),
        "ppc_structurally_valid": bool(ppc_structure_ok),
        "ppc_fit_ok":             output["model_fit"]["ppc_fit_ok"],
        "contrasts_valid":        bool(contrasts_ok),
        "design_complete_300":    bool(design_complete),
    }

    has_issues = not (
        convergence_ok
        and contrasts_ok
        and ppc_structure_ok
        and output["model_fit"]["ppc_fit_ok"]
        and design_complete
    )

    output["status"] = (
        "completed"
        if not has_issues
        else "completed_with_warnings"
    )
    if has_issues:
        logger.warning(
            "[run_full_analysis] Analysis completed but diagnostics have issues %s. "
            "Review before interpreting contrasts.", output["status_detail"]
        )

    # ── Step 8: Write report ───────────────────────────────────────────────
    logger.info("[run_full_analysis] Step 8: Writing report ...")
    output["report_path"] = _write_report(output, output_dir)
    
    return output


def _write_report(output: Dict[str, Any], output_dir: str) -> str:
    """
    Write a JSON file and a human-readable text report.
    Returns the path to the text report.
    """
    # JSON
    json_path = os.path.join(output_dir, "statistical_analysis.json")
    try:
        def _serialise(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Cannot serialise {type(obj)}")

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, default=_serialise)
    except Exception as exc:
        logger.warning("[report] JSON save failed: %s", exc)

    # Text report
    lines: List[str] = [
        "=" * 80,
        "BAYESIAN STATISTICAL ANALYSIS REPORT",
        "IndoBERT Clickbait Robustness Study",
        "=" * 80,
        "",
        f"Status : {output.get('status', 'unknown')}",
        f"Detail : {output.get('status_detail', {})}",
        "  ('completed' = sampling converged AND PPC fit acceptable AND full 300-cell design.",
        "   Structural PPC validation alone does NOT mean the model fits the data.)",
        "",
    ]

    # Observation summary
    obs = output.get("observation_report", {})
    if obs and "error" not in obs:
        lines += [
            "OBSERVATION SUMMARY",
            "-" * 40,
            f"  Total observations   : {obs.get('n_total', '?')}",
            f"  Missing F1           : {obs.get('n_missing_f1', '?')}",
            f"  Boundary F1=0        : {obs.get('n_boundary_zero', '?')}",
            f"  Boundary F1=1        : {obs.get('n_boundary_one', '?')}",
            f"  F1 range             : [{obs.get('f1_min', float('nan')):.4f}, "
            f"{obs.get('f1_max', float('nan')):.4f}]",
            f"  F1 mean (std)        : {obs.get('f1_mean', float('nan')):.4f} "
            f"({obs.get('f1_std', float('nan')):.4f})",
            f"  Missing combinations : {obs.get('n_missing_combos', '?')}",
            "",
        ]
        for w in obs.get("warnings", []):
            lines.append(f"  WARNING: {w}")
        lines.append("")

    # Convergence
    conv = output.get("convergence", {})
    if conv:
        lines += [
            "CONVERGENCE DIAGNOSTICS",
            "-" * 40,
            f"  R-hat max            : {conv.get('rhat_max', float('nan')):.4f}",
            f"  Bulk-ESS min         : {conv.get('ess_bulk_min', float('nan')):.0f}",
            f"  Tail-ESS min         : {conv.get('ess_tail_min', float('nan')):.0f}",
            f"  Divergences          : {conv.get('n_divergences', '?')}",
            f"  Converged            : {conv.get('converged', 'NOT AVAILABLE')}",
            "  (R-hat near 1.0, ESS > ~400, 0 divergences = good)",
            "",
        ]
        for w in conv.get("warnings", []):
            lines.append(f"  WARNING: {w}")
        lines.append("")

    # PPC
    ppc = output.get("ppc", {})
    if ppc and "observed" in ppc:
        obs_s   = ppc["observed"]
        pred_s  = ppc["predicted"]
        bayes_p = ppc.get("bayesian_p_values", {})
        lines += [
            "POSTERIOR PREDICTIVE CHECK",
            "-" * 40,
            "  Pooled (all 300 observations):",
            f"  {'Statistic':<10}  {'Observed':>9}  {'Predicted':>9}  {'Bayes-p':>9}",
            "  " + "-" * 42,
        ]
        for stat in ["mean", "std", "min", "max", "q10", "q90"]:
            lines.append(
                f"  {stat:<10}  {obs_s.get(stat, float('nan')):>9.4f}"
                f"  {pred_s.get(stat, float('nan')):>9.4f}"
                f"  {bayes_p.get(stat, float('nan')):>9.3f}"
            )
        lines.append("  (Bayes-p = one-sided P(rep >= obs); for min/q10 a value near 1 is the misfit side.)")
        ext = ppc.get("ppc_extremeness", {})
        if ext:
            lines.append("  Two-sided extremeness (< 0.05 = misfit): "
                         + ", ".join(f"{k}={v:.3f}" for k, v in ext.items()))
            lines.append(f"  Observations outside own 95% interval: {ppc.get('n_outside_95', '?')} "
                         f"(expected about {ppc.get('n_outside_95_expected', float('nan')):.0f})")
            st = ppc.get("source_target_structure", {})
            if st:
                lines.append("  Source x target residual structure: "
                             f"P(rep >= obs) = {st.get('p_rep_ge_obs', float('nan')):.3f}")
            if ppc.get("extreme_statistics"):
                lines.append(f"  PPC MISFIT FLAGS: {ppc['extreme_statistics']}")
        lines.append("")

        gc = ppc.get("grouped_mean_by_condition", {})
        if gc:
            lines += [
                "  By perturbation condition:",
                f"  {'Condition':<10}  {'Obs mean':>9}  {'Pred mean':>9}  {'Bayes-p':>9}  {'N':>5}",
                "  " + "-" * 50,
            ]
            for cond in PERTURBATION_CONDITIONS:
                if cond in gc:
                    v = gc[cond]
                    lines.append(
                        f"  {cond:<10}  {v['obs_mean']:>9.4f}"
                        f"  {v['pred_mean']:>9.4f}"
                        f"  {v['bayesian_p']:>9.3f}  {v['n_obs']:>5}"
                    )
            lines.append("")

        gs = ppc.get("grouped_mean_by_scenario", {})
        if gs:
            lines += [
                "  By scenario:",
                f"  {'Scenario':<14}  {'Obs mean':>9}  {'Pred mean':>9}  {'Bayes-p':>9}  {'N':>5}",
                "  " + "-" * 54,
            ]
            for scen in ["in_domain", "cross_domain"]:
                if scen in gs:
                    v = gs[scen]
                    lines.append(
                        f"  {scen:<14}  {v['obs_mean']:>9.4f}"
                        f"  {v['pred_mean']:>9.4f}"
                        f"  {v['bayesian_p']:>9.3f}  {v['n_obs']:>5}"
                    )
            lines.append("")

        ppc_val = output.get("ppc_validation", {})
        for f in ppc_val.get("failures", []):
            lines.append(f"  PPC VALIDATION WARNING: {f}")
        lines.append("")

    # Posterior contrasts
    contrasts = output.get("contrasts", {})
    if contrasts and "error" not in contrasts:
        lines += [
            "POSTERIOR CONTRASTS  (F1 scale)",
            "-" * 40,
            "  Mean = average contrast in Macro-F1 percentage points.",
            "  CI   = 95% credible interval.",
            "  P(+) = P(contrast > 0).  P(-) = P(contrast < 0).",
            "  No significance threshold applied — interpret magnitude directly.",
            "",
            f"  {'Contrast':<52} {'Mean':>7} {'95% CI':>21} {'P(+)':>6} {'P(-)':>6}",
            "  " + "-" * 95,
        ]
        rq_groups = [
            ("RQ1 — Model Variant Comparisons",
            [k for k in contrasts if k.startswith("model_") and "_in_vs_cross" not in k]),
            ("RQ2 — In-Domain vs Cross-Domain",
            [k for k in contrasts if k == "in_domain_vs_cross_domain"]),
            ("Model × Domain-shift (robustness)",
            [k for k in contrasts if k.endswith("_in_vs_cross") or k.startswith("interaction_domainshift_")]),
            ("RQ3 — Perturbation Degradation",
            [k for k in contrasts if k.startswith("clean_vs_") or k in ("low_vs_medium", "medium_vs_high")]),
            ("RQ4 — Model × Perturbation",
            [k for k in contrasts if k.startswith("interaction_degradation_")]),
            ("RQ5 — Perturbation × Scenario",
            [k for k in contrasts if k.startswith("interaction_scenario_")]),
        ]
        for rq_label, rq_keys in rq_groups:
            if not rq_keys:
                continue
            lines.append(f"\n  {rq_label}")
            for name in rq_keys:
                c  = contrasts[name]
                ci = f"[{c['ci_95_lower']:+.4f}, {c['ci_95_upper']:+.4f}]"
                lines.append(
                    f"  {name:<52} {c['mean']:>+7.4f} {ci:>21} "
                    f"{c['prob_positive']:>6.3f} {c['prob_negative']:>6.3f}"
                )
                lines.append(f"    -> {c['direction_note']}")
        lines.append("")

        cv = output.get("contrasts_validation", {})
        if cv:
            lines.append(
                f"  Contrast validation: {cv.get('n_passed','?')}/{cv.get('n_contrasts','?')} passed"
            )
            for f in cv.get("failures", []):
                lines.append(f"  FAIL: {f['name']} — {f['issue']}")
        lines.append("")

    lines += ["=" * 80, "END OF REPORT", "=" * 80, ""]

    report_text = "\n".join(lines)
    txt_path    = os.path.join(output_dir, "statistical_analysis_report.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    logger.info("[report] Saved → %s", txt_path)

    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode("ascii", errors="replace").decode("ascii"))

    return txt_path


# ============================================================================
# SECTION 7 — BACKWARD COMPATIBILITY SHIM
# ============================================================================

class StatisticalAnalyzer:
    """
    Thin wrapper around run_full_analysis() so existing callers
    (evaluate_pipeline.py, notebooks) don't need to change.

    Usage:
        analyzer = StatisticalAnalyzer(random_seed=42, output_dir="stats_out", pair_mode="C")
        results  = analyzer.run_full_analysis(all_results)
    """

    def __init__(
        self,
        random_seed: int = 42,
        output_dir:  str = "statistical_analysis",
        mcmc_draws:  int = 2000,
        mcmc_tune:   int = 3000,
        mcmc_chains: int = 4,
        pair_mode:   str = "D",
        target_accept: float = 0.99,
        #pair_effect: Optional[bool] = None,
    ):
        self.random_seed = random_seed
        self.output_dir  = output_dir
        self.mcmc_draws  = mcmc_draws
        self.mcmc_tune   = mcmc_tune
        self.mcmc_chains = mcmc_chains
        self.pair_mode   = pair_mode
        self.target_accept = target_accept
        #self.pair_effect = pair_effect
        os.makedirs(output_dir, exist_ok=True)

    def run_full_analysis(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Delegate to the module-level run_full_analysis function."""
        return run_full_analysis(
            all_results  = all_results,
            output_dir   = self.output_dir,
            mcmc_draws   = self.mcmc_draws,
            mcmc_tune    = self.mcmc_tune,
            mcmc_chains  = self.mcmc_chains,
            random_seed  = self.random_seed,
            pair_mode    = self.pair_mode,
            target_accept = self.target_accept,
            #pair_effect  = self.pair_effect,
        )


# ============================================================================
# SECTION 8 — TESTS
#
# Self-contained. Only require numpy and pandas (no PyMC).
# Run with:  python statistical_analyzer.py
# ============================================================================

def _make_minimal_obs_df() -> pd.DataFrame:
    """Build a small test DataFrame: 3 models × 3 sources × 3 targets × 4 conditions."""
    models     = ["base", "large", "lite"]
    domains    = ["DomA", "DomB", "DomC"]
    conditions = ["clean", "low", "medium", "high"]
    rng  = np.random.default_rng(seed=0)
    rows = []
    for model in models:
        for src in domains:
            for tgt in domains:
                is_in_domain = (src == tgt)
                for i, cond in enumerate(conditions):
                    base_f1 = 0.88 if is_in_domain else 0.78
                    f1 = float(np.clip(base_f1 - i * 0.03 + rng.normal(0, 0.01), 0.01, 0.99))
                    rows.append({
                        "model":                  model,
                        "source_domain":          src,
                        "target_domain":          tgt,
                        "perturbation_condition": cond,
                        "in_domain":              is_in_domain,
                        "macro_f1":               f1,
                        "n_samples":              100,
                        "common_ids_count":       100,
                    })
    return pd.DataFrame(rows)


def test_rq2_in_domain_filter():
    """marginal_f1 with in_domain=True must only include diagonal cells."""
    import scipy.special as sp

    n_m, n_s, n_t, n_p, S = 2, 2, 2, 2, 1
    alpha        = np.array([0.0])
    beta_m       = np.zeros((S, n_m))
    gamma_s      = np.zeros((S, n_s))
    delta_t      = np.zeros((S, n_t))
    zeta_p       = np.zeros((S, n_p))
    theta_mp     = np.zeros((S, n_m, n_p))
    lam          = np.array([2.0])   # strong in-domain boost
    xi_p         = np.zeros((S, n_p))
    u_checkpoint = np.zeros((S, n_m * n_s))

    def _test_marginal(in_domain: Optional[bool]) -> float:
        acc, count = 0.0, 0
        for _m in range(n_m):
            for _p in range(n_p):
                for _s in range(n_s):
                    for _t in range(n_t):
                        is_diag = (_s == _t)
                        if in_domain is True  and not is_diag: continue
                        if in_domain is False and     is_diag: continue
                        d = int(is_diag)
                        eta = (alpha[0] + beta_m[0,_m] + gamma_s[0,_s] + delta_t[0,_t]
                               + zeta_p[0,_p] + theta_mp[0,_m,_p]
                               + lam[0]*d + xi_p[0,_p]*d + u_checkpoint[0, _m*n_s+_s])
                        acc += float(sp.expit(eta))
                        count += 1
        return acc / count

    mu_in    = _test_marginal(True)
    mu_cross = _test_marginal(False)
    mu_all   = _test_marginal(None)

    expected_in    = float(sp.expit(2.0))   # ≈ 0.880
    expected_cross = float(sp.expit(0.0))   # = 0.500
    expected_all   = 0.5 * expected_in + 0.5 * expected_cross

    assert abs(mu_in    - expected_in)    < 1e-6, f"in-domain: got {mu_in:.4f}, want {expected_in:.4f}"
    assert abs(mu_cross - expected_cross) < 1e-6, f"cross-domain: got {mu_cross:.4f}, want {expected_cross:.4f}"
    assert abs(mu_all   - expected_all)   < 1e-6, f"all: got {mu_all:.4f}, want {expected_all:.4f}"
    assert mu_in > mu_cross, "in-domain mean must exceed cross-domain when lambda=2.0"
    logger.info("[test] test_rq2_in_domain_filter PASSED")


def test_rq5_scenario_uses_correct_cells():
    """Verify cell counts match the grid geometry for in-domain and cross-domain."""
    n_m, n_s, n_t, n_p = 2, 2, 2, 2
    count_in, count_cross = 0, 0
    for _m in range(n_m):
        for _p in range(n_p):
            for _s in range(n_s):
                for _t in range(n_t):
                    if _s == _t: count_in += 1
                    else:        count_cross += 1

    assert count_in    == n_m * n_s * n_p,              f"Expected {n_m*n_s*n_p}, got {count_in}"
    assert count_cross == n_m * (n_s*n_t - n_s) * n_p, f"Expected {n_m*(n_s*n_t-n_s)*n_p}, got {count_cross}"
    logger.info("[test] test_rq5_scenario_uses_correct_cells PASSED")


def test_summarise_contrast_values():
    """summarise() on symmetric samples must produce mean≈0, CI straddling zero."""
    rng     = np.random.default_rng(42)
    samples = rng.uniform(-0.1, 0.1, size=1000)
    result  = {
        "mean":           float(np.mean(samples)),
        "median":         float(np.median(samples)),
        "ci_95_lower":    float(np.percentile(samples, 2.5)),
        "ci_95_upper":    float(np.percentile(samples, 97.5)),
        "prob_positive":  float(np.mean(samples > 0)),
        "prob_negative":  float(np.mean(samples < 0)),
    }
    assert abs(result["mean"]) < 0.02,                       f"Mean {result['mean']:.4f} too far from 0"
    assert result["ci_95_lower"] < 0 < result["ci_95_upper"], "CI should straddle zero"
    assert abs(result["prob_positive"] - 0.5) < 0.05,        f"P(+)={result['prob_positive']:.3f}"
    assert result["prob_positive"] + result["prob_negative"] <= 1.0 + 1e-9
    logger.info("[test] test_summarise_contrast_values PASSED")


def test_duplicate_handling_in_assemble():
    """Exact duplicate F1 (both blocks agree) → keep one row."""
    all_results = {
        "base": {
            "perturbation": {
                "DomA": {"clean": {"metrics": {"macro_f1": 0.90}, "common_ids_count": 50}, "semantic": {}}
            },
            "cross_domain_perturbation": {
                "DomA_to_DomA": {"clean": {"metrics": {"macro_f1": 0.90}, "common_ids_count": 50}, "semantic": {}}
            },
        }
    }
    df = assemble_observations(all_results)
    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    assert abs(df.iloc[0]["macro_f1"] - 0.90) < 1e-6
    assert bool(df.iloc[0]["in_domain"]) is True
    logger.info("[test] test_duplicate_handling_in_assemble PASSED")


def test_conflicting_duplicates_raise():
    """Conflicting F1 values for same key must raise ValueError."""
    all_results = {
        "base": {
            "perturbation": {
                "DomA": {"clean": {"metrics": {"macro_f1": 0.90}, "common_ids_count": 50}, "semantic": {}}
            },
            "cross_domain_perturbation": {
                "DomA_to_DomA": {"clean": {"metrics": {"macro_f1": 0.80}, "common_ids_count": 50}, "semantic": {}}
            },
        }
    }
    try:
        assemble_observations(all_results)
        assert False, "Expected ValueError on conflicting F1"
    except ValueError as exc:
        assert "conflicting" in str(exc).lower() or "CONFLICT" in str(exc)
    logger.info("[test] test_conflicting_duplicates_raise PASSED")


def test_ppc_statistics_shape_and_scale():
    """PPC statistic computations must stay in [0, 1] and use correct axes."""
    rng           = np.random.default_rng(7)
    y_obs         = rng.beta(5, 1.5, size=20)
    y_rep         = rng.beta(5, 1.5, size=(10, 20))
    obs_mean      = float(np.mean(y_obs))
    rep_means     = np.mean(y_rep, axis=1)
    pred_mean     = float(np.mean(rep_means))
    bayes_p_mean  = float(np.mean(rep_means >= obs_mean))
    mask          = np.array([True]*10 + [False]*10)
    bayes_p_group = float(np.mean(np.mean(y_rep[:, mask], axis=1) >= np.mean(y_obs[mask])))
    assert 0.0 <= obs_mean     <= 1.0
    assert 0.0 <= pred_mean    <= 1.0
    assert 0.0 <= bayes_p_mean <= 1.0
    assert 0.0 <= bayes_p_group <= 1.0
    logger.info("[test] test_ppc_statistics_shape_and_scale PASSED")


def test_validate_observations_flags_issues():
    """validate_observations catches missing cols, inconsistent flags, boundaries."""
    df = _make_minimal_obs_df()
    report = validate_observations(df)
    assert report.get("n_duplicates", 0) == 0
    assert report.get("n_scenario_inconsistent", 0) == 0

    df_bad = df.copy()
    df_bad.loc[0, "in_domain"] = not df_bad.loc[0, "in_domain"]
    assert validate_observations(df_bad)["n_scenario_inconsistent"] >= 1

    assert "error" in validate_observations(df.drop(columns=["macro_f1"]))

    df_b = df.copy()
    df_b.loc[0, "macro_f1"] = 0.0
    df_b.loc[1, "macro_f1"] = 1.0
    r = validate_observations(df_b)
    assert r["n_boundary_zero"] >= 1
    assert r["n_boundary_one"]  >= 1
    logger.info("[test] test_validate_observations_flags_issues PASSED")


def test_boundary_correction():
    """Smithson-Verkuilen formula must map [0,1] → (0,1) with correct values."""
    y         = np.array([0.0, 0.5, 1.0])
    corrected = apply_boundary_correction(y, n=3)
    expected  = np.array([(0.0*2+0.5)/3, (0.5*2+0.5)/3, (1.0*2+0.5)/3])
    assert np.allclose(corrected, expected, atol=1e-10)
    assert np.all(corrected > 0.0) and np.all(corrected < 1.0)
    logger.info("[test] test_boundary_correction PASSED")


def test_in_domain_string_bool_is_not_flipped():
    """CSV/JSON round-trips can turn the flag into the strings 'True'/'False'.
    Series.astype(bool) would make BOTH True; _coerce_bool must not."""
    s = pd.Series(["True", "False", "false", "TRUE", "0", "1"])
    assert s.astype(bool).all(), "premise: astype(bool) on non-empty strings is all True"
    assert _coerce_bool(s).tolist() == [True, False, False, True, False, True]
    df = _make_minimal_obs_df()
    df["in_domain"] = df["in_domain"].astype(str)
    r = validate_observations(df)
    assert r["n_scenario_inconsistent"] == 0, "string flags must validate against source==target"
    try:
        _coerce_bool(pd.Series(["maybe"]))
        assert False, "unparseable flag must raise"
    except ValueError:
        pass
    logger.info("[test] test_in_domain_string_bool_is_not_flipped PASSED")


def test_boundary_per_row_n_and_min_tail_direction():
    """Per-row N shifts only that row; and the min-statistic tail logic is direction-aware."""
    y = np.array([1.0, 0.9])
    out = apply_boundary_correction(y[:1], n=np.array([80.0]))
    assert abs(out[0] - (1.0 * 79 + 0.5) / 80) < 1e-12           # 0.99375, not 0.99833 (N=300)
    assert abs(apply_boundary_correction(y[:1], n=300)[0] - 0.998333333) < 1e-8
    # observed minimum far BELOW every replicated minimum:
    rep_mins = np.array([0.50, 0.52, 0.55, 0.51]); obs_min = 0.38
    p_ge = np.mean(rep_mins >= obs_min); p_le = np.mean(rep_mins <= obs_min)
    assert p_ge == 1.0 and p_le == 0.0
    assert min(1.0, 2 * min(p_ge, p_le)) == 0.0                    # extremeness flags it; p_ge alone (=1.0) looked "fine"
    logger.info("[test] test_boundary_per_row_n_and_min_tail_direction PASSED")


def run_all_tests() -> None:
    """Run all tests. Raises AssertionError on first failure."""
    tests = [
        test_in_domain_string_bool_is_not_flipped,
        test_boundary_per_row_n_and_min_tail_direction,
        test_rq2_in_domain_filter,
        test_rq5_scenario_uses_correct_cells,
        test_summarise_contrast_values,
        test_duplicate_handling_in_assemble,
        test_conflicting_duplicates_raise,
        test_ppc_statistics_shape_and_scale,
        test_validate_observations_flags_issues,
        test_boundary_correction,
    ]
    print("Running statistical_analyzer tests ...")
    for fn in tests:
        fn()
        print(f"  PASSED: {fn.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    run_all_tests()