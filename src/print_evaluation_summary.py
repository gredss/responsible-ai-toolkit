"""
Comprehensive Evaluation Summary Printer
IndoBERT Clickbait Detection — Cross-Domain & Perturbation Robustness Study

Usage:
    Call print_full_evaluation_summary(results, model_name) after your evaluation pipeline,
    where `results` is the dict returned by EvaluationEngine (complete_evaluation.json).

    Optionally also runs ErrorAnalyzer to explain WHY performance drops:
        print_full_evaluation_summary(results, model_name, run_error_analysis=True)

Expected keys in `results`:
    - in_domain:       {domain: {metrics: {accuracy, f1, precision, recall, mcc, roc_auc, confusion_matrix}}}
    - cross_domain:    {"Src->Tgt": {metrics: {...}, domain_shift: {sd_f1, td_f1, ...}}}
    - perturbation:    {domain: {
                            clean:    {metrics},
                            semantic: {low: {metrics}, medium: {metrics}, high: {metrics}},
                            typo:     {low: {metrics}, medium: {metrics}, high: {metrics}},
                        }}

      The two perturbation mechanisms ('semantic' = synonym substitution,
      'typo' = character typos) share the SAME clean baseline, so 'clean' lives
      once at the top of each domain rather than under each type.
"""

import json
import numpy as np
import os
import argparse
from typing import Dict, Any
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

DOMAINS = ["Education", "Health", "Politics", "Sport", "Technology"]
METRICS = [
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "mcc",
    "roc_auc",
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro_F1",
    "macro_precision": "Macro-Precision",
    "macro_recall": "Macro-Recall",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
}
# The two perturbation mechanisms and their three intensity levels.
# 'clean' is the shared, unperturbed baseline (shown alongside both mechanisms).
PERTURB_TYPES = ["semantic", "typo"]
PERTURB_LEVELS = ["low", "medium", "high"]
# The full ordered sequence used when a chart/table shows clean -> increasing noise.
PERTURB_SEQUENCE = ["clean", "low", "medium", "high"]

# Human-readable labels for the perturbation types (used in titles/headers).
PERTURB_TYPE_LABELS = {
    "semantic": "Semantic (synonym substitution)",
    "typo": "Typo (character noise)",
}
SEP_THICK = "═" * 80
SEP_THIN  = "─" * 80
SEP_MID   = "·" * 80


def _v(val) -> float:
    """Safely cast numpy / None to float."""
    if val is None:
        return 0.0
    if hasattr(val, "item"):          # np scalar
        return float(val.item())
    return float(val)


def _pct(val: float) -> str:
    return f"{val * 100:+.1f}%"


def _bar(val: float, width: int = 20, fill: str = "█", empty: str = "░") -> str:
    """Simple ASCII progress bar, clamped to [0, 1]."""
    val = max(0.0, min(1.0, val))
    filled = round(val * width)
    return fill * filled + empty * (width - filled)


def _hdr(title: str, width: int = 80) -> str:
    pad = (width - len(title) - 2) // 2
    return f"{'═' * pad} {title} {'═' * (width - pad - len(title) - 2)}"


def _section(n: int, title: str) -> None:
    print(f"\n{SEP_THICK}")
    print(_hdr(f"{n}. {title}"))
    print(SEP_THICK)


def _safe_get(d: dict, *keys, default=0.0):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, {})
    return _v(d) if d != {} else default


def _clean_metric(perturbation: dict, in_domain: dict, domain: str,
                  metric: str = "macro_f1") -> float:
    """
    Return the shared CLEAN-baseline metric for one domain.

    The clean condition is stored once at perturbation[domain]['clean'], but for
    robustness it also falls back to the in-domain result (they are the same
    unperturbed evaluation).
    """
    clean = (
        perturbation.get(domain, {})
        .get("clean", {})
        .get("metrics", {})
        .get(metric)
    )
    if clean is None:
        clean = in_domain.get(domain, {}).get("metrics", {}).get(metric, 0)
    return _v(clean)


def _pert_metric(perturbation: dict, domain: str, ptype: str, level: str,
                 metric: str = "macro_f1") -> float:
    """
    Return one perturbed metric from the nested structure:
        perturbation[domain][ptype][level]['metrics'][metric]
    Missing entries return 0.0 so tables/plots never crash on partial results.
    """
    return _v(
        perturbation.get(domain, {})
        .get(ptype, {})
        .get(level, {})
        .get("metrics", {})
        .get(metric, 0)
    )


def load_results(results_dir: str) -> dict:
    """
    Load complete_evaluation.json from results_dir.

    cross_domain keys are stored as "Source->Target" strings by save_results()
    in evaluation_engine.py.  All lookup functions in this module use that same
    format, so no key transformation is needed here.
    """
    file_path = os.path.join(results_dir, "complete_evaluation.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find evaluation file at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# ──────────────────────────────────────────────────────────────────────────────
# 1.  IN-DOMAIN PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────

def _print_in_domain(in_domain: Dict[str, Any]) -> None:
    _section(1, "IN-DOMAIN PERFORMANCE (Specialist → Own Domain)")

    col_w = 11
    hdr = f"{'Domain':<14}" + "".join(f"{METRIC_LABELS[m]:>{col_w}}" for m in METRICS)
    print(hdr)
    print(SEP_THIN)

    for dom in DOMAINS:
        if dom not in in_domain:
            continue
        mets = in_domain[dom].get("metrics", {})
        row = f"{dom:<14}"
        for m in METRICS:
            row += f"{_v(mets.get(m, 0)):>{col_w}.4f}"
        print(row)

    # summary row
    print(SEP_THIN)
    means = {}
    for m in METRICS:
        vals = [_v(in_domain[d]["metrics"].get(m, 0)) for d in DOMAINS if d in in_domain]
        means[m] = np.mean(vals) if vals else 0.0

    row = f"{'  MEAN':<14}"
    for m in METRICS:
        row += f"{means[m]:>{col_w}.4f}"
    print(row)

    # visual bar chart of F1
    print(f"\n  In-Domain F1 (visual):")
    for dom in DOMAINS:
        if dom not in in_domain:
            continue
        f1 = _v(in_domain[dom]["metrics"].get("macro_f1", 0))
        print(f"  {dom:<12} {_bar(f1)} {f1:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CROSS-DOMAIN F1 MATRIX  (rows = source, cols = target)
# ──────────────────────────────────────────────────────────────────────────────

def _print_cross_domain_matrix(cross_domain: dict, metric: str = "macro_f1") -> None:
    _section(2, f"CROSS-DOMAIN MATRIX — {METRIC_LABELS[metric]}  (Row=Source, Col=Target)")

    col_w = 12
    short = [d[:9] for d in DOMAINS]
    header_label = 'Source \\ Target'
    print(f"{header_label:<16}" + "".join(f"{s:>{col_w}}" for s in short))
    print(SEP_THIN)

    for src in DOMAINS:
        row = f"{src:<16}"
        for tgt in DOMAINS:
            key = f"{src}->{tgt}"
            val = _safe_get(cross_domain.get(key, {}), "metrics", metric)
            marker = "◆" if src == tgt else " "
            row += f"{val:>{col_w - 1}.4f}{marker}"
        print(row)

    # column averages (cross-domain only, i.e. src ≠ tgt)
    print(SEP_THIN)
    row = f"{'  col-avg':<16}"
    col_avgs = []
    for tgt in DOMAINS:
        vals = [
            _safe_get(cross_domain.get(f"{src}->{tgt}", {}), "metrics", metric)
            for src in DOMAINS if src != tgt
        ]
        avg = np.mean(vals) if vals else 0.0
        col_avgs.append(avg)
        row += f"{avg:>{col_w - 1}.4f} "
    print(row)

    # row averages
    print(f"\n  Row averages (source generalises to other domains):")
    for src in DOMAINS:
        vals = [
            _safe_get(cross_domain.get(f"{src}->{tgt}", {}), "metrics", metric)
            for tgt in DOMAINS if tgt != src
        ]
        avg = np.mean(vals) if vals else 0.0
        print(f"  {src:<14} avg OOD F1 = {avg:.4f}  {_bar(avg, 16)}")

    print(f"\n  ◆ = in-domain (diagonal) result shown for reference")


# ──────────────────────────────────────────────────────────────────────────────
# 3.  DOMAIN SHIFT METRICS  (SD & TD per source–target pair)
# ──────────────────────────────────────────────────────────────────────────────

def _print_domain_shift(cross_domain: dict, in_domain: dict) -> None:
    _section(3, "DOMAIN SHIFT METRICS  (SD = Source Drop, TD = Target Drop)")

    print(
        "  SD = PID(source)  − P(source→target)   ← how much the SOURCE model degrades\n"
        "  TD = PID(target)  − P(source→target)   ← gap vs target specialist\n"
        "  Positive = degradation  |  Negative = surprisingly better OOD\n"
    )

    # Per-metric SD/TD table
    sub_metrics = [
		"macro_f1",
		"accuracy",
		"macro_precision",
		"macro_recall",
	]
    col_w = 10

    print(f"  {'Pair (src→tgt)':<26}", end="")
    for sm in sub_metrics:
        lbl = METRIC_LABELS[sm][:8]
        print(f"  {'SD_' + lbl:>10}  {'TD_' + lbl:>10}", end="")
    print()
    print("  " + SEP_THIN)

    # collect for aggregate analysis
    all_sd_f1, all_td_f1 = [], []
    worst_sd, worst_td = [], []

    for src in DOMAINS:
        for tgt in DOMAINS:
            if src == tgt:
                continue
            key = f"{src}->{tgt}"
            entry = cross_domain.get(key, {})
            ds = entry.get("domain_shift", {})
            if not ds:
                continue

            label = f"{src[:5]}→{tgt[:5]}"
            print(f"  {label:<26}", end="")
            for sm in sub_metrics:
                sd_val = _v(ds.get(f"sd_{sm}", 0))
                td_val = _v(ds.get(f"td_{sm}", 0))
                print(f"  {sd_val:>+10.4f}  {td_val:>+10.4f}", end="")
            print()

            all_sd_f1.append(_v(ds.get("sd_macro_f1", 0)))
            all_td_f1.append(_v(ds.get("td_macro_f1", 0)))
            worst_sd.append((src, tgt, _v(ds.get("sd_macro_f1", 0))))
            worst_td.append((src, tgt, _v(ds.get("td_macro_f1", 0))))

    # aggregates
    print("  " + SEP_THIN)
    if all_sd_f1:
        print(f"\n  Aggregate (F1-based):")
        print(f"    Mean SD (source degradation)  : {np.mean(all_sd_f1):+.4f}")
        print(f"    Mean TD (target gap)          : {np.mean(all_td_f1):+.4f}")
        print(f"    Max  SD (most vulnerable src) : {max(all_sd_f1):+.4f}")
        print(f"    Max  TD (hardest target)      : {max(all_td_f1):+.4f}")

        worst_sd.sort(key=lambda x: x[2], reverse=True)
        worst_td.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  Top-3 worst Source Drop (F1):")
        for src, tgt, val in worst_sd[:3]:
            print(f"    {src}→{tgt}  SD_Macro_F1 = {val:+.4f}")
        print(f"\n  Top-3 worst Target Drop (F1):")
        for src, tgt, val in worst_td[:3]:
            print(f"    {src}→{tgt}  TD_Macro_F1 = {val:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# 4.  FULL CROSS-DOMAIN METRICS TABLE (all metrics, OOD pairs only)
# ──────────────────────────────────────────────────────────────────────────────

def _print_cross_domain_full(cross_domain: dict) -> None:
    _section(4, "CROSS-DOMAIN FULL METRICS  (OOD pairs, src ≠ tgt)")

    col_w = 10
    print(f"  {'Pair (src→tgt)':<22}", end="")
    for m in METRICS:
        print(f"  {METRIC_LABELS[m]:>{col_w}}", end="")
    print()
    print("  " + SEP_THIN)

    for src in DOMAINS:
        for tgt in DOMAINS:
            if src == tgt:
                continue
            key = f"{src}->{tgt}"
            entry = cross_domain.get(key, {})
            mets = entry.get("metrics", {})
            label = f"{src[:6]}→{tgt[:6]}"
            print(f"  {label:<22}", end="")
            for m in METRICS:
                print(f"  {_v(mets.get(m, 0)):>{col_w}.4f}", end="")
            print()

    # overall OOD averages
    print("  " + SEP_THIN)
    row = f"  {'  OOD MEAN':<22}"
    for m in METRICS:
        vals = [
            _v(cross_domain.get(f"{src}->{tgt}", {}).get("metrics", {}).get(m, 0))
            for src in DOMAINS
            for tgt in DOMAINS
            if src != tgt
        ]
        row += f"  {np.mean(vals):>{col_w}.4f}"
    print(row)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PERTURBATION ROBUSTNESS
# ──────────────────────────────────────────────────────────────────────────────

def _print_perturbation(perturbation: dict, in_domain: dict) -> None:
    _section(5, "PERTURBATION ROBUSTNESS  (Macro-F1 across noise levels)")

    col_w = 10

    # One summary table PER perturbation type. Each shows the shared Clean
    # baseline followed by that type's Low / Medium / High, plus the drop
    # from Clean to High so the intensity trend is easy to read.
    for ptype in PERTURB_TYPES:
        print(f"\n  === {PERTURB_TYPE_LABELS[ptype]} ===")
        print(f"  {'Domain':<14}", end="")
        for lvl in PERTURB_SEQUENCE:
            print(f"  {lvl.capitalize():>{col_w}}", end="")
        print(f"  {'Drop(hi-cl)':>{col_w}}  {'Δ%':>7}")
        print("  " + SEP_THIN)

        all_drops = []
        for dom in DOMAINS:
            clean_f1 = _clean_metric(perturbation, in_domain, dom, "macro_f1")
            row_vals = {"clean": clean_f1}
            for lvl in PERTURB_LEVELS:
                row_vals[lvl] = _pert_metric(perturbation, dom, ptype, lvl, "macro_f1")

            drop = clean_f1 - row_vals["high"]
            drop_pct = (drop / clean_f1 * 100) if clean_f1 > 0 else 0.0
            all_drops.append(drop)

            print(f"  {dom:<14}", end="")
            for lvl in PERTURB_SEQUENCE:
                print(f"  {row_vals[lvl]:>{col_w}.4f}", end="")
            drop_marker = "▼" if drop > 0.05 else ("▲" if drop < -0.01 else "≈")
            print(f"  {drop:>+{col_w}.4f}  {drop_pct:>+6.1f}% {drop_marker}")

        # Mean row across domains.
        print("  " + SEP_THIN)
        print(f"  {'  MEAN':<14}", end="")
        for lvl in PERTURB_SEQUENCE:
            if lvl == "clean":
                vals = [
                    _clean_metric(perturbation, in_domain, d, "macro_f1")
                    for d in DOMAINS
                ]
            else:
                vals = [
                    _pert_metric(perturbation, d, ptype, lvl, "macro_f1")
                    for d in DOMAINS
                ]
            print(f"  {np.mean(vals):>{col_w}.4f}", end="")
        print(f"  {np.mean(all_drops):>+{col_w}.4f}")

    # Per-domain, all-metrics breakdown, still grouped by perturbation type.
    print(f"\n  Perturbation — All Metrics per Domain:\n")
    metric_cols = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
    for dom in DOMAINS:
        clean_mets = in_domain.get(dom, {}).get("metrics", {})
        print(f"  [{dom}]")
        for ptype in PERTURB_TYPES:
            print(f"    {PERTURB_TYPE_LABELS[ptype]}")
            print(f"    {'Level':<10}", end="")
            for m in metric_cols:
                print(f"  {METRIC_LABELS[m]:>{col_w}}", end="")
            print()
            for lvl in PERTURB_SEQUENCE:
                if lvl == "clean":
                    mets = clean_mets
                else:
                    mets = (
                        perturbation.get(dom, {})
                        .get(ptype, {})
                        .get(lvl, {})
                        .get("metrics", {})
                    )
                print(f"    {lvl.capitalize():<10}", end="")
                for m in metric_cols:
                    print(f"  {_v(mets.get(m, 0)):>{col_w}.4f}", end="")
                print()
            print()
        print()


# ──────────────────────────────────────────────────────────────────────────────
# 6.  CONFUSION MATRIX SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def _print_confusion_matrices(in_domain: dict, cross_domain: dict) -> None:
    _section(6, "CONFUSION MATRICES — IN-DOMAIN (Specialist)")

    for dom in DOMAINS:
        if dom not in in_domain:
            continue
        cm = in_domain[dom]["metrics"].get("confusion_matrix")
        if not cm:
            continue
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        print(f"\n  [{dom}]  (total={total})")
        print(f"              Pred 0    Pred 1")
        print(f"  True 0  :  {tn:>6}    {fp:>6}   (TN={tn}, FP={fp})")
        print(f"  True 1  :  {fn:>6}    {tp:>6}   (FN={fn}, TP={tp})")
        if (tp + fn) > 0:
            print(f"  Clickbait recall = {tp/(tp+fn):.4f}   |   "
                  f"Clickbait precision = {tp/(tp+fp):.4f}" if (tp+fp) > 0 else "  precision = undef")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CLASS IMBALANCE CHECK
# ──────────────────────────────────────────────────────────────────────────────

def _print_class_imbalance(in_domain: dict) -> None:
    _section(7, "CLASS DISTRIBUTION — TEST SPLITS  (per domain)")

    print(f"  {'Domain':<14} {'N':>5}  {'Label-0':>8}  {'Label-1':>8}  {'CB-ratio':>10}  Status")
    print("  " + SEP_THIN)

    for dom in DOMAINS:
        if dom not in in_domain:
            continue
        entry = in_domain[dom]
        labels = entry.get("true_labels", [])
        if not labels:
            continue
        n = len(labels)
        n0 = labels.count(0)
        n1 = labels.count(1)
        ratio = n1 / n if n > 0 else 0
        flag = "⚠  highly imbalanced" if ratio < 0.15 or ratio > 0.85 else (
               "△  imbalanced" if ratio < 0.25 or ratio > 0.75 else "✓  balanced")
        print(f"  {dom:<14} {n:>5}  {n0:>8}  {n1:>8}  {ratio:>10.3f}  {flag}")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  GLOBAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def _print_global_summary(in_domain: dict, cross_domain: dict, perturbation: dict,
                           model_name: str) -> None:
    _section(8, f"GLOBAL SUMMARY — {model_name.upper()} MODEL")

    # In-domain averages
    print(f"\n  ── In-Domain (Specialist on Own Data) ──")
    for m in ["accuracy","macro_f1","macro_precision","macro_recall","mcc"]:
        vals = [_v(in_domain[d]["metrics"].get(m, 0)) for d in DOMAINS if d in in_domain]
        lbl = METRIC_LABELS[m]
        mean = np.mean(vals) if vals else 0.0
        print(f"    {lbl:<14}: {mean:.4f}  {_bar(mean, 24)}")

    # Cross-domain OOD averages
    print(f"\n  ── Cross-Domain OOD (src ≠ tgt) ──")
    for m in ["accuracy","macro_f1","macro_precision","macro_recall"]:
        vals = [
            _v(cross_domain.get(f"{src}->{tgt}", {}).get("metrics", {}).get(m, 0))
            for src in DOMAINS
            for tgt in DOMAINS
            if src != tgt
        ]
        lbl = METRIC_LABELS[m]
        mean = np.mean(vals) if vals else 0.0
        print(f"    {lbl:<14}: {mean:.4f}  {_bar(mean, 24)}")

    # Perturbation summary — one block per mechanism, sharing the clean baseline.
    clean_mean = np.mean([
        _clean_metric(perturbation, in_domain, d, "macro_f1") for d in DOMAINS
    ]) if DOMAINS else 0.0
    print(f"\n  ── Perturbation (avg Macro-F1 across domains) ──")
    print(f"    {'Clean':<14}: {clean_mean:.4f}  {_bar(clean_mean, 24)}")
    for ptype in PERTURB_TYPES:
        print(f"    [{ptype}]")
        for lvl in PERTURB_LEVELS:
            vals = [
                _pert_metric(perturbation, d, ptype, lvl, "macro_f1")
                for d in DOMAINS
            ]
            mean = np.mean(vals) if vals else 0.0
            print(f"      {lvl.capitalize():<12}: {mean:.4f}  {_bar(mean, 24)}")

    # Domain shift aggregates
    print(f"\n  ── Domain Shift (mean F1 delta) ──")
    sd_vals = [
        _v(cross_domain.get(f"{src}->{tgt}", {}).get("domain_shift", {}).get("sd_macro_f1", 0))
        for src in DOMAINS for tgt in DOMAINS if src != tgt
        if "domain_shift" in cross_domain.get(f"{src}->{tgt}", {})
    ]
    td_vals = [
        _v(cross_domain.get(f"{src}->{tgt}", {}).get("domain_shift", {}).get("td_macro_f1", 0))
        for src in DOMAINS for tgt in DOMAINS if src != tgt
        if "domain_shift" in cross_domain.get(f"{src}->{tgt}", {})
    ]
    if sd_vals:
        print(f"    Mean Source Drop (SD): {np.mean(sd_vals):+.4f}")
        print(f"    Mean Target Drop (TD): {np.mean(td_vals):+.4f}")

    print(f"\n{SEP_THICK}")

# ──────────────────────────────────────────────────────────────────────────────
# 9.  PLOT PERTURBATION CURVES
# ──────────────────────────────────────────────────────────────────────────────

def _plot_perturbation_curves(perturbation: dict,
                              in_domain: dict,
                              output_dir: str = "."):
    """
    Per-domain robustness curves, drawn as TWO side-by-side subplots:
    one for the semantic mechanism and one for the typo mechanism.

    Each subplot shows the 5 domain lines across Clean -> Low -> Medium -> High.
    The Clean point is the shared unperturbed baseline, so it appears in both
    subplots for reference. The two subplots share the same y-axis so the
    intensity trend within each mechanism is easy to read, without implying the
    two mechanisms are directly comparable point-for-point.
    """
    x_labels = [lvl.capitalize() for lvl in PERTURB_SEQUENCE]  # Clean/Low/Medium/High
    x_base = np.arange(len(PERTURB_SEQUENCE))                   # [0,1,2,3]

    # Small horizontal offset per domain so overlapping points stay readable.
    offsets = np.linspace(-0.06, 0.06, len(DOMAINS))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, ptype in zip(axes, PERTURB_TYPES):
        for idx, dom in enumerate(DOMAINS):
            clean = _clean_metric(perturbation, in_domain, dom, "macro_f1")
            values = [clean] + [
                _pert_metric(perturbation, dom, ptype, lvl, "macro_f1")
                for lvl in PERTURB_LEVELS
            ]
            ax.plot(
                x_base + offsets[idx],
                values,
                marker="o",
                linewidth=2.5,
                alpha=0.85,
                label=dom,
            )

        ax.set_title(PERTURB_TYPE_LABELS[ptype], fontsize=13, fontweight="bold")
        ax.set_xlabel("Perturbation Level", fontsize=12, fontweight="bold")
        ax.set_xticks(x_base)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)

    # Shared y-axis label on the left subplot only.
    axes[0].set_ylabel("Macro-F1 Score", fontsize=12, fontweight="bold")

    # One shared legend for both subplots, placed to the right.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Specialist Domain",
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        frameon=True,
    )

    fig.suptitle(
        "Model Robustness Against Perturbation (per domain)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()

    save_path = os.path.join(output_dir, "perturbation_per_domain.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()      # display in notebook
    plt.close(fig)  # then free memory

    print(f"Saved: {save_path}")

# ──────────────────────────────────────────────────────────────────────────────

def _plot_mean_perturbation(perturbation, in_domain,
                            output_dir: str = "."):
    """
    Mean robustness curve (averaged over all domains), drawn as TWO side-by-side
    subplots: one for semantic, one for typo. Same Clean baseline in both,
    shared y-axis, consistent with _plot_perturbation_curves.
    """
    x_labels = [lvl.capitalize() for lvl in PERTURB_SEQUENCE]
    x_base = np.arange(len(PERTURB_SEQUENCE))

    # Clean mean is shared across both subplots.
    clean_mean = np.mean([
        _clean_metric(perturbation, in_domain, dom, "macro_f1")
        for dom in DOMAINS
    ]) if DOMAINS else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, ptype in zip(axes, PERTURB_TYPES):
        means = [clean_mean]
        for lvl in PERTURB_LEVELS:
            vals = [
                _pert_metric(perturbation, dom, ptype, lvl, "macro_f1")
                for dom in DOMAINS
            ]
            means.append(np.mean(vals) if vals else 0.0)

        ax.plot(
            x_base,
            means,
            marker="o",
            markersize=8,
            linewidth=2.5,
            color="tab:blue",
            label="Average",
        )
        ax.set_title(PERTURB_TYPE_LABELS[ptype], fontsize=13, fontweight="bold")
        ax.set_xlabel("Perturbation Level", fontsize=12, fontweight="bold")
        ax.set_xticks(x_base)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.6)

    axes[0].set_ylabel("Mean Macro-F1 Score", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Mean Model Robustness Against Perturbation (all domains)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()

    save_path = os.path.join(output_dir, "perturbation_mean.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()      # display in notebook
    plt.close(fig)  # then free memory

    print(f"Saved: {save_path}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def print_full_evaluation_summary(
    results: dict,
    model_name: str = "BASE",
    run_error_analysis: bool = True,
    output_dir: str = "."
) -> None:
    """
    Print a complete, formatted evaluation summary to stdout.

    Args:
        results:            Output of EvaluationEngine.run_complete_evaluation()
        model_name:         Label for all section headers
        run_error_analysis: When True, also run ErrorAnalyzer and print its
                            report explaining WHY performance drops
        output_dir:         Directory where plot PNGs are saved.
                            Defaults to '.' (cwd). Pass --results-dir here
                            when running via CLI so plots land next to the JSON.
    """
    in_domain    = results.get("in_domain", {})
    cross_domain = results.get("cross_domain", {})
    perturbation = results.get("perturbation", {})

    banner = f"  EVALUATION SUMMARY  ·  MODEL: {model_name.upper()}"
    print(f"\n{'═' * 80}")
    print(f"{'═' * 80}")
    pad = (80 - len(banner)) // 2
    print(" " * pad + banner)
    print(f"{'═' * 80}")
    print(f"{'═' * 80}")

    _print_in_domain(in_domain)
    _print_cross_domain_matrix(cross_domain, metric="macro_f1")
    _print_domain_shift(cross_domain, in_domain)
    _print_cross_domain_full(cross_domain)
    _print_perturbation(perturbation, in_domain)
    _plot_perturbation_curves(perturbation, in_domain, output_dir=output_dir)
    _plot_mean_perturbation(perturbation, in_domain, output_dir=output_dir)
    _print_confusion_matrices(in_domain, cross_domain)
    _print_class_imbalance(in_domain)
    _print_global_summary(in_domain, cross_domain, perturbation, model_name)

    # Optional: WHY analysis
    if run_error_analysis:
        try:
            from error_analyzer import ErrorAnalyzer
            analyzer = ErrorAnalyzer()
            ea_report = analyzer.analyze(results, model_name=model_name)
            analyzer.print_report(ea_report)
        except ImportError:
            print("\n[WARN] error_analyzer.py not found — skipping error analysis.")
        except Exception as e:
            print(f"\n[WARN] Error analysis failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(
        description="Print evaluation summary from evaluation results."
    )

    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing the evaluation JSON files."
    )

    parser.add_argument(
        "--model",
        default="BASE",
        help="Model name (BASE, LARGE, LITE, etc.)"
    )

    parser.add_argument(
        "--file",
        default="complete_evaluation.json",
        help="Evaluation result JSON filename (default: complete_evaluation.json)."
    )

    parser.add_argument(
        "--no-error-analysis",
        action="store_true",
        help="Skip the ErrorAnalyzer section."
    )

    args = parser.parse_args()

    result_path = os.path.join(args.results_dir, args.file)

    if not os.path.exists(result_path):
        result_path = os.path.join(
            args.results_dir,
            args.model.lower(),
            args.file
        )

    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print_full_evaluation_summary(
        results,
        model_name=args.model.upper(),
        run_error_analysis=not args.no_error_analysis,
        output_dir=args.results_dir      # plots saved alongside the JSON
    )
