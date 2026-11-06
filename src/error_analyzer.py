"""
Error Analysis Module for IndoBERT Clickbait Detection System

Answers the question: *WHY* does performance drop under domain shift or
perturbation — not just *by how much*.

Each misclassified headline is examined for five linguistic patterns known
to distinguish clickbait from non-clickbait in Indonesian news:

  1. Sensational wording   — emotionally charged / superlative language
  2. Numerical claims      — specific numbers used to create false precision
  3. Rhetorical questions  — interrogative hooks that withhold the answer
  4. Named entities        — celebrities, brands, political figures as bait
  5. Domain-specific jargon — terminology dense with domain vocabulary

Usage (standalone):
    python error_analyzer.py \
        --results-dir evaluation_results/base \
        --model BASE

Usage (programmatic, after EvaluationEngine):
    from error_analyzer import ErrorAnalyzer
    analyzer = ErrorAnalyzer()
    report = analyzer.analyze(results, model_name="BASE")
    analyzer.save_report(report, output_dir="evaluation_results/base")
    analyzer.print_report(report)
"""

import re
import os
import json
import argparse
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from debug_logger import (
    dbg_error_input,
    dbg_error_attribution,
    dbg_error_summary,
)


# ──────────────────────────────────────────────────────────────────────────────
# PATTERN CATALOGUE  (Indonesian-specific)
# ──────────────────────────────────────────────────────────────────────────────

# 1. Sensational / emotionally charged words
SENSATIONAL_WORDS = {
    # Superlatives / extremes
    "terkejut", "mengejutkan", "luar biasa", "mencengangkan", "tak terduga",
    "viral", "heboh", "gempar", "geger", "kaget", "dahsyat", "spektakuler",
    "fantastis", "mengerikan", "menakjubkan", "memukau", "terguncang",
    # Urgency / exclusivity
    "breaking", "eksklusif", "terbaru", "terpanas", "segera", "darurat",
    "penting", "wajib", "harus", "segera tahu", "jangan lewatkan",
    # Emotional triggers
    "sedih", "haru", "menangis", "mengharukan", "menyentuh", "miris",
    "marah", "murka", "amarah", "benci", "cinta", "rindu", "khawatir",
    "takut", "panik", "cemas", "trauma",
    # Clickbait qualifiers
    "ternyata", "rupanya", "faktanya", "sebenarnya", "rahasianya",
    "alasannya", "begini cara", "inilah", "inilah dia", "beginilah"
}

# 2. Named-entity indicators (title-case words, known suffixes/prefixes)
NE_TITLE_PATTERN   = re.compile(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+')
NE_SUFFIX_PATTERN  = re.compile(
    r'\b(?:Presiden|Menteri|Gubernur|Bupati|Walikota|Ketua|Direktur|CEO|'
    r'Prof|Dr|Ir|Hj|H|Kiai|Gus|Ustaz|Bapak|Ibu)\b', re.IGNORECASE
)
# Common celebrity / brand markers
NE_BRAND_PATTERN   = re.compile(
    r'\b(?:Instagram|TikTok|YouTube|Twitter|Facebook|WhatsApp|Google|Apple|'
    r'Samsung|Pertamina|Telkom|Gojek|Tokopedia|Shopee|BCA|Mandiri|BNI|BRI)\b',
    re.IGNORECASE
)

# 3. Rhetorical / curiosity-gap questions
RHETORICAL_PATTERN = re.compile(
    r'(?:'
    r'\bapa(?:kah)?\b.*\?|'           # Apa / Apakah … ?
    r'\bsiapa(?:kah)?\b.*\?|'         # Siapakah … ?
    r'\bmengapa\b.*\?|'               # Mengapa … ?
    r'\bkenapa\b.*\?|'                # Kenapa … ?
    r'\bbagaimana\b.*\?|'             # Bagaimana … ?
    r'\bkapan\b.*\?|'                 # Kapan … ?
    r'\bdi mana\b.*\?|'               # Di mana … ?
    r'\bbenarkah\b.*\?|'              # Benarkah … ?
    r'\bsudahkah\b.*\?|'              # Sudahkah … ?
    r'\bbisa(?:kah)?\b.*\?|'          # Bisakah … ?
    r'[Ii]ni (?:yang|adalah).*\?'     # "Ini yang …?"
    r')',
    re.IGNORECASE
)

# 4. Numerical claims (creates false precision / authority)
NUM_PATTERN = re.compile(
    r'(?:'
    r'\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s*(?:persen|%|miliar|triliun|juta|ribu|rb|M|T)\b|'
    r'\b\d+\s*(?:orang|korban|kasus|hari|jam|menit|tahun|bulan|kali|jenis|cara|fakta|alasan|langkah|tips)\b|'
    r'#\d+|'                          # ranking / list items
    r'\bke-\d+\b|'                    # "ke-5 kali"
    r'\bperingkat\s+\d+\b'
    r')',
    re.IGNORECASE
)

# 5. Domain-specific jargon (per domain vocabulary)
DOMAIN_JARGON: Dict[str, List[str]] = {
    "Technology": [
        "ai", "artificial intelligence", "machine learning", "deep learning",
        "blockchain", "cryptocurrency", "bitcoin", "startup", "unicorn",
        "fintech", "e-commerce", "digital", "aplikasi", "platform", "data",
        "server", "cloud", "cybersecurity", "hacker", "ransomware", "malware",
        "5g", "iot", "metaverse", "nft", "robot", "drone", "gadget", "chip",
    ],
    "Politics": [
        "pilkada", "pilpres", "pemilu", "partai", "koalisi", "oposisi",
        "dpr", "mpr", "dprd", "presiden", "gubernur", "bupati", "walikota",
        "menteri", "kabinet", "undang-undang", "perppu", "perda", "kpu",
        "bawaslu", "mahkamah", "konstitusi", "demokrasi", "oligarki",
        "korupsi", "kpk", "kejaksaan", "kepolisian",
    ],
    "Health": [
        "covid", "vaksin", "virus", "bakteri", "pandemi", "epidemi",
        "rumah sakit", "dokter", "pasien", "obat", "terapi", "kanker",
        "diabetes", "hipertensi", "stroke", "jantung", "paru-paru",
        "kesehatan mental", "depresi", "anxietas", "bpjs", "puskesmas",
        "gizi", "nutrisi", "kalori", "diet", "olahraga",
    ],
    "Sport": [
        "gol", "skor", "liga", "turnamen", "piala", "juara", "runner-up",
        "pelatih", "pemain", "transfer", "kontrak", "pertandingan", "laga",
        "stadion", "bola", "basket", "badminton", "atletik", "renang",
        "tinju", "mma", "esports", "gaming", "timnas", "pssi", "pbsi",
    ],
    "Education": [
        "universitas", "perguruan tinggi", "mahasiswa", "siswa", "guru",
        "dosen", "kurikulum", "merdeka belajar", "snmptn", "sbmptn", "utbk",
        "beasiswa", "skripsi", "tesis", "disertasi", "akreditasi", "blt",
        "ppdb", "ujian", "un", "zonasi", "sertifikasi", "pendidikan",
    ],
}
# Flatten into a single set for quick multi-domain lookup
_ALL_JARGON = {word for words in DOMAIN_JARGON.values() for word in words}


# ──────────────────────────────────────────────────────────────────────────────
# PATTERN DETECTOR  (pure functions, no side effects)
# ──────────────────────────────────────────────────────────────────────────────

def detect_patterns(text: str, domain: Optional[str] = None) -> Dict[str, bool]:
    """
    Detect which linguistic error-patterns are present in *text*.

    Args:
        text:   Headline string to inspect
        domain: Optional domain name for domain-jargon lookup

    Returns:
        Dict mapping pattern name → bool
    """
    text_lower = text.lower()

    # 1. Sensational wording
    sensational = any(w in text_lower for w in SENSATIONAL_WORDS)

    # 2. Numerical claims
    numerical = bool(NUM_PATTERN.search(text))

    # 3. Rhetorical questions
    rhetorical = bool(RHETORICAL_PATTERN.search(text))

    # 4. Named entities
    named_entity = (
        bool(NE_TITLE_PATTERN.search(text))
        or bool(NE_SUFFIX_PATTERN.search(text))
        or bool(NE_BRAND_PATTERN.search(text))
    )

    # 5. Domain-specific jargon
    jargon_vocab = DOMAIN_JARGON.get(domain, list(_ALL_JARGON)) if domain else list(_ALL_JARGON)
    domain_jargon = any(j in text_lower for j in jargon_vocab)

    return {
        "sensational_wording": sensational,
        "numerical_claim":     numerical,
        "rhetorical_question": rhetorical,
        "named_entity":        named_entity,
        "domain_jargon":       domain_jargon,
    }


PATTERN_LABELS = {
    "sensational_wording": "Sensational wording",
    "numerical_claim":     "Numerical claim",
    "rhetorical_question": "Rhetorical question",
    "named_entity":        "Named entity",
    "domain_jargon":       "Domain jargon",
}


# ──────────────────────────────────────────────────────────────────────────────
# ERROR TYPE TAXONOMY
# ──────────────────────────────────────────────────────────────────────────────

def classify_error_type(y_true: int, y_pred: int) -> Optional[str]:
    """
    Return the error type name, or None if prediction is correct.

    Args:
        y_true: Ground-truth label (0=non-clickbait, 1=clickbait)
        y_pred: Predicted label

    Returns:
        'false_positive' | 'false_negative' | None
    """
    if y_true == y_pred:
        return None
    return "false_positive" if y_pred == 1 else "false_negative"


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ANALYZER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class ErrorAnalyzer:
    """
    Analyses misclassified headlines to explain *why* performance drops.

    Produces per-domain, per-condition (in-domain / cross-domain /
    perturbation-level) breakdowns showing which linguistic patterns are
    over-represented in errors versus correct predictions.
    """

    PATTERNS = list(PATTERN_LABELS.keys())

    def analyze(
        self,
        results: Dict[str, Any],
        model_name: str = "MODEL"
    ) -> Dict[str, Any]:
        """
        Run full error analysis over a complete evaluation results dict.

        Args:
            results:    Output of EvaluationEngine.run_complete_evaluation()
            model_name: Label for the report header

        Returns:
            Nested dict with error analysis for every evaluation condition
        """
        report: Dict[str, Any] = {
            "model": model_name,
            "in_domain":    self._analyze_section(results.get("in_domain", {})),
            "cross_domain": self._analyze_cross_domain(results.get("cross_domain", {})),
            "perturbation": self._analyze_perturbation(results.get("perturbation", {})),
            "summary":      {}
        }
        report["summary"] = self._build_summary(report)
        return report

    # ── section helpers ───────────────────────────────────────────────────────

    def _analyze_section(
        self,
        section: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyse in-domain results: {domain: {texts, true_labels, predictions, …}}.
        """
        out: Dict[str, Any] = {}
        for domain, data in section.items():
            texts  = data.get("texts") or self._recover_texts(data)
            labels = data.get("true_labels", [])
            preds  = data.get("predictions", [])
            if not texts or not labels or not preds:
                continue
            out[domain] = self._analyse_predictions(
                texts, labels, preds, domain=domain
            )
        return out

    def _analyze_cross_domain(
        self,
        cross_domain: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyse cross-domain results stored with "Source->Target" string keys.
        """
        out: Dict[str, Any] = {}
        for key, data in cross_domain.items():
            src, tgt = (key.split("->") + ["?", "?"])[:2]
            texts  = data.get("texts") or self._recover_texts(data)
            labels = data.get("true_labels", [])
            preds  = data.get("predictions", [])
            if not texts or not labels or not preds:
                continue
            out[key] = self._analyse_predictions(
                texts, labels, preds, domain=tgt
            )
        return out

    def _analyze_perturbation(
        self,
        perturbation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyse perturbation results: {domain: {level: {texts, …}}}.
        """
        out: Dict[str, Any] = {}
        for domain, levels in perturbation.items():
            out[domain] = {}
            for level, data in levels.items():
                if level == "clean":
                    continue          # clean baseline already covered in in_domain
                texts  = data.get("texts") or self._recover_texts(data)
                labels = data.get("true_labels", [])
                preds  = data.get("predictions", [])
                if not texts or not labels or not preds:
                    continue
                out[domain][level] = self._analyse_predictions(
                    texts, labels, preds, domain=domain
                )
        return out

    # ── core per-sample analysis ──────────────────────────────────────────────

    def _analyse_predictions(
        self,
        texts:  List[str],
        labels: List[int],
        preds:  List[int],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        For one (texts, labels, preds) triple, compute:
          - error rate per pattern in errors vs. correct samples
          - example misclassified headlines per pattern

        Returns a dict with counts, rates, pattern_attribution, and examples.
        """
        n = min(len(texts), len(labels), len(preds))
        texts  = texts[:n]
        labels = list(labels[:n])
        preds  = list(preds[:n])

        error_indices   = [i for i in range(n) if labels[i] != preds[i]]
        correct_indices = [i for i in range(n) if labels[i] == preds[i]]

        # ── debug: error analysis input ────────────────────────────────────
        dbg_error_input(
            context=f"domain={domain}",
            n_total=n,
            n_errors=len(error_indices),
            texts_sample=[texts[i] for i in error_indices[:3]],
        )

        # Pattern presence rates in errors vs correct
        def _rate(indices: List[int], pattern: str) -> float:
            if not indices:
                return 0.0
            return sum(
                1 for i in indices
                if detect_patterns(texts[i], domain).get(pattern, False)
            ) / len(indices)

        pattern_error_rate   = {p: _rate(error_indices,   p) for p in self.PATTERNS}
        pattern_correct_rate = {p: _rate(correct_indices, p) for p in self.PATTERNS}

        # Attribution = difference in presence rate (errors − correct)
        pattern_attribution = {
            p: pattern_error_rate[p] - pattern_correct_rate[p]
            for p in self.PATTERNS
        }

        # Collect up to 3 representative error examples per pattern
        examples: Dict[str, List[Dict[str, Any]]] = {p: [] for p in self.PATTERNS}
        for i in error_indices:
            pat = detect_patterns(texts[i], domain)
            error_type = classify_error_type(labels[i], preds[i])
            for p in self.PATTERNS:
                if pat[p] and len(examples[p]) < 3:
                    examples[p].append({
                        "text":       texts[i],
                        "true_label": labels[i],
                        "pred_label": preds[i],
                        "error_type": error_type,
                    })

        result = {
            "n_total":              n,
            "n_errors":             len(error_indices),
            "error_rate":           len(error_indices) / n if n > 0 else 0.0,
            "pattern_error_rate":   pattern_error_rate,
            "pattern_correct_rate": pattern_correct_rate,
            "pattern_attribution":  pattern_attribution,
            "examples":             examples,
        }

        # ── debug: pattern attribution output ─────────────────────────────
        dbg_error_attribution(
            context=f"domain={domain}",
            attribution=pattern_attribution,
        )

        return result

    # ── global summary ────────────────────────────────────────────────────────

    def _build_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate pattern attributions across all domains and conditions.
        Ranks patterns by mean attribution (highest = biggest driver of errors).
        """
        all_attributions: Dict[str, List[float]] = defaultdict(list)

        for domain_data in report["in_domain"].values():
            for p, v in domain_data["pattern_attribution"].items():
                all_attributions[p].append(v)

        for cond_data in report["cross_domain"].values():
            for p, v in cond_data["pattern_attribution"].items():
                all_attributions[p].append(v)

        for domain_levels in report["perturbation"].values():
            for level_data in domain_levels.values():
                for p, v in level_data["pattern_attribution"].items():
                    all_attributions[p].append(v)

        mean_attr = {
            p: float(np.mean(vals)) if vals else 0.0
            for p, vals in all_attributions.items()
        }
        ranked = sorted(mean_attr.items(), key=lambda x: x[1], reverse=True)

        summary = {
            "mean_pattern_attribution": mean_attr,
            "ranked_error_drivers":     [p for p, _ in ranked],
            "top_driver":               ranked[0][0] if ranked else None,
        }

        # ── debug: global summary ──────────────────────────────────────────
        dbg_error_summary(
            top_driver=summary["top_driver"],
            ranked=summary["ranked_error_drivers"],
        )

        return summary

    # ── text recovery (graceful fallback) ─────────────────────────────────────

    @staticmethod
    def _recover_texts(data: Dict[str, Any]) -> List[str]:
        """
        EvaluationEngine does not currently store the raw text in results.
        This method returns an empty list; callers should pass texts explicitly
        when available or integrate text storage into the evaluation engine.
        """
        return []

    # ── I/O helpers ───────────────────────────────────────────────────────────

    def save_report(
        self,
        report: Dict[str, Any],
        output_dir: str = "."
    ) -> str:
        """
        Serialise the error analysis report to JSON.

        Args:
            report:     Return value of analyze()
            output_dir: Directory to write error_analysis.json

        Returns:
            Full path of the saved file
        """
        from utils import make_json_serializable
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "error_analysis.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(report), f, ensure_ascii=False, indent=2)
        logger.info(f"Error analysis saved to {path}")
        return path

    def print_report(self, report: Dict[str, Any]) -> None:
        """
        Print a human-readable error analysis to stdout.

        Args:
            report: Return value of analyze()
        """
        SEP = "═" * 80
        THIN = "─" * 80
        model = report.get("model", "MODEL").upper()

        print(f"\n{SEP}")
        print(f"  ERROR ANALYSIS REPORT  ·  MODEL: {model}")
        print(SEP)

        # ── Summary ───────────────────────────────────────────────────────────
        summary = report.get("summary", {})
        print("\n  GLOBAL ERROR DRIVERS  (mean pattern attribution: errors − correct)")
        print(f"  {'Pattern':<28}  {'Mean Attribution':>18}  {'Rank'}")
        print(f"  {THIN}")
        ranked = summary.get("ranked_error_drivers", [])
        mean_a = summary.get("mean_pattern_attribution", {})
        for rank, p in enumerate(ranked, 1):
            label = PATTERN_LABELS.get(p, p)
            val   = mean_a.get(p, 0.0)
            bar   = "█" * max(0, round(abs(val) * 30))
            sign  = "+" if val >= 0 else ""
            print(f"  {label:<28}  {sign}{val:>+16.4f}  #{rank}  {bar}")

        print(f"\n  Top driver of errors: "
              f"{PATTERN_LABELS.get(summary.get('top_driver', ''), 'N/A')}")

        # ── In-domain per domain ───────────────────────────────────────────────
        print(f"\n{SEP}")
        print(f"  IN-DOMAIN ERROR ANALYSIS")
        print(SEP)
        for domain, data in report.get("in_domain", {}).items():
            self._print_domain_block(domain, data)

        # ── Cross-domain ───────────────────────────────────────────────────────
        print(f"\n{SEP}")
        print(f"  CROSS-DOMAIN ERROR ANALYSIS  (OOD pairs only)")
        print(SEP)
        for key, data in report.get("cross_domain", {}).items():
            src, tgt = (key.split("->") + ["?", "?"])[:2]
            if src == tgt:
                continue
            self._print_domain_block(f"{src} → {tgt}", data)

        # ── Perturbation ───────────────────────────────────────────────────────
        print(f"\n{SEP}")
        print(f"  PERTURBATION ERROR ANALYSIS  (per domain per level)")
        print(SEP)
        for domain, levels in report.get("perturbation", {}).items():
            for level, data in levels.items():
                self._print_domain_block(f"{domain} [{level.upper()}]", data)

        print(f"\n{SEP}\n")

    def _print_domain_block(self, label: str, data: Dict[str, Any]) -> None:
        """Print one block for a single domain / condition."""
        THIN = "─" * 78
        n       = data.get("n_total", 0)
        n_err   = data.get("n_errors", 0)
        err_rt  = data.get("error_rate", 0.0)
        attr    = data.get("pattern_attribution", {})
        ex      = data.get("examples", {})

        print(f"\n  [{label}]  errors={n_err}/{n}  error-rate={err_rt:.1%}")
        print(f"  {'Pattern':<28}  {'Attr (err-correct)':>20}  {'In errors':>10}  {'In correct':>10}")
        print(f"  {THIN}")

        p_err_rate = data.get("pattern_error_rate",   {})
        p_cor_rate = data.get("pattern_correct_rate", {})

        for p in self.PATTERNS:
            label_p  = PATTERN_LABELS.get(p, p)
            a_val    = attr.get(p, 0.0)
            e_val    = p_err_rate.get(p, 0.0)
            c_val    = p_cor_rate.get(p, 0.0)
            sign     = "▲" if a_val > 0.05 else ("▼" if a_val < -0.05 else " ")
            print(f"  {label_p:<28}  {a_val:>+20.4f}{sign}  {e_val:>10.1%}  {c_val:>10.1%}")

        # Print examples for the top attributed pattern
        if attr:
            top_p = max(attr, key=attr.get)
            top_examples = ex.get(top_p, [])
            if top_examples:
                print(f"\n    ↳ Top pattern: {PATTERN_LABELS[top_p]} — example errors:")
                for sample in top_examples[:2]:
                    err_t = sample.get("error_type", "?")
                    true_l = "clickbait" if sample["true_label"] == 1 else "non-clickbait"
                    pred_l = "clickbait" if sample["pred_label"] == 1 else "non-clickbait"
                    print(f'      [{err_t}] true={true_l}, pred={pred_l}')
                    print(f'      "{sample["text"]}"')


# ──────────────────────────────────────────────────────────────────────────────
# INTEGRATION HELPER
# ──────────────────────────────────────────────────────────────────────────────

def attach_texts_to_results(
    results: Dict[str, Any],
    test_data_by_domain: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Inject the original headline texts into the evaluation results dict so that
    ErrorAnalyzer can access them.

    Call this in evaluate_pipeline.py right after
    eval_engine.run_complete_evaluation() returns, before saving results:

        from error_analyzer import attach_texts_to_results
        results = attach_texts_to_results(results, data['test_data_by_domain'])

    Args:
        results:              Output of EvaluationEngine.run_complete_evaluation()
        test_data_by_domain:  {domain: pd.DataFrame} with a 'text' column

    Returns:
        results with 'texts' lists injected into every condition block
    """
    # In-domain
    for domain, data in results.get("in_domain", {}).items():
        df = test_data_by_domain.get(domain)
        if df is not None:
            data["texts"] = df["text"].tolist()

    # Cross-domain — keys may be tuples (source, target) in-memory or
    # "source->target" strings after JSON round-trip; handle both.
    for key, data in results.get("cross_domain", {}).items():
        if isinstance(key, tuple):
            tgt = key[1]
        else:
            _, tgt = (key.split("->") + ["?", "?"])[:2]
        df = test_data_by_domain.get(tgt)
        if df is not None:
            data["texts"] = df["text"].tolist()

    # Perturbation — texts are the same domain test split
    for domain, levels in results.get("perturbation", {}).items():
        df = test_data_by_domain.get(domain)
        for level, data in levels.items():
            if df is not None:
                data["texts"] = df["text"].tolist()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run error analysis on saved evaluation results."
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory containing complete_evaluation.json"
    )
    parser.add_argument(
        "--model", default="BASE",
        help="Model label for the report header (default: BASE)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save error_analysis.json alongside the evaluation results"
    )
    args = parser.parse_args()

    result_path = os.path.join(args.results_dir, "complete_evaluation.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Not found: {result_path}")

    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    analyzer = ErrorAnalyzer()
    report   = analyzer.analyze(results, model_name=args.model)
    analyzer.print_report(report)

    if args.save:
        analyzer.save_report(report, output_dir=args.results_dir)
