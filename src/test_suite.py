"""
End-to-End Test Suite for IndoBERT Clickbait Detection System
=============================================================
Tests every module in src/ WITHOUT requiring a GPU, HuggingFace downloads,
or a live model.  Heavy dependencies (torch, transformers) are mocked so the
suite can run in any CI environment.

Run:
    cd responsible-ai-toolkit/src
    python -m pytest test_suite.py -v

Or directly:
    python test_suite.py
"""

import ast
import json
import os
import random
import sys
import tempfile
import textwrap
import types
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# HEAVYWEIGHT MOCK SETUP
# Intercept torch / transformers BEFORE any project import so modules that do
# `import torch` at the top level get a lightweight stand-in.
# ──────────────────────────────────────────────────────────────────────────────

# torch.optim must exist because model_trainer.py does `from torch.optim import AdamW`
def _add_optim_mock(torch_mod):
    optim = types.ModuleType("torch.optim")
    optim.AdamW = MagicMock(return_value=MagicMock())
    torch_mod.optim = optim
    sys.modules["torch.optim"] = optim

def _make_torch_mock():
    torch = types.ModuleType("torch")
    torch.device = lambda x: x
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda s: None,
    )
    torch.manual_seed = lambda s: None
    torch.no_grad = MagicMock(return_value=MagicMock(
        __enter__=lambda s: s, __exit__=MagicMock(return_value=False)))
    torch.tensor = lambda x, **kw: np.array(x)
    torch.softmax = lambda t, dim: t                  # identity for tests
    torch.argmax = lambda t, dim: np.array([0] * len(t))
    torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(deterministic=True, benchmark=False))
    # Tensor class — required so scipy's array_api compat doesn't crash
    class _FakeTensor:
        pass
    torch.Tensor = _FakeTensor
    # nn sub-module
    nn = types.ModuleType("torch.nn")
    nn.Module = object
    nn.Dropout = lambda p: lambda x: x
    nn.Linear  = lambda a, b: None
    torch.nn = nn
    sys.modules["torch.nn"] = nn
    # utils.data sub-module
    data = types.ModuleType("torch.utils.data")
    data.Dataset   = object
    data.DataLoader = lambda ds, **kw: iter([])
    utils_mod = types.ModuleType("torch.utils")
    utils_mod.data = data
    torch.utils = utils_mod
    sys.modules["torch.utils"]      = utils_mod
    sys.modules["torch.utils.data"] = data
    _add_optim_mock(torch)
    return torch

def _make_transformers_mock():
    tr = types.ModuleType("transformers")
    dummy_tokenizer = MagicMock()
    dummy_tokenizer.return_value = {
        "input_ids": np.zeros((1, 128), dtype=int),
        "attention_mask": np.ones((1, 128), dtype=int),
    }
    dummy_tokenizer.__call__ = dummy_tokenizer.return_value
    tr.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=MagicMock(return_value=dummy_tokenizer)
    )
    tr.BertTokenizer = types.SimpleNamespace(
        from_pretrained=MagicMock(return_value=dummy_tokenizer)
    )
    dummy_model = MagicMock()
    dummy_model.config = types.SimpleNamespace(hidden_size=768)
    tr.AutoModel = types.SimpleNamespace(
        from_pretrained=MagicMock(return_value=dummy_model)
    )
    tr.get_linear_schedule_with_warmup = MagicMock(return_value=MagicMock())
    return tr

def _make_tqdm_mock():
    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda x, **kw: x
    return tqdm_mod

def _make_streamlit_mock():
    st = types.ModuleType("streamlit")
    for attr in ["title", "header", "subheader", "write", "sidebar",
                 "columns", "tabs", "cache_resource", "cache_data",
                 "text_input", "selectbox", "button", "spinner",
                 "success", "warning", "error", "info", "metric",
                 "plotly_chart", "set_page_config"]:
        setattr(st, attr, MagicMock())
    return st

def _make_sastrawi_mock():
    """Minimal stub so perturbation_engine.py can be imported without PySastrawi installed."""
    sastrawi_root     = types.ModuleType("Sastrawi")
    sastrawi_stemmer  = types.ModuleType("Sastrawi.Stemmer")
    sastrawi_factory  = types.ModuleType("Sastrawi.Stemmer.StemmerFactory")

    class _FakeStemmer:
        def stem(self, word: str) -> str:
            # Naive fallback: strip common Indonesian suffixes so the stub
            # behaves deterministically without the real stemmer.
            for suffix in ("kan", "an", "i", "nya", "ku", "mu"):
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    return word[: -len(suffix)]
            return word

    class _FakeStemmerFactory:
        def create_stemmer(self):
            return _FakeStemmer()

    sastrawi_factory.StemmerFactory = _FakeStemmerFactory
    sastrawi_root.Stemmer = sastrawi_stemmer
    sastrawi_stemmer.StemmerFactory = sastrawi_factory
    return sastrawi_root, sastrawi_stemmer, sastrawi_factory


for _name, _factory in [
    ("torch", _make_torch_mock),
    ("torch.nn", lambda: _make_torch_mock().nn),
    ("torch.utils", lambda: _make_torch_mock().utils),
    ("torch.utils.data", lambda: _make_torch_mock().utils.data),
    ("transformers", _make_transformers_mock),
    ("tqdm", _make_tqdm_mock),
    ("tqdm.auto", _make_tqdm_mock),
    ("streamlit", _make_streamlit_mock),
    ("plotly", types.ModuleType("plotly")),
    ("plotly.express", types.ModuleType("plotly.express")),
    ("plotly.graph_objects", types.ModuleType("plotly.graph_objects")),
    ("plotly.subplots", types.ModuleType("plotly.subplots")),
    ("seaborn", types.ModuleType("seaborn")),
]:
    if _name not in sys.modules:
        sys.modules[_name] = _factory() if callable(_factory) else _factory

# Sastrawi must be mocked before perturbation_engine is imported
if "Sastrawi" not in sys.modules:
    _s_root, _s_stemmer, _s_factory = _make_sastrawi_mock()
    sys.modules["Sastrawi"]                        = _s_root
    sys.modules["Sastrawi.Stemmer"]                = _s_stemmer
    sys.modules["Sastrawi.Stemmer.StemmerFactory"] = _s_factory

# ──────────────────────────────────────────────────────────────────────────────
# Now safe to import project modules
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    Config, ModelConfig, TrainingConfig, DataConfig,
    PerturbationConfig, EvaluationConfig, StatisticalConfig,
    PathConfig, get_config,
)
from utils import (
    FileManager, Timer, safe_divide, flatten_dict, batch_iterator,
    get_timestamp, reproducibility,
)
from data_manager import DataManager
from perturbation_engine import (
    PerturbationEngine,
    SemanticWordSubstitution,
    IndonesianThesaurus,
    PERTURBATION_INTENSITIES,
    SIM_MIN,
    SIM_MAX,
    tokenize_words,
)
from evaluation_engine import (
    calculate_metrics,
    calculate_robustness_metrics,
    calculate_domain_shift_metrics,
    calculate_prediction_flip,
    evaluate_in_domain,
    evaluate_all_cross_domain_combinations,
    EvaluationEngine,
    PERTURBATION_TYPES,
    PERTURBATION_LEVELS,
)
from statistical_analyzer import (
    assemble_observations,
    validate_observations,
    apply_boundary_correction,
    StatisticalAnalyzer,
)
from error_analyzer import (
    ErrorAnalyzer, detect_patterns, attach_texts_to_results,
    SENSATIONAL_WORDS, DOMAIN_JARGON,
)


# ──────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

DOMAINS = ["Technology", "Politics", "Health", "Sport", "Education"]

def _make_domain_df(domain: str, n: int = 40) -> pd.DataFrame:
    """
    Minimal domain DataFrame.
    Uses ONE array for both 'Label' and 'label' so value_counts() gets a
    1-D Series (not 2-D), which was causing GrouperError in Pandas.
    """
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=n).tolist()
    return pd.DataFrame({
        "id":     range(n),
        "source": ["TestSource"] * n,
        "date":   ["2024-01-01"] * n,
        "title":  [f"Sample headline {i} about {domain}" for i in range(n)],
        "Label":  labels,
        "url":    [f"https://example.com/{i}" for i in range(n)],
        "text":   [f"Sample headline {i} about {domain}" for i in range(n)],
        "label":  labels,
        "domain": [domain] * n,
    })

def _make_fake_trainer(domain: str = "Technology") -> MagicMock:
    """Mock ModelTrainer whose predict() adapts to the actual input length."""
    trainer = MagicMock()

    def _predict(texts, batch_size=32):
        n = len(texts)
        preds = np.array([i % 2 for i in range(n)])
        proba = np.column_stack([
            np.clip(1 - preds * 0.4, 0, 1),
            np.clip(preds * 0.4 + 0.3, 0, 1),
        ])
        return preds, proba

    trainer.predict.side_effect = _predict
    return trainer

def _make_fake_results(domains=None, include_perturbation=True) -> dict:
    """Build a minimal complete_evaluation results dict."""
    if domains is None:
        domains = DOMAINS
    n = 30
    rng = np.random.default_rng(0)

    labels = rng.integers(0, 2, n).tolist()
    preds  = rng.integers(0, 2, n).tolist()
    proba  = rng.random((n, 2)).tolist()
    texts  = [f"Apakah {d} dapat mengubah dunia?" for d in domains for _ in range(n // len(domains))][:n]

    def _domain_result(domain):
        return {
            "domain": domain,
            "num_samples": n,
            "metrics": {
                "accuracy": 0.75,
                "macro_precision": 0.74, "macro_recall": 0.76, "macro_f1": 0.75,
                "mcc": 0.50, "roc_auc": 0.80,
                "confusion_matrix": [[10, 5], [3, 12]],
            },
            "predictions": preds,
            "probabilities": proba,
            "true_labels": labels,
            "texts": texts[:n],
        }

    in_domain = {d: _domain_result(d) for d in domains}

    cross_domain = {}
    for src in domains:
        for tgt in domains:
            key = f"{src}->{tgt}"
            entry = _domain_result(tgt)
            if src != tgt:
                entry["domain_shift"] = {
                    "sd_f1": 0.05, "td_f1": 0.08,
                    "sd_accuracy": 0.04, "td_accuracy": 0.06,
                    "sd_precision": 0.03, "td_precision": 0.05,
                    "sd_recall": 0.04, "td_recall": 0.07,
                }
            cross_domain[key] = entry

    # Nested perturbation structure:
    #   perturbation[domain] = {
    #       'clean': {...},
    #       'semantic': {'low': {...}, 'medium': {...}, 'high': {...}},
    #   }
    # The shared 'clean' baseline lives once at the top of each domain.
    _levels = ["low", "medium", "high"]

    def _level_block(f1):
        return {
            "metrics": {
                "accuracy": f1 - 0.01,
                "macro_f1": f1,
                "macro_precision": 0.74, "macro_recall": 0.76,
                # dashboard reads 'f1'/'precision'/'recall'; include them too
                "f1": f1, "precision": 0.74, "recall": 0.76,
            },
            "predictions": preds,
            "probabilities": proba,
            "true_labels": labels,
            "texts": texts[:n],
        }

    perturbation = {}
    if include_perturbation:
        for d in domains:
            domain_block = {"clean": _level_block(0.75)}
            for ptype in ["semantic"]:
                domain_block[ptype] = {}
                for i, lvl in enumerate(_levels):
                    # F1 decreases as intensity rises.
                    step = 0.05
                    domain_block[ptype][lvl] = _level_block(0.75 - step * (i + 1))
            perturbation[d] = domain_block

    return {
        "timestamp": "2024-01-01T00:00:00",
        "domains": domains,
        "in_domain": in_domain,
        "cross_domain": cross_domain,
        "perturbation": perturbation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):

    def test_config_instantiates(self):
        cfg = Config()
        self.assertIsNotNone(cfg.model)
        self.assertIsNotNone(cfg.training)
        self.assertIsNotNone(cfg.data)

    def test_split_proportions_sum_to_one(self):
        cfg = Config()
        total = cfg.data.TRAIN_SIZE + cfg.data.VAL_SIZE + cfg.data.TEST_SIZE
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_five_domains_defined(self):
        cfg = Config()
        self.assertEqual(len(cfg.data.DOMAINS), 5)
        self.assertIn("Technology", cfg.data.DOMAINS)
        self.assertIn("Education", cfg.data.DOMAINS)

    def test_three_model_variants(self):
        cfg = Config()
        self.assertEqual(len(cfg.model.AVAILABLE_MODELS), 3)

    def test_perturbation_intensities_ordered(self):
        cfg = Config()
        self.assertLess(cfg.perturbation.LOW_INTENSITY_MAX, cfg.perturbation.MEDIUM_INTENSITY_MIN)
        self.assertLess(cfg.perturbation.MEDIUM_INTENSITY_MAX, cfg.perturbation.HIGH_INTENSITY_MIN)

    def test_validate_passes(self):
        self.assertTrue(Config().validate())

    def test_to_dict_roundtrip(self):
        cfg = Config()
        d = cfg.to_dict()
        self.assertIn("model", d)
        self.assertIn("training", d)

    def test_save_load_roundtrip(self):
        cfg = Config()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save_to_file(path)
            loaded = Config.load_from_file(path)
            self.assertEqual(cfg.data.DOMAINS, loaded.data.DOMAINS)
        finally:
            os.unlink(path)

    def test_get_config_singleton(self):
        self.assertIsInstance(get_config(), Config)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  UTILS
# ══════════════════════════════════════════════════════════════════════════════

class TestUtils(unittest.TestCase):

    def test_file_manager_ensure_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = os.path.join(tmp, "a", "b", "c")
            FileManager.ensure_directory(new_dir)
            self.assertTrue(os.path.isdir(new_dir))

    def test_file_manager_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.json")
            data = {"key": "value", "num": 42}
            FileManager.save_json(data, path)
            loaded = FileManager.load_json(path)
            self.assertEqual(loaded, data)

    def test_file_manager_load_json_missing(self):
        with self.assertRaises(FileNotFoundError):
            FileManager.load_json("/nonexistent/path.json")

    def test_file_manager_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.csv")
            df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
            FileManager.save_csv(df, path)
            loaded = FileManager.load_csv(path)
            pd.testing.assert_frame_equal(df, loaded)

    def test_file_manager_text_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.txt")
            FileManager.save_text("hello world", path)
            self.assertEqual(FileManager.load_text(path), "hello world")

    def test_file_manager_file_exists(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            self.assertTrue(FileManager.file_exists(path))
        finally:
            os.unlink(path)
        self.assertFalse(FileManager.file_exists(path))

    def test_timer_context_manager(self):
        with Timer() as t:
            _ = sum(range(1000))
        self.assertGreaterEqual(t.elapsed(), 0.0)

    def test_safe_divide(self):
        self.assertAlmostEqual(safe_divide(10, 4), 2.5)
        self.assertEqual(safe_divide(1, 0), 0.0)
        self.assertEqual(safe_divide(1, 0, default=-1.0), -1.0)

    def test_flatten_dict(self):
        # flatten_dict uses '_' as default separator, not '.'
        nested = {"a": {"b": 1, "c": {"d": 2}}}
        flat = flatten_dict(nested)
        self.assertEqual(flat["a_b"], 1)
        self.assertEqual(flat["a_c_d"], 2)

    def test_batch_iterator(self):
        items = list(range(10))
        batches = list(batch_iterator(items, batch_size=3))
        self.assertEqual(len(batches), 4)          # ceil(10/3)
        self.assertEqual(batches[0], [0, 1, 2])
        self.assertEqual(batches[-1], [9])

    def test_get_timestamp_format(self):
        ts = get_timestamp()
        self.assertEqual(len(ts), 15)              # YYYYMMDD_HHMMSS

    def test_reproducibility_set_seed(self):
        reproducibility.set_seed(42)               # should not raise


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATA MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class TestDataManager(unittest.TestCase):

    def _write_domain_csvs(self, tmpdir: str) -> list:
        """Write 5 minimal domain CSVs to tmpdir and return paths.

        The CSV files must only have the raw schema columns (id, source, date,
        title, Label, url) — NOT the derived 'text'/'label'/'domain' columns
        that data_manager creates internally.  Having both 'Label' and 'label'
        in the CSV makes pandas value_counts() receive a 2-D DataFrame column.
        """
        paths = []
        for domain in DOMAINS:
            rng = np.random.default_rng(42)
            n = 30
            df = pd.DataFrame({
                "id":     range(n),
                "source": ["TestSource"] * n,
                "date":   ["2024-01-01"] * n,
                "title":  [f"Sample headline {i} about {domain}" for i in range(n)],
                "Label":  rng.integers(0, 2, size=n).tolist(),
                "url":    [f"https://example.com/{i}" for i in range(n)],
            })
            path = os.path.join(tmpdir, f"{domain.lower()}.csv")
            df.to_csv(path, index=False, sep=";")  # data_manager reads sep=';'
            paths.append(path)
        return paths

    def _load(self, tmp):
        """DataManager never touches a tokenizer at load time, so no mock is needed."""
        paths = self._write_domain_csvs(tmp)
        dm = DataManager(tokenizer_name="mock", max_length=32, random_seed=42)
        dm.load_datasets(paths, domain_names=DOMAINS)
        return dm

    def test_load_datasets_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm = self._load(tmp)
            self.assertEqual(len(dm.raw_data), 150)  # 5 × 30
            self.assertEqual(len(dm.domains), 5)
            for col in ("text", "label", "domain"):
                self.assertIn(col, dm.raw_data.columns)

    def test_from_dataset_directory_raises_missing(self):
        with self.assertRaises(FileNotFoundError):
            DataManager.from_dataset_directory("/nonexistent/dir")

    def test_from_dataset_directory_raises_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                DataManager.from_dataset_directory(tmp)   # dir exists, CSVs do not

    def test_load_datasets_rejects_non_binary_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "bad.csv")
            pd.DataFrame({"id": [1], "source": ["s"], "date": ["d"], "title": ["t"],
                          "Label": [2], "url": ["u"]}).to_csv(p, sep=";", index=False)
            with self.assertRaises(ValueError):
                DataManager().load_datasets([p], domain_names=["X"])

    def test_load_datasets_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "bad.csv")
            pd.DataFrame({"id": [1], "title": ["t"]}).to_csv(p, sep=";", index=False)
            with self.assertRaises(ValueError):
                DataManager().load_datasets([p], domain_names=["X"])

    def test_stratified_split_proportions(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm = self._load(tmp)
            splits = dm.stratified_split_by_domain(
                train_size=0.70, val_size=0.15, test_size=0.15
            )
            for domain in DOMAINS:
                n_total = (len(splits[domain]["train"])
                           + len(splits[domain]["val"])
                           + len(splits[domain]["test"]))
                self.assertEqual(n_total, 30)

    def test_stratified_split_has_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm = self._load(tmp)
            splits = dm.stratified_split_by_domain()
            for domain in DOMAINS:
                ids = [set(splits[domain][s]["id"]) for s in ("train", "val", "test")]
                self.assertEqual(len(ids[0] & ids[1]), 0)
                self.assertEqual(len(ids[0] & ids[2]), 0)
                self.assertEqual(len(ids[1] & ids[2]), 0)

    def test_stratified_split_rejects_bad_proportions(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm = self._load(tmp)
            with self.assertRaises(ValueError):
                dm.stratified_split_by_domain(train_size=0.5, val_size=0.5, test_size=0.5)

    def test_export_and_load_domain_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm = self._load(tmp)
            splits = dm.stratified_split_by_domain()
            export_dir = os.path.join(tmp, "splits")
            dm.export_domain_splits(splits, export_dir)

            # Verify files written
            for domain in DOMAINS:
                for split in ["train", "val", "test"]:
                    p = os.path.join(export_dir, domain.lower(), f"{split}.csv")
                    self.assertTrue(os.path.exists(p), f"Missing {p}")

            # Reload
            loaded = dm.load_domain_splits(export_dir)
            self.assertEqual(set(loaded.keys()), {d.capitalize() for d in DOMAINS} |
                             set(DOMAINS))  # flexible capitalisation

    def test_get_summary_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            dm = self._load(tmp)
            s = dm.get_summary()
            for key in ("total_samples", "num_domains", "domains"):
                self.assertIn(key, s)

    def test_get_summary_before_load(self):
        self.assertEqual(DataManager().get_summary(), {"status": "No data loaded"})


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PERTURBATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_HEADLINES = [
    "Apakah Jokowi Tiba-Tiba Mengundurkan Diri dari Jabatannya?",
    "Teknologi AI Terbaru Mengubah 10 Bidang Kehidupan Manusia",
    "Vaksin Covid-19 Terbukti 95% Efektif Menurut Penelitian Baru",
    "Timnas Indonesia Kalah Telak 3-0 dari Malaysia di Semifinal",
    "Beasiswa S2 Luar Negeri Gratis Dibuka, Ini Syaratnya!",
]

# ── Shared thesaurus path for perturbation tests (file lives in the repo) ────
_THESAURUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "data", "dict.json"
)
_THESAURUS_PATH = os.path.normpath(_THESAURUS_PATH)


class TestPerturbationConstants(unittest.TestCase):
    """The engine-level constants must match the experiment design."""

    def test_intensities_keys(self):
        self.assertEqual(set(PERTURBATION_INTENSITIES.keys()), {"low", "medium", "high"})

    def test_intensities_ordered(self):
        """Low < Medium < High."""
        self.assertLess(PERTURBATION_INTENSITIES["low"], PERTURBATION_INTENSITIES["medium"])
        self.assertLess(PERTURBATION_INTENSITIES["medium"], PERTURBATION_INTENSITIES["high"])

    def test_intensities_values(self):
        self.assertAlmostEqual(PERTURBATION_INTENSITIES["low"],    0.10)
        self.assertAlmostEqual(PERTURBATION_INTENSITIES["medium"], 0.20)
        self.assertAlmostEqual(PERTURBATION_INTENSITIES["high"],   0.30)

    def test_nested_structure_constants(self):
        """The evaluation engine drives its perturbation loops from these."""
        self.assertEqual(PERTURBATION_TYPES, ["semantic"])
        self.assertEqual(PERTURBATION_LEVELS, ["low", "medium", "high"])

    def test_sim_bounds(self):
        self.assertGreater(SIM_MIN, 0.0)
        self.assertLess(SIM_MAX, 1.0)
        self.assertLess(SIM_MIN, SIM_MAX)

    def test_tokenize_words_returns_list(self):
        tokens = tokenize_words("IndoBERT mengubah dunia NLP")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        for tok in tokens:
            self.assertIsInstance(tok, str)

    def test_tokenize_words_strips_punctuation(self):
        tokens = tokenize_words("Halo, dunia! Ini tes.")
        for tok in tokens:
            self.assertNotIn(",", tok)
            self.assertNotIn("!", tok)
            self.assertNotIn(".", tok)


class TestPerturbationEngine(unittest.TestCase):
    """Tests for PerturbationEngine using a mocked SemanticWordSubstitution
    so no live IndoBERT model or thesaurus file is needed."""

    def _make_engine(self):
        """Return a PerturbationEngine whose inner SemanticWordSubstitution is
        replaced by a deterministic stub that substitutes every word with its
        reverse (preserving word count and token positions)."""
        engine = PerturbationEngine.__new__(PerturbationEngine)
        engine.random_seed = 42
        engine.sim_min = SIM_MIN
        engine.sim_max = SIM_MAX

        stub = MagicMock()

        def _stub_perturb_with_metadata(text, intensity):
            words = tokenize_words(text)
            n = len(words)
            n_change = max(1, int(n * intensity)) if n else 0
            result_text = text
            replacements = []
            for i, w in enumerate(words[:n_change]):
                replacement = w[::-1] if w[::-1] != w else w + "x"
                result_text = result_text.replace(w, replacement, 1)
                replacements.append({
                    "position": i, "original": w, "replacement": replacement,
                    "word_cosine_similarity": 0.88,
                    "lookup_source": "direct", "parent_word": None,
                    "original_lookup_source": "direct", "original_parent_word": None,
                    "candidate_pos": "NN", "pos_verified": True, "pos_source": "stub",
                })
            words_changed = len(replacements)
            return {
                "original_text": text,
                "perturbed_text": result_text,
                "perturbation_level": None,
                "target_intensity": intensity,
                "total_words": n,
                "failed_lookup_words": max(0, n - words_changed),
                "eligible_words": n,
                "target_words": n_change,
                "words_changed": words_changed,
                "actual_ratio_all_words": words_changed / n if n else 0.0,
                "is_same_as_original": result_text == text,
                "similarity_in_range": True,
                "perturbation_in_range": words_changed == n_change,
                "insufficient_eligible_words": False,
                "similarity_score_mean": 0.88,
                "similarity_score_min": 0.88,
                "similarity_score_max": 0.88,
                "replacements": replacements,
            }

        stub.perturb_with_metadata.side_effect = _stub_perturb_with_metadata
        engine.perturbation = stub
        return engine

    def setUp(self):
        self.engine = self._make_engine()

    # ── apply_perturbation returns string ─────────────────────────────────────

    def test_low_level_returns_string(self):
        for text in SAMPLE_HEADLINES:
            result = self.engine.apply_perturbation(text, "low")
            self.assertIsInstance(result, str)

    def test_medium_level_returns_string(self):
        for text in SAMPLE_HEADLINES:
            result = self.engine.apply_perturbation(text, "medium")
            self.assertIsInstance(result, str)

    def test_high_level_returns_string(self):
        for text in SAMPLE_HEADLINES:
            result = self.engine.apply_perturbation(text, "high")
            self.assertIsInstance(result, str)

    def test_low_level_nonempty(self):
        for text in SAMPLE_HEADLINES:
            result = self.engine.apply_perturbation(text, "low")
            self.assertGreater(len(result), 0)

    # ── Same underlying method, different intensities ─────────────────────────

    def test_all_levels_use_same_method(self):
        """All three levels must delegate to the same perturbation object."""
        for level in ("low", "medium", "high"):
            self.engine.apply_perturbation(SAMPLE_HEADLINES[0], level)
        # stub.perturb_with_metadata was called exactly once per level call
        self.assertEqual(
            self.engine.perturbation.perturb_with_metadata.call_count,
            3,
        )

    def test_intensity_increases_with_level(self):
        """Higher levels must request higher intensity from the underlying method."""
        calls = {}
        for level in ("low", "medium", "high"):
            self.engine.perturbation.perturb_with_metadata.reset_mock()
            self.engine.apply_perturbation(SAMPLE_HEADLINES[0], level)
            _, kwargs = self.engine.perturbation.perturb_with_metadata.call_args
            calls[level] = kwargs.get("intensity") or self.engine.perturbation.perturb_with_metadata.call_args[0][1]
        self.assertLess(calls["low"], calls["medium"])
        self.assertLess(calls["medium"], calls["high"])

    # ── Invalid level raises ──────────────────────────────────────────────────

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError):
            self.engine.apply_perturbation("text", "extreme")

    # ── Removed typo levels are rejected ──────────────────────────────────────

    def test_typo_level_is_rejected(self):
        for level in ("typo_low", "typo_medium", "typo_high"):
            with self.assertRaises(ValueError):
                self.engine.apply_perturbation(SAMPLE_HEADLINES[0], level)

    # ── apply_perturbation_with_metadata ─────────────────────────────────────

    def test_metadata_has_required_keys(self):
        result = self.engine.apply_perturbation_with_metadata(
            SAMPLE_HEADLINES[0], "low"
        )
        for key in ("original_text", "perturbed_text", "replacements",
                    "words_changed", "perturbation_level", "target_intensity",
                    "perturbation_in_range"):
            self.assertIn(key, result)

    def test_metadata_level_field_matches_requested(self):
        for level in ("low", "medium", "high"):
            result = self.engine.apply_perturbation_with_metadata(
                SAMPLE_HEADLINES[0], level
            )
            self.assertEqual(result["perturbation_level"], level)

    # ── DataFrame API ─────────────────────────────────────────────────────────

    def test_apply_to_dataframe(self):
        df = pd.DataFrame({"text": SAMPLE_HEADLINES, "label": [0, 1, 0, 1, 0]})
        result = self.engine.apply_to_dataframe(df.copy(), text_column="text", level="low")
        self.assertEqual(len(result), len(df))
        self.assertIn("text", result.columns)

    def test_apply_to_dataframe_invalid_level_raises(self):
        df = pd.DataFrame({"text": SAMPLE_HEADLINES[:2], "label": [0, 1]})
        with self.assertRaises(ValueError):
            self.engine.apply_to_dataframe(df, text_column="text", level="extreme")

    # ── get_perturbation_stats ────────────────────────────────────────────────

    def test_perturbation_stats_keys(self):
        original  = "Teknologi AI mengubah dunia"
        perturbed = self.engine.apply_perturbation(original, "low")
        stats = self.engine.get_perturbation_stats(original, perturbed)
        for key in ("word_change_ratio", "words_changed",
                    "original_word_count", "perturbed_word_count"):
            self.assertIn(key, stats)

    def test_perturbation_stats_ratio_range(self):
        original  = "Teknologi AI mengubah dunia dengan cepat"
        perturbed = self.engine.apply_perturbation(original, "medium")
        stats = self.engine.get_perturbation_stats(original, perturbed)
        self.assertGreaterEqual(stats["word_change_ratio"], 0.0)
        self.assertLessEqual(stats["word_change_ratio"], 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricFunctions(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        self.y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        self.y_proba = np.column_stack([
            [0.8, 0.2, 0.6, 0.9, 0.1, 0.3, 0.7, 0.15],
            [0.2, 0.8, 0.4, 0.1, 0.9, 0.7, 0.3, 0.85],
        ])

    def test_calculate_metrics_keys(self):
        m = calculate_metrics(self.y_true, self.y_pred, self.y_proba)
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "mcc", "roc_auc", "confusion_matrix",
                    "f1_0", "f1_1", "precision_0", "recall_1"):
            self.assertIn(key, m)

    def test_accuracy_and_f1_range(self):
        m = calculate_metrics(self.y_true, self.y_pred)
        self.assertGreaterEqual(m["accuracy"], 0.0)
        self.assertLessEqual(m["accuracy"], 1.0)
        self.assertGreaterEqual(m["macro_f1"], 0.0)
        self.assertLessEqual(m["macro_f1"], 1.0)

    def test_roc_auc_none_without_proba(self):
        self.assertIsNone(calculate_metrics(self.y_true, self.y_pred)["roc_auc"])

    def test_roc_auc_present_with_proba(self):
        auc = calculate_metrics(self.y_true, self.y_pred, self.y_proba)["roc_auc"]
        self.assertIsNotNone(auc)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_confusion_matrix_shape(self):
        m = calculate_metrics(self.y_true, self.y_pred)
        self.assertEqual(len(m["confusion_matrix"]), 2)
        self.assertEqual(len(m["confusion_matrix"][0]), 2)

    def test_confusion_matrix_2x2_on_single_class_subset(self):
        """Regression: a matched subset holding one class used to give a 1x1 matrix."""
        import warnings
        y = np.array([1, 1, 1, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # sklearn MCC single-label notice
            m = calculate_metrics(y, y)
        self.assertEqual(np.array(m["confusion_matrix"]).shape, (2, 2))
        self.assertEqual(m["confusion_matrix"], [[0, 0], [0, 4]])

    def test_perfect_prediction(self):
        m = calculate_metrics(self.y_true, self.y_true)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["macro_f1"], 1.0)

    def test_robustness_metrics_drop_direction(self):
        clean     = {"accuracy": 0.90, "macro_precision": 0.88, "macro_recall": 0.91, "macro_f1": 0.89}
        perturbed = {"accuracy": 0.75, "macro_precision": 0.73, "macro_recall": 0.76, "macro_f1": 0.74}
        rob = calculate_robustness_metrics(clean, perturbed)
        self.assertGreater(rob["macro_f1_drop"], 0)
        self.assertGreater(rob["accuracy_drop"], 0)
        self.assertIn("macro_f1_drop_pct", rob)

    def test_robustness_metrics_no_drop(self):
        m = {"accuracy": 0.80, "macro_precision": 0.80, "macro_recall": 0.80, "macro_f1": 0.80}
        rob = calculate_robustness_metrics(m, m)
        self.assertAlmostEqual(rob["macro_f1_drop"], 0.0)

    def test_domain_shift_sd_td(self):
        src    = {"accuracy": 0.88, "macro_f1": 0.87, "macro_precision": 0.86, "macro_recall": 0.88}
        cross  = {"accuracy": 0.70, "macro_f1": 0.69, "macro_precision": 0.68, "macro_recall": 0.70}
        tgt_in = {"accuracy": 0.85, "macro_f1": 0.84, "macro_precision": 0.83, "macro_recall": 0.85}
        shift = calculate_domain_shift_metrics(src, cross, tgt_in)
        self.assertGreater(shift["sd_macro_f1"], 0)
        self.assertGreater(shift["td_macro_f1"], 0)
        self.assertAlmostEqual(shift["sd_macro_f1"], 0.87 - 0.69)
        self.assertAlmostEqual(shift["td_macro_f1"], 0.84 - 0.69)


class TestEvaluateFunctions(unittest.TestCase):

    def test_evaluate_in_domain_keys(self):
        result = evaluate_in_domain(_make_fake_trainer(), _make_domain_df("Technology", n=40), "Technology")
        for key in ("domain", "num_samples", "metrics", "predictions", "probabilities", "true_labels"):
            self.assertIn(key, result)

    def test_evaluate_in_domain_sample_count(self):
        result = evaluate_in_domain(_make_fake_trainer(), _make_domain_df("Health", n=40))
        self.assertEqual(result["num_samples"], 40)
        self.assertEqual(len(result["true_labels"]), 40)
        self.assertEqual(len(result["predictions"]), 40)

    def test_cross_domain_all_combinations_count(self):
        trainers = {d: _make_fake_trainer(d) for d in DOMAINS}
        test_data = {d: _make_domain_df(d, n=20) for d in DOMAINS}
        results = evaluate_all_cross_domain_combinations(trainers, test_data)
        self.assertEqual(len(results), 25)

    def test_cross_domain_keys_are_tuples(self):
        trainers = {d: _make_fake_trainer(d) for d in DOMAINS}
        test_data = {d: _make_domain_df(d, n=20) for d in DOMAINS}
        results = evaluate_all_cross_domain_combinations(trainers, test_data)
        self.assertIn(("Technology", "Politics"), results)
        entry = results[("Technology", "Politics")]
        for key in ("source_domain", "target_domain", "num_samples", "metrics"):
            self.assertIn(key, entry)

    def test_cross_domain_skips_source_without_trainer(self):
        test_data = {d: _make_domain_df(d, n=10) for d in ("Technology", "Politics")}
        results = evaluate_all_cross_domain_combinations(
            {"Technology": _make_fake_trainer()}, test_data
        )
        self.assertEqual(set(results), {("Technology", "Technology"), ("Technology", "Politics")})


class _FakePerturbationEngine:
    """Minimal stand-in for PerturbationEngine: every row is perturbed successfully."""

    def apply_to_dataframe(self, df, text_column="text", level="low", **_):
        out = df.copy()
        out[text_column] = out[text_column].astype(str) + " x"
        out["is_same_as_original"] = False
        out["similarity_in_range"] = True
        out["perturbation_in_range"] = True
        out["insufficient_eligible_words"] = False
        out["actual_ratio_all_words"] = 0.2
        out["similarity_score_mean"] = 0.88
        out["similarity_score_min"] = 0.85
        out["similarity_score_max"] = 0.9
        out["replacements"] = [[{"word_cosine_similarity": 0.88}]] * len(out)
        return out


class TestEvaluationEngine(unittest.TestCase):

    def _build_engine(self, tmpdir):
        trainers = {d: _make_fake_trainer(d) for d in DOMAINS}
        return EvaluationEngine(
            model_trainers=trainers,
            perturbation_engine=_FakePerturbationEngine(),
            output_dir=tmpdir,
            perturbation_output_dir=os.path.join(tmpdir, "perturbed"),
        )

    def test_run_complete_evaluation_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._build_engine(tmp)
            test_data = {d: _make_domain_df(d, n=20) for d in DOMAINS}
            results = ev.run_complete_evaluation(test_data, include_perturbations=False)
            for key in ("in_domain", "cross_domain", "domains", "perturbation"):
                self.assertIn(key, results)
            self.assertEqual(set(results["in_domain"]), set(DOMAINS))
            self.assertEqual(len(results["cross_domain"]), 25)
            self.assertEqual(results["perturbation"], {})

    def test_run_complete_evaluation_with_perturbations(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._build_engine(tmp)
            test_data = {d: _make_domain_df(d, n=16) for d in DOMAINS}
            results = ev.run_complete_evaluation(test_data, include_perturbations=True)
            self.assertEqual(set(results["perturbation"]), set(DOMAINS))
            block = results["perturbation"]["Sport"]
            self.assertIn("clean", block)
            for level in PERTURBATION_LEVELS:
                self.assertIn(level, block["semantic"])
            self.assertEqual(len(results["cross_domain_perturbation"]), 25)
            self.assertIn("Sport_to_Health", results["cross_domain_perturbation"])

    def test_perturbation_csvs_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._build_engine(tmp)
            test_data = {d: _make_domain_df(d, n=10) for d in DOMAINS}
            ev.generate_all_perturbations(test_data)
            for d in DOMAINS:
                for level in PERTURBATION_LEVELS:
                    path = os.path.join(tmp, "perturbed", d.lower(), "semantic", f"{level}.csv")
                    self.assertTrue(os.path.exists(path), path)

    def test_domain_shift_metrics_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._build_engine(tmp)
            test_data = {d: _make_domain_df(d, n=15) for d in DOMAINS}
            results = ev.run_complete_evaluation(test_data, include_perturbations=False)
            ood_with_shift = [
                v for k, v in results["cross_domain"].items()
                if k[0] != k[1] and "domain_shift" in v
            ]
            self.assertEqual(len(ood_with_shift), 20)

    def test_save_results_creates_json_with_arrow_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._build_engine(tmp)
            test_data = {d: _make_domain_df(d, n=10) for d in DOMAINS}
            results = ev.run_complete_evaluation(test_data, include_perturbations=False)
            path = ev.save_results(results, "test_out.json")
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                loaded = json.load(f)
            self.assertIn("in_domain", loaded)
            for key in loaded["cross_domain"]:
                self.assertIn("->", key, f"Key {key!r} is not in 'A->B' format")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  STATISTICAL ANALYZER  (pure-numpy helpers; the PyMC fit is not exercised)
# ══════════════════════════════════════════════════════════════════════════════

def _make_stat_results(domains=("Technology", "Politics"), f1_offset=0.0) -> dict:
    """Results dict in the shape assemble_observations() reads."""
    def block(f1):
        return {"metrics": {"macro_f1": f1}, "common_ids_count": 50}

    results = {"perturbation": {}, "cross_domain_perturbation": {}}
    for d in domains:
        results["perturbation"][d] = {
            "clean": block(0.80 + f1_offset),
            "semantic": {"low": block(0.75), "medium": block(0.70), "high": block(0.65)},
        }
    for s in domains:
        for t in domains:
            sem = {"low": block(0.75), "medium": block(0.70), "high": block(0.65)}
            results["cross_domain_perturbation"][f"{s}_to_{t}"] = {
                "clean": block(0.80 + f1_offset), "semantic": sem,
            }
    return results


class TestAssembleObservations(unittest.TestCase):

    def test_row_count_and_columns(self):
        df = assemble_observations({"base": _make_stat_results()})
        # 2x2 source/target cells x 4 conditions (clean/low/medium/high)
        self.assertEqual(len(df), 16)
        for col in ("model", "source_domain", "target_domain",
                    "perturbation_condition", "in_domain", "macro_f1", "n_samples"):
            self.assertIn(col, df.columns)

    def test_diagonal_duplicates_collapsed(self):
        df = assemble_observations({"base": _make_stat_results()})
        keys = ["model", "source_domain", "target_domain", "perturbation_condition"]
        self.assertFalse(df.duplicated(subset=keys).any())

    def test_in_domain_flag_matches_diagonal(self):
        df = assemble_observations({"base": _make_stat_results()})
        self.assertTrue(((df["source_domain"] == df["target_domain"]) == df["in_domain"]).all())

    def test_conflicting_duplicates_raise(self):
        res = _make_stat_results()
        res["cross_domain_perturbation"]["Technology_to_Technology"]["clean"]["metrics"]["macro_f1"] = 0.10
        with self.assertRaises(ValueError):
            assemble_observations({"base": res})

    def test_empty_results_give_empty_frame(self):
        self.assertTrue(assemble_observations({"base": {}}).empty)

    def test_multiple_models(self):
        df = assemble_observations({"base": _make_stat_results(), "large": _make_stat_results()})
        self.assertEqual(set(df["model"]), {"base", "large"})
        self.assertEqual(len(df), 32)


class TestValidateObservations(unittest.TestCase):

    def setUp(self):
        self.df = assemble_observations({"base": _make_stat_results()})

    def test_clean_frame_has_no_duplicates_or_inconsistency(self):
        report = validate_observations(self.df)
        self.assertEqual(report.get("n_duplicates", 0), 0)
        self.assertEqual(report.get("n_scenario_inconsistent", 0), 0)

    def test_empty_frame_reports_error(self):
        self.assertIn("error", validate_observations(pd.DataFrame()))

    def test_missing_column_reports_error(self):
        self.assertIn("error", validate_observations(self.df.drop(columns=["macro_f1"])))

    def test_inconsistent_in_domain_flag_detected(self):
        bad = self.df.copy()
        bad.loc[0, "in_domain"] = not bad.loc[0, "in_domain"]
        self.assertGreaterEqual(validate_observations(bad)["n_scenario_inconsistent"], 1)

    def test_boundary_values_counted(self):
        df = self.df.copy()
        df.loc[0, "macro_f1"] = 0.0
        df.loc[1, "macro_f1"] = 1.0
        r = validate_observations(df)
        self.assertGreaterEqual(r["n_boundary_zero"], 1)
        self.assertGreaterEqual(r["n_boundary_one"], 1)


class TestBoundaryCorrection(unittest.TestCase):

    def test_maps_into_open_unit_interval(self):
        out = apply_boundary_correction(np.array([0.0, 0.5, 1.0]), n=3)
        np.testing.assert_allclose(out, [0.5 / 3, 1.5 / 3, 2.5 / 3])
        self.assertTrue(((out > 0) & (out < 1)).all())

    def test_default_n_is_length(self):
        y = np.array([0.0, 1.0, 0.5, 0.5])
        np.testing.assert_allclose(
            apply_boundary_correction(y), apply_boundary_correction(y, n=4)
        )

    def test_per_row_n(self):
        out = apply_boundary_correction(np.array([1.0]), n=np.array([80.0]))
        self.assertAlmostEqual(out[0], (1.0 * 79 + 0.5) / 80)


class TestStatisticalAnalyzerWrapper(unittest.TestCase):

    def test_constructor_creates_output_dir_and_stores_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "stats")
            analyzer = StatisticalAnalyzer(output_dir=out, mcmc_chains=2, random_seed=7)
            self.assertTrue(os.path.isdir(out))
            self.assertEqual(analyzer.mcmc_chains, 2)
            self.assertEqual(analyzer.random_seed, 7)

    def test_run_full_analysis_delegates(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = StatisticalAnalyzer(output_dir=tmp)
            with patch("statistical_analyzer.run_full_analysis", return_value={"ok": 1}) as m:
                self.assertEqual(analyzer.run_full_analysis({"base": {}}), {"ok": 1})
            self.assertEqual(m.call_args.kwargs["output_dir"], tmp)
            self.assertEqual(m.call_args.kwargs["mcmc_chains"], 4)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ERROR ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectPatterns(unittest.TestCase):

    def test_sensational_detected(self):
        text = "Mengejutkan! Viral video ini bikin semua orang terkejut"
        p = detect_patterns(text)
        self.assertTrue(p["sensational_wording"])

    def test_numerical_claim_detected(self):
        text = "10 cara mudah turunkan berat badan 5 kg dalam 7 hari"
        p = detect_patterns(text)
        self.assertTrue(p["numerical_claim"])

    def test_numerical_percentage_detected(self):
        text = "Vaksin terbukti 95 persen efektif cegah kematian"
        p = detect_patterns(text)
        self.assertTrue(p["numerical_claim"])

    def test_rhetorical_question_detected(self):
        text = "Apakah Presiden benar-benar akan mundur dari jabatannya?"
        p = detect_patterns(text)
        self.assertTrue(p["rhetorical_question"])

    def test_named_entity_detected(self):
        text = "Presiden Joko Widodo umumkan kebijakan baru hari ini"
        p = detect_patterns(text)
        self.assertTrue(p["named_entity"])

    def test_domain_jargon_technology(self):
        text = "startup unicorn fintech Indonesia capai valuasi 1 miliar dolar"
        p = detect_patterns(text, domain="Technology")
        self.assertTrue(p["domain_jargon"])

    def test_domain_jargon_health(self):
        text = "rumah sakit kehabisan vaksin covid di tengah pandemi"
        p = detect_patterns(text, domain="Health")
        self.assertTrue(p["domain_jargon"])

    def test_clean_headline_no_flags(self):
        text = "hasil pertandingan sepak bola hari ini"
        p = detect_patterns(text)
        # Non-sensational, no numbers, no question, no NE, may have jargon
        self.assertFalse(p["sensational_wording"])
        self.assertFalse(p["rhetorical_question"])

    def test_returns_all_five_keys(self):
        p = detect_patterns("test")
        self.assertEqual(set(p.keys()), {
            "sensational_wording", "numerical_claim",
            "rhetorical_question", "named_entity", "domain_jargon"
        })


class TestErrorAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = ErrorAnalyzer()
        self.results  = _make_fake_results()

    def test_analyze_returns_top_keys(self):
        report = self.analyzer.analyze(self.results, model_name="BASE")
        for key in ("model", "in_domain", "cross_domain", "perturbation", "summary"):
            self.assertIn(key, report)

    def test_in_domain_per_domain(self):
        report = self.analyzer.analyze(self.results)
        # Every domain that has texts should appear
        for domain in DOMAINS:
            if domain in report["in_domain"]:
                d = report["in_domain"][domain]
                self.assertIn("n_errors", d)
                self.assertIn("pattern_attribution", d)

    def test_pattern_attribution_all_five_patterns(self):
        report = self.analyzer.analyze(self.results)
        first_domain = DOMAINS[0]
        if first_domain in report["in_domain"]:
            attr = report["in_domain"][first_domain]["pattern_attribution"]
            self.assertEqual(set(attr.keys()), {
                "sensational_wording", "numerical_claim",
                "rhetorical_question", "named_entity", "domain_jargon"
            })

    def test_summary_ranked_drivers(self):
        report = self.analyzer.analyze(self.results)
        ranked = report["summary"]["ranked_error_drivers"]
        self.assertEqual(len(ranked), 5)

    def test_save_report_creates_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.analyzer.analyze(self.results, "BASE")
            path = self.analyzer.save_report(report, output_dir=tmp)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertIn("model", data)

    def test_print_report_runs_without_error(self):
        report = self.analyzer.analyze(self.results, "BASE")
        captured = StringIO()
        with patch("sys.stdout", captured):
            self.analyzer.print_report(report)
        output = captured.getvalue()
        self.assertIn("ERROR ANALYSIS", output)

    def test_attach_texts_to_results(self):
        results = _make_fake_results(include_perturbation=False)
        # Remove texts to simulate missing data
        for d in results["in_domain"].values():
            d.pop("texts", None)
        test_data = {dom: _make_domain_df(dom, n=10) for dom in DOMAINS}
        patched = attach_texts_to_results(results, test_data)
        for domain in DOMAINS:
            self.assertIn("texts", patched["in_domain"][domain])
            self.assertEqual(len(patched["in_domain"][domain]["texts"]), 10)

    def test_error_rate_consistent_with_labels(self):
        report = self.analyzer.analyze(self.results)
        for domain, data in report["in_domain"].items():
            n = data["n_total"]
            n_err = data["n_errors"]
            rate = data["error_rate"]
            if n > 0:
                self.assertAlmostEqual(rate, n_err / n, places=5)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PIPELINE-LEVEL INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluatePipelineIntegration(unittest.TestCase):
    """
    Tests evaluate_pipeline.py's EvaluationPipeline class using mocks for
    everything that touches the filesystem or live models.
    """

    def _make_pipeline(self, tmpdir):
        from evaluate_pipeline import EvaluationPipeline
        return EvaluationPipeline(
            checkpoint_dir=os.path.join(tmpdir, "checkpoints"),
            dataset_dir=os.path.join(tmpdir, "dataset"),
            output_dir=os.path.join(tmpdir, "results"),
            train_output_dir=os.path.join(tmpdir, "output"),
            device="cpu",
            random_seed=42,
        )

    def test_train_output_dir_attribute_set(self):
        """Fix #3: train_output_dir must be stored on the pipeline."""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_pipeline(tmp)
            self.assertEqual(p.train_output_dir, os.path.join(tmp, "output"))

    def test_perform_statistical_analysis_skipped_when_no_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_pipeline(tmp)
            result = p.perform_statistical_analysis({})
            self.assertEqual(result["status"], "skipped")

    def test_perform_statistical_analysis_delegates_to_analyzer(self):
        """The pipeline must hand all model results to StatisticalAnalyzer
        (PyMC is mocked out; the real fit is not exercised here)."""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_pipeline(tmp)
            all_results = {"base": _make_fake_results()}
            with patch("evaluate_pipeline.StatisticalAnalyzer") as analyzer_cls:
                analyzer_cls.return_value.run_full_analysis.return_value = {"status": "completed"}
                result = p.perform_statistical_analysis(all_results)
            self.assertEqual(result, {"status": "completed"})
            analyzer_cls.return_value.run_full_analysis.assert_called_once_with(all_results)
            self.assertEqual(analyzer_cls.call_args.kwargs["random_seed"], 42)
            self.assertEqual(
                analyzer_cls.call_args.kwargs["output_dir"],
                os.path.join(tmp, "results", "statistical_analysis"),
            )

    def test_generate_summary_contains_model(self):
        from evaluate_pipeline import EvaluationPipeline
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_pipeline(tmp)
            # generate_summary iterates cross_domain with tuple unpacking
            # (source, target) — the fake results use "Src->Tgt" string keys
            # so we must use in-memory tuple keys here.
            results = _make_fake_results()
            results["cross_domain"] = {
                (src, tgt): v
                for k, v in results["cross_domain"].items()
                for src, tgt in [k.split("->")]
            }
            all_results = {"base": results}
            stat = {"status": "skipped"}
            summary = p.generate_summary(all_results, stat)
            self.assertIn("base", summary["model_summaries"])

    def test_generate_summary_f1_averages_are_floats(self):
        from evaluate_pipeline import EvaluationPipeline
        with tempfile.TemporaryDirectory() as tmp:
            p = self._make_pipeline(tmp)
            results = _make_fake_results()
            results["cross_domain"] = {
                (src, tgt): v
                for k, v in results["cross_domain"].items()
                for src, tgt in [k.split("->")]
            }
            summary = p.generate_summary({"base": results}, {})
            ms = summary["model_summaries"]["base"]
            self.assertIsInstance(ms["avg_in_domain_f1"],    float)
            self.assertIsInstance(ms["avg_cross_domain_f1"], float)


class TestTrainPipelineIntegration(unittest.TestCase):
    """
    Smoke-tests the TrainingPipeline class structure without executing real training.
    """

    def test_train_pipeline_instantiates(self):
        from train_pipeline import TrainingPipeline
        with tempfile.TemporaryDirectory() as tmp:
            p = TrainingPipeline(
                dataset_dir=os.path.join(tmp, "dataset"),
                output_dir=os.path.join(tmp, "output"),
                checkpoint_dir=os.path.join(tmp, "checkpoints"),
                device="cpu",
                random_seed=42,
            )
            self.assertEqual(p.random_seed, 42)

    def test_model_mapping_contains_three_variants(self):
        from train_pipeline import TrainingPipeline
        with tempfile.TemporaryDirectory() as tmp:
            p = TrainingPipeline(dataset_dir=tmp, output_dir=tmp,
                                 checkpoint_dir=tmp, device="cpu")
            # Verify the internal model mapping has base/large/lite
            from config import config
            self.assertIn("indobert-base-p1",
                          config.model.INDOBERT_BASE)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  PRINT EVALUATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

class TestPrintEvaluationSummary(unittest.TestCase):

    def setUp(self):
        self.results = _make_fake_results()

    def _run_summary(self, **kwargs):
        from print_evaluation_summary import print_full_evaluation_summary
        captured = StringIO()
        # Plots go to a temp dir so the test never writes PNGs into the working tree.
        with tempfile.TemporaryDirectory() as plot_dir:
            with patch("sys.stdout", captured):
                with patch("matplotlib.pyplot.show"):
                    print_full_evaluation_summary(
                        self.results,
                        model_name="BASE",
                        run_error_analysis=False,
                        output_dir=plot_dir,
                        **kwargs
                    )
        return captured.getvalue()

    def test_runs_without_exception(self):
        self._run_summary()   # must not raise

    def test_output_contains_in_domain_section(self):
        out = self._run_summary()
        self.assertIn("IN-DOMAIN", out)

    def test_output_contains_cross_domain_section(self):
        out = self._run_summary()
        self.assertIn("CROSS-DOMAIN", out)

    def test_output_contains_perturbation_section(self):
        out = self._run_summary()
        self.assertIn("PERTURBATION", out)

    def test_output_contains_model_name(self):
        out = self._run_summary()
        self.assertIn("BASE", out)

    def test_global_summary_section(self):
        out = self._run_summary()
        self.assertIn("GLOBAL SUMMARY", out)

    def test_no_perturbation_data_no_crash(self):
        """If perturbation was skipped, _plot_mean_perturbation must not crash."""
        results_no_pert = _make_fake_results(include_perturbation=False)
        from print_evaluation_summary import print_full_evaluation_summary
        with tempfile.TemporaryDirectory() as plot_dir:
            with patch("sys.stdout", StringIO()):
                with patch("matplotlib.pyplot.show"):
                    # Should not raise KeyError
                    print_full_evaluation_summary(
                        results_no_pert,
                        model_name="BASE",
                        run_error_analysis=False,
                        output_dir=plot_dir,
                    )


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SYNTAX CHECK — all .py files in src/
# ══════════════════════════════════════════════════════════════════════════════

class TestSyntaxAllFiles(unittest.TestCase):
    """Parse every .py file in src/ with ast.parse — catches typos instantly."""

    SRC_DIR = os.path.dirname(__file__)

    def _py_files(self):
        return [
            os.path.join(self.SRC_DIR, f)
            for f in os.listdir(self.SRC_DIR)
            if f.endswith(".py") and not f.startswith("__")
        ]

    def test_all_files_parse(self):
        errors = []
        for path in self._py_files():
            try:
                with open(path) as f:
                    ast.parse(f.read(), filename=path)
            except SyntaxError as e:
                errors.append(f"{os.path.basename(path)}: {e}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_no_bare_random_seed_in_perturbation(self):
        """Regression: no global random.seed() calls in perturbation_engine.py."""
        path = os.path.join(self.SRC_DIR, "perturbation_engine.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        bare_seeds = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(getattr(node.func, "attr", None), str)
            and node.func.attr == "seed"
            and isinstance(getattr(node.func.value, "id", None), str)
            and node.func.value.id == "random"
        ]
        self.assertEqual(bare_seeds, [],
                         f"Bare random.seed() found at lines: {bare_seeds}")

    def test_dead_make_serializable_removed(self):
        """Regression: _make_serializable() method must be gone from evaluation_engine."""
        path = os.path.join(self.SRC_DIR, "evaluation_engine.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        defs = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_make_serializable"
        ]
        self.assertEqual(defs, [],
                         "_make_serializable() still present in evaluation_engine.py")
    def test_semantic_engine_has_no_apply_typo(self):
        """Regression: new semantic engine must NOT contain the old _apply_typo helper."""
        path = os.path.join(self.SRC_DIR, "perturbation_engine.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        old_fns = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_apply_typo"
        ]
        self.assertEqual(old_fns, [],
                         "_apply_typo still present — old character-typo engine not removed")

    def test_semantic_engine_has_perturbation_intensities(self):
        """Regression: PERTURBATION_INTENSITIES constant must be defined at module level."""
        path = os.path.join(self.SRC_DIR, "perturbation_engine.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        assigned = [
            node.targets[0].id for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(getattr(node.targets[0], "id", None), str)
            and node.targets[0].id == "PERTURBATION_INTENSITIES"
        ]
        self.assertGreater(len(assigned), 0,
                           "PERTURBATION_INTENSITIES not found in perturbation_engine.py")


# ══════════════════════════════════════════════════════════════════════════════
# 11.  _copy_to_drive  (retry helper in model_trainer.py)
# ══════════════════════════════════════════════════════════════════════════════

class TestCopyToDrive(unittest.TestCase):
    """
    Tests for the _copy_to_drive() helper in model_trainer.py.

    _copy_to_drive() wraps shutil.copy2() with exponential back-off retries.
    Failures are simulated by patching model_trainer.shutil.copy2 directly —
    shutil.copy2 uses C-level I/O and does not go through builtins.open.

    No Google Drive, no GPU, no torch needed — only real local file I/O.
    """

    @staticmethod
    def _get_fn():
        """Grab _copy_to_drive from the already-loaded model_trainer module."""
        import model_trainer as mt
        return mt._copy_to_drive

    def _make_src(self, tmp, content=b"fake-checkpoint-data" * 1000):
        p = os.path.join(tmp, "src.pt")
        with open(p, "wb") as f:
            f.write(content)
        return p

    # ------------------------------------------------------------------
    # 1. Happy path: file is copied correctly on first attempt
    # ------------------------------------------------------------------
    def test_copy_succeeds_first_attempt(self):
        fn = self._get_fn()
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_src(tmp)
            dst = os.path.join(tmp, "dst.pt")
            fn(src, dst)
            with open(dst, "rb") as f:
                self.assertEqual(f.read(), open(src, "rb").read())

    # ------------------------------------------------------------------
    # 2. Content is identical after copy (byte-for-byte)
    # ------------------------------------------------------------------
    def test_copy_content_identical(self):
        fn = self._get_fn()
        payload = bytes(range(256)) * 4096  # 1 MB of known bytes
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_src(tmp, payload)
            dst = os.path.join(tmp, "dst.pt")
            fn(src, dst)
            with open(dst, "rb") as f:
                self.assertEqual(f.read(), payload)

    # ------------------------------------------------------------------
    # 3. Retry: first 2 attempts raise OSError, 3rd attempt succeeds
    # ------------------------------------------------------------------
    def test_retry_succeeds_after_transient_failures(self):
        import model_trainer as mt
        fn = self._get_fn()
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_src(tmp)
            dst = os.path.join(tmp, "dst.pt")

            call_count = {"n": 0}
            real_copy2 = mt.shutil.copy2

            def flaky_copy2(s, d):
                call_count["n"] += 1
                if call_count["n"] < 3:          # fail first 2 attempts
                    raise OSError(95, "Operation not supported")
                real_copy2(s, d)                 # succeed on 3rd

            with patch.object(mt.shutil, "copy2", side_effect=flaky_copy2), \
                 patch.object(mt.time, "sleep"):
                fn(src, dst, retries=5, base_delay=0.0)

            self.assertEqual(call_count["n"], 3)    # failed twice, succeeded third

    # ------------------------------------------------------------------
    # 4. All retries exhausted → RuntimeError is raised
    # ------------------------------------------------------------------
    def test_raises_after_all_retries_exhausted(self):
        import model_trainer as mt
        fn = self._get_fn()
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_src(tmp)
            dst = os.path.join(tmp, "dst.pt")

            with patch.object(mt.shutil, "copy2",
                               side_effect=OSError(95, "Operation not supported")), \
                 patch.object(mt.time, "sleep"):
                with self.assertRaises(RuntimeError) as ctx:
                    fn(src, dst, retries=3, base_delay=0.0)
            self.assertIn("3 attempts", str(ctx.exception))

    # ------------------------------------------------------------------
    # 5. Back-off delays increase exponentially
    # ------------------------------------------------------------------
    def test_backoff_delays_are_exponential(self):
        import model_trainer as mt
        fn = self._get_fn()
        sleep_calls = []

        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_src(tmp)
            dst = os.path.join(tmp, "dst.pt")

            with patch.object(mt.shutil, "copy2",
                               side_effect=OSError(95, "Operation not supported")), \
                 patch.object(mt.time, "sleep",
                               side_effect=lambda d: sleep_calls.append(d)):
                with self.assertRaises(RuntimeError):
                    fn(src, dst, retries=4, base_delay=2.0)

        # 4 attempts → 4 sleeps (sleep called after every failure including last)
        # delays: 2, 4, 8, 16
        self.assertEqual(len(sleep_calls), 4)
        self.assertAlmostEqual(sleep_calls[0], 2.0)
        self.assertAlmostEqual(sleep_calls[1], 4.0)
        self.assertAlmostEqual(sleep_calls[2], 8.0)
        self.assertAlmostEqual(sleep_calls[3], 16.0)

    # ------------------------------------------------------------------
    # 6. When all retries fail, dst is not created (no partial writes)
    # ------------------------------------------------------------------
    def test_dst_not_created_when_all_retries_fail(self):
        import model_trainer as mt
        fn = self._get_fn()
        with tempfile.TemporaryDirectory() as tmp:
            src = self._make_src(tmp)
            dst = os.path.join(tmp, "dst.pt")
            # dst does not exist yet

            with patch.object(mt.shutil, "copy2",
                               side_effect=OSError(95, "Operation not supported")), \
                 patch.object(mt.time, "sleep"):
                with self.assertRaises(RuntimeError):
                    fn(src, dst, retries=2, base_delay=0.0)

            # shutil.copy2 was fully intercepted — dst was never written
            self.assertFalse(os.path.exists(dst))


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Pretty output when run directly
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
