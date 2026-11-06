# Robustness Evaluation of IndoBERT for Cross-Domain Indonesian Clickbait Detection

## Overview

This repository provides a robustness evaluation framework for Indonesian clickbait detection using IndoBERT model variants. The system fine-tunes IndoBERT on a domain-split dataset, then assesses performance under cross-domain conditions and linguistic perturbations using Bayesian statistical tests.

## Key Features

- **Multi-Model Support:** Fine-tune and compare IndoBERT base, large, and lite variants
- **Cross-Domain Evaluation:** 5×5 matrix across Technology, Politics, Health, Sport, and Education domains
- **Perturbation Testing:** Three-level stress testing — all using the same contextual semantic word substitution method at 10% (low), 20% (medium), and 30% (high) intensity
- **Error Analysis:** Linguistic pattern breakdown explaining *why* performance drops (sensational wording, numerical claims, rhetorical questions, named entities, domain jargon)
- **Statistical Validation:** Bayesian signed-rank test with ROPE analysis, bootstrap confidence intervals, and paired/non-parametric significance tests
- **Interactive Dashboard:** Streamlit interface for real-time predictions, cross-domain heatmaps, degradation curves, and automated reliability reports
- **Pipeline Debug Logging:** Comprehensive, structured observability for every processing stage — enable with a single flag, zero cost when off

## Project Structure

```
responsible-ai-toolkit/
├── dataset/
│   ├── data/                        # Per-domain CSV files + thesaurus
│   │   ├── technology.csv
│   │   ├── politic.csv
│   │   ├── health.csv
│   │   ├── sport.csv
│   │   ├── education.csv
│   │   └── dict.json                # Indonesian Thesaurus (victoriasovereigne/tesaurus)
│   ├── Dataset-description.md       # Dataset specifications and annotation details
│   ├── dataset_eda.ipynb            # Exploratory data analysis notebook
│   └── scrap.py                     # Web scraping utilities
├── src/                             # Core system modules
│   ├── config.py                    # Centralised configuration (models, training, paths, etc.)
│   ├── data_manager.py              # Data loading, validation, stratified splitting, tokenisation
│   ├── model_trainer.py             # Training loop, hyperparameter grid search, checkpoint saving
│   ├── perturbation_engine.py       # Contextual semantic perturbation engine (3 intensity levels)
│   ├── evaluation_engine.py         # Metrics calculation and cross-domain evaluation orchestration
│   ├── statistical_analyzer.py      # Bayesian tests, ROPE analysis, bootstrap CI
│   ├── error_analyzer.py            # Linguistic error pattern analysis for misclassifications
│   ├── print_evaluation_summary.py  # Pretty-print evaluation results to console / plots
│   ├── dashboard_app.py             # Interactive Streamlit dashboard
│   ├── debug_logger.py              # Configurable debug observability for all pipeline stages
│   ├── train_pipeline.py            # Training orchestration script (CLI)
│   ├── evaluate_pipeline.py         # Evaluation orchestration script (CLI)
│   ├── utils.py                     # Shared helper functions
│   └── test_suite.py                # End-to-end unit/integration tests (no GPU required)
├── requirements.txt
├── run_complete_pipeline.ipynb      # End-to-end execution notebook
└── README.md
```

## System Workflow

```mermaid
graph TD
    A[Raw Dataset<br/>5 × 1,000 articles] --> B[Data Manager<br/>load · validate · split · tokenise]
    B --> C{Domain Split 70/15/15}
    C --> D[Technology]
    C --> E[Politics]
    C --> F[Health]
    C --> G[Sport]
    C --> H[Education]

    D & E & F & G & H --> I[Model Trainer<br/>AdamW · linear scheduler · early stopping]
    I --> J[IndoBERT Base / Large / Lite]

    J --> K[Evaluation Engine]
    K --> L[In-Domain Testing]
    K --> M[Cross-Domain 5×5 Matrix]

    J --> N[Perturbation Engine<br/>Contextual Semantic Substitution]
    N --> O[Low: 10% words substituted]
    N --> P[Medium: 20% words substituted]
    N --> Q[High: 30% words substituted]

    O & P & Q --> R[Perturbation Robustness Testing]

    L & M & R --> S[Statistical Analyzer<br/>Bayesian Signed-Rank · ROPE · Bootstrap]
    S --> T[Error Analyzer<br/>Linguistic Pattern Breakdown]

    S & T --> U[Dashboard App / Summary Printer]
    U --> V[Real-time Predictions]
    U --> W[Domain-Shift Heatmaps]
    U --> X[Reliability Reports]
```

## Dataset

5,000 Indonesian news headlines scraped from CNN Indonesia, Kompas, Detik, Tempo, and Tribun News (2022–2026). Each domain (Technology, Politics, Health, Sport, Education) contributes 1,000 articles. Headlines are labelled `clickbait` / `non-clickbait` by three independent annotators with majority voting (Fleiss' κ = 0.72, observed agreement 87%).

CSV schema: `id`, `source`, `date`, `title`, `Label`, `url`

See [`dataset/Dataset-description.md`](dataset/Dataset-description.md) for full annotation guidelines and quality criteria.

## Perturbation Engine Design

All three perturbation levels use **the same underlying method**: contextual semantic word substitution via IndoBERT cosine similarity. The levels differ only in the fraction of words targeted.

| Level | Intensity | Method |
|-------|-----------|--------|
| Low | 10% of words | Semantic synonym substitution |
| Medium | 20% of words | Semantic synonym substitution |
| High | 30% of words | Semantic synonym substitution |

**Candidate generation:** Indonesian Thesaurus (`dataset/data/dict.json`, sourced from `victoriasovereigne/tesaurus`). Lookup cascade: direct → reverse synonym → Sastrawi-stemmed form.

**Candidate filtering:**
1. Same POS tag as original word
2. Not the original word itself
3. Not an antonym
4. IndoBERT contextual cosine similarity in **[0.80, 0.95]** — scored as `original_word_in_original_context` vs `candidate_word_in_modified_context`

The candidate with the highest similarity ≤ 0.95 is selected.

## Technology Stack

| Category | Libraries |
|---|---|
| Model / Training | PyTorch, Hugging Face Transformers |
| Data Processing | Pandas, NumPy, scikit-learn |
| Indonesian NLP | PySastrawi (Stemmer for thesaurus lookup) |
| Statistical Analysis | SciPy (Wilcoxon, Mann-Whitney U, paired t-test), custom Bayesian signed-rank |
| Visualisation / Dashboard | Streamlit, Plotly |
| Utilities | tqdm |

> **Note:** `baycomp` is **not** a dependency. Bayesian analysis is implemented directly in [`src/statistical_analyzer.py`](src/statistical_analyzer.py).

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training; evaluation can run on CPU)

### Setup

```bash
git clone https://github.com/gredss/responsible-ai-toolkit
cd responsible-ai-toolkit
pip install -r requirements.txt
```

Edit [`src/config.py`](src/config.py) to change model variants, hyperparameters, or output paths.

## Usage

### Train a model

```bash
# Single model (base / large / lite)
python src/train_pipeline.py --model base

# All three variants
python src/train_pipeline.py --model all

# With hyperparameter grid search
python src/train_pipeline.py --model base --grid-search

# On Colab / GPU
python src/train_pipeline.py --model base --device cuda
```

### Run evaluation

```bash
# Evaluate single model (thesaurus path defaults to dataset/data/dict.json)
python src/evaluate_pipeline.py --model base

# Evaluate all models
python src/evaluate_pipeline.py --model all

# Override thesaurus path (e.g. on Colab with a Drive copy)
python src/evaluate_pipeline.py --model base --thesaurus-path /content/dict.json

# Skip perturbation testing (faster)
python src/evaluate_pipeline.py --model base --skip-perturbation
```

### Launch the dashboard

```bash
streamlit run src/dashboard_app.py
```

### Full end-to-end notebook

```bash
jupyter notebook run_complete_pipeline.ipynb
```

### Run tests (no GPU needed)

```bash
cd src
python -m pytest test_suite.py -v
```

## Debug Logging

Every major processing stage emits structured, human-readable debug output that can be enabled without touching any pipeline logic. All debug calls are no-ops by default — zero performance impact on normal runs.

### Activation

```bash
# Environment variable (works everywhere, no code changes)
DEBUG_PIPELINE=1 python src/train_pipeline.py --model base
DEBUG_PIPELINE=1 python src/evaluate_pipeline.py --model base

# CLI flag
python src/train_pipeline.py --model base --debug
python src/evaluate_pipeline.py --model base --debug

# Programmatic (e.g. notebook)
from debug_logger import set_debug
set_debug(True)
```

### What gets logged

Each stage is identified by a bracketed header so output can be grepped by stage:

| Header | Stage | Logged information |
|---|---|---|
| `[DataManager]` | CSV loading & splitting | Raw row counts and label distributions per file; combined dataset summary; data-quality issues (nulls, duplicates, length outliers); per-domain split sizes and class distributions; 3 representative sample texts per split |
| `[Tokenizer]` | Text tokenisation | Token IDs, attention mask, real vs. padded token counts, and decoded token string for 2 example texts |
| `[ClickbaitDataset]` | Dataset construction | Total samples, clickbait ratio, and tokenisation samples at the point a `ClickbaitDataset` is created |
| `[ModelTrainer]` | Training & inference | Model init parameters (device, max length, dropout); every 50th batch: input shape, raw logits, predicted labels, and loss; per-epoch train and validation metrics; predict call summary, per-batch logits/predictions, and final label counts |
| `[Evaluation]` | In-domain & cross-domain | Full metric dict (accuracy, precision, recall, F1, MCC, ROC-AUC) per domain; confusion matrix; Source Drop and Target Drop domain-shift values |
| `[Perturbation]` | Text perturbation | 3 before/after text pairs per level and domain; mean word change ratios per level; actual vs. target intensity; per-level metrics; absolute and relative robustness drop |
| `[Statistics]` | Statistical tests | Input score arrays (mean, std, range) for both models; Bayesian signed-rank output (p-value, effect size, credible interval, ROPE decision); ROPE analysis (probability in ROPE, all credible interval levels); bootstrap CI; paired t-test and Mann-Whitney U results |
| `[ErrorAnalysis]` | Linguistic error patterns | Error count and rate per condition; pattern attribution ranked from highest to lowest driver; top global error driver |

## Module Reference

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth for all hyperparameters, paths, and thresholds |
| `data_manager.py` | Loads per-domain CSVs, validates schema, stratified train/val/test split (70/15/15), tokenises with IndoBERT tokeniser |
| `model_trainer.py` | Fine-tunes `BertForSequenceClassification`, AdamW + linear warmup scheduler, optional grid search, checkpoint saving |
| `perturbation_engine.py` | Contextual semantic word substitution at three intensities (10/20/30%) using IndoBERT cosine similarity + Indonesian Thesaurus |
| `evaluation_engine.py` | Computes accuracy, precision, recall, F1, MCC, ROC-AUC; orchestrates in-domain, cross-domain, and perturbation scenarios |
| `statistical_analyzer.py` | Bayesian signed-rank test, ROPE analysis, bootstrap CI, Wilcoxon, Mann-Whitney U, Cohen's d |
| `error_analyzer.py` | Identifies linguistic patterns in misclassified headlines (sensational wording, numerical claims, rhetorical questions, named entities, domain jargon) |
| `debug_logger.py` | Configurable observability layer; all hooks are no-ops unless `DEBUG_PIPELINE=1` or `--debug` is passed |
| `print_evaluation_summary.py` | Console pretty-printer and matplotlib plots for evaluation results |
| `dashboard_app.py` | Streamlit UI: single-text prediction, 5×5 heatmap, degradation curves, automated reliability report |
| `train_pipeline.py` | CLI wrapper for end-to-end training |
| `evaluate_pipeline.py` | CLI wrapper for end-to-end evaluation |
| `utils.py` | Shared helper functions |
| `test_suite.py` | Mocked unit/integration tests covering all modules without GPU or model downloads |

## Evaluation Metrics

- **Classification:** Accuracy, Precision, Recall, F1-Score, MCC, ROC-AUC
- **Robustness:** Source Drop (SD), Target Drop (TD), Performance Degradation across perturbation levels, word change ratio per level
- **Statistical:** Bayesian signed-rank with ROPE (threshold = 0.01), bootstrap 95% CI, Wilcoxon signed-rank, Mann-Whitney U, paired t-test, Cohen's d effect size

## License

See [`LICENSE`](LICENSE) for details.

## Citation

If you use this system in your research, please cite:

```
TBA
```
