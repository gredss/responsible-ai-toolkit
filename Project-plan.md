# IndoBERT Clickbait Detection: Robustness Evaluation System
## Project Plan and Technical Specification

---

## 1. Executive Summary

This document outlines the complete implementation plan for an end-to-end robustness evaluation system for Indonesian clickbait detection using IndoBERT model variants. The system addresses four primary research questions through a comprehensive pipeline encompassing data preparation, model fine-tuning, perturbation testing, statistical validation, and interactive visualization.

**Primary Objectives:**
- Evaluate domain shift impact on model performance (RQ1)
- Assess vulnerability to linguistic perturbations (RQ2)
- Analyze interaction effects between domain shift and textual noise (RQ3)
- Deploy functional evaluation dashboard for practical use (RQ4)

**Technology Stack:** Python 3.8+, PyTorch, Transformers, PySastrawi, Streamlit, NumPy, Pandas, SciPy

---

## 2. System Architecture Overview

The system follows a modular pipeline architecture with five core components:

**Component 1: Data Management Module**
- Dataset loading and validation
- Stratified train-validation-test splitting (70-15-15)
- Domain-specific data organization
- Reproducibility control (seed=42)

**Component 2: Model Training Module**
- IndoBERT variant initialization (base, large, lite)
- Grid search hyperparameter optimization
- Fine-tuning with optimal configuration
- Model checkpoint management

**Component 3: Evaluation Engine**
- In-domain performance assessment
- Cross-domain evaluation (5x5 matrix)
- Perturbation stress testing (3 intensity levels)
- Metric computation (Accuracy, Precision, Recall, F1-Score, MCC, ROC-AUC)

**Component 4: Statistical Analysis Module**
- Source Drop (SD) and Target Drop (TD) calculation
- Bayesian significance testing with ROPE
- Robustness comparison across variants

**Component 5: Interactive Dashboard**
- Single-text prediction with real-time perturbation
- Domain shift visualization matrix
- Robustness saturation analysis charts
- Automated reliability summary generation

---

## 3. Data Pipeline Design

**3.1 Dataset Structure**
- Total entries: 5,000 samples
- Distribution: 1,000 samples per domain (5 domains)
- Labels: Binary (0=non-clickbait, 1=clickbait)
- Format: CSV with columns [id, source, date, title, Label, url]

**3.2 Preprocessing Workflow**
- Text normalization and cleaning
- Tokenization using IndoBERT tokenizer
- Maximum sequence length: 128 tokens
- Padding and truncation handling
- Attention mask generation

**3.3 Data Splitting Strategy**
- Stratified split maintaining label and domain distribution
- Training set: 3,500 samples (70%)
- Validation set: 750 samples (15%)
- Test set: 750 samples (15%)
- Fixed random seed for reproducibility

**3.4 Domain Organization**
- Each domain treated as separate evaluation unit
- Cross-domain test sets prepared for all combinations

---

## 4. Model Training Framework

**4.1 Model Variants**
- IndoBERT Base: `indobenchmark/indobert-base-p1`
- IndoBERT Large: `indobenchmark/indobert-large-p1`
- IndoBERT Lite: `indobenchmark/indobert-lite-base-p1`

**4.2 Training Configuration**
- Optimizer: AdamW (ε = 1e-8, weight decay = 0.01)
- Learning rate: 2e-5 (searched over [1e-5, 2e-5, 3e-5])
- Batch size: 32 (searched over [16, 32])
- Epochs: 4 (early stopping patience = 2, monitored on F1)
- Warmup ratio: 10%
- Gradient clipping: max_norm = 1.0
- Dropout: 0.1 (searched over [0.1, 0.2])

**4.3 Checkpoint Management**
- Best checkpoint saved per domain specialist
- Drive copy with retry logic (3 attempts, exponential back-off)
- Training summary written to `output/training_summary.json`

---

## 5. Perturbation Testing Methodology

**5.1 Design Principle**

All three perturbation levels use **the same underlying method**: contextual semantic word substitution. The levels differ only in the fraction of word tokens targeted. This ensures the degradation curve is a true intensity gradient rather than a confound of three unrelated linguistic operations.

**5.2 Perturbation Levels**

| Level | Target Intensity | Words Substituted |
|-------|-----------------|-------------------|
| Low | 10% | ~10% of eligible words |
| Medium | 20% | ~20% of eligible words |
| High | 30% | ~30% of eligible words |

**5.3 Substitution Algorithm**

For each selected word:
1. Look up the Indonesian Thesaurus (`dataset/data/dict.json`) via: direct → reverse synonym → Sastrawi-stemmed lookup
2. Filter candidates: same POS, not the original word, not an antonym
3. Score candidates using IndoBERT contextual cosine similarity in **[0.80, 0.95]** — comparing `original_word_in_original_context` vs `candidate_word_in_modified_context`
4. Select the candidate with the highest similarity ≤ 0.95
5. Apply substitution, preserving case and punctuation
6. Record rich metadata per replacement: `word_cosine_similarity`, `lookup_source`, `position`

**5.4 Perturbation Application**
- Systematic application to test sets
- Preservation of original labels
- Domain-specific perturbation tracking
- Reproducible via seeded `random.Random` instance

**5.5 Evaluation Scenarios**
- In-domain clean baseline
- In-domain with perturbations (3 levels)
- Cross-domain clean
- Cross-domain with perturbations (3 levels)
- Total: 80 evaluation scenarios per model

---

## 6. Evaluation Metrics and Analysis

**6.1 Performance Metrics**
- Accuracy: Overall correctness
- Precision: Positive prediction reliability
- Recall: True positive detection rate
- F1-Score: Harmonic mean (primary metric)
- MCC: Matthews Correlation Coefficient
- ROC-AUC: Area under ROC curve

**6.2 Robustness Metrics**
- Source Drop (SD): Performance degradation from source domain
- Target Drop (TD): Performance degradation in target domain
- Word Change Ratio: Actual fraction of words substituted (honest per-word count, not symmetric-difference)
- Confidence Mean: Mean max-class prediction probability
- Confidence Drop: Reduction in confidence under perturbation

**6.3 Statistical Validation**
- Bayesian signed-rank test for paired comparisons
- ROPE threshold: 0.01 (1% practical equivalence)
- Credible interval analysis
- Posterior probability distributions
- Bootstrap 95% CI

**6.4 Comparative Analysis**
- Inter-variant robustness comparison
- Domain-specific vulnerability patterns
- Perturbation sensitivity profiles
- Interaction effect quantification (RQ3: cross-domain + perturbation)

---

## 7. Dashboard Application Design

**7.1 Module 1: Single-Text Prediction**
- Text input field with real-time prediction
- Perturbation preview: show original vs. perturbed text
- Confidence score and classification output
- Model variant selector

**7.2 Module 2: Domain Shift Matrix**
- Interactive 5×5 heatmap (source domain × target domain)
- F1-Score color-coding
- Source Drop and Target Drop annotations
- Per-cell drill-down

**7.3 Module 3: Robustness Curves**
- Degradation curves: Clean → Low → Medium → High
- Per-domain lines on a single chart
- Confidence interval bands
- Comparison across model variants

**7.4 Module 4: Automated Reliability Report**
- Plain-language summary of findings
- Top vulnerability patterns
- Statistical significance of degradation
- Recommended use-case constraints

---

## 8. Module Structure and File Organization

```
src/
├── config.py                # ModelConfig, TrainingConfig, DataConfig,
│                            # PerturbationConfig, EvaluationConfig,
│                            # StatisticalConfig, PathConfig, Config singleton
├── data_manager.py          # DataManager, DatasetValidator, ClickbaitDataset
├── model_trainer.py         # IndoBERTClassifier, ModelTrainer
├── perturbation_engine.py   # _SimilarityChecker, IndonesianThesaurus,
│                            # SemanticWordSubstitution, PerturbationEngine
├── evaluation_engine.py     # MetricsCalculator, InDomainEvaluator,
│                            # CrossDomainEvaluator, PerturbationEvaluator,
│                            # EvaluationEngine
├── statistical_analyzer.py  # BayesianTester, ROPEAnalyzer, SignificanceTester,
│                            # ComparativeStatistics, StatisticalAnalyzer
├── error_analyzer.py        # detect_patterns, ErrorAnalyzer,
│                            # attach_texts_to_results
├── print_evaluation_summary.py  # print_evaluation_summary
├── dashboard_app.py         # Streamlit app
├── debug_logger.py          # set_debug, dbg_* hooks
├── train_pipeline.py        # TrainingPipeline, main()
├── evaluate_pipeline.py     # EvaluationPipeline, main()
├── utils.py                 # FileManager, Timer, safe_divide, reproducibility, …
└── test_suite.py            # 40+ unit/integration tests (no GPU)
```

---

## 9. Implementation Workflow

**Phase 1: Environment Setup**
1. Clone repo (`git clone https://github.com/gredss/responsible-ai-toolkit`)
2. Install dependencies (`pip install -r requirements.txt`)
3. Verify GPU availability and Drive mount (Colab)

**Phase 2: Data Preparation**
1. Confirm 5 domain CSV files at `dataset/data/*.csv`
2. Confirm thesaurus at `dataset/data/dict.json`
3. Run data inspection cell to verify row counts and label distributions
4. Run `train_pipeline.py` — splits are saved to `output/data_splits/` for exact reproduction

**Phase 3: Model Training**
1. Train IndoBERT base specialist per domain (5 models)
2. Validate checkpoints saved to Drive
3. Repeat for large and lite variants

**Phase 4: Evaluation**
1. Run `evaluate_pipeline.py --model base`
2. Inspect in-domain F1, cross-domain matrix, perturbation degradation curves
3. Run statistical analysis (Bayesian signed-rank, ROPE)
4. Run error analysis (linguistic pattern attribution)
5. Repeat for large and lite

**Phase 5: Analysis and Reporting**
1. Execute `print_evaluation_summary.py` for console report
2. Launch Streamlit dashboard for interactive exploration
3. Export results and figures

---

## 10. Technical Specifications

**Hardware Requirements**
- Training: NVIDIA T4 (minimum) or A100 (recommended for large variant)
- Evaluation: T4 sufficient; CPU feasible for perturbation-only runs

**Software Dependencies**
- Python 3.8+
- PyTorch ≥ 1.10
- Transformers ≥ 4.20
- PySastrawi ≥ 0.0.2
- Pandas ≥ 1.3, NumPy ≥ 1.21, scikit-learn ≥ 1.0
- SciPy ≥ 1.7
- Streamlit ≥ 1.15, Plotly ≥ 5.0
- tqdm ≥ 4.62

**Reproducibility Controls**
- Global seed: 42 throughout (set via `reproducibility.set_seed`)
- All random operations in `PerturbationEngine` use a seeded `random.Random` instance — no global `random.seed()` calls
- Data splits exported and reloaded so train and eval use identical test partitions

---

## 11. Research Question Mapping

**RQ1: Domain Shift Impact**
- Addressed by: Cross-domain evaluation (5×5 matrix)
- Metrics: SD, TD, F1-Score degradation
- Visualization: Domain shift heatmap
- Analysis: Bayesian significance testing across specialist pairs

**RQ2: Linguistic Perturbation Vulnerability**
- Addressed by: Perturbation testing (3 intensity levels — contextual semantic substitution)
- Metrics: F1 degradation curve, word change ratio, confidence drop
- Visualization: Robustness degradation curves per domain
- Analysis: Intensity gradient analysis; ROPE for practical equivalence

**RQ3: Interaction Effects**
- Addressed by: Combined cross-domain + perturbation testing
- Metrics: Cumulative F1 degradation, domain × perturbation interaction
- Visualization: Multi-dimensional comparison (domain shift + noise)
- Analysis: Whether domain shift and perturbation effects are additive or super-additive

**RQ4: System Deployment**
- Addressed by: Streamlit dashboard
- Features: All four dashboard modules
- Usability: Interactive and user-friendly
- Functionality: Complete evaluation pipeline accessible without CLI

---

## 12. Quality Assurance

**12.1 Testing Strategy**
- Unit tests for individual functions (no GPU, no model downloads)
- Integration tests for module interactions
- Syntax validation for all .py files via `ast.parse`
- Regression guards: no `_apply_typo` (old engine removed), `PERTURBATION_INTENSITIES` present, no bare `random.seed()`, no `_make_serializable`

**12.2 Validation Checks**
- Data integrity verification (null counts, duplicate detection, length outliers)
- Model output validation (logit shapes, label distributions)
- Metric calculation accuracy (perfect prediction → F1=1.0)
- Statistical test correctness (identical scores → posterior in ROPE)

**12.3 Documentation Standards**
- Inline code documentation with type hints
- Function docstrings (Args / Returns / Raises)
- Module-level documentation
- User guide in README.md

---

## 13. Deliverables

**13.1 Code Artifacts**
- Complete Python implementation in `src/`
- Configuration files (`config.py`, `requirements.txt`)
- Test suite (`test_suite.py`)
- End-to-end notebook (`run_complete_pipeline.ipynb`)

**13.2 Trained Models**
- Fine-tuned IndoBERT base specialists (5 domains × 1 variant = 5 checkpoints)
- Fine-tuned IndoBERT large and lite specialists (pending)
- Training logs and hyperparameter records

**13.3 Evaluation Outputs**
- In-domain F1 table (5 domains × 3 variants)
- Cross-domain 5×5 F1 matrix per variant
- Perturbation degradation curves (Clean → Low → Medium → High)
- Statistical analysis report (Bayesian signed-rank + ROPE decisions)
- Error analysis report (linguistic pattern attribution per condition)

**13.4 Visualizations**
- Domain shift heatmaps
- Robustness degradation curves
- Confidence distribution plots
- Automated reliability report

---

## 14. Success Criteria

**14.1 Functional Criteria**
- All five domain specialist models trained and evaluated
- 5×5 cross-domain matrix computed
- Three-level perturbation curves produced with interpretable word-change ratios (10/20/30%)
- Bayesian signed-rank tests run on F1-per-condition vectors
- Dashboard functional with live prediction and heatmap

**14.2 Scientific Criteria**
- Perturbation word-change ratios match target intensities (±5%)
- Degradation curves show monotonic or near-monotonic decline for most domains
- Statistical decisions are ROPE-informed (not just p < 0.05)
- Error analysis attributes ≥ 50% of degradation to identifiable linguistic patterns

**14.3 Reproducibility Criteria**
- `python -m pytest test_suite.py -v` passes with zero failures
- Two independent runs with seed=42 produce identical results
- README installation steps work on a fresh Colab runtime

---

## 15. Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Thesaurus lookup miss-rate too high (few eligible words per headline) | Medium | Report actual intensity alongside target; domains with low hit rate are a finding, not a bug |
| IndoBERT large/lite OOM on T4 | Medium | Use A100 runtime or reduce batch size; large/lite evaluation can be deferred |
| Drive I/O latency slows evaluation | Low | Write results to local `/content/` first, copy to Drive after; retry logic in `model_trainer.py` |
| Cross-domain + perturbation run time excessive | Low | `--skip-perturbation` flag allows decoupled runs; use `tqdm` progress bars |
| test_suite.py fails on import in CI (no Sastrawi) | Low | Test suite mocks torch/transformers; PySastrawi is mocked implicitly because perturbation tests use a stub engine |

---

## 16. Timeline Estimation

| Phase | Task | Estimated Time |
|-------|------|---------------|
| Setup | Environment, dependencies, repo clone | 15 min |
| Data | Load, inspect, validate all 5 CSVs | 10 min |
| Training | IndoBERT base × 5 domains | ~3–4 hours (T4) |
| Training | IndoBERT large + lite × 5 domains each | ~8–12 hours (A100 recommended) |
| Evaluation | Base model: in-domain + cross-domain + perturbation | ~1 hour |
| Evaluation | Large + lite models | ~2 hours each |
| Statistics | Bayesian tests, ROPE, error analysis | ~15 min |
| Reporting | Summary printer, dashboard | ~15 min |

---

## 17. Conclusion

This system provides a rigorous, reproducible, and methodologically sound framework for evaluating IndoBERT robustness on Indonesian clickbait detection. The perturbation engine uses contextual semantic substitution — not character-level typos or informal slang injection — ensuring that all three perturbation levels probe the same linguistic phenomenon (semantic paraphrase resistance) at increasing intensity. This design produces interpretable, defensible results for all four research questions.
