# 29 AUG - Evaluation Pipeline
"""
Evaluation Pipeline for IndoBERT Clickbait Detection System

This script orchestrates comprehensive evaluation.  EvaluationEngine runs a
5-phase pipeline per model (phase 3 only generates perturbed data; phases
1, 2, 4 and 5 produce metrics):
1. In-domain evaluation (train and test on same domain)
2. Cross-domain evaluation (5×5 matrix) + domain-shift metrics
3. Perturbation generation (3 intensity levels; data only, no metrics)
4. In-domain perturbation evaluation
5. Cross-domain × perturbation evaluation
Afterwards this script runs the Bayesian statistical analysis across models
and writes the aggregated evaluation summary JSON.

Usage:
    # Evaluate single model
    python evaluate_pipeline.py --model base

    # Evaluate all models
    python evaluate_pipeline.py --model all

    # Skip perturbation testing (faster)
    python evaluate_pipeline.py --model base --skip-perturbation

    # On Colab
    !python evaluate_pipeline.py --model base --device cuda
"""

import glob
import os
import re
import sys
import argparse
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from config import config
from data_manager import DataManager
from model_trainer import ModelTrainer
from evaluation_engine import (
    EvaluationEngine,
    PERTURBATION_TYPES,
    PERTURBATION_LEVELS,
)
from perturbation_engine import PerturbationEngine
from statistical_analyzer import StatisticalAnalyzer
from error_analyzer import ErrorAnalyzer, attach_texts_to_results
from utils import file_manager, reproducibility, Timer, get_timestamp
import debug_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Orchestrates comprehensive evaluation of trained IndoBERT models."""

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        dataset_dir: str = "dataset",
        output_dir: str = "evaluation_results",
        train_output_dir: str = "output",
        device: str = "auto",
        random_seed: int = 42,
        thesaurus_path: Optional[str] = None,
        perturbation_output_dir: str = "/content/drive/MyDrive/thesis/results",
    ):
        """
        Initialize evaluation pipeline.

        Args:
            checkpoint_dir: Directory containing trained model checkpoints
            dataset_dir: Directory containing CSV files
            output_dir: Directory for evaluation results
            train_output_dir: Directory used by train_pipeline.py (where data_splits/ were saved)
            device: Device to use ('cuda', 'cpu', or 'auto')
            random_seed: Random seed for reproducibility
            thesaurus_path: Path to the Indonesian thesaurus dict.json.
                Defaults to <dataset_dir>/data/dict.json (the copy committed in the repo).
            perturbation_output_dir: Root for perturbed CSVs; each model writes
                to <root>/<model_name>.
        """
        self.perturbation_output_dir_root = perturbation_output_dir
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.train_output_dir = train_output_dir
        self.device = device
        self.random_seed = random_seed
        # Resolve thesaurus path: prefer explicit arg, then derive from dataset_dir
        if thesaurus_path is not None:
            self.thesaurus_path = thesaurus_path
        else:
            self.thesaurus_path = os.path.join(dataset_dir, "dict.json")

        reproducibility.set_seed(random_seed)
        file_manager.ensure_directory(output_dir)

        logger.info("Evaluation Pipeline initialized")
        logger.info(f"Checkpoints: {checkpoint_dir}, Dataset: {dataset_dir}, Output: {output_dir}")
        logger.info(f"Device: {device}, Random seed: {random_seed}")

    def load_data(self) -> Dict[str, Any]:
        """
        Load datasets and organize by domain.

        Splits are loaded from train_output_dir/data_splits/ (written by train_pipeline.py)
        so that evaluation uses the exact same test split the model never saw during training.
        If that directory does not exist, splits are recreated and saved there.

        Returns:
            Dictionary with data_manager and domain-organized test data
        """
        logger.info("\n[STEP 1] Loading Data")

        with Timer() as timer:
            data_manager = DataManager.from_dataset_directory(
                dataset_dir=self.dataset_dir,
                tokenizer_name=config.model.DEFAULT_MODEL,
                max_length=config.model.MAX_SEQUENCE_LENGTH,
                random_seed=self.random_seed,
            )

            summary = data_manager.get_summary()
            logger.info(f"Loaded {summary['total_samples']} samples from {len(summary['domains'])} domains")

            # Always look in the training output dir first so we reuse the same
            # test split that was held out during train_pipeline.py.
            splits_dir = os.path.join(self.train_output_dir, "data_splits")

            # Load the ONE shared test split held out during training.
            #
            # All model notebooks (base / large / lite) MUST evaluate on the
            # exact same split, so evaluation never creates its own split.
            # If the pre-saved splits are missing we FAIL FAST rather than
            # silently generating a fresh (possibly different) split.
            if not os.path.exists(splits_dir):
                raise FileNotFoundError(
                    f"No pre-saved data splits found at '{splits_dir}'. "
                    "Evaluation will NOT create a new split, because all model "
                    "notebooks must evaluate on the same split. "
                    "Run train_pipeline.py once to create data_splits/, then "
                    "pass --train-output-dir pointing at that same folder for "
                    "every model you evaluate."
                )

            data_manager.check_splits_preprocess(splits_dir)
            logger.info(f"Loading the shared test split from {splits_dir}")
            domain_splits = data_manager.load_domain_splits(splits_dir)

            # Evaluation only needs the test split of each domain
            test_data_by_domain = {
                domain: splits["test"]
                for domain, splits in domain_splits.items()
            }

        logger.info(f"Data loading completed in {timer.elapsed():.2f}s")

        return {
            "data_manager": data_manager,
            "domain_splits": domain_splits,
            "test_data_by_domain": test_data_by_domain
        }

    def load_models(self, model_names: List[str]) -> Dict[str, Dict[str, ModelTrainer]]:
        """
        Load trained specialist checkpoints for each requested model variant.

        For each model name (e.g. ``'base'``, ``'large'``, ``'lite'``) this
        method iterates over all five domains and attempts to load the best
        checkpoint saved by :class:`train_pipeline.TrainingPipeline`.  Domains
        whose checkpoint file is missing are skipped with a warning so that a
        partial evaluation can still proceed.

        Args:
            model_names: List of model variant names to load
                         (subset of ``['base', 'large', 'lite']``).

        Returns:
            Nested dict ``{model_name: {domain: ModelTrainer}}`` containing
            one :class:`ModelTrainer` per successfully loaded specialist.
        """
        logger.info("\n[STEP 2] Loading SPECIALIST Models")

        model_mapping = {
            'base': config.model.INDOBERT_BASE,
            'large': config.model.INDOBERT_LARGE,
            'lite': config.model.INDOBERT_LITE
        }

        all_models = {}

        for model_name in model_names:
            full_model_name = model_mapping[model_name]
            domain_trainers = {}

            # FIX: Loop through the 5 domains and load the SPECIFIC model for each
            for domain in config.data.DOMAINS:
                model_checkpoint_dir = os.path.join(
                    self.checkpoint_dir,
                    model_name,
                    domain.capitalize()
                )

                checkpoint_path = os.path.join(
                    model_checkpoint_dir,
                    "best_model.pt"
                )

                if not os.path.exists(checkpoint_path):
                    logger.warning(f"No checkpoint found for {model_name} -> {domain}")
                    continue

                logger.info(f"Loading {domain} specialist from {checkpoint_path}")

                trainer = ModelTrainer(
                    model_name=full_model_name,
                    max_length=config.model.MAX_SEQUENCE_LENGTH,
                    device=self.device,
                    random_seed=self.random_seed
                )
                trainer.load_checkpoint(checkpoint_path)

                # Assign the specific specialist trainer to its specific domain key
                domain_trainers[domain] = trainer

            all_models[model_name] = domain_trainers

        return all_models

    def evaluate_single_model(
        self,
        model_name: str,
        domain_trainers: Dict[str, ModelTrainer],
        test_data_by_domain: Dict[str, pd.DataFrame],
        skip_perturbation: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively.

        Args:
            model_name: Name of model variant
            domain_trainers: Dictionary mapping domains to trainers
            test_data_by_domain: Test data organized by domain
            skip_perturbation: Whether to skip perturbation testing

        Returns:
            Dictionary with all evaluation results
        """
        logger.info(f"\n[STEP 3] Evaluating {model_name.upper()} Model")

        total_timer = Timer()
        total_timer.start()

        perturbation_engine = PerturbationEngine(
            thesaurus_path=self.thesaurus_path,
            random_seed=self.random_seed,
        )

        model_output_dir = os.path.join(self.output_dir, model_name)
        perturbation_output_dir = os.path.join(
            self.perturbation_output_dir_root,
            model_name
        )

        eval_engine = EvaluationEngine(
            model_trainers=domain_trainers,
            perturbation_engine=perturbation_engine,
            output_dir=model_output_dir,
            perturbation_output_dir=perturbation_output_dir
        )

        results = eval_engine.run_complete_evaluation(
            test_data=test_data_by_domain,
            include_perturbations=not skip_perturbation
        )

        # Inject raw texts into results so ErrorAnalyzer can access headlines
        results = attach_texts_to_results(
            results,
            test_data_by_domain
        )

        # Capture the output of save_results into the results_file variable
        results_file = eval_engine.save_results(results, filename='complete_evaluation.json')
        logger.info(f"Results saved to {results_file}")

        # Run error analysis — explains WHY performance drops per condition
        logger.info(f"\n[STEP 3b] Error Analysis for {model_name.upper()}")
        try:
            error_analyzer = ErrorAnalyzer()
            error_report = error_analyzer.analyze(results, model_name=model_name)
            error_analyzer.save_report(error_report, output_dir=model_output_dir)
            error_analyzer.print_report(error_report)
        except Exception as e:
            logger.warning(f"Error analysis failed (non-fatal): {e}")

        # Run pattern-based robustness analysis on the synonym-perturbation CSVs
        # that were just written to perturbation_output_dir. We reuse the
        # already-loaded specialist trainers (no checkpoint reload, no training).
        # This step is skipped when perturbation testing was skipped, because it
        # needs the per-level perturbed CSVs.
        if not skip_perturbation:
            logger.info(f"\n[STEP 3c] Pattern Analysis for {model_name.upper()}")
            try:
                self.run_pattern_analysis_for_model(
                    model_name=model_name,
                    domain_trainers=domain_trainers,
                    perturbation_results_root=perturbation_output_dir,
                    test_data_by_domain=test_data_by_domain,
                )
            except Exception as e:
                logger.warning(f"Pattern analysis failed (non-fatal): {e}")

        total_time = total_timer.stop()
        logger.info(f"Evaluation completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")

        return results

    def run_pattern_analysis_for_model(
        self,
        model_name: str,
        domain_trainers: Dict[str, ModelTrainer],
        perturbation_results_root: str,
        test_data_by_domain: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        Run the SUCCESS-BASED pattern robustness analysis for one model.

        This reads the per-domain Low/Medium/High *semantic* (synonym)
        perturbation CSVs that ``EvaluationEngine.generate_all_perturbations``
        wrote under ``perturbation_results_root``. It then:

          * Report A - reports per-level perturbation validity (success/fail
            counts and rates), keeping failures visible; and
          * Report B - aligns ONLY the samples successfully perturbed at all
            three levels (by the standardized common-id method), classifies the
            five text-equality patterns, and scores the model on each pattern.

        When ``test_data_by_domain`` is provided the original (unperturbed)
        headlines of every matched id are also scored so Report B shows clean
        alongside low/medium/high — a true apple-to-apple comparison.

        The model is NOT retrained and NOT reloaded: we build a predict function
        that reuses the specialist ``ModelTrainer`` already loaded for each
        domain. The two reports are printed to the log/notebook; the returned
        DataFrames are available if a caller wants to persist them.
        """
        # Imported here (not at module top) so the rest of the evaluation
        # pipeline does not depend on this analysis module being importable.
        import pattern_analysis

        # Build a factory: given a domain, return a predict function that uses
        # that domain's already-loaded specialist trainer.
        def predict_fn_factory_for_domain(domain: str):
            trainer_for_domain = domain_trainers[domain]

            def predict_texts(texts: List[str]) -> List[int]:
                if len(texts) == 0:
                    return []
                # predict() already returns a 1-D array of label ints.
                predicted_labels, _probabilities = trainer_for_domain.predict(texts)
                return [int(label) for label in predicted_labels]

            return predict_texts

        # Only analyse domains whose specialist checkpoint was actually loaded.
        domains_to_analyse = list(domain_trainers.keys())

        # Save the three result tables so the notebook can simply read+plot them.
        pattern_analysis_output_dir = os.path.join(
            perturbation_results_root, "pattern_analysis"
        )

        pattern_analysis.run_success_based_pattern_analysis(
            results_root=perturbation_results_root,
            predict_fn_factory=predict_fn_factory_for_domain,
            domains=domains_to_analyse,
            ptype="semantic",
            print_reports=True,
            output_dir=pattern_analysis_output_dir,
            test_data_by_domain=test_data_by_domain,
        )
        logger.info(
            f"Success-based pattern analysis for {model_name.upper()} complete "
            f"(source CSVs under {perturbation_results_root}; "
            f"tables saved to {pattern_analysis_output_dir})"
        )

    def perform_statistical_analysis(
        self,
        all_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run the Bayesian hierarchical statistical analysis.

        Assembles up to 300 aggregate F1 observations
        (3 models × 5 source domains × 5 target domains × 4 perturbation
        conditions) from the nested evaluation result dict, then fits a
        Beta regression with logit link using PyMC.

        Returns the full analysis dict produced by
        StatisticalAnalyzer.run_full_analysis(), including:
          - observation_report   : data-quality summary
          - convergence          : R-hat, ESS, divergences
          - ppc                  : posterior predictive check
          - loo                  : PSIS-LOO cross-validation
          - contrasts            : posterior contrasts for all 5 RQs
          - descriptive          : degradation and domain-shift summaries
          - report_path          : path to the saved text report

        If fewer than 2 model variants are present the analysis is skipped.
        If PyMC is not installed only descriptive statistics are returned.
        """
        logger.info("\n[STEP 4] Statistical Analysis — Bayesian Hierarchical Model")

        if len(all_results) < 1:
            logger.info("Skipping statistical analysis (no model results).")
            return {'status': 'skipped', 'reason': 'no_results'}

        analyzer = StatisticalAnalyzer(
            random_seed=self.random_seed,
            output_dir=os.path.join(self.output_dir, 'statistical_analysis'),
        )

        return analyzer.run_full_analysis(all_results)

    def generate_summary(
        self,
        all_results: Dict[str, Dict[str, Any]],
        statistical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate evaluation summary.

        Args:
            all_results: Dictionary mapping model names to their results
            statistical_analysis: Statistical analysis results

        Returns:
            Summary dictionary
        """
        logger.info("\n[STEP 5] Generating Summary")

        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': list(all_results.keys()),
            'model_summaries': {}
        }

        for model_name, results in all_results.items():
            # ── in-domain F1: use the subset clean from perturbation block ──
            # results['perturbation'][domain]['clean'] is the common-subset
            # baseline (same N as low/medium/high).  Fall back to in_domain
            # if the perturbation block is absent.
            in_domain_f1 = []
            pert_block = results.get('perturbation', {})
            if pert_block:
                for domain, domain_block in pert_block.items():
                    if domain == 'clean':          # skip the top-level 'clean' key
                        continue
                    clean_b = domain_block.get('clean', {})
                    if clean_b:
                        in_domain_f1.append(
                            float(clean_b.get('metrics', {}).get('macro_f1', 0))
                        )
            if not in_domain_f1:
                # fallback: full-test-set in-domain scores
                for domain, domain_results in results.get('in_domain', {}).items():
                    in_domain_f1.append(
                        float(domain_results['metrics']['macro_f1'])
                    )

            avg_in_domain_f1 = (
                sum(in_domain_f1) / len(in_domain_f1) if in_domain_f1 else 0.0
            )

            # ── cross-domain F1 ───────────────────────────────────────────
            # Keys in results['cross_domain'] are "Source->Target" strings.
            cross_domain_f1 = []
            for key, metrics_data in results.get('cross_domain', {}).items():
                # skip in-domain diagonal (source == target)
                if '->' in key:
                    src, tgt = key.split('->', 1)
                    if src.strip() == tgt.strip():
                        continue
                cross_domain_f1.append(
                    float(metrics_data.get('metrics', {}).get('macro_f1', 0))
                )
            avg_cross_domain_f1 = (
                sum(cross_domain_f1) / len(cross_domain_f1)
                if cross_domain_f1 else 0.0
            )

            model_summary = {
                'avg_in_domain_f1':    avg_in_domain_f1,
                'avg_cross_domain_f1': avg_cross_domain_f1,
                'in_domain_f1_scores': in_domain_f1,
            }

            # ── perturbation F1 ───────────────────────────────────────────
            # Structure: results['perturbation'][domain]['semantic'][level]
            if pert_block:
                perturbation_f1 = []
                for domain, domain_block in pert_block.items():
                    if domain == 'clean':
                        continue
                    semantic = domain_block.get('semantic', {})
                    for level in ['low', 'medium', 'high']:
                        lvl = semantic.get(level, {})
                        if lvl:
                            perturbation_f1.append(
                                float(lvl.get('metrics', {}).get('macro_f1', 0))
                            )
                model_summary['avg_perturbation_f1'] = (
                    sum(perturbation_f1) / len(perturbation_f1)
                    if perturbation_f1 else 0.0
                )

            summary['model_summaries'][model_name] = model_summary

        summary['statistical_analysis'] = statistical_analysis

        summary_file = os.path.join(self.output_dir, 'evaluation_summary.json')
        file_manager.save_json(summary, summary_file)
        logger.info(f"Summary saved to {summary_file}")

        logger.info("\nEvaluation Summary:")
        for model_name, metrics in summary['model_summaries'].items():
            logger.info(f"  {model_name.upper()}:")
            logger.info(
                f"    Avg In-Domain Macro-F1 (subset):  "
                f"{metrics['avg_in_domain_f1']:.4f}"
            )
            logger.info(
                f"    Avg Cross-Domain Macro-F1:        "
                f"{metrics['avg_cross_domain_f1']:.4f}"
            )
            if 'avg_perturbation_f1' in metrics:
                logger.info(
                    f"    Avg Perturbation Macro-F1:        "
                    f"{metrics['avg_perturbation_f1']:.4f}"
                )

        return summary

    def run(
        self,
        models: List[str] = ['base'],
        skip_perturbation: bool = False,
        skip_statistics: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.

        Args:
            models: List of model names to evaluate
            skip_perturbation: Whether to skip perturbation testing
            skip_statistics: Skip the CPU-bound Bayesian MCMC step (run it later
                on a CPU machine from the saved complete_evaluation.json files)

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("\nStarting Evaluation Pipeline")
        logger.info(f"Models: {models}, Skip perturbation: {skip_perturbation}")

        total_timer = Timer()
        total_timer.start()

        data = self.load_data()

        if 'all' in models:
            models = ['base', 'large', 'lite']

        all_model_trainers = self.load_models(models)

        if not all_model_trainers:
            logger.error("No models loaded. Please train models first using train_pipeline.py")
            return {'status': 'failed', 'reason': 'no_models_loaded'}

        all_results = {}
        for model_name, domain_trainers in all_model_trainers.items():
            try:
                results = self.evaluate_single_model(
                    model_name=model_name,
                    domain_trainers=domain_trainers,
                    test_data_by_domain=data['test_data_by_domain'],
                    skip_perturbation=skip_perturbation
                )
                all_results[model_name] = results
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                traceback.print_exc()

        if not all_results:
            # Every model failed: do NOT report success (exit code must be non-zero).
            logger.error("All requested models failed to evaluate; see errors above.")
            return {'status': 'failed', 'reason': 'all_models_failed'}

        if skip_statistics:
            logger.info("Skipping statistical analysis (--skip-statistics).")
            statistical_analysis = {'status': 'skipped', 'reason': 'skip_statistics_flag'}
        else:
            statistical_analysis = self.perform_statistical_analysis(all_results)
        summary = self.generate_summary(all_results, statistical_analysis)

        total_time = total_timer.stop()
        logger.info(f"\nPipeline completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")

        return {
            'status': 'success',
            'results': all_results,
            'statistical_analysis': statistical_analysis,
            'summary': summary
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained IndoBERT models for clickbait detection'
    )

    parser.add_argument(
        '--model', type=str, default='base',
        choices=['base', 'large', 'lite', 'all'],
        help='Model variant to evaluate (default: base)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory containing trained model checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--dataset-dir', type=str, default='dataset',
        help='Directory containing CSV files (default: dataset)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='evaluation_results',
        help='Directory for evaluation results (default: evaluation_results)'
    )
    parser.add_argument(
        '--train-output-dir', type=str, default='output',
        help='Directory written by train_pipeline.py containing data_splits/ (default: output)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for evaluation (default: auto)'
    )
    parser.add_argument(
        '--skip-perturbation', action='store_true',
        help='Skip perturbation robustness testing (faster evaluation)'
    )
    parser.add_argument(
        '--skip-statistics', action='store_true',
        help='Skip the Bayesian MCMC analysis (CPU-bound; do not pay GPU rates for it)'
    )
    parser.add_argument(
        '--amp-mode', type=str, default=None,
        choices=['auto', 'bf16', 'fp16', 'off'],
        help='Mixed precision on CUDA (default: config.system.AMP_MODE = auto -> bf16 on A100, fp16 on T4)'
    )
    parser.add_argument(
        '--perturbation-output-dir', type=str,
        default='/content/drive/MyDrive/thesis/results',
        help='Root directory for perturbed CSVs (written to <root>/<model>). '
             'Use a separate root per scenario to protect the baseline.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--thesaurus-path', type=str, default=None,
        help='Path to dict.json thesaurus file. '
             'Defaults to <dataset-dir>/data/dict.json (committed in the repo). '
             'Override with e.g. /content/dict.json on Colab if using a Drive copy.'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable comprehensive debug logging for all pipeline stages '
             '(also enabled via DEBUG_PIPELINE=1 env var)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()

        # ── activate debug logging if requested ────────────────────────────
        if args.debug:
            debug_logger.set_debug(True)
            logger.info("[debug_logger] Debug output ENABLED for this run")

        if args.amp_mode:
            config.system.AMP_MODE = args.amp_mode

        pipeline = EvaluationPipeline(
            checkpoint_dir=args.checkpoint_dir,
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            train_output_dir=args.train_output_dir,
            device=args.device,
            random_seed=args.seed,
            thesaurus_path=args.thesaurus_path,
            perturbation_output_dir=args.perturbation_output_dir,
        )

        models = [args.model] if args.model != 'all' else ['base', 'large', 'lite']

        results = pipeline.run(
            models=models,
            skip_perturbation=args.skip_perturbation,
            skip_statistics=args.skip_statistics,
        )

        if results['status'] == 'success':
            logger.info("Evaluation pipeline completed successfully")
            sys.exit(0)
        else:
            logger.error(f"Evaluation pipeline failed: {results.get('reason', 'unknown')}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nEvaluation pipeline failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()