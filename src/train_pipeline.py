"""
Training Pipeline for IndoBERT Clickbait Detection System

This script orchestrates the complete training process:
1. Loads and prepares data from 5 CSV files
2. Trains IndoBERT models (base/large/lite)
3. Performs hyperparameter optimization (optional)
4. Saves model checkpoints
5. Tracks training progress

Usage:
    # Basic training (single model)
    python train_pipeline.py --model base
    
    # Train all models
    python train_pipeline.py --model all
    
    # With hyperparameter search
    python train_pipeline.py --model base --grid-search
    
    # On Colab
    !python train_pipeline.py --model base --device cuda
"""

import os
import sys
import argparse
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

from config import config
from data_manager import DataManager
from model_trainer import ModelTrainer, HyperparameterSearch, _copy_to_drive
from utils import file_manager, reproducibility, Timer, get_timestamp
import debug_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the complete training pipeline for IndoBERT models."""
    
    def __init__(
        self,
        dataset_dir: str = "dataset",
        output_dir: str = "output",
        checkpoint_dir: str = "checkpoints",
        device: str = "auto",
        random_seed: int = 42
    ):
        """
        Initialize training pipeline.
        
        Args:
            dataset_dir: Directory containing CSV files
            output_dir: Directory for output files
            checkpoint_dir: Directory for model checkpoints
            device: Device to use ('cuda', 'cpu', or 'auto')
            random_seed: Random seed for reproducibility
        """
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.random_seed = random_seed
        
        reproducibility.set_seed(random_seed)
        file_manager.ensure_directory(output_dir)
        file_manager.ensure_directory(checkpoint_dir)
        
        logger.info("Training Pipeline initialized")
        logger.info(f"Dataset: {dataset_dir}, Output: {output_dir}, Checkpoints: {checkpoint_dir}")
        logger.info(f"Device: {device}, Random seed: {random_seed}")
    
    def load_and_prepare_data(self) -> Dict[str, Any]:
        """
        Load datasets, perform domain-isolated stratified splitting, 
        and reconstruct global DataFrames.
        """
        logger.info("\n[STEP 1] Loading and Preparing Data")
        
        with Timer() as timer:
            data_manager = DataManager.from_dataset_directory(
                dataset_dir=self.dataset_dir,
                tokenizer_name=config.model.DEFAULT_MODEL,
                max_length=config.model.MAX_SEQUENCE_LENGTH,
                random_seed=self.random_seed
            )
            
            summary = data_manager.get_summary()
            logger.info(f"Loaded {summary['total_samples']} samples from {len(summary['domains'])} domains")
            
            splits_dir = os.path.join(self.output_dir, 'data_splits')

            # 1. Get the isolated dictionary of splits.
            #
            # IMPORTANT: create the split ONCE and reuse it. When training the
            # second and third model variants, we must NOT re-split, because a
            # new split would change the test set the earlier models were
            # evaluated on. So: if data_splits/ already exists, load it;
            # otherwise split now and save it as the canonical split.
            if os.path.exists(splits_dir):
                logger.info(
                    f"Reusing existing data splits from {splits_dir} "
                    "(no re-splitting)."
                )
                domain_splits = data_manager.load_domain_splits(splits_dir)
                created_new_split = False
            else:
                logger.info(
                    "No existing data splits found. Creating the canonical "
                    f"split once and saving it to {splits_dir}."
                )
                domain_splits = data_manager.stratified_split_by_domain(
                    train_size=config.data.TRAIN_SIZE,
                    val_size=config.data.VAL_SIZE,
                    test_size=config.data.TEST_SIZE
                )
                # 2. Export the isolated domain splits to folders (once).
                data_manager.export_domain_splits(domain_splits, splits_dir)
                created_new_split = True

            # 3. Stitch them back together to create the Global DataFrames!
            train_df = pd.concat([splits['train'] for splits in domain_splits.values()]).reset_index(drop=True)
            val_df = pd.concat([splits['val'] for splits in domain_splits.values()]).reset_index(drop=True)
            test_df = pd.concat([splits['test'] for splits in domain_splits.values()]).reset_index(drop=True)

            # Export the global split CSVs ONLY when we just created the split.
            # If we reused an existing split, we must not overwrite these files.
            if created_new_split:
                train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
                val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
                test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
            
        logger.info(f"Data preparation completed in {timer.elapsed():.2f}s")
        
        # 4. Return BOTH the isolated splits (for training) and global splits (for summaries)
        return {
            'data_manager': data_manager,
            'domain_splits': domain_splits,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df
        }
    
    def train_single_model(
        self,
        model_name: str,
        domain: str, # ADDED: The domain name
        train_df: Any,
        val_df: Any,
        perform_grid_search: bool = False
    ) -> Dict[str, Any]:
        """Train a single IndoBERT model for a specific domain."""
        logger.info(f"\n[STEP 2] Training {model_name.upper()} Model for {domain.upper()}")
        
        model_mapping = {
            'base': config.model.INDOBERT_BASE,
            'large': config.model.INDOBERT_LARGE,
            'lite': config.model.INDOBERT_LITE
        }
        
        if model_name not in model_mapping:
            raise ValueError(f"Invalid model name: {model_name}. Choose from: {list(model_mapping.keys())}")
        
        full_model_name = model_mapping[model_name]
        
        with Timer() as timer:
            if perform_grid_search:
                logger.info("Performing hyperparameter grid search")
                results = self._train_with_grid_search(
                    full_model_name, model_name, domain, train_df, val_df
                )
            else:
                logger.info("Training with default hyperparameters")
                results = self._train_with_defaults(
                    full_model_name, model_name, domain, train_df, val_df
                )
        
        logger.info(
            f"Training completed for {domain} in {timer.elapsed():.2f}s | "
            f"Macro-F1={results['best_macro_f1']:.4f} | "
            f"F1_0={results['best_f1_0']:.4f} | "
            f"F1_1={results['best_f1_1']:.4f}"
        )
        
        return results
    
    def _train_with_defaults(
        self,
        full_model_name: str,
        model_name: str,
        domain: str,
        train_df: Any,
        val_df: Any
    ) -> Dict[str, Any]:
        """Train model with default hyperparameters.

        Checkpoints are written to a local staging directory on /content/ (fast
        local SSD) during training.  Only after the full domain training run
        completes are the files copied once to the Drive destination.  This
        keeps all per-epoch torch.save() calls off the Drive FUSE mount.
        """
        trainer = ModelTrainer(
            model_name=full_model_name,
            max_length=config.model.MAX_SEQUENCE_LENGTH,
            device=self.device,
            random_seed=self.random_seed
        )

        # Local staging dir — fast local SSD, no FUSE/Drive involved.
        # Falls back to a system temp directory when /content/ is not available
        # (i.e. when running outside Google Colab).
        _staging_root = (
            "/content/checkpoints_staging"
            if os.path.isdir("/content")
            else os.path.join(os.path.dirname(self.checkpoint_dir), "checkpoints_staging")
        )
        local_checkpoint_dir = os.path.join(_staging_root, model_name, domain)
        file_manager.ensure_directory(local_checkpoint_dir)

        # Final Drive destination (may be the same path when running locally)
        drive_checkpoint_dir = os.path.join(self.checkpoint_dir, model_name, domain)
        file_manager.ensure_directory(drive_checkpoint_dir)

        history = trainer.train(
            train_df=train_df,
            val_df=val_df,
            num_epochs=config.training.NUM_EPOCHS,
            batch_size=config.training.BATCH_SIZE,
            learning_rate=config.training.LEARNING_RATE,
            weight_decay=config.training.WEIGHT_DECAY,
            dropout_rate=config.model.DROPOUT_RATE,
            early_stopping_patience=config.training.EARLY_STOPPING_PATIENCE,
            checkpoint_dir=local_checkpoint_dir
        )

        history_file = os.path.join(local_checkpoint_dir, 'training_history.json')
        file_manager.save_json(history, history_file)

        # Copy all checkpoint files from local staging → Drive once, after
        # training completes.  One copy per domain instead of one per epoch.
        for filename in os.listdir(local_checkpoint_dir):
            src = os.path.join(local_checkpoint_dir, filename)
            dst = os.path.join(drive_checkpoint_dir, filename)
            if os.path.isfile(src):
                logger.info(f"Copying {filename} → Drive ({domain})")
                _copy_to_drive(src, dst)

        best_macro_f1 = max(history['val_macro_f1'])
        best_epoch = history['val_macro_f1'].index(best_macro_f1)

        return {
            'model_name': model_name,
            'domain': domain,
            'full_model_name': full_model_name,
            'history': history,
            
            'best_epoch': best_epoch + 1,

            'best_macro_f1': best_macro_f1,
            'best_f1_0': history['val_f1_0'][best_epoch],
            'best_f1_1': history['val_f1_1'][best_epoch],

            'checkpoint_dir': drive_checkpoint_dir,
            'hyperparameters': {
                'learning_rate': config.training.LEARNING_RATE,
                'batch_size': config.training.BATCH_SIZE,
                'num_epochs': config.training.NUM_EPOCHS,
                'dropout_rate': config.model.DROPOUT_RATE
            }
        }
    
    def _train_with_grid_search(
        self,
        full_model_name: str,
        model_name: str,
        domain: str,
        train_df: Any,
        val_df: Any
    ) -> Dict[str, Any]:
        """Train model with hyperparameter grid search for a specific domain.

        Same local-staging → Drive-copy pattern as _train_with_defaults.
        """
        # Local staging dir (same Colab-aware fallback as _train_with_defaults)
        _staging_root = (
            "/content/checkpoints_staging"
            if os.path.isdir("/content")
            else os.path.join(os.path.dirname(self.checkpoint_dir), "checkpoints_staging")
        )
        local_checkpoint_dir = os.path.join(_staging_root, model_name, domain)
        file_manager.ensure_directory(local_checkpoint_dir)

        # Final Drive destination
        drive_checkpoint_dir = os.path.join(self.checkpoint_dir, model_name, domain)
        file_manager.ensure_directory(drive_checkpoint_dir)

        # 2. Run Grid Search (results saved locally)
        grid_search = HyperparameterSearch(
            model_name=full_model_name,
            param_grid=config.training.GRID_SEARCH_PARAMS,
            max_length=config.model.MAX_SEQUENCE_LENGTH,
            device=self.device,
            random_seed=self.random_seed
        )

        search_results = grid_search.search(
            train_df=train_df,
            val_df=val_df,
            num_epochs=config.training.NUM_EPOCHS
        )

        search_file = os.path.join(local_checkpoint_dir, 'grid_search_results.json')
        grid_search.save_results(search_file)

        logger.info(f"Training final {domain} specialist with best params: {search_results['best_params']}")

        # 3. Train the final specialist model (local staging)
        trainer = ModelTrainer(
            model_name=full_model_name,
            max_length=config.model.MAX_SEQUENCE_LENGTH,
            device=self.device,
            random_seed=self.random_seed
        )

        history = trainer.train(
            train_df=train_df,
            val_df=val_df,
            num_epochs=config.training.NUM_EPOCHS,
            checkpoint_dir=local_checkpoint_dir,
            **search_results['best_params']
        )

        file_manager.save_json(history, os.path.join(local_checkpoint_dir, 'training_history.json'))

        # Copy all files from local staging → Drive once
        for filename in os.listdir(local_checkpoint_dir):
            src = os.path.join(local_checkpoint_dir, filename)
            dst = os.path.join(drive_checkpoint_dir, filename)
            if os.path.isfile(src):
                logger.info(f"Copying {filename} → Drive ({domain})")
                _copy_to_drive(src, dst)
		
        best_macro_f1 = max(history['val_macro_f1'])
        best_epoch = history['val_macro_f1'].index(best_macro_f1)

        return {
            'model_name': model_name,
            'domain': domain,
            'best_epoch': best_epoch + 1,
            'best_macro_f1': max(history['val_macro_f1']),
            'best_f1_0': history['val_f1_0'][best_epoch],
            'best_f1_1': history['val_f1_1'][best_epoch],
            'checkpoint_dir': drive_checkpoint_dir,
            'hyperparameters': search_results['best_params'],
            'grid_search_results': search_results
        }
    
    def save_pipeline_summary(self, results: Dict[str, Any]) -> None:
        """
        Save pipeline execution summary.

        Args:
            results: Dictionary with all training results
        """
        logger.info("\n[STEP 3] Saving Pipeline Summary")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset_dir": self.dataset_dir,
            "total_samples": results["data_info"]["total_samples"],
            "domains": results["data_info"]["domains"],
            "models_trained": list(results["models"].keys()),
            "training_results": {}
        }

        # Save training results for every model and every domain
        for model_name, domain_results in results["models"].items():

            summary["training_results"][model_name] = {}

            for domain_name, info in domain_results.items():

                summary["training_results"][model_name][domain_name] = {
                    "best_epoch": info["best_epoch"],
                    "best_macro_f1": info["best_macro_f1"],
                    "best_f1_0": info["best_f1_0"],
                    "best_f1_1": info["best_f1_1"],
                    "checkpoint_dir": info["checkpoint_dir"],
                    "hyperparameters": info["hyperparameters"]
                }

        summary_file = os.path.join(self.output_dir, "training_summary.json")
        file_manager.save_json(summary, summary_file)

        logger.info(f"Summary saved to {summary_file}")
        logger.info("\nTraining Summary:")
        logger.info(f"  Total samples: {summary['total_samples']}")
        logger.info(f"  Domains: {', '.join(summary['domains'])}")
        logger.info(f"  Models trained: {', '.join(summary['models_trained'])}")
        logger.info("  Best Macro-F1 Scores:")
        for model_name, domain_results in summary["training_results"].items():
            logger.info(f"    {model_name}:")
            for domain_name, info in domain_results.items():
                logger.info(
                    f"      {domain_name}: "
                    f"Epoch={info['best_epoch']}, "
                    f"Macro-F1={info['best_macro_f1']:.4f}, "
                    f"F1_0={info['best_f1_0']:.4f}, "
                    f"F1_1={info['best_f1_1']:.4f}"
                )
    
    def run(self, models: List[str] = ['base'], perform_grid_search: bool = False) -> Dict[str, Any]:
        logger.info("\nStarting SPECIALIST Training Pipeline")
        total_timer = Timer()
        total_timer.start()
        
        # 1. Load the full dataset first
        data_results = self.load_and_prepare_data()
        
        if 'all' in models:
            models = ['base', 'large', 'lite']
        
        training_results = {}

        # 2. Get the unique domains from your data
        domains = data_results['data_manager'].domains
        domain_splits = data_results['domain_splits']  # isolated splits

        for model_name in models:
            training_results[model_name] = {}
            failed_domains = []  # collect per-domain failures; raise at end

            for domain in domains:
                logger.info(f"\n--- Training Specialist: {model_name} for Domain: {domain} ---")

                # 3. Pull directly from the new dictionary structure!
                train_df_domain = domain_splits[domain]['train']
                val_df_domain = domain_splits[domain]['val']

                # 4. Train the specialist
                try:
                    model_results = self.train_single_model(
                        model_name=model_name,
                        domain=domain,
                        train_df=train_df_domain,
                        val_df=val_df_domain,
                        perform_grid_search=perform_grid_search
                    )
                    training_results[model_name][domain] = model_results
                except Exception as e:
                    logger.error(f"Failed to train {model_name} on {domain}: {str(e)}")
                    traceback.print_exc()
                    failed_domains.append((domain, str(e)))

            # Raise after the full model loop so all domains are attempted, but
            # the pipeline does NOT silently continue to evaluation with gaps.
            if failed_domains:
                missing = ", ".join(d for d, _ in failed_domains)
                raise RuntimeError(
                    f"Training failed for {model_name} on domain(s): {missing}. "
                    "Fix checkpoint save errors before running evaluate_pipeline.py. "
                    f"Details: {failed_domains}"
                )
        
        results = {
            'data_info': {
                'total_samples': data_results['data_manager'].get_summary()['total_samples'],
                'domains': domains
            },
            'models': training_results
        }
        
        self.save_pipeline_summary(results)
        
        total_time = total_timer.stop()
        logger.info(f"\nPipeline completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        return results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train IndoBERT models for clickbait detection'
    )
    
    parser.add_argument(
        '--model', type=str, default='base',
        choices=['base', 'large', 'lite', 'all'],
        help='Model variant to train (default: base)'
    )
    parser.add_argument(
        '--dataset-dir', type=str, default='dataset',
        help='Directory containing CSV files (default: dataset)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='output',
        help='Directory for output files (default: output)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory for model checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for training (default: auto)'
    )
    parser.add_argument(
        '--grid-search', action='store_true',
        help='Perform hyperparameter grid search'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
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
        
        pipeline = TrainingPipeline(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            random_seed=args.seed
        )
        
        models = [args.model] if args.model != 'all' else ['base', 'large', 'lite']
        
        results = pipeline.run(
            models=models,
            perform_grid_search=args.grid_search
        )
        
        logger.info("Training pipeline completed successfully")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTraining pipeline failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
