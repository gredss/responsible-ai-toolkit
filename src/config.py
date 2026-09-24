"""
Configuration Module for IndoBERT Clickbait Detection System

This module centralizes all configuration settings including model hyperparameters,
training configuration, evaluation settings, perturbation parameters, and file paths.
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for model architecture and variants."""
    
    # Model variants
    INDOBERT_BASE: str = "indobenchmark/indobert-base-p1"
    INDOBERT_LARGE: str = "indobenchmark/indobert-large-p1"
    INDOBERT_LITE: str = "indobenchmark/indobert-lite-base-p1"
    
    # Default model
    DEFAULT_MODEL: str = INDOBERT_BASE
    
    # Model architecture
    NUM_LABELS: int = 2
    MAX_SEQUENCE_LENGTH: int = 128
    DROPOUT_RATE: float = 0.1
    
    # Available models
    AVAILABLE_MODELS: List[str] = field(default_factory=lambda: [
        "indobenchmark/indobert-base-p1",
        "indobenchmark/indobert-large-p1",
        "indobenchmark/indobert-lite-base-p1"
    ])


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Hyperparameters
    LEARNING_RATE: float = 2e-5
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 4
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.1
    
    # Optimizer
    OPTIMIZER: str = "AdamW"
    EPSILON: float = 1e-8
    MAX_GRAD_NORM: float = 1.0
    
    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 2
    EARLY_STOPPING_METRIC: str = "f1"
    
    # Learning rate scheduler
    SCHEDULER_TYPE: str = "linear"
    
    # Grid search parameters
    GRID_SEARCH_PARAMS: Dict[str, List[Any]] = field(default_factory=lambda: {
        'learning_rate': [1e-5, 2e-5, 3e-5],
        'batch_size': [16, 32],
        'dropout_rate': [0.1, 0.2]
    })


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Data splits
    TRAIN_SIZE: float = 0.70
    VAL_SIZE: float = 0.15
    TEST_SIZE: float = 0.15
    
    # Domains
    DOMAINS: List[str] = field(default_factory=lambda: [
        'Technology',
        'Politics',
        'Health',
        'Sport',
        'Education'
    ])
    
    # Domain file mapping
    DOMAIN_FILES: Dict[str, str] = field(default_factory=lambda: {
        'Technology': 'technology.csv',
        'Politics': 'politic.csv',
        'Health': 'health.csv',
        'Sport': 'sport.csv',
        'Education': 'education.csv'
    })
    
    # CSV schema
    REQUIRED_COLUMNS: List[str] = field(default_factory=lambda: [
        'id', 'source', 'date', 'title', 'Label', 'url'
    ])
    
    TEXT_COLUMN: str = 'title'
    LABEL_COLUMN: str = 'Label'
    
    # Data validation
    MIN_TEXT_LENGTH: int = 10
    MAX_TEXT_LENGTH: int = 500
    BALANCE_THRESHOLD: float = 0.3


@dataclass
class PerturbationConfig:
    """Configuration for text perturbations."""
    
    # Perturbation levels
    LEVELS: List[str] = field(default_factory=lambda: ['low', 'medium', 'high'])
    
    # Low-level perturbation (character-level typos)
    LOW_INTENSITY_MIN: float = 0.05
    LOW_INTENSITY_MAX: float = 0.10
    LOW_PERTURBATION_TYPES: List[str] = field(default_factory=lambda: [
        'substitute', 'delete', 'insert', 'swap'
    ])
    
    # Medium-level perturbation (informal language)
    MEDIUM_INTENSITY_MIN: float = 0.15
    MEDIUM_INTENSITY_MAX: float = 0.25
    MEDIUM_SLANG_PROBABILITY: float = 0.3
    MEDIUM_ABBREVIATION_PROBABILITY: float = 0.5
    
    # High-level perturbation (synonym replacement)
    HIGH_INTENSITY_MIN: float = 0.40
    HIGH_INTENSITY_MAX: float = 0.60
    HIGH_STRUCTURE_MODIFICATION_PROBABILITY: float = 0.3
    
    # Perturbation reproducibility
    APPLY_SEED: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Metrics
    PRIMARY_METRIC: str = "f1"
    METRICS: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1'
    ])
    
    # Evaluation types
    EVALUATION_TYPES: List[str] = field(default_factory=lambda: [
        'in_domain',
        'cross_domain',
        'perturbation',
        'cross_domain_perturbation'
    ])
    
    # Cross-domain evaluation
    CROSS_DOMAIN_MATRIX_SIZE: int = 5  # 5x5 matrix
    
    # Robustness metrics
    CALCULATE_SOURCE_DROP: bool = True
    CALCULATE_TARGET_DROP: bool = True
    CALCULATE_DEGRADATION: bool = True
    
    # Performance thresholds
    GOOD_PERFORMANCE_THRESHOLD: float = 0.80
    ACCEPTABLE_PERFORMANCE_THRESHOLD: float = 0.70
    POOR_PERFORMANCE_THRESHOLD: float = 0.60


@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis."""
    
    # Bayesian analysis
    ROPE_THRESHOLD: float = 0.01  # 1% practical equivalence
    CREDIBLE_INTERVAL_LEVELS: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    
    # Significance testing
    ALPHA: float = 0.05
    SIGNIFICANCE_TESTS: List[str] = field(default_factory=lambda: [
        'bayesian_signed_rank',
        'paired_t_test',
        'mann_whitney_u'
    ])
    
    # Bootstrap
    N_BOOTSTRAP_SAMPLES: int = 10000
    BOOTSTRAP_CONFIDENCE_LEVEL: float = 0.95
    
    # Effect size thresholds
    SMALL_EFFECT_SIZE: float = 0.2
    MEDIUM_EFFECT_SIZE: float = 0.5
    LARGE_EFFECT_SIZE: float = 0.8


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    
    # Base directories
    BASE_DIR: str = os.getcwd()
    DATA_DIR: str = "dataset"
    OUTPUT_DIR: str = "output"
    
    # Model directories
    CHECKPOINT_DIR: str = "checkpoints"
    MODEL_CACHE_DIR: str = "model_cache"
    
    # Evaluation directories
    EVALUATION_DIR: str = "evaluation_results"
    STATISTICAL_DIR: str = "statistical_analysis"
    
    # Export directories
    REPORTS_DIR: str = "reports"
    FIGURES_DIR: str = "figures"
    LOGS_DIR: str = "logs"
    
    # Data files
    TRAIN_FILE: str = "train.csv"
    VAL_FILE: str = "val.csv"
    TEST_FILE: str = "test.csv"
    
    # Result files
    EVALUATION_RESULTS_FILE: str = "evaluation_results.json"
    STATISTICAL_RESULTS_FILE: str = "statistical_analysis.json"
    AGGREGATED_RESULTS_FILE: str = "aggregated_results.csv"
    SUMMARY_REPORT_FILE: str = "summary_report.txt"
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.DATA_DIR,
            self.OUTPUT_DIR,
            self.CHECKPOINT_DIR,
            self.MODEL_CACHE_DIR,
            self.EVALUATION_DIR,
            self.STATISTICAL_DIR,
            self.REPORTS_DIR,
            self.FIGURES_DIR,
            self.LOGS_DIR
        ]
        
        for directory in directories:
            full_path = os.path.join(self.BASE_DIR, directory)
            os.makedirs(full_path, exist_ok=True)


@dataclass
class DashboardConfig:
    """Configuration for dashboard application."""
    
    # Dashboard settings
    PAGE_TITLE: str = "IndoBERT Clickbait Detection Dashboard"
    PAGE_ICON: str = "🔍"
    LAYOUT: str = "wide"
    INITIAL_SIDEBAR_STATE: str = "expanded"
    
    # Visualization settings
    HEATMAP_COLORSCALE: str = "RdYlGn"
    DEFAULT_FIGURE_HEIGHT: int = 400
    DEFAULT_FIGURE_WIDTH: int = 700
    
    # Color schemes
    DEGRADATION_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'clean': '#2ecc71',
        'low': '#f39c12',
        'medium': '#e67e22',
        'high': '#e74c3c'
    })
    
    PERFORMANCE_COLORS: Dict[str, str] = field(default_factory=lambda: {
        'good': '#2ecc71',
        'acceptable': '#f39c12',
        'poor': '#e74c3c'
    })
    
    # Cache settings
    ENABLE_MODEL_CACHE: bool = True
    ENABLE_DATA_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour in seconds


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    # Logging levels
    LOG_LEVEL: str = "INFO"
    FILE_LOG_LEVEL: str = "DEBUG"
    CONSOLE_LOG_LEVEL: str = "INFO"
    
    # Log format
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # Log files
    MAIN_LOG_FILE: str = "main.log"
    TRAINING_LOG_FILE: str = "training.log"
    EVALUATION_LOG_FILE: str = "evaluation.log"
    ERROR_LOG_FILE: str = "error.log"
    
    # Log rotation
    MAX_LOG_SIZE: int = 10 * 1024 * 1024  # 10 MB
    BACKUP_COUNT: int = 5


@dataclass
class SystemConfig:
    """Configuration for system settings."""
    
    # Reproducibility
    RANDOM_SEED: int = 42
    DETERMINISTIC: bool = True
    
    # Device settings
    USE_CUDA: bool = True
    CUDA_DEVICE: int = 0
    NUM_WORKERS: int = 0  # For DataLoader
    
    # Performance
    MIXED_PRECISION: bool = False
    GRADIENT_ACCUMULATION_STEPS: int = 1
    
    # Memory management
    EMPTY_CACHE_FREQUENCY: int = 100  # Empty CUDA cache every N batches
    MAX_MEMORY_ALLOCATED: float = 0.9  # Maximum GPU memory usage (90%)


class Config:
    """
    Main configuration class that aggregates all configuration components.
    """
    
    def __init__(self):
        """Initialize all configuration components."""
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.perturbation = PerturbationConfig()
        self.evaluation = EvaluationConfig()
        self.statistical = StatisticalConfig()
        self.paths = PathConfig()
        self.dashboard = DashboardConfig()
        self.logging = LoggingConfig()
        self.system = SystemConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary with all configuration settings
        """
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'perturbation': self.perturbation.__dict__,
            'evaluation': self.evaluation.__dict__,
            'statistical': self.statistical.__dict__,
            'paths': {k: v for k, v in self.paths.__dict__.items() if not k.startswith('get_')},
            'dashboard': self.dashboard.__dict__,
            'logging': self.logging.__dict__,
            'system': self.system.__dict__
        }
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration
        """
        import json
        
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Config instance with loaded settings
        """
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # Update configuration from loaded dictionary
        for section, values in config_dict.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        return config
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Validate data splits sum to 1.0
        total_split = self.data.TRAIN_SIZE + self.data.VAL_SIZE + self.data.TEST_SIZE
        if not (0.99 <= total_split <= 1.01):
            return False
        
        # Validate perturbation intensities
        if not (0 < self.perturbation.LOW_INTENSITY_MIN < self.perturbation.LOW_INTENSITY_MAX < 1):
            return False
        
        if not (0 < self.perturbation.MEDIUM_INTENSITY_MIN < self.perturbation.MEDIUM_INTENSITY_MAX < 1):
            return False
        
        if not (0 < self.perturbation.HIGH_INTENSITY_MIN < self.perturbation.HIGH_INTENSITY_MAX < 1):
            return False
        
        # Validate thresholds
        if not (0 < self.statistical.ALPHA < 1):
            return False
        
        if not (0 < self.statistical.ROPE_THRESHOLD < 1):
            return False
        
        # Validate performance thresholds
        if not (0 < self.evaluation.POOR_PERFORMANCE_THRESHOLD < 
                self.evaluation.ACCEPTABLE_PERFORMANCE_THRESHOLD < 
                self.evaluation.GOOD_PERFORMANCE_THRESHOLD < 1):
            return False
        
        return True
    
    def setup_environment(self) -> None:
        """Setup environment based on configuration."""
        import random
        import numpy as np
        import torch
        
        # Set random seeds for reproducibility
        if self.system.DETERMINISTIC:
            random.seed(self.system.RANDOM_SEED)
            np.random.seed(self.system.RANDOM_SEED)
            torch.manual_seed(self.system.RANDOM_SEED)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.system.RANDOM_SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        # Create necessary directories
        self.paths.create_directories()
    
    def get_summary(self) -> str:
        """
        Get configuration summary.
        
        Returns:
            String with configuration summary
        """
        lines = [
            "=" * 80,
            "CONFIGURATION SUMMARY",
            "=" * 80,
            "",
            "MODEL CONFIGURATION:",
            f"  Default Model: {self.model.DEFAULT_MODEL}",
            f"  Max Sequence Length: {self.model.MAX_SEQUENCE_LENGTH}",
            f"  Dropout Rate: {self.model.DROPOUT_RATE}",
            "",
            "TRAINING CONFIGURATION:",
            f"  Learning Rate: {self.training.LEARNING_RATE}",
            f"  Batch Size: {self.training.BATCH_SIZE}",
            f"  Epochs: {self.training.NUM_EPOCHS}",
            f"  Early Stopping Patience: {self.training.EARLY_STOPPING_PATIENCE}",
            "",
            "DATA CONFIGURATION:",
            f"  Domains: {', '.join(self.data.DOMAINS)}",
            f"  Train/Val/Test Split: {self.data.TRAIN_SIZE}/{self.data.VAL_SIZE}/{self.data.TEST_SIZE}",
            "",
            "PERTURBATION CONFIGURATION:",
            f"  Levels: {', '.join(self.perturbation.LEVELS)}",
            f"  Low Intensity: {self.perturbation.LOW_INTENSITY_MIN}-{self.perturbation.LOW_INTENSITY_MAX}",
            f"  Medium Intensity: {self.perturbation.MEDIUM_INTENSITY_MIN}-{self.perturbation.MEDIUM_INTENSITY_MAX}",
            f"  High Intensity: {self.perturbation.HIGH_INTENSITY_MIN}-{self.perturbation.HIGH_INTENSITY_MAX}",
            "",
            "EVALUATION CONFIGURATION:",
            f"  Primary Metric: {self.evaluation.PRIMARY_METRIC}",
            f"  Good Performance Threshold: {self.evaluation.GOOD_PERFORMANCE_THRESHOLD}",
            "",
            "STATISTICAL CONFIGURATION:",
            f"  ROPE Threshold: {self.statistical.ROPE_THRESHOLD}",
            f"  Alpha: {self.statistical.ALPHA}",
            "",
            "SYSTEM CONFIGURATION:",
            f"  Random Seed: {self.system.RANDOM_SEED}",
            f"  Use CUDA: {self.system.USE_CUDA}",
            f"  Deterministic: {self.system.DETERMINISTIC}",
            "",
            "=" * 80
        ]
        
        return "\n".join(lines)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Return the global :class:`Config` singleton.

    All subsections are accessible via ``config.<section>``, e.g.::

        from config import config
        config.model.DEFAULT_MODEL
        config.training.LEARNING_RATE
    """
    return config