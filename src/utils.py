"""
Utility Module for IndoBERT Clickbait Detection System

This module provides common helper functions, logging utilities, file I/O operations,
and reproducibility helpers used across the entire system.
"""

import os
import json
import random
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime


class FileManager:
    """Handles file I/O operations with error handling."""
    
    @staticmethod
    def ensure_directory(directory: str) -> None:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            directory: Path to directory
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
        """
        Save dictionary to JSON file.
        
        Args:
            data: Dictionary to save
            filepath: Path to output file
            indent: JSON indentation level
        """
        FileManager.ensure_directory(os.path.dirname(filepath))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """
        Load dictionary from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary with loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
        """
        Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Path to output file
            index: Whether to save index
        """
        FileManager.ensure_directory(os.path.dirname(filepath))
        df.to_csv(filepath, index=index, encoding='utf-8')
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """
        Load DataFrame from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return pd.read_csv(filepath, encoding='utf-8')
    
    @staticmethod
    def save_text(text: str, filepath: str) -> None:
        """
        Save text to file.
        
        Args:
            text: Text content to save
            filepath: Path to output file
        """
        FileManager.ensure_directory(os.path.dirname(filepath))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
    
    @staticmethod
    def load_text(filepath: str) -> str:
        """
        Load text from file.
        
        Args:
            filepath: Path to text file
            
        Returns:
            Text content
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def file_exists(filepath: str) -> bool:
        """
        Check if file exists.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file exists, False otherwise
        """
        return os.path.exists(filepath)


class ReproducibilityHelper:
    """Ensures reproducibility across the system."""
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set random seeds for reproducibility.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        
        try:
            import torch
            torch.manual_seed(seed)
            
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.end_time = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = datetime.now()
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        self.end_time = datetime.now()
        return self.elapsed()
    
    def elapsed(self) -> float:
        """
        Get elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time else datetime.now()
        delta = end - self.start_time
        return delta.total_seconds()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, *args):
        """Context manager exit."""
        self.stop()


def get_timestamp(format_string: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as formatted string.
    
    Args:
        format_string: strftime format string
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format_string)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    
    return numerator / denominator


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = '',
    sep: str = '_'
) -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def batch_iterator(
    items: List[Any],
    batch_size: int
) -> List[List[Any]]:
    """
    Create batches from list of items.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    
    return batches


# Convenience instances used by train_pipeline.py and evaluate_pipeline.py
file_manager = FileManager()
reproducibility = ReproducibilityHelper()

# ──────────────────────────────────────────────────────────────────────────────
# JSON serialisation helper (shared across evaluation_engine and error_analyzer)
# ──────────────────────────────────────────────────────────────────────────────

def make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serialisable form.

    Handles:
    - dict  : converts tuple keys ``(a, b)`` to ``"a->b"`` strings
    - numpy : np.integer → int, np.floating → float, np.ndarray → list
    - list / tuple : recurses element-wise

    Args:
        obj: Any Python object to serialise

    Returns:
        A JSON-safe equivalent of *obj*
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            key = f"{k[0]}->{k[1]}" if isinstance(k, tuple) else k
            new_dict[key] = make_json_serializable(v)
        return new_dict
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    return obj
