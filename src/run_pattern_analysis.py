"""
Thin compatibility shim for the pattern-based robustness analysis.

The real implementation now lives in ``pattern_analysis.py`` (the analysis and
the checkpoint-wiring "runner" were consolidated into that single file). This
module is kept only so older entry points that do ``from run_pattern_analysis
import main`` keep working. It adds no logic of its own and has no CLI.

Notebook usage
--------------
    from run_pattern_analysis import main
    main(
        results_root="/content/drive/MyDrive/thesis/results/base",
        checkpoint_dir="/content/drive/MyDrive/thesis/checkpoints",
        model_name="base",
    )

Prefer calling ``pattern_analysis.main(...)`` directly in new code. This shim
can be removed once nothing depends on it.
"""

from __future__ import annotations

import os
import sys

# Make sure this folder (src/) is importable so "import pattern_analysis" works
# whether launched from the repo root or from inside src/.
THIS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
if THIS_DIRECTORY not in sys.path:
    sys.path.insert(0, THIS_DIRECTORY)

import pattern_analysis as pattern_analysis_module

# Re-export the consolidated entry point under this module's name.
main = pattern_analysis_module.main

__all__ = ["main"]
