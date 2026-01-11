"""
Framework-specific detection patterns.

This module contains patterns organized by language/ecosystem:
- python.py: Python AI/ML frameworks
- model_files.py: Model file extensions
"""

from aigrc.detection.scanner import (
    FrameworkCategory,
    FrameworkType,
    MODEL_EXTENSIONS,
)

__all__ = [
    "FrameworkType",
    "FrameworkCategory",
    "MODEL_EXTENSIONS",
]
