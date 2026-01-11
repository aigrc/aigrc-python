"""
Detection strategies for framework detection.

Each strategy implements a specific detection approach:
- ImportAnalysisStrategy: Detect via import statements
- PatternMatchingStrategy: Detect via code patterns
- AnnotationStrategy: Detect AIGRC annotations
"""

from aigrc.detection.patterns import (
    AnnotationStrategy,
    DetectionStrategy,
    ImportAnalysisStrategy,
    PatternMatchingStrategy,
)

__all__ = [
    "DetectionStrategy",
    "ImportAnalysisStrategy",
    "PatternMatchingStrategy",
    "AnnotationStrategy",
]
