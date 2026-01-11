"""
Detection Engine for AI/ML framework detection.

Implements SPEC-SDK-PYTHON-AIGRC Section 4: Detection Engine.

Usage:
    from aigrc.detection import Scanner, ScanResult

    scanner = Scanner()
    result = await scanner.scan_directory(Path("."))
    print(f"Found {len(result.detections)} frameworks")
"""

from aigrc.detection.scanner import (
    Detection,
    DetectionConfidence,
    FrameworkCategory,
    FrameworkType,
    ScanResult,
    Scanner,
)
from aigrc.detection.patterns import (
    DetectionStrategy,
    FrameworkPattern,
    PatternRegistry,
)
from aigrc.detection.suggestions import suggest_asset_card

__all__ = [
    # Scanner
    "Scanner",
    "ScanResult",
    "Detection",
    "DetectionConfidence",
    "FrameworkType",
    "FrameworkCategory",
    # Patterns
    "PatternRegistry",
    "FrameworkPattern",
    "DetectionStrategy",
    # Suggestions
    "suggest_asset_card",
]
