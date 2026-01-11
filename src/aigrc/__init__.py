"""
AIGRC - AI Governance, Risk, and Compliance toolkit for Python.

This package provides tools for managing AI governance, including:
- Schema validation for AI asset cards
- Framework detection engine
- Risk classification
- Golden Thread protocol
- Policy engine
- Command-line interface
"""

from aigrc.schemas import (
    ApprovalStatus,
    Artifact,
    AssetCard,
    Control,
    ControlStatus,
    GoldenThread,
    GovernanceStatus,
    JurisdictionClassification,
    Owner,
    RiskFactors,
    RiskLevel,
    Technical,
)
from aigrc.golden_thread import (
    compute_golden_thread_hash,
    verify_golden_thread_hash,
)
from aigrc.asset_card import (
    create_asset_card,
    generate_id,
    load_asset_card,
    save_asset_card,
    find_asset_cards,
)
from aigrc.detection import (
    Detection,
    DetectionConfidence,
    FrameworkCategory,
    FrameworkType,
    PatternRegistry,
    ScanResult,
    Scanner,
    suggest_asset_card,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Schemas
    "ApprovalStatus",
    "Artifact",
    "AssetCard",
    "Control",
    "ControlStatus",
    "GoldenThread",
    "GovernanceStatus",
    "JurisdictionClassification",
    "Owner",
    "RiskFactors",
    "RiskLevel",
    "Technical",
    # Golden Thread
    "compute_golden_thread_hash",
    "verify_golden_thread_hash",
    # Asset Card CRUD
    "create_asset_card",
    "generate_id",
    "load_asset_card",
    "save_asset_card",
    "find_asset_cards",
    # Detection Engine
    "Scanner",
    "ScanResult",
    "Detection",
    "DetectionConfidence",
    "FrameworkType",
    "FrameworkCategory",
    "PatternRegistry",
    "suggest_asset_card",
]
