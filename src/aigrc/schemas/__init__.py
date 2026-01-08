"""
Pydantic v2 schemas for AIGRC.

These schemas are designed to be compatible with the TypeScript Zod schemas
in @aigrc/core. All field names and validation rules must match exactly.
"""

from aigrc.schemas.asset_card import (
    ApprovalStatus,
    Artifact,
    AssetCard,
    GoldenThread,
    GovernanceStatus,
)
from aigrc.schemas.risk_factors import RiskFactors
from aigrc.schemas.jurisdiction import (
    Control,
    ControlStatus,
    JurisdictionClassification,
    RiskLevel,
)
from aigrc.schemas.owner import Owner
from aigrc.schemas.technical import Technical

__all__ = [
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
]
