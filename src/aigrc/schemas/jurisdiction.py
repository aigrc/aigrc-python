"""
Jurisdiction and regulatory classification schemas.

Corresponds to TypeScript: JurisdictionClassificationSchema in @aigrc/core/schemas
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk levels aligned with EU AI Act."""

    MINIMAL = "minimal"
    LIMITED = "limited"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"


class ControlStatus(str, Enum):
    """Implementation status of a control."""

    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_APPLICABLE = "not_applicable"


class Control(BaseModel):
    """A governance control requirement."""

    id: str = Field(description="Unique control identifier")
    name: str = Field(description="Human-readable control name")
    description: str | None = Field(default=None, description="Control description")
    status: ControlStatus = Field(
        default=ControlStatus.NOT_IMPLEMENTED,
        description="Implementation status",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Links or references to implementation evidence",
    )
    notes: str | None = Field(default=None, description="Additional notes")

    model_config = {"frozen": True, "extra": "forbid"}


class JurisdictionClassification(BaseModel):
    """
    Classification under a specific regulatory framework.

    Supports multiple frameworks:
    - EU AI Act
    - US OMB M-24-10
    - NIST AI RMF
    - ISO 42001
    """

    framework: Literal["eu_ai_act", "us_omb_m24", "nist_ai_rmf", "iso_42001"] = Field(
        description="Regulatory framework identifier"
    )
    risk_level: RiskLevel = Field(description="Risk level under this framework")
    category: str | None = Field(
        default=None,
        description="Framework-specific category (e.g., 'Annex III' for EU AI Act)",
    )
    controls: list[Control] = Field(
        default_factory=list,
        description="Required controls and their status",
    )
    compliance_percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage of controls implemented",
    )
    last_assessed: str | None = Field(
        default=None,
        description="ISO 8601 timestamp of last assessment",
    )
    assessor: str | None = Field(
        default=None,
        description="Person or system that performed assessment",
    )
    notes: str | None = Field(default=None, description="Additional notes")

    model_config = {
        "frozen": True,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "framework": "eu_ai_act",
                    "risk_level": "high",
                    "category": "Annex III",
                    "controls": [
                        {
                            "id": "AIA-HIGH-001",
                            "name": "Risk Management System",
                            "status": "implemented",
                        }
                    ],
                    "compliance_percentage": 85.0,
                    "last_assessed": "2024-01-15T10:30:00Z",
                    "assessor": "compliance@example.com",
                }
            ]
        },
    }

    def update_compliance_percentage(self) -> "JurisdictionClassification":
        """
        Recalculate compliance percentage based on control statuses.

        Returns:
            New JurisdictionClassification with updated percentage
        """
        if not self.controls:
            return self

        applicable = [c for c in self.controls if c.status != ControlStatus.NOT_APPLICABLE]
        if not applicable:
            return self.model_copy(update={"compliance_percentage": 100.0})

        implemented = sum(
            1 for c in applicable if c.status == ControlStatus.IMPLEMENTED
        )
        partial = sum(1 for c in applicable if c.status == ControlStatus.PARTIAL)

        percentage = ((implemented + partial * 0.5) / len(applicable)) * 100
        return self.model_copy(update={"compliance_percentage": round(percentage, 1)})
