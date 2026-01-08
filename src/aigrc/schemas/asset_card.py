"""
AssetCard schema - the fundamental unit of governance in AIGRC.

Corresponds to TypeScript: AssetCardSchema in @aigrc/core/schemas
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator

from aigrc.schemas.jurisdiction import JurisdictionClassification
from aigrc.schemas.owner import Owner
from aigrc.schemas.risk_factors import RiskFactors
from aigrc.schemas.technical import Technical


class GovernanceStatus(str, Enum):
    """Lifecycle status of an asset card."""

    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ApprovalStatus(str, Enum):
    """Approval status for an asset card."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class GoldenThread(BaseModel):
    """
    Golden Thread tracking for audit trail.

    Links the asset card to external approval systems.
    """

    ticket_id: str = Field(description="External ticket/approval ID")
    approver: str = Field(description="Person or system that approved")
    approved_at: str = Field(description="ISO 8601 timestamp of approval")
    hash: str = Field(description="SHA-256 hash of canonical approval string")
    signature: str | None = Field(
        default=None,
        description="Optional cryptographic signature",
    )

    model_config = {"frozen": True, "extra": "forbid"}


class Artifact(BaseModel):
    """
    Reference to an artifact associated with the asset.

    Can be model files, documentation, test results, etc.
    """

    name: str = Field(description="Artifact name")
    type: str = Field(description="Artifact type (model, documentation, test_report, etc.)")
    url: str | None = Field(default=None, description="URL or path to artifact")
    hash: str | None = Field(default=None, description="SHA-256 hash of artifact")
    size_bytes: int | None = Field(default=None, description="Size in bytes")
    created_at: str | None = Field(default=None, description="ISO 8601 creation timestamp")

    model_config = {"frozen": True, "extra": "forbid"}


class AssetCard(BaseModel):
    """
    AI Asset Card - the fundamental unit of governance in AIGRC.

    An asset card describes an AI system, its capabilities, risks,
    and governance status. It serves as the primary artifact for
    tracking and managing AI systems throughout their lifecycle.
    """

    # Identity
    id: Annotated[
        str,
        Field(
            pattern=r"^aigrc-\d{4}-[a-f0-9]{8}$",
            description="Unique identifier in format aigrc-YYYY-XXXXXXXX",
        ),
    ]
    name: Annotated[str, Field(min_length=1, max_length=256)]
    version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+.*$")]
    description: str | None = Field(default=None, max_length=2000)

    # Ownership
    owner: Owner

    # Technical
    technical: Technical

    # Risk
    risk_factors: RiskFactors

    # Jurisdictions
    jurisdiction_classifications: dict[str, JurisdictionClassification] = Field(
        default_factory=dict,
        description="Classifications by jurisdiction/framework",
    )

    # Governance
    governance_status: GovernanceStatus = Field(default=GovernanceStatus.DRAFT)
    approval_status: ApprovalStatus = Field(default=ApprovalStatus.PENDING)
    approved_by: str | None = Field(default=None)
    approved_at: datetime | None = Field(default=None)

    # Golden Thread
    golden_thread: GoldenThread | None = Field(default=None)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Artifacts
    artifacts: list[Artifact] = Field(default_factory=list)

    # Extensions
    extensions: dict[str, Any] = Field(
        default_factory=dict,
        description="Organization-specific extensions",
    )

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "id": "aigrc-2024-a1b2c3d4",
                    "name": "Customer Support Agent",
                    "version": "1.0.0",
                    "description": "AI agent for handling customer support inquiries",
                    "owner": {"team": "AI Platform", "contact": "ai-team@example.com"},
                    "technical": {
                        "language": "python",
                        "frameworks": ["langchain"],
                        "model_providers": ["openai"],
                    },
                    "risk_factors": {
                        "autonomous_decisions": False,
                        "customer_facing": True,
                        "tool_execution": True,
                        "external_data_access": True,
                    },
                }
            ]
        },
    }

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Ensure ID follows AIGRC format."""
        if not re.match(r"^aigrc-\d{4}-[a-f0-9]{8}$", v):
            raise ValueError(f"Invalid ID format: {v}. Expected aigrc-YYYY-XXXXXXXX")
        return v

    @field_validator("version")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Ensure version follows semver format."""
        if not re.match(r"^\d+\.\d+\.\d+.*$", v):
            raise ValueError(f"Invalid version format: {v}. Expected semver (e.g., 1.0.0)")
        return v

    def with_updated_timestamp(self) -> "AssetCard":
        """Return a copy with updated_at set to now."""
        return self.model_copy(update={"updated_at": datetime.utcnow()})

    def approve(self, approver: str) -> "AssetCard":
        """
        Mark this asset card as approved.

        Args:
            approver: Email or identifier of the approver

        Returns:
            New AssetCard with updated approval status
        """
        now = datetime.utcnow()
        return self.model_copy(
            update={
                "approval_status": ApprovalStatus.APPROVED,
                "approved_by": approver,
                "approved_at": now,
                "updated_at": now,
            }
        )

    def publish(self) -> "AssetCard":
        """
        Publish this asset card.

        Raises:
            ValueError: If not approved

        Returns:
            New AssetCard with published status
        """
        if self.approval_status != ApprovalStatus.APPROVED:
            raise ValueError("Cannot publish unapproved asset card")
        return self.model_copy(
            update={
                "governance_status": GovernanceStatus.PUBLISHED,
                "updated_at": datetime.utcnow(),
            }
        )

    def deprecate(self) -> "AssetCard":
        """Mark this asset card as deprecated."""
        return self.model_copy(
            update={
                "governance_status": GovernanceStatus.DEPRECATED,
                "updated_at": datetime.utcnow(),
            }
        )
