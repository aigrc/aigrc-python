"""
Owner schema for AI asset cards.

Corresponds to TypeScript: OwnerSchema in @aigrc/core/schemas
"""

from __future__ import annotations

from pydantic import BaseModel, Field, EmailStr


class Owner(BaseModel):
    """
    Ownership information for an AI asset.

    Defines the responsible team and contact information.
    """

    team: str = Field(
        min_length=1,
        max_length=256,
        description="Team or department responsible for this asset",
    )
    contact: str = Field(
        min_length=1,
        max_length=256,
        description="Contact email or identifier for the responsible party",
    )
    department: str | None = Field(
        default=None,
        description="Department within the organization",
    )
    cost_center: str | None = Field(
        default=None,
        description="Cost center for billing purposes",
    )

    model_config = {
        "frozen": True,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "team": "AI Platform",
                    "contact": "ai-team@example.com",
                    "department": "Engineering",
                    "cost_center": "ENG-001",
                }
            ]
        },
    }
