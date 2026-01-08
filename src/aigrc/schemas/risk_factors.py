"""
Risk factors schema for AI asset cards.

Corresponds to TypeScript: RiskFactorsSchema in @aigrc/core/schemas
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RiskFactors(BaseModel):
    """
    Risk factors that influence classification.

    These factors determine the risk level of an AI system and
    which governance controls are required.
    """

    # Autonomy
    autonomous_decisions: bool = Field(
        default=False,
        description="Agent can make decisions without human approval",
    )

    # User interaction
    customer_facing: bool = Field(
        default=False,
        description="Agent interacts directly with customers/end-users",
    )

    # Tool execution
    tool_execution: bool = Field(
        default=False,
        description="Agent can execute tools, functions, or code",
    )

    # Data access
    external_data_access: bool = Field(
        default=False,
        description="Agent accesses external APIs or data sources",
    )

    # PII
    pii_processing: bool = Field(
        default=False,
        description="Agent processes personally identifiable information",
    )

    # Stakes
    high_stakes_decisions: bool = Field(
        default=False,
        description="Agent makes decisions with significant consequences",
    )

    # Spawning
    can_spawn_agents: bool = Field(
        default=False,
        description="Agent can create child agents",
    )

    # Custom factors
    custom_factors: dict[str, bool] = Field(
        default_factory=dict,
        description="Organization-specific risk factors",
    )

    model_config = {
        "frozen": True,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "autonomous_decisions": False,
                    "customer_facing": True,
                    "tool_execution": True,
                    "external_data_access": True,
                    "pii_processing": False,
                    "high_stakes_decisions": False,
                    "can_spawn_agents": False,
                }
            ]
        },
    }

    def risk_score(self) -> int:
        """
        Calculate a simple risk score based on factors.

        Returns:
            Score from 0 (minimal) to 7 (high risk)
        """
        factors = [
            self.autonomous_decisions,
            self.customer_facing,
            self.tool_execution,
            self.external_data_access,
            self.pii_processing,
            self.high_stakes_decisions,
            self.can_spawn_agents,
        ]
        return sum(1 for f in factors if f) + sum(1 for v in self.custom_factors.values() if v)
