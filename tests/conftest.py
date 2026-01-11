"""
Pytest configuration and fixtures for AIGRC tests.
"""

from datetime import datetime
from pathlib import Path

import pytest

from aigrc.schemas import (
    AssetCard,
    Owner,
    RiskFactors,
    Technical,
)


@pytest.fixture
def sample_owner() -> Owner:
    """Sample owner for testing."""
    return Owner(
        team="AI Platform",
        contact="ai-team@example.com",
        department="Engineering",
    )


@pytest.fixture
def sample_technical() -> Technical:
    """Sample technical metadata for testing."""
    return Technical(
        language="python",
        frameworks=["langchain", "langgraph"],
        model_providers=["openai", "anthropic"],
        runtime="python3.11",
    )


@pytest.fixture
def sample_risk_factors() -> RiskFactors:
    """Sample risk factors for testing."""
    return RiskFactors(
        autonomous_decisions=False,
        customer_facing=True,
        tool_execution=True,
        external_data_access=True,
        pii_processing=False,
        high_stakes_decisions=False,
        can_spawn_agents=False,
    )


@pytest.fixture
def sample_asset_card(
    sample_owner: Owner,
    sample_technical: Technical,
    sample_risk_factors: RiskFactors,
) -> AssetCard:
    """Sample asset card for testing."""
    return AssetCard(
        id="aigrc-2024-a1b2c3d4",  # Must be hex: aigrc-YYYY-XXXXXXXX
        name="Test Agent",
        version="1.0.0",
        description="A test agent for unit testing",
        owner=sample_owner,
        technical=sample_technical,
        risk_factors=sample_risk_factors,
    )


@pytest.fixture
def temp_asset_card_dir(tmp_path: Path) -> Path:
    """Temporary directory for asset card files."""
    return tmp_path / "assets"
