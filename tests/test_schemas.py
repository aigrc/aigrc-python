"""
Unit tests for AIGRC Pydantic schemas.

Tests cover:
- Valid data acceptance
- Invalid data rejection
- Field validation
- Default values
- Model methods
"""

import pytest
from pydantic import ValidationError

from aigrc.schemas import (
    ApprovalStatus,
    AssetCard,
    Control,
    ControlStatus,
    GovernanceStatus,
    JurisdictionClassification,
    Owner,
    RiskFactors,
    RiskLevel,
    Technical,
)


class TestRiskFactors:
    """Tests for RiskFactors schema."""

    def test_default_values(self) -> None:
        """All factors default to False."""
        rf = RiskFactors()
        assert rf.autonomous_decisions is False
        assert rf.customer_facing is False
        assert rf.tool_execution is False
        assert rf.external_data_access is False
        assert rf.pii_processing is False
        assert rf.high_stakes_decisions is False
        assert rf.can_spawn_agents is False
        assert rf.custom_factors == {}

    def test_all_true(self) -> None:
        """Can set all factors to True."""
        rf = RiskFactors(
            autonomous_decisions=True,
            customer_facing=True,
            tool_execution=True,
            external_data_access=True,
            pii_processing=True,
            high_stakes_decisions=True,
            can_spawn_agents=True,
        )
        assert rf.autonomous_decisions is True
        assert rf.can_spawn_agents is True

    def test_custom_factors(self) -> None:
        """Can add custom factors."""
        rf = RiskFactors(
            custom_factors={"financial_impact": True, "regulatory_scope": False}
        )
        assert rf.custom_factors["financial_impact"] is True
        assert rf.custom_factors["regulatory_scope"] is False

    def test_risk_score(self) -> None:
        """Risk score calculation."""
        rf = RiskFactors()
        assert rf.risk_score() == 0

        rf = RiskFactors(autonomous_decisions=True, tool_execution=True)
        assert rf.risk_score() == 2

        rf = RiskFactors(
            autonomous_decisions=True,
            tool_execution=True,
            custom_factors={"custom1": True, "custom2": True},
        )
        assert rf.risk_score() == 4

    def test_immutable(self) -> None:
        """RiskFactors should be immutable."""
        rf = RiskFactors()
        with pytest.raises(ValidationError):
            rf.autonomous_decisions = True


class TestOwner:
    """Tests for Owner schema."""

    def test_valid_owner(self) -> None:
        """Valid owner data."""
        owner = Owner(team="AI Platform", contact="ai@example.com")
        assert owner.team == "AI Platform"
        assert owner.contact == "ai@example.com"

    def test_optional_fields(self) -> None:
        """Optional fields are None by default."""
        owner = Owner(team="AI Platform", contact="ai@example.com")
        assert owner.department is None
        assert owner.cost_center is None

    def test_empty_team_rejected(self) -> None:
        """Empty team string rejected."""
        with pytest.raises(ValidationError):
            Owner(team="", contact="ai@example.com")

    def test_extra_fields_rejected(self) -> None:
        """Extra fields are rejected."""
        with pytest.raises(ValidationError):
            Owner(team="AI", contact="ai@example.com", unknown_field="value")


class TestTechnical:
    """Tests for Technical schema."""

    def test_valid_technical(self) -> None:
        """Valid technical metadata."""
        tech = Technical(
            language="python",
            frameworks=["langchain"],
            model_providers=["openai"],
        )
        assert tech.language == "python"
        assert "langchain" in tech.frameworks

    def test_valid_languages(self) -> None:
        """All valid language values."""
        for lang in ["python", "javascript", "typescript", "go", "rust", "java", "other"]:
            tech = Technical(language=lang)
            assert tech.language == lang

    def test_invalid_language_rejected(self) -> None:
        """Invalid language rejected."""
        with pytest.raises(ValidationError):
            Technical(language="cobol")

    def test_default_empty_lists(self) -> None:
        """Lists default to empty."""
        tech = Technical(language="python")
        assert tech.frameworks == []
        assert tech.model_providers == []
        assert tech.dependencies == []


class TestJurisdictionClassification:
    """Tests for JurisdictionClassification schema."""

    def test_valid_classification(self) -> None:
        """Valid jurisdiction classification."""
        jc = JurisdictionClassification(
            framework="eu_ai_act",
            risk_level=RiskLevel.HIGH,
            category="Annex III",
        )
        assert jc.framework == "eu_ai_act"
        assert jc.risk_level == RiskLevel.HIGH

    def test_valid_frameworks(self) -> None:
        """All valid framework values."""
        for framework in ["eu_ai_act", "us_omb_m24", "nist_ai_rmf", "iso_42001"]:
            jc = JurisdictionClassification(
                framework=framework, risk_level=RiskLevel.MINIMAL
            )
            assert jc.framework == framework

    def test_invalid_framework_rejected(self) -> None:
        """Invalid framework rejected."""
        with pytest.raises(ValidationError):
            JurisdictionClassification(
                framework="invalid_framework",
                risk_level=RiskLevel.MINIMAL,
            )

    def test_compliance_percentage_range(self) -> None:
        """Compliance percentage must be 0-100."""
        jc = JurisdictionClassification(
            framework="eu_ai_act",
            risk_level=RiskLevel.HIGH,
            compliance_percentage=85.5,
        )
        assert jc.compliance_percentage == 85.5

        with pytest.raises(ValidationError):
            JurisdictionClassification(
                framework="eu_ai_act",
                risk_level=RiskLevel.HIGH,
                compliance_percentage=101.0,
            )

    def test_update_compliance_percentage(self) -> None:
        """Compliance percentage recalculation."""
        controls = [
            Control(id="C1", name="Control 1", status=ControlStatus.IMPLEMENTED),
            Control(id="C2", name="Control 2", status=ControlStatus.PARTIAL),
            Control(id="C3", name="Control 3", status=ControlStatus.NOT_IMPLEMENTED),
            Control(id="C4", name="Control 4", status=ControlStatus.NOT_APPLICABLE),
        ]
        jc = JurisdictionClassification(
            framework="eu_ai_act",
            risk_level=RiskLevel.HIGH,
            controls=controls,
        )
        updated = jc.update_compliance_percentage()
        # 1 implemented + 0.5 partial out of 3 applicable = 50%
        assert updated.compliance_percentage == 50.0


class TestAssetCard:
    """Tests for AssetCard schema."""

    def test_valid_asset_card(self, sample_asset_card: AssetCard) -> None:
        """Valid asset card creation."""
        assert sample_asset_card.id == "aigrc-2024-a1b2c3d4"
        assert sample_asset_card.name == "Test Agent"
        assert sample_asset_card.version == "1.0.0"

    def test_id_format_validation(self) -> None:
        """ID must match aigrc-YYYY-XXXXXXXX format."""
        # Valid IDs
        for id_val in ["aigrc-2024-a1b2c3d4", "aigrc-2025-00000000", "aigrc-1999-ffffffff"]:
            card = AssetCard(
                id=id_val,
                name="Test",
                version="1.0.0",
                owner=Owner(team="AI", contact="ai@example.com"),
                technical=Technical(language="python"),
                risk_factors=RiskFactors(),
            )
            assert card.id == id_val

        # Invalid IDs
        for id_val in ["aigrc-24-a1b2c3d4", "AIGRC-2024-a1b2c3d4", "aigrc-2024-a1b2", "random-id"]:
            with pytest.raises(ValidationError):
                AssetCard(
                    id=id_val,
                    name="Test",
                    version="1.0.0",
                    owner=Owner(team="AI", contact="ai@example.com"),
                    technical=Technical(language="python"),
                    risk_factors=RiskFactors(),
                )

    def test_version_format_validation(self) -> None:
        """Version must be semver format."""
        # Valid versions
        for version in ["1.0.0", "0.1.0", "2.3.4-beta", "1.0.0-rc.1+build.123"]:
            card = AssetCard(
                id="aigrc-2024-a1b2c3d4",
                name="Test",
                version=version,
                owner=Owner(team="AI", contact="ai@example.com"),
                technical=Technical(language="python"),
                risk_factors=RiskFactors(),
            )
            assert card.version == version

        # Invalid versions
        for version in ["1.0", "v1.0.0", "1", "latest"]:
            with pytest.raises(ValidationError):
                AssetCard(
                    id="aigrc-2024-a1b2c3d4",
                    name="Test",
                    version=version,
                    owner=Owner(team="AI", contact="ai@example.com"),
                    technical=Technical(language="python"),
                    risk_factors=RiskFactors(),
                )

    def test_default_statuses(self, sample_asset_card: AssetCard) -> None:
        """Default governance and approval status."""
        assert sample_asset_card.governance_status == GovernanceStatus.DRAFT
        assert sample_asset_card.approval_status == ApprovalStatus.PENDING

    def test_approve_method(self, sample_asset_card: AssetCard) -> None:
        """Approve method updates status correctly."""
        approved = sample_asset_card.approve("approver@example.com")
        assert approved.approval_status == ApprovalStatus.APPROVED
        assert approved.approved_by == "approver@example.com"
        assert approved.approved_at is not None

    def test_publish_requires_approval(self, sample_asset_card: AssetCard) -> None:
        """Cannot publish unapproved card."""
        with pytest.raises(ValueError, match="Cannot publish unapproved"):
            sample_asset_card.publish()

    def test_publish_after_approval(self, sample_asset_card: AssetCard) -> None:
        """Can publish after approval."""
        approved = sample_asset_card.approve("approver@example.com")
        published = approved.publish()
        assert published.governance_status == GovernanceStatus.PUBLISHED

    def test_deprecate_method(self, sample_asset_card: AssetCard) -> None:
        """Deprecate method updates status."""
        deprecated = sample_asset_card.deprecate()
        assert deprecated.governance_status == GovernanceStatus.DEPRECATED
