"""
Unit tests for Asset Card CRUD operations.

Tests cover:
- ID generation
- Card creation
- YAML serialization/deserialization
- Finding cards in directory
"""

from pathlib import Path

import pytest
import yaml

from aigrc.asset_card import (
    create_asset_card,
    generate_id,
    load_asset_card,
    save_asset_card,
    find_asset_cards,
)
from aigrc.asset_card.crud import update_asset_card, delete_asset_card
from aigrc.schemas import AssetCard, Owner, Technical, RiskFactors


class TestGenerateId:
    """Tests for ID generation."""

    def test_id_format(self) -> None:
        """Generated ID matches expected format."""
        id_val = generate_id()
        assert id_val.startswith("aigrc-")
        parts = id_val.split("-")
        assert len(parts) == 3
        assert len(parts[1]) == 4  # Year
        assert len(parts[2]) == 8  # Hex chars

    def test_id_with_year(self) -> None:
        """Can specify year in ID."""
        id_val = generate_id(year=2025)
        assert "aigrc-2025-" in id_val

    def test_ids_are_unique(self) -> None:
        """Generated IDs are unique."""
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestCreateAssetCard:
    """Tests for asset card creation."""

    def test_create_with_dicts(self) -> None:
        """Can create card with dict inputs."""
        card = create_asset_card(
            name="Test Agent",
            owner={"team": "AI", "contact": "ai@example.com"},
            technical={"language": "python", "frameworks": ["langchain"]},
        )
        assert card.name == "Test Agent"
        assert card.owner.team == "AI"
        assert card.technical.language == "python"
        assert card.id.startswith("aigrc-")

    def test_create_with_models(self) -> None:
        """Can create card with model inputs."""
        owner = Owner(team="AI", contact="ai@example.com")
        technical = Technical(language="python")
        card = create_asset_card(
            name="Test Agent",
            owner=owner,
            technical=technical,
        )
        assert card.name == "Test Agent"
        assert card.owner == owner

    def test_create_with_custom_version(self) -> None:
        """Can specify version."""
        card = create_asset_card(
            name="Test Agent",
            owner={"team": "AI", "contact": "ai@example.com"},
            technical={"language": "python"},
            version="2.0.0",
        )
        assert card.version == "2.0.0"

    def test_create_with_risk_factors(self) -> None:
        """Can specify risk factors."""
        card = create_asset_card(
            name="Test Agent",
            owner={"team": "AI", "contact": "ai@example.com"},
            technical={"language": "python"},
            risk_factors={"autonomous_decisions": True, "tool_execution": True},
        )
        assert card.risk_factors.autonomous_decisions is True
        assert card.risk_factors.tool_execution is True


class TestSaveLoadAssetCard:
    """Tests for saving and loading asset cards."""

    def test_save_and_load(self, sample_asset_card: AssetCard, tmp_path: Path) -> None:
        """Save and load round-trip."""
        path = tmp_path / "test.aigrc.yaml"
        save_asset_card(sample_asset_card, path)

        loaded = load_asset_card(path)
        assert loaded.id == sample_asset_card.id
        assert loaded.name == sample_asset_card.name
        assert loaded.owner.team == sample_asset_card.owner.team

    def test_save_creates_directories(self, sample_asset_card: AssetCard, tmp_path: Path) -> None:
        """Save creates parent directories."""
        path = tmp_path / "nested" / "dirs" / "test.aigrc.yaml"
        save_asset_card(sample_asset_card, path)
        assert path.exists()

    def test_yaml_format(self, sample_asset_card: AssetCard, tmp_path: Path) -> None:
        """Saved YAML is valid and readable."""
        path = tmp_path / "test.aigrc.yaml"
        save_asset_card(sample_asset_card, path)

        with open(path) as f:
            data = yaml.safe_load(f)

        assert data["id"] == sample_asset_card.id
        assert data["name"] == sample_asset_card.name

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_asset_card(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        """Loading invalid YAML raises error."""
        path = tmp_path / "invalid.yaml"
        path.write_text("not: valid: yaml: [[[")
        with pytest.raises(Exception):
            load_asset_card(path)


class TestFindAssetCards:
    """Tests for finding asset cards."""

    def test_find_cards_in_directory(self, tmp_path: Path) -> None:
        """Find all cards in directory."""
        # Create multiple cards
        for i in range(3):
            card = create_asset_card(
                name=f"Agent {i}",
                owner={"team": "AI", "contact": "ai@example.com"},
                technical={"language": "python"},
            )
            save_asset_card(card, tmp_path / f"agent{i}.aigrc.yaml")

        found = list(find_asset_cards(tmp_path))
        assert len(found) == 3

    def test_find_cards_nested(self, tmp_path: Path) -> None:
        """Find cards in nested directories."""
        # Create cards in nested dirs
        for subdir in ["a", "b", "c"]:
            card = create_asset_card(
                name=f"Agent {subdir}",
                owner={"team": "AI", "contact": "ai@example.com"},
                technical={"language": "python"},
            )
            save_asset_card(card, tmp_path / subdir / "agent.aigrc.yaml")

        found = list(find_asset_cards(tmp_path))
        assert len(found) == 3

    def test_find_skips_invalid(self, tmp_path: Path) -> None:
        """Invalid files are skipped."""
        # Create valid card
        card = create_asset_card(
            name="Valid Agent",
            owner={"team": "AI", "contact": "ai@example.com"},
            technical={"language": "python"},
        )
        save_asset_card(card, tmp_path / "valid.aigrc.yaml")

        # Create invalid file
        (tmp_path / "invalid.aigrc.yaml").write_text("not: a: valid: card")

        found = list(find_asset_cards(tmp_path))
        assert len(found) == 1
        assert found[0][1].name == "Valid Agent"


class TestUpdateDeleteAssetCard:
    """Tests for update and delete operations."""

    def test_update_card(self, sample_asset_card: AssetCard, tmp_path: Path) -> None:
        """Update card fields."""
        path = tmp_path / "test.aigrc.yaml"
        save_asset_card(sample_asset_card, path)

        updated = update_asset_card(path, version="2.0.0")
        assert updated.version == "2.0.0"
        assert updated.id == sample_asset_card.id

        # Verify saved
        loaded = load_asset_card(path)
        assert loaded.version == "2.0.0"

    def test_delete_card(self, sample_asset_card: AssetCard, tmp_path: Path) -> None:
        """Delete card file."""
        path = tmp_path / "test.aigrc.yaml"
        save_asset_card(sample_asset_card, path)
        assert path.exists()

        result = delete_asset_card(path)
        assert result is True
        assert not path.exists()

    def test_delete_nonexistent(self, tmp_path: Path) -> None:
        """Delete returns False for nonexistent file."""
        result = delete_asset_card(tmp_path / "nonexistent.yaml")
        assert result is False
