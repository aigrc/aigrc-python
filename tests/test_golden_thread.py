"""
Unit tests for Golden Thread hash computation.

Tests cover:
- Canonical string generation
- Hash computation
- Cross-language compatibility (test vectors)
- Hash verification
"""

import pytest

from aigrc.golden_thread import (
    compute_golden_thread_hash,
    compute_canonical_string,
    verify_golden_thread_hash,
)
from aigrc.golden_thread.hash import compute_approval_hash
from aigrc.golden_thread.verify import verify_approval_hash
from aigrc.schemas import AssetCard


class TestCanonicalString:
    """Tests for canonical string generation."""

    def test_basic_canonical_string(self) -> None:
        """Basic canonical string format."""
        canonical = compute_canonical_string(
            ticket_id="FIN-1234",
            approver="ciso@corp.com",
            timestamp="2025-01-15T10:30:00Z",
        )
        assert canonical == "FIN-1234 | ciso@corp.com | 2025-01-15T10:30:00Z"

    def test_canonical_string_with_spaces(self) -> None:
        """Canonical string preserves spaces."""
        canonical = compute_canonical_string(
            ticket_id="FIN 1234",
            approver="John Doe <john@corp.com>",
            timestamp="2025-01-15T10:30:00Z",
        )
        assert canonical == "FIN 1234 | John Doe <john@corp.com> | 2025-01-15T10:30:00Z"


class TestApprovalHash:
    """Tests for approval hash computation."""

    def test_spec_test_vector(self) -> None:
        """
        Test vector from specification.

        Input: "FIN-1234 | ciso@corp.com | 2025-01-15T10:30:00Z"
        Expected: "sha256:7d865e959b2466918c9863afca942d0fb89d7c9ac0c99bafc3749504ded97730"
        """
        canonical = compute_canonical_string(
            ticket_id="FIN-1234",
            approver="ciso@corp.com",
            timestamp="2025-01-15T10:30:00Z",
        )
        hash_result = compute_approval_hash(canonical)
        expected = "sha256:7d865e959b2466918c9863afca942d0fb89d7c9ac0c99bafc3749504ded97730"
        assert hash_result == expected

    def test_different_inputs_different_hashes(self) -> None:
        """Different inputs produce different hashes."""
        hash1 = compute_approval_hash("input1")
        hash2 = compute_approval_hash("input2")
        assert hash1 != hash2

    def test_same_input_same_hash(self) -> None:
        """Same input always produces same hash."""
        input_str = "FIN-1234 | ciso@corp.com | 2025-01-15T10:30:00Z"
        hash1 = compute_approval_hash(input_str)
        hash2 = compute_approval_hash(input_str)
        assert hash1 == hash2


class TestApprovalHashVerification:
    """Tests for approval hash verification."""

    def test_valid_hash_verification(self) -> None:
        """Valid hash verifies correctly."""
        result = verify_approval_hash(
            ticket_id="FIN-1234",
            approver="ciso@corp.com",
            timestamp="2025-01-15T10:30:00Z",
            expected_hash="sha256:7d865e959b2466918c9863afca942d0fb89d7c9ac0c99bafc3749504ded97730",
        )
        assert result is True

    def test_invalid_hash_fails(self) -> None:
        """Invalid hash fails verification."""
        result = verify_approval_hash(
            ticket_id="FIN-1234",
            approver="ciso@corp.com",
            timestamp="2025-01-15T10:30:00Z",
            expected_hash="sha256:0000000000000000000000000000000000000000000000000000000000000000",
        )
        assert result is False

    def test_modified_input_fails(self) -> None:
        """Modified input fails verification."""
        # Original hash
        original_hash = "sha256:7d865e959b2466918c9863afca942d0fb89d7c9ac0c99bafc3749504ded97730"

        # Modified ticket ID
        result = verify_approval_hash(
            ticket_id="FIN-1235",  # Changed
            approver="ciso@corp.com",
            timestamp="2025-01-15T10:30:00Z",
            expected_hash=original_hash,
        )
        assert result is False


class TestAssetCardHash:
    """Tests for asset card hash computation."""

    def test_hash_computation(self, sample_asset_card: AssetCard) -> None:
        """Hash is computed for asset card."""
        hash_result = compute_golden_thread_hash(sample_asset_card)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA-256 hex string

    def test_deterministic_hash(self, sample_asset_card: AssetCard) -> None:
        """Same card produces same hash."""
        hash1 = compute_golden_thread_hash(sample_asset_card)
        hash2 = compute_golden_thread_hash(sample_asset_card)
        assert hash1 == hash2

    def test_different_cards_different_hashes(
        self, sample_asset_card: AssetCard, sample_owner, sample_technical
    ) -> None:
        """Different cards produce different hashes."""
        from aigrc.schemas import RiskFactors

        card2 = AssetCard(
            id="aigrc-2024-different",
            name="Different Agent",
            version="2.0.0",
            owner=sample_owner,
            technical=sample_technical,
            risk_factors=RiskFactors(),
        )
        hash1 = compute_golden_thread_hash(sample_asset_card)
        hash2 = compute_golden_thread_hash(card2)
        assert hash1 != hash2

    def test_hash_verification(self, sample_asset_card: AssetCard) -> None:
        """Hash verification works correctly."""
        expected_hash = compute_golden_thread_hash(sample_asset_card)
        assert verify_golden_thread_hash(sample_asset_card, expected_hash) is True
        assert verify_golden_thread_hash(sample_asset_card, "wrong_hash") is False

    def test_hash_sensitive_to_risk_factors(
        self, sample_asset_card: AssetCard, sample_owner, sample_technical
    ) -> None:
        """Hash changes when risk factors change."""
        from aigrc.schemas import RiskFactors

        # Create card with different risk factors but same everything else
        card2 = AssetCard(
            id=sample_asset_card.id,
            name=sample_asset_card.name,
            version=sample_asset_card.version,
            owner=sample_owner,
            technical=sample_technical,
            risk_factors=RiskFactors(autonomous_decisions=True),  # Different
        )
        hash1 = compute_golden_thread_hash(sample_asset_card)
        hash2 = compute_golden_thread_hash(card2)
        assert hash1 != hash2
