"""
Golden Thread hash verification.

Provides constant-time verification to prevent timing attacks.
"""

from __future__ import annotations

import hmac
from typing import TYPE_CHECKING

from aigrc.golden_thread.hash import (
    compute_approval_hash,
    compute_canonical_string,
    compute_golden_thread_hash,
)

if TYPE_CHECKING:
    from aigrc.schemas import AssetCard


def verify_golden_thread_hash(card: "AssetCard", expected_hash: str) -> bool:
    """
    Verify that an asset card matches its expected hash.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        card: Asset card to verify
        expected_hash: Expected SHA-256 hash

    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = compute_golden_thread_hash(card)
    return hmac.compare_digest(actual_hash, expected_hash)


def verify_approval_hash(
    ticket_id: str,
    approver: str,
    timestamp: str,
    expected_hash: str,
) -> bool:
    """
    Verify a Golden Thread approval hash.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        ticket_id: External ticket/approval ID
        approver: Approver email or identifier
        timestamp: ISO 8601 timestamp
        expected_hash: Expected hash in format "sha256:XXXX..."

    Returns:
        True if hash matches, False otherwise
    """
    canonical = compute_canonical_string(ticket_id, approver, timestamp)
    computed = compute_approval_hash(canonical)
    return hmac.compare_digest(computed, expected_hash)
