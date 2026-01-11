"""
Golden Thread hash computation.

MUST produce identical output to TypeScript implementation for cross-language compatibility.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aigrc.schemas import AssetCard


def compute_canonical_string(
    ticket_id: str,
    approver: str,
    timestamp: str,
) -> str:
    """
    Compute canonical string for Golden Thread approval.

    MUST match TypeScript implementation for cross-language compatibility.
    Format: "approved_at={timestamp}|approved_by={email}|ticket_id={id}"
    Fields are sorted alphabetically with pipe delimiters (no spaces).

    Args:
        ticket_id: External ticket/approval ID
        approver: Approver email or identifier
        timestamp: ISO 8601 timestamp

    Returns:
        Canonical string for hashing

    Example:
        >>> compute_canonical_string("FIN-1234", "ciso@corp.com", "2025-01-15T10:30:00Z")
        'approved_at=2025-01-15T10:30:00Z|approved_by=ciso@corp.com|ticket_id=FIN-1234'
    """
    # Normalize timestamp - remove milliseconds if present
    normalized_ts = _normalize_timestamp(timestamp)

    # Build key-value pairs sorted alphabetically by key
    # (approved_at, approved_by, ticket_id)
    return f"approved_at={normalized_ts}|approved_by={approver}|ticket_id={ticket_id}"


def _normalize_timestamp(timestamp: str) -> str:
    """
    Normalize timestamp to UTC ISO 8601 format without milliseconds.

    Args:
        timestamp: ISO 8601 timestamp string

    Returns:
        Normalized timestamp ending with Z
    """
    from datetime import datetime

    # Parse various timestamp formats
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]:
        try:
            dt = datetime.strptime(timestamp, fmt)
            # Return without milliseconds
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue

    # If parsing fails, return as-is (let hash be computed)
    # This matches TypeScript behavior where invalid timestamps are passed through
    return timestamp


def compute_approval_hash(canonical: str) -> str:
    """
    Compute SHA-256 hash with prefix for approval canonical string.

    Args:
        canonical: Canonical string from compute_canonical_string()

    Returns:
        Hash in format "sha256:XXXX..."

    Test vector (matching TypeScript):
        Input: "approved_at=2025-01-15T10:30:00Z|approved_by=ciso@corp.com|ticket_id=FIN-1234"
        Output: "sha256:7d865e959b2466918c9863afca942d0fb89d7c9ac0c99bafc3749504ded97730"
    """
    hash_bytes = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{hash_bytes}"


def compute_golden_thread_hash(card: "AssetCard") -> str:
    """
    Compute Golden Thread hash for an asset card.

    The hash is SHA-256 of the canonical JSON representation.
    MUST produce identical output to TypeScript implementation.

    Args:
        card: Asset card to hash

    Returns:
        Hex-encoded SHA-256 hash

    Example:
        >>> hash = compute_golden_thread_hash(card)
        >>> print(hash)
        'a1b2c3d4e5f6...'
    """
    canonical = _canonical_json(card)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_json(card: "AssetCard") -> str:
    """
    Generate canonical JSON string for hashing.

    Must match TypeScript's goldenThreadCanonicalString() exactly.
    Only hash-relevant fields are included, sorted consistently.
    """
    # Extract hash-relevant fields only
    hash_data = {
        "id": card.id,
        "name": card.name,
        "version": card.version,
        "owner": {
            "team": card.owner.team,
            "contact": card.owner.contact,
        },
        "technical": {
            "language": card.technical.language,
            "frameworks": sorted(card.technical.frameworks),
            "model_providers": sorted(card.technical.model_providers),
        },
        "risk_factors": {
            "autonomous_decisions": card.risk_factors.autonomous_decisions,
            "customer_facing": card.risk_factors.customer_facing,
            "tool_execution": card.risk_factors.tool_execution,
            "external_data_access": card.risk_factors.external_data_access,
            "pii_processing": card.risk_factors.pii_processing,
            "high_stakes_decisions": card.risk_factors.high_stakes_decisions,
        },
    }

    # JSON with sorted keys, no whitespace (matches TypeScript JSON.stringify behavior)
    return json.dumps(
        hash_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
