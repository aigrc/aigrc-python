"""
Asset Card CRUD operations.

Provides functions for creating, reading, updating, and deleting asset cards.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import yaml

from aigrc.schemas import AssetCard, Owner, RiskFactors, Technical


def generate_id(year: int | None = None) -> str:
    """
    Generate a unique asset card ID.

    Format: aigrc-YYYY-XXXXXXXX (8 hex chars)

    Args:
        year: Year to use in ID (defaults to current year)

    Returns:
        Unique asset card ID

    Example:
        >>> generate_id()
        'aigrc-2024-a1b2c3d4'
        >>> generate_id(2025)
        'aigrc-2025-e5f6a7b8'
    """
    year = year or datetime.now().year
    hex_part = uuid.uuid4().hex[:8]
    return f"aigrc-{year}-{hex_part}"


def create_asset_card(
    name: str,
    owner: Owner | dict[str, Any],
    technical: Technical | dict[str, Any],
    version: str = "1.0.0",
    risk_factors: RiskFactors | dict[str, Any] | None = None,
    **kwargs: Any,
) -> AssetCard:
    """
    Create a new asset card with generated ID.

    Args:
        name: Human-readable name
        owner: Owner information
        technical: Technical metadata
        version: Semantic version (default: "1.0.0")
        risk_factors: Risk factors (default: empty RiskFactors)
        **kwargs: Additional AssetCard fields

    Returns:
        New AssetCard instance

    Example:
        >>> card = create_asset_card(
        ...     name="My Agent",
        ...     owner={"team": "AI", "contact": "ai@example.com"},
        ...     technical={"language": "python", "frameworks": ["langchain"]},
        ... )
        >>> print(card.id)
        'aigrc-2024-...'
    """
    # Convert dicts to models if needed
    if isinstance(owner, dict):
        owner = Owner(**owner)
    if isinstance(technical, dict):
        technical = Technical(**technical)
    if risk_factors is None:
        risk_factors = RiskFactors()
    elif isinstance(risk_factors, dict):
        risk_factors = RiskFactors(**risk_factors)

    return AssetCard(
        id=generate_id(),
        name=name,
        version=version,
        owner=owner,
        technical=technical,
        risk_factors=risk_factors,
        **kwargs,
    )


def load_asset_card(path: Path | str) -> AssetCard:
    """
    Load an asset card from YAML file.

    Args:
        path: Path to .aigrc.yaml or similar

    Returns:
        Parsed AssetCard

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If YAML is invalid or doesn't match schema

    Example:
        >>> card = load_asset_card("my-agent.aigrc.yaml")
        >>> print(card.name)
        'My Agent'
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AssetCard.model_validate(data)


def save_asset_card(card: AssetCard, path: Path | str) -> None:
    """
    Save an asset card to YAML file.

    Creates parent directories if they don't exist.
    Updates the updated_at timestamp.

    Args:
        card: AssetCard to save
        path: Destination path

    Example:
        >>> save_asset_card(card, "my-agent.aigrc.yaml")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict with JSON-compatible types
    data = card.model_dump(mode="json", exclude_none=True)

    # Update timestamp
    data["updated_at"] = datetime.utcnow().isoformat()

    with path.open("w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def find_asset_cards(
    root: Path | str = ".",
    pattern: str = "**/*.aigrc.yaml",
) -> Iterator[tuple[Path, AssetCard]]:
    """
    Find and load all asset cards in a directory.

    Searches recursively using the given glob pattern.
    Skips files that fail to parse or validate.

    Args:
        root: Root directory to search
        pattern: Glob pattern for asset card files

    Yields:
        (path, card) tuples for each found and valid card

    Example:
        >>> for path, card in find_asset_cards("."):
        ...     print(f"{card.name} at {path}")
        'My Agent at ./agents/my-agent.aigrc.yaml'
    """
    root = Path(root)
    for path in root.glob(pattern):
        try:
            card = load_asset_card(path)
            yield path, card
        except Exception:
            continue  # Skip invalid files


def update_asset_card(
    path: Path | str,
    **updates: Any,
) -> AssetCard:
    """
    Update an existing asset card.

    Loads the card, applies updates, and saves it back.

    Args:
        path: Path to asset card file
        **updates: Fields to update

    Returns:
        Updated AssetCard

    Example:
        >>> card = update_asset_card(
        ...     "my-agent.aigrc.yaml",
        ...     version="1.1.0",
        ... )
    """
    path = Path(path)
    card = load_asset_card(path)

    # Apply updates
    updated = card.model_copy(update=updates)
    updated = updated.with_updated_timestamp()

    save_asset_card(updated, path)
    return updated


def delete_asset_card(path: Path | str) -> bool:
    """
    Delete an asset card file.

    Args:
        path: Path to asset card file

    Returns:
        True if deleted, False if file didn't exist
    """
    path = Path(path)
    if path.exists():
        path.unlink()
        return True
    return False
