# AIGRC Python SDK

AI Governance, Risk, and Compliance toolkit for Python.

## Installation

```bash
pip install aigrc
```

## Quick Start

### Create an Asset Card

```python
from aigrc import create_asset_card, save_asset_card

# Create a new asset card
card = create_asset_card(
    name="Customer Support Agent",
    owner={
        "team": "AI Platform",
        "contact": "ai-team@example.com",
    },
    technical={
        "language": "python",
        "frameworks": ["langchain", "langgraph"],
        "model_providers": ["openai", "anthropic"],
    },
    risk_factors={
        "customer_facing": True,
        "tool_execution": True,
        "external_data_access": True,
    },
)

# Save to YAML
save_asset_card(card, "my-agent.aigrc.yaml")
```

### Load and Validate

```python
from aigrc import load_asset_card

# Load from YAML
card = load_asset_card("my-agent.aigrc.yaml")

print(f"Agent: {card.name}")
print(f"Risk Score: {card.risk_factors.risk_score()}")
```

### Golden Thread Verification

```python
from aigrc import compute_golden_thread_hash, verify_golden_thread_hash

# Compute hash
hash_value = compute_golden_thread_hash(card)
print(f"Golden Thread Hash: {hash_value}")

# Verify hash
is_valid = verify_golden_thread_hash(card, hash_value)
print(f"Hash Valid: {is_valid}")
```

### Find All Asset Cards

```python
from aigrc import find_asset_cards

# Find all cards in current directory
for path, card in find_asset_cards("."):
    print(f"{card.name} ({card.id}) at {path}")
```

## Features

- **Pydantic v2 Schemas**: Type-safe, validated models for AI asset cards
- **Golden Thread Protocol**: Cryptographic verification of approval chains
- **YAML Serialization**: Human-readable storage format
- **TypeScript Compatibility**: Produces identical output to @aigrc/core

## Schemas

### AssetCard

The fundamental unit of governance in AIGRC.

```python
from aigrc.schemas import AssetCard

card = AssetCard(
    id="aigrc-2024-a1b2c3d4",
    name="My Agent",
    version="1.0.0",
    owner=Owner(team="AI", contact="ai@example.com"),
    technical=Technical(language="python"),
    risk_factors=RiskFactors(customer_facing=True),
)
```

### RiskFactors

Risk factors that influence classification.

```python
from aigrc.schemas import RiskFactors

factors = RiskFactors(
    autonomous_decisions=False,
    customer_facing=True,
    tool_execution=True,
    external_data_access=True,
    pii_processing=False,
    high_stakes_decisions=False,
    can_spawn_agents=False,
)

print(f"Risk Score: {factors.risk_score()}")
```

### JurisdictionClassification

Classification under regulatory frameworks (EU AI Act, NIST AI RMF, etc.).

```python
from aigrc.schemas import JurisdictionClassification, RiskLevel

classification = JurisdictionClassification(
    framework="eu_ai_act",
    risk_level=RiskLevel.HIGH,
    category="Annex III",
    compliance_percentage=85.0,
)
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/aigrc/aigrc-python.git
cd aigrc-python

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Type Checking

```bash
mypy src/aigrc
```

### Linting

```bash
ruff check src/aigrc
black src/aigrc
```

## License

Apache 2.0

## Related Projects

- [@aigrc/core](https://github.com/aigrc/aigrc) - TypeScript SDK
- [aigos](https://github.com/ai-gos/aigos-python) - Runtime governance SDK
