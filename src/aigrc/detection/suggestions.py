"""
Asset Card suggestion generation from scan results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aigrc.detection.scanner import ScanResult

from aigrc.detection.scanner import FrameworkCategory, FrameworkType


def suggest_asset_card(result: "ScanResult") -> dict[str, Any]:
    """
    Generate asset card suggestion from scan results.

    Analyzes detected frameworks and model files to suggest
    appropriate risk factors and classifications.

    Args:
        result: Scan result from Scanner

    Returns:
        Dictionary with suggested asset card fields

    Example:
        >>> result = await scanner.scan_directory(Path("."))
        >>> suggestion = suggest_asset_card(result)
        >>> print(suggestion["riskFactors"]["autonomousDecisions"])
    """
    frameworks = result.frameworks_found
    detections = result.detections

    # Determine risk level based on detected frameworks
    risk_level = _infer_risk_level(frameworks, result.model_files)

    # Infer risk factors
    risk_factors = _infer_risk_factors(detections, frameworks)

    # Determine primary framework
    primary_framework = _determine_primary_framework(detections)

    # Generate description
    description = _generate_description(frameworks, result.model_files)

    return {
        "name": "Detected AI Asset",
        "description": description,
        "version": "1.0.0",
        "technical": {
            "language": "python",
            "frameworks": sorted(frameworks),
            "model_providers": _extract_providers(frameworks),
        },
        "riskFactors": risk_factors,
        "classification": {
            "level": risk_level,
            "primaryFramework": primary_framework,
        },
        "metadata": {
            "filesScanned": result.files_scanned,
            "detectionsCount": len(result.detections),
            "modelFilesCount": len(result.model_files),
            "scanDurationMs": result.scan_duration_ms,
        },
    }


def _infer_risk_level(frameworks: set[str], model_files: list) -> str:
    """
    Infer risk level from detected frameworks.

    Risk levels (EU AI Act aligned):
    - minimal: Basic ML, no autonomy
    - limited: LLM usage, some autonomy
    - high: Agentic systems, tool execution
    - unacceptable: Prohibited use cases (requires manual review)
    """
    agent_frameworks = {"autogen", "crewai", "agency_swarm", "langgraph"}
    llm_frameworks = {"langchain", "llamaindex", "openai", "anthropic"}

    # Check for agent frameworks (highest risk)
    if frameworks & agent_frameworks:
        return "high"

    # Check for LLM frameworks with potential autonomy
    if frameworks & llm_frameworks:
        return "limited"

    # Check for custom models (medium risk)
    if model_files:
        return "limited"

    # Basic ML libraries
    return "minimal"


def _infer_risk_factors(detections: list, frameworks: set[str]) -> dict[str, Any]:
    """Infer risk factors from detections."""
    agent_frameworks = {"autogen", "crewai", "agency_swarm", "langgraph"}
    api_clients = {"openai", "anthropic", "google_ai", "cohere", "mistral", "groq"}
    tool_frameworks = {"langchain", "autogen", "crewai", "agency_swarm"}

    # Collect implications from detections
    autonomous = bool(frameworks & agent_frameworks)
    tool_execution = bool(frameworks & tool_frameworks)
    external_data = bool(frameworks & api_clients)

    # Check for tool decorators/patterns
    for d in detections:
        if "@tool" in d.evidence or "BaseTool" in d.evidence:
            tool_execution = True

    return {
        "autonomousDecisions": autonomous,
        "customerFacing": False,  # Cannot be inferred from code
        "toolExecution": tool_execution,
        "externalDataAccess": external_data,
        "piiProcessing": "unknown",  # Cannot be inferred from code
        "highStakesDecisions": False,  # Cannot be inferred from code
    }


def _determine_primary_framework(detections: list) -> str | None:
    """Determine the primary framework based on detection count."""
    if not detections:
        return None

    # Count by framework
    counts: dict[str, int] = {}
    for d in detections:
        fw = d.framework.value if isinstance(d.framework, FrameworkType) else d.framework
        counts[fw] = counts.get(fw, 0) + 1

    # Return most common
    return max(counts, key=counts.get)  # type: ignore


def _extract_providers(frameworks: set[str]) -> list[str]:
    """Extract model providers from frameworks."""
    providers = []
    provider_map = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google_ai": "Google",
        "cohere": "Cohere",
        "mistral": "Mistral",
        "groq": "Groq",
        "together": "Together",
        "replicate": "Replicate",
        "huggingface": "HuggingFace",
    }

    for fw in frameworks:
        if fw in provider_map:
            providers.append(provider_map[fw])

    return sorted(set(providers))


def _generate_description(frameworks: set[str], model_files: list) -> str:
    """Generate a description for the asset card."""
    parts = []

    if frameworks:
        parts.append(f"Uses {len(frameworks)} AI framework(s): {', '.join(sorted(frameworks))}")

    if model_files:
        parts.append(f"Contains {len(model_files)} model file(s)")

    if not parts:
        return "No AI components detected"

    return ". ".join(parts)
