"""
Technical metadata schema for AI asset cards.

Corresponds to TypeScript: TechnicalSchema in @aigrc/core/schemas
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Technical(BaseModel):
    """
    Technical metadata about an AI asset.

    Describes the programming language, frameworks, and providers used.
    """

    language: Literal["python", "javascript", "typescript", "go", "rust", "java", "other"] = Field(
        description="Primary programming language"
    )
    frameworks: list[str] = Field(
        default_factory=list,
        description="AI/ML frameworks used (e.g., langchain, autogen, crewai)",
    )
    model_providers: list[str] = Field(
        default_factory=list,
        description="LLM providers used (e.g., openai, anthropic, cohere)",
    )
    runtime: str | None = Field(
        default=None,
        description="Runtime environment (e.g., python3.11, node20)",
    )
    repository: str | None = Field(
        default=None,
        description="Source code repository URL",
    )
    entry_point: str | None = Field(
        default=None,
        description="Main entry point file or function",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Key dependencies with versions",
    )

    model_config = {
        "frozen": True,
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "language": "python",
                    "frameworks": ["langchain", "langgraph"],
                    "model_providers": ["openai", "anthropic"],
                    "runtime": "python3.11",
                    "repository": "https://github.com/example/my-agent",
                    "entry_point": "src/main.py",
                    "dependencies": ["langchain>=0.1.0", "openai>=1.0.0"],
                }
            ]
        },
    }
