"""
Pattern Registry for AI/ML framework detection.

Contains detection strategies and framework patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aigrc.detection.scanner import Detection

from aigrc.detection.scanner import (
    DetectionConfidence,
    FrameworkCategory,
    FrameworkType,
)


class DetectionStrategy(ABC):
    """Base class for detection strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @abstractmethod
    def detect(self, path: Path, content: str) -> list["Detection"]:
        """
        Detect frameworks in file content.

        Args:
            path: File path
            content: File content

        Returns:
            List of detections
        """
        pass


@dataclass
class FrameworkPattern:
    """Pattern for detecting a specific framework."""

    framework: FrameworkType | str
    """Framework identifier."""

    category: FrameworkCategory
    """Framework category."""

    import_patterns: list[str] = field(default_factory=list)
    """Import statement patterns (e.g., ["import torch", "from torch"])."""

    code_patterns: list[str] = field(default_factory=list)
    """Code usage patterns (e.g., ["torch.nn.", "model.forward("])."""

    config_patterns: list[str] = field(default_factory=list)
    """Patterns in config files (pyproject.toml, requirements.txt)."""

    risk_implications: dict = field(default_factory=dict)
    """Implied risk factors when this framework is detected."""


class ImportAnalysisStrategy(DetectionStrategy):
    """Detect frameworks via import statements."""

    def __init__(self, patterns: list[FrameworkPattern]) -> None:
        self._patterns = patterns

    @property
    def name(self) -> str:
        return "import_analysis"

    def detect(self, path: Path, content: str) -> list["Detection"]:
        from aigrc.detection.scanner import Detection

        detections: list[Detection] = []
        lines = content.split("\n")

        for pattern in self._patterns:
            for i, line in enumerate(lines):
                stripped = line.strip()

                # Skip comments and empty lines
                if not stripped or stripped.startswith("#"):
                    continue

                for import_pat in pattern.import_patterns:
                    if import_pat in line:
                        detections.append(
                            Detection(
                                framework=pattern.framework,
                                category=pattern.category,
                                confidence=DetectionConfidence.HIGH,
                                source_file=path,
                                line_number=i + 1,
                                evidence=stripped,
                                strategy=self.name,
                            )
                        )
                        # Only one detection per pattern per file
                        break

        return detections


class PatternMatchingStrategy(DetectionStrategy):
    """Detect frameworks via code patterns."""

    def __init__(self, patterns: list[FrameworkPattern]) -> None:
        self._patterns = patterns

    @property
    def name(self) -> str:
        return "pattern_matching"

    def detect(self, path: Path, content: str) -> list["Detection"]:
        from aigrc.detection.scanner import Detection

        detections: list[Detection] = []
        lines = content.split("\n")

        for pattern in self._patterns:
            for code_pat in pattern.code_patterns:
                for i, line in enumerate(lines):
                    if code_pat in line:
                        detections.append(
                            Detection(
                                framework=pattern.framework,
                                category=pattern.category,
                                confidence=DetectionConfidence.MEDIUM,
                                source_file=path,
                                line_number=i + 1,
                                evidence=f"Pattern: {code_pat}",
                                strategy=self.name,
                            )
                        )
                        # Only one detection per pattern per file
                        break

        return detections


class AnnotationStrategy(DetectionStrategy):
    """Detect AIGRC annotations in comments."""

    @property
    def name(self) -> str:
        return "annotation"

    def detect(self, path: Path, content: str) -> list["Detection"]:
        from aigrc.detection.scanner import Detection

        detections: list[Detection] = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Look for AIGRC annotations
            if "@aigrc:" in stripped or "# aigrc:" in stripped.lower():
                detections.append(
                    Detection(
                        framework="aigrc_annotation",
                        category=FrameworkCategory.LLM_FRAMEWORK,
                        confidence=DetectionConfidence.HIGH,
                        source_file=path,
                        line_number=i + 1,
                        evidence=stripped,
                        strategy=self.name,
                    )
                )

        return detections


class PatternRegistry:
    """Registry of framework detection patterns."""

    def __init__(self) -> None:
        self._patterns: list[FrameworkPattern] = []
        self._strategies: list[DetectionStrategy] = []
        self._register_default_patterns()
        self._build_strategies()

    @property
    def patterns(self) -> list[FrameworkPattern]:
        """All registered patterns."""
        return self._patterns

    @property
    def strategies(self) -> list[DetectionStrategy]:
        """All detection strategies."""
        return self._strategies

    def register_pattern(self, pattern: FrameworkPattern) -> None:
        """Register a custom framework pattern."""
        self._patterns.append(pattern)
        self._build_strategies()

    def get_patterns_by_category(
        self, category: FrameworkCategory
    ) -> list[FrameworkPattern]:
        """Get patterns for a specific category."""
        return [p for p in self._patterns if p.category == category]

    def _build_strategies(self) -> None:
        """Build detection strategies from patterns."""
        self._strategies = [
            ImportAnalysisStrategy(self._patterns),
            PatternMatchingStrategy(self._patterns),
            AnnotationStrategy(),
        ]

    def _register_default_patterns(self) -> None:
        """Register 40+ Python framework patterns."""

        # ============================================================
        # Deep Learning Frameworks
        # ============================================================
        self._patterns.extend(
            [
                FrameworkPattern(
                    framework=FrameworkType.PYTORCH,
                    category=FrameworkCategory.DEEP_LEARNING,
                    import_patterns=[
                        "import torch",
                        "from torch",
                        "import torchvision",
                        "from torchvision",
                    ],
                    code_patterns=[
                        "torch.nn.",
                        "torch.Tensor",
                        "torch.cuda",
                        "nn.Module",
                        "torch.optim",
                        ".to(device)",
                        "model.train()",
                        "model.eval()",
                    ],
                    risk_implications={"autonomous_decisions": False},
                ),
                FrameworkPattern(
                    framework=FrameworkType.TENSORFLOW,
                    category=FrameworkCategory.DEEP_LEARNING,
                    import_patterns=[
                        "import tensorflow",
                        "from tensorflow",
                        "import tf",
                    ],
                    code_patterns=[
                        "tf.keras",
                        "tf.data",
                        "tf.function",
                        "tf.Variable",
                        "tf.constant",
                        "model.compile(",
                        "model.fit(",
                    ],
                    risk_implications={"autonomous_decisions": False},
                ),
                FrameworkPattern(
                    framework=FrameworkType.KERAS,
                    category=FrameworkCategory.DEEP_LEARNING,
                    import_patterns=[
                        "import keras",
                        "from keras",
                        "from tensorflow.keras",
                    ],
                    code_patterns=[
                        "Sequential(",
                        "Dense(",
                        "Conv2D(",
                        "model.compile(",
                        "model.fit(",
                    ],
                    risk_implications={"autonomous_decisions": False},
                ),
                FrameworkPattern(
                    framework=FrameworkType.JAX,
                    category=FrameworkCategory.DEEP_LEARNING,
                    import_patterns=[
                        "import jax",
                        "from jax",
                        "import flax",
                        "from flax",
                    ],
                    code_patterns=[
                        "jax.numpy",
                        "jax.grad",
                        "jax.jit",
                        "@jax.jit",
                        "flax.linen",
                    ],
                    risk_implications={"autonomous_decisions": False},
                ),
            ]
        )

        # ============================================================
        # LLM Frameworks
        # ============================================================
        self._patterns.extend(
            [
                FrameworkPattern(
                    framework=FrameworkType.LANGCHAIN,
                    category=FrameworkCategory.LLM_FRAMEWORK,
                    import_patterns=[
                        "from langchain",
                        "import langchain",
                        "from langchain_core",
                        "from langchain_openai",
                        "from langchain_anthropic",
                        "from langchain_community",
                        "from langchain_google",
                    ],
                    code_patterns=[
                        "ChatOpenAI(",
                        "ChatAnthropic(",
                        "LLMChain(",
                        "RetrievalQA(",
                        "ConversationChain(",
                        ".invoke(",
                        "PromptTemplate(",
                        "ChatPromptTemplate",
                    ],
                    risk_implications={
                        "tool_execution": True,
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.LLAMAINDEX,
                    category=FrameworkCategory.LLM_FRAMEWORK,
                    import_patterns=[
                        "from llama_index",
                        "import llama_index",
                        "from llama_index.core",
                        "from llama_index.llms",
                    ],
                    code_patterns=[
                        "VectorStoreIndex(",
                        "ServiceContext(",
                        "SimpleDirectoryReader(",
                        "QueryEngine(",
                        "index.as_query_engine(",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.LANGGRAPH,
                    category=FrameworkCategory.LLM_FRAMEWORK,
                    import_patterns=[
                        "from langgraph",
                        "import langgraph",
                        "from langgraph.graph",
                        "from langgraph.prebuilt",
                    ],
                    code_patterns=[
                        "StateGraph(",
                        "add_node(",
                        "add_edge(",
                        "add_conditional_edges(",
                        ".compile(",
                    ],
                    risk_implications={
                        "autonomous_decisions": True,
                        "tool_execution": True,
                    },
                ),
            ]
        )

        # ============================================================
        # Agent Frameworks
        # ============================================================
        self._patterns.extend(
            [
                FrameworkPattern(
                    framework=FrameworkType.AUTOGEN,
                    category=FrameworkCategory.AGENT_FRAMEWORK,
                    import_patterns=[
                        "from autogen",
                        "import autogen",
                        "from pyautogen",
                        "import pyautogen",
                        "from autogen_agentchat",
                    ],
                    code_patterns=[
                        "AssistantAgent(",
                        "UserProxyAgent(",
                        "GroupChat(",
                        "GroupChatManager(",
                        "ConversableAgent(",
                    ],
                    risk_implications={
                        "autonomous_decisions": True,
                        "tool_execution": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.CREWAI,
                    category=FrameworkCategory.AGENT_FRAMEWORK,
                    import_patterns=[
                        "from crewai",
                        "import crewai",
                    ],
                    code_patterns=[
                        "Crew(",
                        "Agent(",
                        "Task(",
                        "@agent",
                        "@task",
                        "@crew",
                        "crew.kickoff(",
                    ],
                    risk_implications={
                        "autonomous_decisions": True,
                        "tool_execution": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.AGENCY_SWARM,
                    category=FrameworkCategory.AGENT_FRAMEWORK,
                    import_patterns=[
                        "from agency_swarm",
                        "import agency_swarm",
                    ],
                    code_patterns=[
                        "Agency(",
                        "Agent(",
                        "BaseTool(",
                        "agency.run(",
                    ],
                    risk_implications={
                        "autonomous_decisions": True,
                        "tool_execution": True,
                    },
                ),
            ]
        )

        # ============================================================
        # API Clients
        # ============================================================
        self._patterns.extend(
            [
                FrameworkPattern(
                    framework=FrameworkType.OPENAI,
                    category=FrameworkCategory.API_CLIENT,
                    import_patterns=[
                        "from openai",
                        "import openai",
                    ],
                    code_patterns=[
                        "OpenAI(",
                        "AsyncOpenAI(",
                        "client.chat.completions",
                        "ChatCompletion",
                        ".create(",
                        "openai.api_key",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.ANTHROPIC,
                    category=FrameworkCategory.API_CLIENT,
                    import_patterns=[
                        "from anthropic",
                        "import anthropic",
                    ],
                    code_patterns=[
                        "Anthropic(",
                        "AsyncAnthropic(",
                        "client.messages",
                        "claude-",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.GOOGLE_AI,
                    category=FrameworkCategory.API_CLIENT,
                    import_patterns=[
                        "import google.generativeai",
                        "from google.generativeai",
                        "import vertexai",
                        "from vertexai",
                    ],
                    code_patterns=[
                        "genai.GenerativeModel(",
                        "GenerativeModel(",
                        "gemini-",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.COHERE,
                    category=FrameworkCategory.API_CLIENT,
                    import_patterns=[
                        "import cohere",
                        "from cohere",
                    ],
                    code_patterns=[
                        "cohere.Client(",
                        "co.chat(",
                        "co.generate(",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.MISTRAL,
                    category=FrameworkCategory.API_CLIENT,
                    import_patterns=[
                        "from mistralai",
                        "import mistralai",
                    ],
                    code_patterns=[
                        "MistralClient(",
                        "Mistral(",
                        "mistral-",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.GROQ,
                    category=FrameworkCategory.API_CLIENT,
                    import_patterns=[
                        "from groq",
                        "import groq",
                    ],
                    code_patterns=[
                        "Groq(",
                        "groq.Client(",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.TOGETHER,
                    category=FrameworkCategory.API_CLIENT,
                    import_patterns=[
                        "from together",
                        "import together",
                    ],
                    code_patterns=[
                        "Together(",
                        "together.Complete(",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.REPLICATE,
                    category=FrameworkCategory.API_CLIENT,
                    import_patterns=[
                        "import replicate",
                        "from replicate",
                    ],
                    code_patterns=[
                        "replicate.run(",
                        "replicate.predictions",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
            ]
        )

        # ============================================================
        # ML Libraries
        # ============================================================
        self._patterns.extend(
            [
                FrameworkPattern(
                    framework=FrameworkType.HUGGINGFACE,
                    category=FrameworkCategory.ML_LIBRARY,
                    import_patterns=[
                        "from transformers",
                        "import transformers",
                        "from huggingface_hub",
                        "import huggingface_hub",
                        "from datasets",
                        "from sentence_transformers",
                    ],
                    code_patterns=[
                        "AutoModel",
                        "AutoTokenizer",
                        "pipeline(",
                        "from_pretrained(",
                        "Trainer(",
                    ],
                    risk_implications={},
                ),
                FrameworkPattern(
                    framework=FrameworkType.SKLEARN,
                    category=FrameworkCategory.ML_LIBRARY,
                    import_patterns=[
                        "from sklearn",
                        "import sklearn",
                        "from scikit-learn",
                    ],
                    code_patterns=[
                        ".fit(",
                        ".predict(",
                        ".transform(",
                        "train_test_split(",
                    ],
                    risk_implications={},
                ),
                FrameworkPattern(
                    framework=FrameworkType.SPACY,
                    category=FrameworkCategory.ML_LIBRARY,
                    import_patterns=[
                        "import spacy",
                        "from spacy",
                    ],
                    code_patterns=[
                        "spacy.load(",
                        "nlp(",
                        "doc.ents",
                    ],
                    risk_implications={},
                ),
                FrameworkPattern(
                    framework=FrameworkType.NLTK,
                    category=FrameworkCategory.ML_LIBRARY,
                    import_patterns=[
                        "import nltk",
                        "from nltk",
                    ],
                    code_patterns=[
                        "nltk.download(",
                        "word_tokenize(",
                        "sent_tokenize(",
                    ],
                    risk_implications={},
                ),
            ]
        )

        # ============================================================
        # Vector Databases
        # ============================================================
        self._patterns.extend(
            [
                FrameworkPattern(
                    framework=FrameworkType.PINECONE,
                    category=FrameworkCategory.VECTOR_DB,
                    import_patterns=[
                        "import pinecone",
                        "from pinecone",
                    ],
                    code_patterns=[
                        "pinecone.init(",
                        "Pinecone(",
                        ".upsert(",
                        ".query(",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.WEAVIATE,
                    category=FrameworkCategory.VECTOR_DB,
                    import_patterns=[
                        "import weaviate",
                        "from weaviate",
                    ],
                    code_patterns=[
                        "weaviate.Client(",
                        "WeaviateClient(",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.CHROMADB,
                    category=FrameworkCategory.VECTOR_DB,
                    import_patterns=[
                        "import chromadb",
                        "from chromadb",
                    ],
                    code_patterns=[
                        "chromadb.Client(",
                        "chromadb.PersistentClient(",
                        ".get_or_create_collection(",
                    ],
                    risk_implications={},
                ),
                FrameworkPattern(
                    framework=FrameworkType.QDRANT,
                    category=FrameworkCategory.VECTOR_DB,
                    import_patterns=[
                        "from qdrant_client",
                        "import qdrant_client",
                    ],
                    code_patterns=[
                        "QdrantClient(",
                        ".upsert(",
                        ".search(",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
                FrameworkPattern(
                    framework=FrameworkType.MILVUS,
                    category=FrameworkCategory.VECTOR_DB,
                    import_patterns=[
                        "from pymilvus",
                        "import pymilvus",
                    ],
                    code_patterns=[
                        "connections.connect(",
                        "Collection(",
                        ".insert(",
                    ],
                    risk_implications={
                        "external_data_access": True,
                    },
                ),
            ]
        )
