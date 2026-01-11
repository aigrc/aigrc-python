"""
AI/ML Framework Scanner.

Scans codebases to detect AI frameworks, model files,
and generate asset card suggestions.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Callable

if TYPE_CHECKING:
    from aigrc.detection.patterns import PatternRegistry


class FrameworkType(str, Enum):
    """Known AI/ML framework types."""

    # Deep Learning
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    JAX = "jax"

    # LLM Frameworks
    LANGCHAIN = "langchain"
    LLAMAINDEX = "llamaindex"
    LANGGRAPH = "langgraph"

    # Agent Frameworks
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    AGENCY_SWARM = "agency_swarm"

    # API Clients
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_AI = "google_ai"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GROQ = "groq"
    TOGETHER = "together"
    REPLICATE = "replicate"

    # ML Libraries
    HUGGINGFACE = "huggingface"
    SKLEARN = "sklearn"
    SPACY = "spacy"
    NLTK = "nltk"

    # Vector DBs (often used with AI)
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMADB = "chromadb"
    QDRANT = "qdrant"
    MILVUS = "milvus"


class FrameworkCategory(str, Enum):
    """Categories of AI/ML frameworks."""

    DEEP_LEARNING = "deep_learning"
    LLM_FRAMEWORK = "llm_framework"
    AGENT_FRAMEWORK = "agent_framework"
    API_CLIENT = "api_client"
    ML_LIBRARY = "ml_library"
    VECTOR_DB = "vector_db"
    MODEL_FILE = "model_file"


class DetectionConfidence(str, Enum):
    """Confidence level of a detection."""

    HIGH = "high"  # Direct import or config file
    MEDIUM = "medium"  # Pattern match
    LOW = "low"  # File extension or indirect


@dataclass
class Detection:
    """A detected AI/ML framework or pattern."""

    framework: FrameworkType | str
    """The detected framework."""

    category: FrameworkCategory
    """Category of the framework."""

    confidence: DetectionConfidence
    """Confidence level of the detection."""

    source_file: Path
    """File where detection was found."""

    line_number: int | None = None
    """Line number of detection (if applicable)."""

    column: int | None = None
    """Column number of detection (if applicable)."""

    evidence: str = ""
    """The matched text or pattern."""

    strategy: str = ""
    """Detection strategy that found this."""

    version: str | None = None
    """Detected version (if available)."""


@dataclass
class ScanProgress:
    """Progress information during scanning."""

    current_file: str
    scanned_files: int
    total_files: int
    detections_found: int


@dataclass
class ScanResult:
    """Result of scanning a directory or file."""

    directory: str = ""
    """Scanned directory path."""

    detections: list[Detection] = field(default_factory=list)
    """List of framework detections."""

    model_files: list[Path] = field(default_factory=list)
    """List of detected model files."""

    scan_duration_ms: float = 0.0
    """Scan duration in milliseconds."""

    files_scanned: int = 0
    """Number of files scanned."""

    errors: list[str] = field(default_factory=list)
    """Errors encountered during scan."""

    @property
    def frameworks_found(self) -> set[str]:
        """Unique frameworks detected."""
        return {
            d.framework.value if isinstance(d.framework, FrameworkType) else d.framework
            for d in self.detections
        }

    @property
    def has_ai_components(self) -> bool:
        """Whether any AI components were detected."""
        return len(self.detections) > 0 or len(self.model_files) > 0

    @property
    def summary(self) -> dict:
        """Summary of detections by framework, category, and confidence."""
        by_framework: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_confidence: dict[str, int] = {"high": 0, "medium": 0, "low": 0}

        for d in self.detections:
            fw = d.framework.value if isinstance(d.framework, FrameworkType) else d.framework
            by_framework[fw] = by_framework.get(fw, 0) + 1
            by_category[d.category.value] = by_category.get(d.category.value, 0) + 1
            by_confidence[d.confidence.value] += 1

        return {
            "byFramework": by_framework,
            "byCategory": by_category,
            "byConfidence": by_confidence,
        }

    def deduplicate(self) -> "ScanResult":
        """Remove duplicate detections (same framework + file)."""
        seen: set[tuple] = set()
        unique: list[Detection] = []

        for d in self.detections:
            key = (
                d.framework.value if isinstance(d.framework, FrameworkType) else d.framework,
                str(d.source_file),
            )
            if key not in seen:
                seen.add(key)
                unique.append(d)

        return ScanResult(
            directory=self.directory,
            detections=unique,
            model_files=self.model_files,
            scan_duration_ms=self.scan_duration_ms,
            files_scanned=self.files_scanned,
            errors=self.errors,
        )


# Model file extensions
MODEL_EXTENSIONS: set[str] = {
    ".pt",
    ".pth",
    ".bin",  # PyTorch
    ".safetensors",  # SafeTensors
    ".onnx",  # ONNX
    ".h5",
    ".keras",  # TensorFlow/Keras
    ".gguf",
    ".ggml",  # llama.cpp / GGML
    ".mlmodel",  # CoreML
    ".pb",  # TensorFlow protobuf
    ".tflite",  # TensorFlow Lite
    ".mar",  # TorchServe
}


class Scanner:
    """
    AI/ML framework detection scanner.

    Scans codebases to detect AI frameworks, model files,
    and generate asset card suggestions.

    Example:
        >>> scanner = Scanner()
        >>> result = await scanner.scan_directory(Path("."))
        >>> for detection in result.detections:
        ...     print(f"{detection.framework}: {detection.source_file}")

        >>> # Sync version
        >>> result = scanner.scan_directory_sync(Path("."))
    """

    def __init__(
        self,
        *,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        max_file_size_mb: float = 10.0,
    ) -> None:
        """
        Initialize the scanner.

        Args:
            include_patterns: File patterns to include (default: Python files)
            exclude_patterns: File patterns to exclude (default: venv, node_modules)
            max_file_size_mb: Maximum file size to scan in MB
        """
        self._include = include_patterns or ["*.py", "*.ipynb"]
        self._exclude = exclude_patterns or [
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/__pycache__/**",
            "**/.git/**",
            "**/dist/**",
            "**/build/**",
            "**/.tox/**",
            "**/.pytest_cache/**",
            "**/.mypy_cache/**",
        ]
        self._max_size = int(max_file_size_mb * 1024 * 1024)

        # Lazy import to avoid circular dependency
        from aigrc.detection.patterns import PatternRegistry

        self._registry = PatternRegistry()

    async def scan_directory(
        self,
        path: Path,
        progress_callback: Callable[[ScanProgress], None] | None = None,
    ) -> ScanResult:
        """
        Scan a directory for AI/ML frameworks.

        Args:
            path: Directory to scan
            progress_callback: Optional callback for progress updates

        Returns:
            ScanResult with all detections
        """
        start = time.perf_counter()
        path = Path(path).resolve()

        result = ScanResult(directory=str(path))

        # Collect all files first for progress tracking
        files_to_scan: list[Path] = []
        model_files: list[Path] = []

        for item in path.rglob("*"):
            if not item.is_file():
                continue

            # Skip excluded patterns
            rel_path = str(item.relative_to(path))
            if self._should_exclude(rel_path):
                continue

            # Check for model files
            if self._is_model_file(item):
                model_files.append(item)
                continue

            # Check for code files
            if self._should_include(item.name):
                files_to_scan.append(item)

        result.model_files = model_files
        total_files = len(files_to_scan)

        # Scan each file
        for i, file_path in enumerate(files_to_scan):
            try:
                detections = await self.scan_file(file_path)
                result.detections.extend(detections)
                result.files_scanned += 1

                if progress_callback:
                    progress_callback(
                        ScanProgress(
                            current_file=str(file_path.relative_to(path)),
                            scanned_files=i + 1,
                            total_files=total_files,
                            detections_found=len(result.detections),
                        )
                    )
            except Exception as e:
                result.errors.append(f"{file_path}: {e}")

        result.scan_duration_ms = (time.perf_counter() - start) * 1000
        return result

    def scan_directory_sync(
        self,
        path: Path,
        progress_callback: Callable[[ScanProgress], None] | None = None,
    ) -> ScanResult:
        """
        Synchronous version of scan_directory.

        Args:
            path: Directory to scan
            progress_callback: Optional callback for progress updates

        Returns:
            ScanResult with all detections
        """
        return asyncio.run(self.scan_directory(path, progress_callback))

    async def scan_file(self, path: Path) -> list[Detection]:
        """
        Scan a single file for AI/ML frameworks.

        Args:
            path: File to scan

        Returns:
            List of detections found
        """
        path = Path(path)

        # Skip large files
        try:
            if path.stat().st_size > self._max_size:
                return []
        except OSError:
            return []

        # Read file content
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        # Run all detection strategies
        detections: list[Detection] = []
        for strategy in self._registry.strategies:
            try:
                strategy_detections = strategy.detect(path, content)
                detections.extend(strategy_detections)
            except Exception:
                # Skip failed strategies
                pass

        return detections

    def scan_file_sync(self, path: Path) -> list[Detection]:
        """Synchronous version of scan_file."""
        return asyncio.run(self.scan_file(path))

    def _should_exclude(self, rel_path: str) -> bool:
        """Check if path should be excluded."""
        # Normalize path separators
        normalized = rel_path.replace("\\", "/")

        for pattern in self._exclude:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            # Also check with forward slashes for Windows compatibility
            if fnmatch.fnmatch(normalized, pattern):
                return True

        # Also check path components for common exclusions
        parts = normalized.split("/")
        excluded_dirs = {".venv", "venv", "node_modules", "__pycache__", ".git", ".tox"}
        if any(part in excluded_dirs for part in parts):
            return True

        return False

    def _should_include(self, filename: str) -> bool:
        """Check if filename matches include patterns."""
        return any(fnmatch.fnmatch(filename, pattern) for pattern in self._include)

    def _is_model_file(self, path: Path) -> bool:
        """Check if file is a model file by extension."""
        return path.suffix.lower() in MODEL_EXTENSIONS
