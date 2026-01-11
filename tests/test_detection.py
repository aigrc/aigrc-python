"""
Tests for the Detection Engine.

Tests cover:
- Scanner functionality
- Pattern registry
- Detection strategies
- Asset card suggestions
- Model file detection
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aigrc.detection import (
    Detection,
    DetectionConfidence,
    FrameworkCategory,
    FrameworkType,
    PatternRegistry,
    ScanResult,
    Scanner,
    suggest_asset_card,
)
from aigrc.detection.patterns import (
    AnnotationStrategy,
    FrameworkPattern,
    ImportAnalysisStrategy,
    PatternMatchingStrategy,
)


class TestFrameworkType:
    """Tests for FrameworkType enum."""

    def test_framework_types_exist(self):
        """All expected framework types exist."""
        assert FrameworkType.PYTORCH.value == "pytorch"
        assert FrameworkType.TENSORFLOW.value == "tensorflow"
        assert FrameworkType.LANGCHAIN.value == "langchain"
        assert FrameworkType.AUTOGEN.value == "autogen"
        assert FrameworkType.CREWAI.value == "crewai"
        assert FrameworkType.OPENAI.value == "openai"
        assert FrameworkType.ANTHROPIC.value == "anthropic"

    def test_framework_category_values(self):
        """Framework categories have expected values."""
        assert FrameworkCategory.DEEP_LEARNING.value == "deep_learning"
        assert FrameworkCategory.LLM_FRAMEWORK.value == "llm_framework"
        assert FrameworkCategory.AGENT_FRAMEWORK.value == "agent_framework"
        assert FrameworkCategory.API_CLIENT.value == "api_client"


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self):
        """Detection can be created with required fields."""
        detection = Detection(
            framework=FrameworkType.OPENAI,
            category=FrameworkCategory.API_CLIENT,
            confidence=DetectionConfidence.HIGH,
            source_file=Path("test.py"),
            line_number=5,
            evidence="import openai",
        )

        assert detection.framework == FrameworkType.OPENAI
        assert detection.category == FrameworkCategory.API_CLIENT
        assert detection.confidence == DetectionConfidence.HIGH
        assert detection.line_number == 5

    def test_detection_with_custom_framework(self):
        """Detection works with custom framework strings."""
        detection = Detection(
            framework="my_custom_framework",
            category=FrameworkCategory.LLM_FRAMEWORK,
            confidence=DetectionConfidence.MEDIUM,
            source_file=Path("custom.py"),
        )

        assert detection.framework == "my_custom_framework"


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_empty_scan_result(self):
        """Empty scan result has expected defaults."""
        result = ScanResult()

        assert result.detections == []
        assert result.model_files == []
        assert result.files_scanned == 0
        assert not result.has_ai_components
        assert result.frameworks_found == set()

    def test_scan_result_with_detections(self):
        """Scan result correctly reports frameworks found."""
        result = ScanResult(
            detections=[
                Detection(
                    framework=FrameworkType.OPENAI,
                    category=FrameworkCategory.API_CLIENT,
                    confidence=DetectionConfidence.HIGH,
                    source_file=Path("a.py"),
                ),
                Detection(
                    framework=FrameworkType.LANGCHAIN,
                    category=FrameworkCategory.LLM_FRAMEWORK,
                    confidence=DetectionConfidence.HIGH,
                    source_file=Path("b.py"),
                ),
            ]
        )

        assert result.has_ai_components
        assert result.frameworks_found == {"openai", "langchain"}

    def test_scan_result_summary(self):
        """Scan result generates correct summary."""
        result = ScanResult(
            detections=[
                Detection(
                    framework=FrameworkType.OPENAI,
                    category=FrameworkCategory.API_CLIENT,
                    confidence=DetectionConfidence.HIGH,
                    source_file=Path("a.py"),
                ),
                Detection(
                    framework=FrameworkType.OPENAI,
                    category=FrameworkCategory.API_CLIENT,
                    confidence=DetectionConfidence.MEDIUM,
                    source_file=Path("b.py"),
                ),
            ]
        )

        summary = result.summary
        assert summary["byFramework"]["openai"] == 2
        assert summary["byCategory"]["api_client"] == 2
        assert summary["byConfidence"]["high"] == 1
        assert summary["byConfidence"]["medium"] == 1

    def test_scan_result_deduplicate(self):
        """Deduplicate removes duplicate framework+file combinations."""
        result = ScanResult(
            detections=[
                Detection(
                    framework=FrameworkType.OPENAI,
                    category=FrameworkCategory.API_CLIENT,
                    confidence=DetectionConfidence.HIGH,
                    source_file=Path("a.py"),
                    line_number=1,
                ),
                Detection(
                    framework=FrameworkType.OPENAI,
                    category=FrameworkCategory.API_CLIENT,
                    confidence=DetectionConfidence.MEDIUM,
                    source_file=Path("a.py"),
                    line_number=10,
                ),
            ]
        )

        deduped = result.deduplicate()
        assert len(deduped.detections) == 1


class TestPatternRegistry:
    """Tests for PatternRegistry."""

    def test_registry_has_default_patterns(self):
        """Registry initializes with default patterns."""
        registry = PatternRegistry()

        assert len(registry.patterns) >= 25  # 27 patterns registered
        assert len(registry.strategies) == 3  # import, pattern, annotation

    def test_registry_has_pytorch_pattern(self):
        """Registry includes PyTorch pattern."""
        registry = PatternRegistry()

        pytorch_patterns = [
            p for p in registry.patterns if p.framework == FrameworkType.PYTORCH
        ]
        assert len(pytorch_patterns) == 1
        assert "import torch" in pytorch_patterns[0].import_patterns

    def test_registry_has_langchain_pattern(self):
        """Registry includes LangChain pattern."""
        registry = PatternRegistry()

        langchain_patterns = [
            p for p in registry.patterns if p.framework == FrameworkType.LANGCHAIN
        ]
        assert len(langchain_patterns) == 1
        assert any("langchain" in p for p in langchain_patterns[0].import_patterns)

    def test_register_custom_pattern(self):
        """Custom patterns can be registered."""
        registry = PatternRegistry()
        initial_count = len(registry.patterns)

        registry.register_pattern(
            FrameworkPattern(
                framework="my_custom_ai",
                category=FrameworkCategory.LLM_FRAMEWORK,
                import_patterns=["import my_custom_ai"],
            )
        )

        assert len(registry.patterns) == initial_count + 1

    def test_get_patterns_by_category(self):
        """Patterns can be filtered by category."""
        registry = PatternRegistry()

        agent_patterns = registry.get_patterns_by_category(
            FrameworkCategory.AGENT_FRAMEWORK
        )
        assert len(agent_patterns) >= 3  # autogen, crewai, agency_swarm


class TestImportAnalysisStrategy:
    """Tests for ImportAnalysisStrategy."""

    def test_detects_simple_import(self):
        """Detects simple import statements."""
        patterns = [
            FrameworkPattern(
                framework=FrameworkType.OPENAI,
                category=FrameworkCategory.API_CLIENT,
                import_patterns=["import openai"],
            )
        ]
        strategy = ImportAnalysisStrategy(patterns)

        content = "import openai\n\nclient = openai.OpenAI()"
        detections = strategy.detect(Path("test.py"), content)

        assert len(detections) == 1
        assert detections[0].framework == FrameworkType.OPENAI
        assert detections[0].confidence == DetectionConfidence.HIGH
        assert detections[0].line_number == 1

    def test_detects_from_import(self):
        """Detects from ... import statements."""
        patterns = [
            FrameworkPattern(
                framework=FrameworkType.LANGCHAIN,
                category=FrameworkCategory.LLM_FRAMEWORK,
                import_patterns=["from langchain"],
            )
        ]
        strategy = ImportAnalysisStrategy(patterns)

        content = "from langchain.llms import OpenAI"
        detections = strategy.detect(Path("test.py"), content)

        assert len(detections) == 1
        assert detections[0].framework == FrameworkType.LANGCHAIN

    def test_skips_comments(self):
        """Skips commented import lines."""
        patterns = [
            FrameworkPattern(
                framework=FrameworkType.OPENAI,
                category=FrameworkCategory.API_CLIENT,
                import_patterns=["import openai"],
            )
        ]
        strategy = ImportAnalysisStrategy(patterns)

        content = "# import openai\n# from openai import Client"
        detections = strategy.detect(Path("test.py"), content)

        assert len(detections) == 0


class TestPatternMatchingStrategy:
    """Tests for PatternMatchingStrategy."""

    def test_detects_code_patterns(self):
        """Detects code usage patterns."""
        patterns = [
            FrameworkPattern(
                framework=FrameworkType.PYTORCH,
                category=FrameworkCategory.DEEP_LEARNING,
                import_patterns=[],
                code_patterns=["torch.nn."],
            )
        ]
        strategy = PatternMatchingStrategy(patterns)

        content = "model = torch.nn.Linear(10, 5)"
        detections = strategy.detect(Path("test.py"), content)

        assert len(detections) == 1
        assert detections[0].framework == FrameworkType.PYTORCH
        assert detections[0].confidence == DetectionConfidence.MEDIUM


class TestAnnotationStrategy:
    """Tests for AnnotationStrategy."""

    def test_detects_aigrc_annotation(self):
        """Detects @aigrc: annotations."""
        strategy = AnnotationStrategy()

        content = "# @aigrc:asset my-component\ndef process():\n    pass"
        detections = strategy.detect(Path("test.py"), content)

        assert len(detections) == 1
        assert "@aigrc:" in detections[0].evidence


class TestScanner:
    """Tests for Scanner class."""

    def test_scanner_initialization(self):
        """Scanner initializes with default settings."""
        scanner = Scanner()
        assert scanner._include == ["*.py", "*.ipynb"]
        assert len(scanner._exclude) > 0

    def test_scanner_custom_patterns(self):
        """Scanner accepts custom include/exclude patterns."""
        scanner = Scanner(
            include_patterns=["*.py"],
            exclude_patterns=["**/test/**"],
        )

        assert scanner._include == ["*.py"]
        assert "**/test/**" in scanner._exclude

    @pytest.mark.asyncio
    async def test_scan_file_with_imports(self):
        """Scanner detects imports in a file."""
        scanner = Scanner()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("import openai\nfrom langchain import chains\n")
            f.flush()

            detections = await scanner.scan_file(Path(f.name))

        assert len(detections) >= 2
        frameworks = {
            d.framework.value if hasattr(d.framework, "value") else d.framework
            for d in detections
        }
        assert "openai" in frameworks
        assert "langchain" in frameworks

    @pytest.mark.asyncio
    async def test_scan_directory(self):
        """Scanner scans a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "main.py").write_text("import torch\n")
            (Path(tmpdir) / "api.py").write_text("from openai import OpenAI\n")

            scanner = Scanner()
            result = await scanner.scan_directory(Path(tmpdir))

            assert result.files_scanned == 2
            assert len(result.detections) >= 2
            assert "pytorch" in result.frameworks_found
            assert "openai" in result.frameworks_found

    def test_scan_directory_sync(self):
        """Sync scan works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("import anthropic\n")

            scanner = Scanner()
            result = scanner.scan_directory_sync(Path(tmpdir))

            assert "anthropic" in result.frameworks_found

    @pytest.mark.asyncio
    async def test_scan_excludes_venv(self):
        """Scanner excludes venv directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create venv directory with file
            venv_dir = Path(tmpdir) / ".venv"
            venv_dir.mkdir()
            (venv_dir / "openai_wrapper.py").write_text("import openai\n")

            # Create main file
            (Path(tmpdir) / "main.py").write_text("print('hello')\n")

            scanner = Scanner()
            result = await scanner.scan_directory(Path(tmpdir))

            # Should only scan main.py, not .venv contents
            assert result.files_scanned == 1
            assert "openai" not in result.frameworks_found

    @pytest.mark.asyncio
    async def test_scan_detects_model_files(self):
        """Scanner detects model files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model files
            (Path(tmpdir) / "model.pt").touch()
            (Path(tmpdir) / "weights.safetensors").touch()
            (Path(tmpdir) / "model.onnx").touch()

            scanner = Scanner()
            result = await scanner.scan_directory(Path(tmpdir))

            assert len(result.model_files) == 3

    @pytest.mark.asyncio
    async def test_scan_with_progress_callback(self):
        """Progress callback is called during scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("import torch\n")
            (Path(tmpdir) / "b.py").write_text("import keras\n")

            progress_calls = []

            def on_progress(progress):
                progress_calls.append(progress)

            scanner = Scanner()
            await scanner.scan_directory(Path(tmpdir), progress_callback=on_progress)

            assert len(progress_calls) == 2


class TestSuggestAssetCard:
    """Tests for suggest_asset_card function."""

    def test_suggest_from_empty_result(self):
        """Suggestion works with empty result."""
        result = ScanResult()
        suggestion = suggest_asset_card(result)

        assert suggestion["name"] == "Detected AI Asset"
        assert suggestion["riskFactors"]["autonomousDecisions"] is False
        assert suggestion["classification"]["level"] == "minimal"

    def test_suggest_from_agent_frameworks(self):
        """Agent frameworks result in high risk."""
        result = ScanResult(
            detections=[
                Detection(
                    framework=FrameworkType.AUTOGEN,
                    category=FrameworkCategory.AGENT_FRAMEWORK,
                    confidence=DetectionConfidence.HIGH,
                    source_file=Path("agent.py"),
                )
            ]
        )
        suggestion = suggest_asset_card(result)

        assert suggestion["classification"]["level"] == "high"
        assert suggestion["riskFactors"]["autonomousDecisions"] is True
        assert suggestion["riskFactors"]["toolExecution"] is True

    def test_suggest_from_api_clients(self):
        """API clients result in limited risk."""
        result = ScanResult(
            detections=[
                Detection(
                    framework=FrameworkType.OPENAI,
                    category=FrameworkCategory.API_CLIENT,
                    confidence=DetectionConfidence.HIGH,
                    source_file=Path("api.py"),
                )
            ]
        )
        suggestion = suggest_asset_card(result)

        assert suggestion["classification"]["level"] == "limited"
        assert suggestion["riskFactors"]["externalDataAccess"] is True
        assert "OpenAI" in suggestion["technical"]["model_providers"]

    def test_suggest_includes_metadata(self):
        """Suggestion includes scan metadata."""
        result = ScanResult(
            files_scanned=10,
            scan_duration_ms=150.5,
            detections=[
                Detection(
                    framework=FrameworkType.PYTORCH,
                    category=FrameworkCategory.DEEP_LEARNING,
                    confidence=DetectionConfidence.HIGH,
                    source_file=Path("model.py"),
                )
            ],
        )
        suggestion = suggest_asset_card(result)

        assert suggestion["metadata"]["filesScanned"] == 10
        assert suggestion["metadata"]["detectionsCount"] == 1
        assert suggestion["metadata"]["scanDurationMs"] == 150.5


class TestDetectionIntegration:
    """Integration tests for the detection engine."""

    def test_full_scan_workflow(self):
        """Full scan workflow from directory to suggestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a realistic project structure
            (Path(tmpdir) / "main.py").write_text(
                """
import openai
from langchain.llms import ChatOpenAI
from crewai import Agent, Task, Crew

client = openai.OpenAI()
llm = ChatOpenAI()

agent = Agent(role="researcher", goal="research")
task = Task(description="do research", agent=agent)
crew = Crew(agents=[agent], tasks=[task])
"""
            )

            (Path(tmpdir) / "models" / "placeholder").parent.mkdir()
            (Path(tmpdir) / "models" / "weights.pt").touch()

            scanner = Scanner()
            result = scanner.scan_directory_sync(Path(tmpdir))

            # Check detections
            assert result.has_ai_components
            assert "openai" in result.frameworks_found
            assert "langchain" in result.frameworks_found
            assert "crewai" in result.frameworks_found
            assert len(result.model_files) == 1

            # Generate suggestion
            suggestion = suggest_asset_card(result)
            assert suggestion["classification"]["level"] == "high"  # crewai = agent
            assert suggestion["riskFactors"]["autonomousDecisions"] is True

    def test_scan_all_supported_frameworks(self):
        """Scanner detects all supported framework types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with many imports
            imports = """
import torch
import tensorflow
from langchain import chains
from autogen import AssistantAgent
from crewai import Crew
import openai
import anthropic
from transformers import AutoModel
import chromadb
"""
            (Path(tmpdir) / "all_frameworks.py").write_text(imports)

            scanner = Scanner()
            result = scanner.scan_directory_sync(Path(tmpdir))

            expected = {
                "pytorch",
                "tensorflow",
                "langchain",
                "autogen",
                "crewai",
                "openai",
                "anthropic",
                "huggingface",
                "chromadb",
            }
            assert expected.issubset(result.frameworks_found)
