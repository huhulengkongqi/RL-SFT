"""Tests for TaskGenerator functionality."""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent_sft.task_generator import (
    Difficulty,
    Domain,
    SeedPrompt,
    SeedPromptPool,
    SourceType,
    TaskGenerator,
    TaskGeneratorConfig,
    TaskTestCase,
)
from src.agent_sft.task_generator.models import ValidationReport


def create_test_prompt(**kwargs) -> SeedPrompt:
    """Create a test SeedPrompt with default values."""
    defaults = {
        "id": str(uuid.uuid4()),
        "domain": Domain.CODE_DEBUG,
        "difficulty": Difficulty.MEDIUM,
        "prompt": "Fix the bug in this Python function: def add(a, b): return a - b",
        "test_cases": [TaskTestCase(input=[1, 2], expected_output=3)],
        "validator_code": "def validate(fn): return fn(1, 2) == 3",
        "source": SourceType.HUMAN_CURATED,
        "quality_score": 0.9,
    }
    defaults.update(kwargs)
    return SeedPrompt(**defaults)


class TestTaskGeneratorConfig:
    def test_default_config(self):
        """Test default config values."""
        config = TaskGeneratorConfig()
        assert config.max_concurrent_requests == 5
        assert config.temperature == 0.7
        assert config.enable_mutation is True
        assert config.mutation_rate == 0.3
        assert config.min_quality_score == 0.8


class TestTaskGeneratorInit:
    def test_init_with_defaults(self):
        """Test initializing TaskGenerator with minimal args."""
        pool = SeedPromptPool()
        llm_client = MagicMock()
        generator = TaskGenerator(pool, llm_client)
        assert generator.seed_pool == pool
        assert generator.llm_client == llm_client
        assert isinstance(generator.config, TaskGeneratorConfig)
        assert generator.validators == {}

    def test_init_with_custom_config(self):
        """Test initializing with custom config."""
        pool = SeedPromptPool()
        llm_client = MagicMock()
        config = TaskGeneratorConfig(max_concurrent_requests=10, temperature=0.5)
        generator = TaskGenerator(pool, llm_client, config=config)
        assert generator.config.max_concurrent_requests == 10
        assert generator.config.temperature == 0.5

    def test_init_with_validators(self):
        """Test initializing with pre-registered validators."""
        pool = SeedPromptPool()
        llm_client = MagicMock()
        validator = lambda x: True
        validators = {Domain.CODE_DEBUG: validator}
        generator = TaskGenerator(pool, llm_client, validators=validators)
        assert Domain.CODE_DEBUG in generator.validators

    def test_register_validator(self):
        """Test registering a validator after initialization."""
        pool = SeedPromptPool()
        llm_client = MagicMock()
        generator = TaskGenerator(pool, llm_client)
        validator = lambda x: True
        generator.register_validator(Domain.CODE_DEBUG, validator)
        assert Domain.CODE_DEBUG in generator.validators


class TestGenerateBatch:
    @pytest.mark.asyncio
    async def test_generate_batch_with_validation(self):
        """Test batch generation with comprehensive validation enabled."""
        from src.agent_sft.task_generator.validator import TaskValidator
        from src.infra.sandbox.execution_manager import SandboxExecutor
        from unittest.mock import Mock

        pool = SeedPromptPool()
        pool.add_batch([create_test_prompt() for _ in range(3)])

        llm_client = MagicMock()

        # Mock validator
        mock_validator = Mock(spec=TaskValidator)
        mock_validator.validate_task = AsyncMock(return_value=ValidationReport(
            function_signature_valid=True,
            sandbox_execution_passed=True,
            execution_time=1.0
        ))
        mock_validator.validate_batch = AsyncMock(return_value=[
            ValidationReport(
                function_signature_valid=True,
                sandbox_execution_passed=True,
                execution_time=1.0
            ) for _ in range(3)
        ])

        generator = TaskGenerator(
            pool,
            llm_client,
            task_validator=mock_validator
        )

        tasks = await generator.generate_batch(
            batch_size=3,
            mode="seed_only",
            enable_validation=True
        )

        assert len(tasks) == 3
        assert all(t.validation_passed is True for t in tasks)
        assert mock_validator.validate_batch.called

    @pytest.mark.asyncio
    async def test_generate_batch_without_validation(self):
        """Test batch generation without validation."""
        pool = SeedPromptPool()
        pool.add_batch([create_test_prompt() for _ in range(3)])

        llm_client = MagicMock()
        generator = TaskGenerator(pool, llm_client)

        tasks = await generator.generate_batch(
            batch_size=3,
            mode="seed_only",
            enable_validation=False
        )

        assert len(tasks) == 3
        assert all(t.validation_passed is True for t in tasks)

    @pytest.mark.asyncio
    async def test_generate_batch_validation_failure(self):
        """Test batch generation with validation failures."""
        from src.agent_sft.task_generator.validator import TaskValidator
        from unittest.mock import Mock

        pool = SeedPromptPool()
        pool.add_batch([create_test_prompt() for _ in range(3)])

        llm_client = MagicMock()

        # Mock validator that returns failure
        mock_validator = Mock(spec=TaskValidator)
        mock_validator.validate_batch = AsyncMock(return_value=[
            ValidationReport(
                function_signature_valid=False,
                sandbox_execution_passed=False,
                execution_time=1.0
            ) for _ in range(3)
        ])

        generator = TaskGenerator(
            pool,
            llm_client,
            task_validator=mock_validator
        )

        tasks = await generator.generate_batch(
            batch_size=3,
            mode="seed_only",
            enable_validation=True
        )

        # Should filter out invalid tasks
        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_generate_batch_partial_validation(self):
        """Test batch generation with some validation failures."""
        from src.agent_sft.task_generator.validator import TaskValidator
        from unittest.mock import Mock

        pool = SeedPromptPool()
        pool.add_batch([create_test_prompt() for _ in range(5)])

        llm_client = MagicMock()

        # Mock validator with mixed results
        reports = [
            ValidationReport(
                function_signature_valid=True,
                sandbox_execution_passed=True,
                execution_time=1.0
            ),
            ValidationReport(
                function_signature_valid=True,
                sandbox_execution_passed=False,
                execution_time=1.0
            ),
            ValidationReport(
                function_signature_valid=False,
                sandbox_execution_passed=True,
                execution_time=1.0
            ),
            ValidationReport(
                function_signature_valid=True,
                sandbox_execution_passed=True,
                execution_time=1.0
            ),
            ValidationReport(
                function_signature_valid=False,
                sandbox_execution_passed=False,
                execution_time=1.0
            )
        ]

        mock_validator = Mock(spec=TaskValidator)
        mock_validator.validate_batch = AsyncMock(return_value=reports)

        generator = TaskGenerator(
            pool,
            llm_client,
            task_validator=mock_validator
        )

        tasks = await generator.generate_batch(
            batch_size=5,
            mode="seed_only",
            enable_validation=True
        )

        # Should only include tasks that pass both validations
        assert len(tasks) == 2
        assert all(t.validation_passed is True for t in tasks)
    @pytest.mark.asyncio
    async def test_generate_batch_seed_only_mode(self):
        """Test batch generation in seed_only mode."""
        pool = SeedPromptPool()
        pool.add_batch([create_test_prompt() for _ in range(5)])

        llm_client = MagicMock()
        generator = TaskGenerator(pool, llm_client)

        tasks = await generator.generate_batch(batch_size=3, mode="seed_only")
        assert len(tasks) == 3
        assert all(t.validation_passed is True for t in tasks)

    @pytest.mark.asyncio
    async def test_generate_batch_empty_pool(self):
        """Test generation with empty pool returns empty list."""
        pool = SeedPromptPool()
        llm_client = MagicMock()
        generator = TaskGenerator(pool, llm_client)

        tasks = await generator.generate_batch(batch_size=10)
        assert tasks == []

    @pytest.mark.asyncio
    async def test_invalid_mode_raises_error(self):
        """Test invalid generation mode raises ValueError."""
        pool = SeedPromptPool()
        llm_client = MagicMock()
        generator = TaskGenerator(pool, llm_client)

        with pytest.raises(ValueError, match="Invalid mode"):
            await generator.generate_batch(batch_size=1, mode="invalid_mode")

    @pytest.mark.asyncio
    async def test_generate_with_domain_filter(self):
        """Test generation with domain filtering."""
        pool = SeedPromptPool()
        pool.add_batch(
            [
                create_test_prompt(domain=Domain.CODE_DEBUG),
                create_test_prompt(domain=Domain.CODE_DEBUG),
                create_test_prompt(domain=Domain.MATH_REASONING),
            ]
        )
        llm_client = MagicMock()
        generator = TaskGenerator(pool, llm_client)

        tasks = await generator.generate_batch(
            batch_size=10, domains=[Domain.CODE_DEBUG], mode="seed_only"
        )
        assert len(tasks) == 2
        assert all(t.domain == Domain.CODE_DEBUG for t in tasks)

    @pytest.mark.asyncio
    async def test_generate_with_difficulty_filter(self):
        """Test generation with difficulty filtering."""
        pool = SeedPromptPool()
        pool.add_batch(
            [
                create_test_prompt(difficulty=Difficulty.EASY),
                create_test_prompt(difficulty=Difficulty.HARD),
            ]
        )
        llm_client = MagicMock()
        generator = TaskGenerator(pool, llm_client)

        tasks = await generator.generate_batch(
            batch_size=10, difficulty=Difficulty.EASY, mode="seed_only"
        )
        assert len(tasks) == 1
        assert all(t.difficulty == Difficulty.EASY for t in tasks)

    @pytest.mark.asyncio
    async def test_mutate_prompt_calls_llm(self):
        """Test that mutation mode calls the LLM client."""
        pool = SeedPromptPool()
        pool.add(create_test_prompt())

        llm_client = MagicMock()
        llm_client.achat = AsyncMock(return_value="Mutated prompt content")

        config = TaskGeneratorConfig(enable_mutation=True, mutation_rate=1.0)
        generator = TaskGenerator(pool, llm_client, config=config)

        tasks = await generator.generate_batch(batch_size=1, mode="seed_based")
        assert len(tasks) == 1

    @pytest.mark.asyncio
    async def test_validator_execution(self):
        """Test that validator code is executed during validation."""
        pool = SeedPromptPool()
        pool.add(create_test_prompt(validator_code="def validate(x): return True"))

        llm_client = MagicMock()
        generator = TaskGenerator(pool, llm_client)

        tasks = await generator.generate_batch(batch_size=1, mode="seed_only")
        assert len(tasks) == 1
        assert tasks[0].validation_passed is True

    @pytest.mark.asyncio
    async def test_custom_validator_used(self):
        """Test that custom registered validators are used."""
        pool = SeedPromptPool()
        pool.add(create_test_prompt(domain=Domain.CODE_DEBUG))

        llm_client = MagicMock()
        generator = TaskGenerator(pool, llm_client)

        call_count = 0

        def custom_validator(data):
            nonlocal call_count
            call_count += 1
            return True

        generator.register_validator(Domain.CODE_DEBUG, custom_validator)

        tasks = await generator.generate_batch(batch_size=1, mode="seed_only")
        assert len(tasks) == 1
        assert call_count == 1
