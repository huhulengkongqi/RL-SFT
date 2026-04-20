"""Tests for vLLM client."""

import pytest

from infra.vllm_client.client import Message, VLLMClient
from infra.vllm_client.server import VLLMServerConfig


def test_message_model():
    """Test Message Pydantic model."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_server_config():
    """Test vLLM server config."""
    config = VLLMServerConfig(model="test-model", port=8001)
    assert config.model == "test-model"
    assert config.port == 8001
    assert config.gpu_memory_utilization == 0.9


def test_client_initialization():
    """Test client initialization."""
    client = VLLMClient(base_url="http://test:8000/v1", api_key="test")
    assert client.base_url == "http://test:8000/v1"
    assert client.api_key == "test"


def test_is_server_ready_when_down():
    """Test server ready check when server is down."""
    client = VLLMClient(base_url="http://localhost:9999/v1")
    assert not client.is_server_ready()
