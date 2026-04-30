"""Tests for temperature defaults and request propagation."""

from types import SimpleNamespace

import pytest

from mini_agent.config import Config
from mini_agent.llm import AnthropicClient, LLMClient, OpenAIClient
from mini_agent.schema import LLMProvider, Message


def test_config_defaults_temperature_to_point_seven(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text('api_key: "test-key"\n', encoding="utf-8")

    config = Config.from_yaml(config_path)

    assert config.llm.temperature == pytest.approx(0.7)


def test_config_reads_temperature_override(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text('api_key: "test-key"\ntemperature: 0.55\n', encoding="utf-8")

    config = Config.from_yaml(config_path)

    assert config.llm.temperature == pytest.approx(0.55)


def test_wrapper_propagates_temperature_to_provider_clients():
    anthropic_client = LLMClient(
        api_key="test-key",
        provider=LLMProvider.ANTHROPIC,
        temperature=0.65,
    )
    openai_client = LLMClient(
        api_key="test-key",
        provider=LLMProvider.OPENAI,
        temperature=0.45,
    )

    assert anthropic_client._client.temperature == pytest.approx(0.65)
    assert openai_client._client.temperature == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_anthropic_client_passes_temperature_to_api(monkeypatch):
    client = AnthropicClient(api_key="test-key")
    captured: dict[str, float] = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(client.client.messages, "create", fake_create)

    await client._make_api_request(
        None,
        [{"role": "user", "content": "hello"}],
        None,
    )

    assert captured["temperature"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_openai_client_passes_temperature_to_api(monkeypatch):
    client = OpenAIClient(api_key="test-key")
    captured: dict[str, float] = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)

    await client._make_api_request(
        [{"role": "user", "content": "hello"}],
        None,
    )

    assert captured["temperature"] == pytest.approx(0.7)


def test_anthropic_client_merges_multiple_system_messages():
    client = AnthropicClient(api_key="test-key")

    system_message, api_messages = client._convert_messages(
        [
            Message(role="system", content="MAIN SYSTEM"),
            Message(role="user", content="hello"),
            Message(role="system", content="EPHEMERAL SKILL"),
        ]
    )

    assert system_message == "MAIN SYSTEM\n\nEPHEMERAL SKILL"
    assert api_messages == [{"role": "user", "content": "hello"}]