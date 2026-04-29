"""Tests for optional max_steps behavior."""

from types import SimpleNamespace

import pytest

from mini_agent.acp import MiniMaxACPAgent
from mini_agent.agent import Agent
from mini_agent.config import AgentConfig, Config, LLMConfig, ToolsConfig
from mini_agent.schema import FunctionCall, LLMResponse, ToolCall
from mini_agent.tools.base import Tool, ToolResult


class DummyConn:
    def __init__(self):
        self.updates = []

    async def sessionUpdate(self, payload):
        self.updates.append(payload)


class TwoStepLLM:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, tools=None):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                thinking="calling echo",
                tool_calls=[
                    ToolCall(
                        id="tool1",
                        type="function",
                        function=FunctionCall(name="echo", arguments={"text": "ping"}),
                    )
                ],
                finish_reason="tool",
            )
        return LLMResponse(content="done", thinking=None, tool_calls=None, finish_reason="stop")


class EchoTool(Tool):
    @property
    def name(self):
        return "echo"

    @property
    def description(self):
        return "Echo helper"

    @property
    def parameters(self):
        return {"type": "object", "properties": {"text": {"type": "string"}}}

    async def execute(self, text: str):
        return ToolResult(success=True, content=f"tool:{text}")


def test_config_without_max_steps_defaults_to_unbounded(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        '\n'.join(
            [
                'api_key: "test-key"',
                'api_base: "https://api.minimax.io"',
                'model: "MiniMax-M2.7"',
            ]
        ),
        encoding="utf-8",
    )

    config = Config.from_yaml(config_path)

    assert config.agent.max_steps is None


@pytest.mark.asyncio
async def test_agent_without_max_steps_runs_until_completion(tmp_path, capsys):
    llm = TwoStepLLM()
    agent = Agent(
        llm_client=llm,
        system_prompt="system",
        tools=[EchoTool()],
        max_steps=None,
        workspace_dir=str(tmp_path),
    )
    agent.add_user_message("hello")

    result = await agent.run()

    assert result == "done"
    assert llm.calls == 2
    assert "Step 1/∞" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_acp_without_max_steps_runs_until_completion(tmp_path):
    config = Config(
        llm=LLMConfig(api_key="test-key"),
        agent=AgentConfig(workspace_dir=str(tmp_path)),
        tools=ToolsConfig(),
    )
    llm = TwoStepLLM()
    conn = DummyConn()
    agent = MiniMaxACPAgent(conn, config, llm, [EchoTool()], "system")

    session = await agent.newSession(SimpleNamespace(cwd=None))
    prompt = SimpleNamespace(sessionId=session.sessionId, prompt=[{"text": "hello"}])
    response = await agent.prompt(prompt)

    assert response.stopReason == "end_turn"
    assert llm.calls == 2
    assert any("tool:ping" in str(update) for update in conn.updates)
