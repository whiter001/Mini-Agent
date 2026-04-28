"""Focused tests for agent history compaction behaviour."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mini_agent.agent import Agent
from mini_agent.schema import FunctionCall, LLMResponse, Message, ToolCall


def _append_round(agent: Agent, round_num: int) -> None:
    agent.add_user_message(f"request {round_num}")
    agent.messages.append(
        Message(
            role="assistant",
            content=f"assistant {round_num}",
            tool_calls=[
                ToolCall(
                    id=f"call-{round_num}",
                    type="function",
                    function=FunctionCall(name="bash", arguments={"command": f"echo {round_num}"}),
                )
            ],
        )
    )
    agent.messages.append(
        Message(
            role="tool",
            content=f"result {round_num}",
            tool_call_id=f"call-{round_num}",
            name="bash",
        )
    )


def _actual_user_messages(agent: Agent) -> list[str]:
    messages: list[str] = []
    for message in agent.messages:
        if message.role != "user" or not isinstance(message.content, str):
            continue
        if message.content.startswith("[Conversation Summary Anchor]"):
            continue
        if message.content.startswith("[Assistant Execution Summary]"):
            continue
        messages.append(message.content)
    return messages


@pytest.mark.asyncio
async def test_summary_creates_anchor_and_keeps_recent_rounds(tmp_path):
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=LLMResponse(content="merged anchor summary", finish_reason="stop"))

    agent = Agent(
        llm_client=llm,
        system_prompt="system",
        tools=[],
        workspace_dir=str(tmp_path),
        token_limit=1,
        summary_recent_rounds=2,
    )

    for round_num in range(1, 5):
        _append_round(agent, round_num)

    agent.api_total_tokens = agent.token_limit + 1
    await agent._summarize_messages()

    assert agent.messages[1].content.startswith("[Conversation Summary Anchor]")
    assert _actual_user_messages(agent) == ["request 3", "request 4"]
    assert llm.generate.await_count == 1


@pytest.mark.asyncio
async def test_summary_merges_existing_anchor_before_archiving_more_rounds(tmp_path):
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=LLMResponse(content="updated rolling anchor", finish_reason="stop"))

    agent = Agent(
        llm_client=llm,
        system_prompt="system",
        tools=[],
        workspace_dir=str(tmp_path),
        token_limit=1,
        summary_recent_rounds=2,
    )
    agent.messages.append(Message(role="user", content="[Conversation Summary Anchor]\n\nolder context"))

    for round_num in range(3, 6):
        _append_round(agent, round_num)

    agent.api_total_tokens = agent.token_limit + 1
    await agent._summarize_messages()

    assert agent.messages[1].content == "[Conversation Summary Anchor]\n\nupdated rolling anchor"
    assert _actual_user_messages(agent) == ["request 4", "request 5"]
    assert llm.generate.await_count == 1


@pytest.mark.asyncio
async def test_summary_falls_back_to_round_summary_when_only_one_round_exists(tmp_path):
    llm = MagicMock()
    llm.generate = AsyncMock(return_value=LLMResponse(content="single round summary", finish_reason="stop"))

    agent = Agent(
        llm_client=llm,
        system_prompt="system",
        tools=[],
        workspace_dir=str(tmp_path),
        token_limit=1,
        summary_recent_rounds=2,
    )
    _append_round(agent, 1)

    agent.api_total_tokens = agent.token_limit + 1
    await agent._summarize_messages()

    assert [message.role for message in agent.messages] == ["system", "user", "user"]
    assert agent.messages[1].content == "request 1"
    assert agent.messages[2].content == "[Assistant Execution Summary]\n\nsingle round summary"