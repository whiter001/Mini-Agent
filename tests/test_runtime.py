"""Tests for runtime prompt composition helpers."""

from mini_agent.runtime import inject_optional_prompt_block


def test_inject_optional_prompt_block_replaces_placeholder():
    prompt = "System prompt\n\n{SKILLS_METADATA}\n\nWorkspace info"

    result = inject_optional_prompt_block(prompt, "skill metadata", placeholder="{SKILLS_METADATA}")

    assert result == "System prompt\n\nskill metadata\n\nWorkspace info"


def test_inject_optional_prompt_block_appends_when_placeholder_missing():
    prompt = "System prompt"

    result = inject_optional_prompt_block(prompt, "skill metadata", placeholder="{SKILLS_METADATA}")

    assert result == "System prompt\n\nskill metadata"


def test_inject_optional_prompt_block_removes_placeholder_when_block_empty():
    prompt = "System prompt\n\n{SKILLS_METADATA}\n\nWorkspace info"

    result = inject_optional_prompt_block(prompt, "", placeholder="{SKILLS_METADATA}")

    assert result == "System prompt\n\n\n\nWorkspace info"