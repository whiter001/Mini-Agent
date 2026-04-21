"""Test auto skill selection and context injection."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_agent.agent import Agent
from mini_agent.config import ToolsConfig
from mini_agent.schema import FunctionCall, Message, ToolCall
from mini_agent.tools.auto_skill import build_auto_skill_context, maybe_create_auto_skill
from mini_agent.tools.skill_loader import SkillLoader


def create_test_skill(skill_dir: Path, name: str, description: str, content: str, metadata: str = ""):
    """Create a test skill file."""
    skill_file = skill_dir / "SKILL.md"
    skill_content = f"""---
name: {name}
description: {description}
{metadata}---

{content}
"""
    skill_file.write_text(skill_content, encoding="utf-8")


def test_select_relevant_skills_prefers_matching_skill():
    """Auto skill selection should favor the most relevant skill."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_dir = Path(tmpdir) / "pdf"
        pdf_dir.mkdir()
        create_test_skill(
            pdf_dir,
            "pdf",
            "Create and edit PDF documents",
            "Use this skill for PDF forms and document processing.",
        )

        github_dir = Path(tmpdir) / "github"
        github_dir.mkdir()
        create_test_skill(
            github_dir,
            "github",
            "Work with GitHub pull requests and issues",
            "Use this skill for GitHub workflows.",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        selected = loader.select_relevant_skills("Please help me create a PDF form", max_skills=1)

        assert len(selected) == 1
        assert selected[0].name == "pdf"
        assert loader.get_auto_skills_prompt("hello there") == ""


def test_select_relevant_skills_uses_metadata_tags():
    """Auto skill selection should consider metadata tags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        slides_dir = Path(tmpdir) / "slides"
        slides_dir.mkdir()
        create_test_skill(
            slides_dir,
            "slides",
            "Create presentation decks",
            "Use this skill for presentation workflows.",
            metadata="""metadata:\n  hermes:\n    tags:\n      - slides\n      - presentation\n""",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        selected = loader.select_relevant_skills("Need slides for a talk", max_skills=1)

        assert len(selected) == 1
        assert selected[0].name == "slides"


def test_build_auto_skill_context_returns_system_message():
    """Auto skill context should be wrapped as a temporary system message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_dir = Path(tmpdir) / "pdf"
        pdf_dir.mkdir()
        create_test_skill(
            pdf_dir,
            "pdf",
            "Create and edit PDF documents",
            "Use this skill for PDF forms and document processing.",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        context = build_auto_skill_context(loader, "Need to generate a PDF", max_skills=1)

        assert len(context) == 1
        assert isinstance(context[0], Message)
        assert context[0].role == "system"
        assert "Auto-loaded Skills" in context[0].content
        assert "pdf" in context[0].content.lower()


def test_agent_ephemeral_context_isolated_from_history():
    """Temporary skill context should not be stored in the persisted history."""
    agent = Agent(llm_client=object(), system_prompt="system", tools=[], max_steps=1)
    agent.add_user_message("Need help with docs")
    agent.set_ephemeral_context([Message(role="system", content="auto skills")])

    active_messages = agent._get_active_messages()

    assert len(active_messages) == 3
    assert active_messages[0].role == "system"
    assert active_messages[1].role == "user"
    assert active_messages[2].role == "system"

    agent.clear_ephemeral_context()
    assert len(agent._get_active_messages()) == 2


def test_skill_loader_scans_extra_skill_directories():
    """User skill directories should be scanned alongside bundled skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundled_dir = Path(tmpdir) / "bundled"
        generated_dir = Path(tmpdir) / "generated"
        bundled_dir.mkdir()
        generated_dir.mkdir()

        create_test_skill(
            bundled_dir,
            "bundled",
            "Bundled skill",
            "Bundled workflow.",
        )
        create_test_skill(
            generated_dir,
            "generated",
            "Generated skill",
            "Generated workflow.",
        )

        loader = SkillLoader(str(bundled_dir), [str(generated_dir)])
        skills = loader.discover_skills()

        assert {skill.name for skill in skills} == {"bundled", "generated"}


def test_auto_skill_creation_persists_and_deduplicates():
    """Auto skill creation should write skills to disk and reuse identical content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))

        turn_messages = [Message(role="user", content="Create a PDF form workflow")]
        for idx in range(5):
            turn_messages.append(
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"call-{idx}",
                            type="function",
                            function=FunctionCall(name="bash", arguments={"command": f"step {idx}"}),
                        )
                    ],
                )
            )
            turn_messages.append(Message(role="tool", content=f"done {idx}", tool_call_id=f"call-{idx}", name="bash"))

        first = maybe_create_auto_skill(
            loader,
            turn_messages,
            "Task completed successfully.",
            auto_skill_dir=str(skill_dir),
            min_tool_calls=5,
        )
        second = maybe_create_auto_skill(
            loader,
            turn_messages,
            "Task completed successfully.",
            auto_skill_dir=str(skill_dir),
            min_tool_calls=5,
        )

        assert first.created is True
        assert first.skill_path is not None and first.skill_path.exists()
        assert second.created is True
        assert second.skill_path == first.skill_path


def test_auto_skill_creation_suffixed_when_content_changes():
    """Changed workflow content should create a new skill file instead of overwriting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))

        def build_turn_messages(command: str) -> list[Message]:
            messages = [Message(role="user", content="Create a PDF form workflow")]
            for idx in range(5):
                messages.append(
                    Message(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ToolCall(
                                id=f"call-{idx}-{command}",
                                type="function",
                                function=FunctionCall(name="bash", arguments={"command": f"{command} {idx}"}),
                            )
                        ],
                    )
                )
                messages.append(
                    Message(role="tool", content=f"done {command} {idx}", tool_call_id=f"call-{idx}-{command}", name="bash")
                )
            return messages

        first = maybe_create_auto_skill(
            loader,
            build_turn_messages("alpha"),
            "Task completed successfully.",
            auto_skill_dir=str(skill_dir),
            min_tool_calls=5,
        )
        second = maybe_create_auto_skill(
            loader,
            build_turn_messages("beta"),
            "Task completed successfully.",
            auto_skill_dir=str(skill_dir),
            min_tool_calls=5,
        )

        assert first.skill_path is not None
        assert second.skill_path is not None
        assert first.skill_path != second.skill_path


def test_auto_skill_prompt_truncates_large_skill_content():
    """Auto-loaded skills should stay compact to protect the request context window."""
    with tempfile.TemporaryDirectory() as tmpdir:
        large_dir = Path(tmpdir) / "large"
        large_dir.mkdir()
        create_test_skill(
            large_dir,
            "large-skill",
            "Very large skill",
            "X" * 5000,
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        prompt = loader.get_auto_skills_prompt("need a large skill", max_skills=1)

        assert "large-skill" in prompt
        assert "use get_skill for the full version" in prompt.lower()
        assert len(prompt) < 3500


@pytest.mark.parametrize("invalid_value", [0, -1])
def test_tools_config_rejects_non_positive_auto_skill_min_tool_calls(invalid_value):
    """Auto skill creation threshold must stay positive."""
    with pytest.raises(ValidationError):
        ToolsConfig(auto_skill_min_tool_calls=invalid_value)
