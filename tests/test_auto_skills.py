"""Test auto skill selection and context injection."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_agent.agent import Agent
from mini_agent.cli import build_turn_context
from mini_agent.config import ToolsConfig
from mini_agent.schema import FunctionCall, Message, ToolCall
from mini_agent.tools.auto_skill import _collect_turn_trace, build_auto_skill_context, maybe_create_auto_skill
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


def build_turn_messages(user_request: str, steps: list[tuple[str, dict[str, object], str]]) -> list[Message]:
    """Build a turn trace with assistant tool calls followed by tool responses."""
    messages = [Message(role="user", content=user_request)]
    for idx, (tool_name, arguments, result) in enumerate(steps):
        call_id = f"call-{idx}-{tool_name}"
        messages.append(
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id=call_id,
                        type="function",
                        function=FunctionCall(name=tool_name, arguments=arguments),
                    )
                ],
            )
        )
        messages.append(Message(role="tool", content=result, tool_call_id=call_id, name=tool_name))
    return messages


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


def test_build_turn_context_respects_auto_skill_toggle():
    """Per-turn context should skip auto skills when the feature is disabled."""

    class DummyMemoryStore:
        def build_turn_context(self, query: str):
            return [Message(role="system", content=f"memory:{query}")]

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

        context = build_turn_context(
            DummyMemoryStore(),
            loader,
            "Need to generate a PDF",
            1,
            enable_auto_skills=False,
        )

        assert len(context) == 1
        assert context[0].content == "memory:Need to generate a PDF"


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


def test_collect_turn_trace_matches_tool_results_by_call_id():
    """Tool results should be matched to their originating calls even if messages arrive out of order."""
    turn_messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-first",
                    type="function",
                    function=FunctionCall(name="first_tool", arguments={"step": 1}),
                ),
                ToolCall(
                    id="call-second",
                    type="function",
                    function=FunctionCall(name="second_tool", arguments={"step": 2}),
                ),
            ],
        ),
        Message(role="tool", content="second result", tool_call_id="call-second", name="second_tool"),
        Message(role="tool", content="first result", tool_call_id="call-first", name="first_tool"),
    ]

    trace = _collect_turn_trace(turn_messages)

    assert [step["name"] for step in trace] == ["first_tool", "second_tool"]
    assert [step["result"] for step in trace] == ["first result", "second result"]


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
        turn_messages = build_turn_messages(
            "Create a PDF form workflow",
            [("bash", {"command": f"step {idx}"}, f"done {idx}") for idx in range(5)],
        )

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
        assert first.tier == "approved"
        assert second.created is True
        assert second.skill_path == first.skill_path


def test_auto_skill_creation_suffixed_when_content_changes():
    """Changed workflow content should create a new skill file instead of overwriting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))

        def build_workflow(command: str) -> list[Message]:
            return build_turn_messages(
                "Create a PDF form workflow",
                [("bash", {"command": f"{command} {idx}"}, f"done {command} {idx}") for idx in range(5)],
            )

        first = maybe_create_auto_skill(
            loader,
            build_workflow("alpha"),
            "Task completed successfully.",
            auto_skill_dir=str(skill_dir),
            min_tool_calls=5,
        )
        second = maybe_create_auto_skill(
            loader,
            build_workflow("beta"),
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


def test_auto_skill_creation_handles_multiline_requests():
    """Generated skill frontmatter should remain parseable for multiline user requests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))

        user_request = "Create a workflow:\n- collect inputs\n- validate: yes"
        turn_messages = build_turn_messages(
            user_request,
            [("bash", {"command": f"step {idx}"}, f"done {idx}") for idx in range(5)],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "Task completed successfully.",
            auto_skill_dir=str(skill_dir),
            min_tool_calls=5,
        )

        assert result.created is True
        assert result.skill_path is not None

        loaded_skill = loader.load_skill(result.skill_path)

        assert loaded_skill is not None
        assert "collect inputs" in loaded_skill.content
        assert "validate: yes" in loaded_skill.content


def test_auto_skill_creation_records_quality_metadata_for_approved_skills():
    """High-quality workflows should be approved and carry structured quality metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "Update a document and verify the result",
            [
                ("read_file", {"path": "docs/spec.md"}, "Read the current spec."),
                ("edit_file", {"path": "docs/spec.md", "old_str": "old", "new_str": "new"}, "Updated the spec."),
                ("bash", {"command": "uv run pytest tests/test_auto_skills.py -q"}, "11 passed in 1.20s"),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "Updated the document and verified the tests still pass.",
            auto_skill_dir=str(skill_dir),
        )

        assert result.created is True
        assert result.tier == "approved"
        assert result.quality_score >= 8
        assert result.skill_path is not None

        loaded_skill = loader.load_skill(result.skill_path)

        assert loaded_skill is not None
        assert loaded_skill.metadata is not None
        assert loaded_skill.metadata["auto_skill"]["tier"] == "approved"
        assert loaded_skill.metadata["auto_skill"]["score"] == result.quality_score
        assert "discovery" in loaded_skill.metadata["auto_skill"]["metrics"]["categories"]
        assert "validation" in loaded_skill.metadata["auto_skill"]["metrics"]["categories"]


def test_auto_skill_creation_candidate_stays_out_of_loader_discovery():
    """Lower-confidence generated workflows should be stored as candidates and stay out of auto-loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "Quickly update a note",
            [
                ("bash", {"command": "echo step-1"}, "step 1 complete"),
                ("bash", {"command": "echo step-2"}, "step 2 complete"),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "Completed after updating the note contents.",
            auto_skill_dir=str(skill_dir),
        )

        assert result.created is True
        assert result.tier == "candidate"
        assert result.skill_path is not None
        assert "_candidates" in result.skill_path.parts

        fresh_loader = SkillLoader(str(skill_dir))
        discovered = fresh_loader.discover_skills()

        assert discovered == []


def test_auto_skill_creation_rejects_low_quality_single_step_workflow():
    """Single-step generic workflows should fail the quality gate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "Run one quick command",
            [("bash", {"command": "echo done"}, "done")],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "Done",
            auto_skill_dir=str(skill_dir),
        )

        assert result.created is False
        assert result.reason == "quality-gate"
        assert result.quality_score > 0


def test_auto_skill_content_sanitizes_environment_specific_values():
    """Generated skill content should replace absolute paths, timestamps, and IDs with placeholders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "Repair a local script and rerun it",
            [
                ("bash", {"command": r'python C:\Users\white\temp\repair.py'}, r"Updated C:\Users\white\temp\repair.py successfully"),
                ("bash", {"command": "pytest tests/test_auto_skills.py -q"}, "11 passed at 2026-04-24T17:42:49Z for run 123e4567-e89b-12d3-a456-426614174000"),
                ("bash", {"command": "echo clean"}, "Cleanup complete"),
                ("bash", {"command": "echo package"}, "Packaged output"),
                ("bash", {"command": "echo verify"}, "Verification complete"),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            r"Finished repairing C:\Users\white\temp\repair.py at 2026-04-24T17:42:49Z for run 123e4567-e89b-12d3-a456-426614174000",
            auto_skill_dir=str(skill_dir),
        )

        assert result.created is True
        assert result.skill_path is not None

        content = result.skill_path.read_text(encoding="utf-8")

        assert r"C:\Users\white\temp\repair.py" not in content
        assert "2026-04-24T17:42:49Z" not in content
        assert "123e4567-e89b-12d3-a456-426614174000" not in content
        assert "<path>" in content
        assert "<timestamp>" in content
        assert "<id>" in content


@pytest.mark.parametrize("invalid_value", [0, -1])
def test_tools_config_rejects_non_positive_auto_skill_min_tool_calls(invalid_value):
    """Auto skill creation threshold must stay positive."""
    with pytest.raises(ValidationError):
        ToolsConfig(auto_skill_min_tool_calls=invalid_value)


def test_tools_config_rejects_auto_skill_approved_score_below_candidate_score():
    """Approved quality threshold should never be lower than candidate threshold."""
    with pytest.raises(ValidationError):
        ToolsConfig(auto_skill_candidate_score=6, auto_skill_approved_score=5)
