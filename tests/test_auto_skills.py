"""Test auto skill selection and context injection."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mini_agent.agent import Agent
from mini_agent.cli import build_turn_context
from mini_agent.config import ToolsConfig
from mini_agent.schema import FunctionCall, Message, ToolCall
from mini_agent.tools.auto_skill import _collect_turn_trace, _merge_guidance_lists, _normalize_string_list, _summarize_final_outcome, build_auto_skill_context, cleanup_auto_skills, maybe_create_auto_skill
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


def create_generated_auto_skill(
    skill_dir: Path,
    directory_name: str,
    *,
    internal_name: str,
    score: int = 8,
    generated_at: str = "2026-04-29T00:00:00+00:00",
    warnings: list[str] | None = None,
    family_key: str | None = None,
    body: str = "Auto-generated workflow body.",
):
    """Create a generated auto skill fixture with structured metadata."""
    target_dir = skill_dir / directory_name
    target_dir.mkdir(parents=True, exist_ok=True)

    auto_skill_metadata: dict[str, object] = {
        "tier": "approved",
        "score": score,
        "warnings": warnings or [],
    }
    if family_key:
        auto_skill_metadata["family_key"] = family_key

    frontmatter = {
        "name": internal_name,
        "description": f"Auto-generated workflow for {directory_name}",
        "metadata": {
            "source": "mini-agent",
            "generated_at": generated_at,
            "auto_skill": auto_skill_metadata,
        },
    }
    content = (
        "---\n"
        f"{yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()}\n"
        "---\n\n"
        f"# {directory_name}\n\n"
        "## When to use\n\n"
        f"{directory_name}\n\n"
        "## Procedure\n\n"
        "1. Execute `bash`\n\n"
        "## Final outcome\n\n"
        f"{body}\n"
    )
    (target_dir / "SKILL.md").write_text(content, encoding="utf-8")


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


def test_select_relevant_skills_prefers_mixed_language_triggers():
    """Mixed Chinese/English prompts should favor skills with matching triggers and tool metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        browser_dir = Path(tmpdir) / "browser-flow"
        browser_dir.mkdir()
        create_test_skill(
            browser_dir,
            "browser-flow",
            "Autobrowser workflow helper",
            "# Overview\n\nUse this skill when you need to control Chrome with autobrowser.",
            metadata="""tools:\n  - autobrowser\ntriggers:\n  - 执行autobrowser help\ntags:\n  - browser\n  - automation\nplatform: windows\n""",
        )

        generic_dir = Path(tmpdir) / "generic-help"
        generic_dir.mkdir()
        create_test_skill(
            generic_dir,
            "generic-help",
            "Generic troubleshooting notes",
            "Use this skill for general command line tasks.",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        selected = loader.select_relevant_skills("执行autobrowser help", max_skills=1)

        assert len(selected) == 1
        assert selected[0].name == "browser-flow"


def test_select_relevant_skills_ignores_generic_url_fragment_noise():
    """Generic URL fragments like `question` or `com` should not pull in unrelated skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        autobrowser_dir = Path(tmpdir) / "autobrowser"
        autobrowser_dir.mkdir()
        create_test_skill(
            autobrowser_dir,
            "autobrowser",
            "Autobrowser workflow helper",
            "Use this skill when you need to drive autobrowser from the CLI.",
            metadata="tools:\n  - autobrowser\ntriggers:\n  - 用autobrowser答题\n",
        )

        internal_dir = Path(tmpdir) / "internal-comms"
        internal_dir.mkdir()
        create_test_skill(
            internal_dir,
            "internal-comms",
            "Internal communication helper",
            "Use this skill for FAQ answers, updates, and common questions.",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        selected = loader.select_relevant_skills(
            "用autobrowser帮我在https://zhidao.baidu.com/ihome/homepage/recommendquestion里答3道题",
            max_skills=2,
        )

        assert [skill.name for skill in selected] == ["autobrowser"]


def test_normalize_string_list_preserves_sentence_items_with_commas():
    """Sentence-style notes should stay intact when reloading existing auto-skill metadata."""
    assert _normalize_string_list("bash, write_file") == ["bash", "write_file"]
    assert _normalize_string_list(
        "Inspect the current state first, then act; avoid editing or submitting before you confirm the real target."
    ) == ["Inspect the current state first, then act; avoid editing or submitting before you confirm the real target."]


def test_summarize_final_outcome_keeps_reusable_lead_paragraph_only():
    """Long enumerated outcomes should collapse to a short reusable success summary."""
    final_result = (
        "成功获取了 **11 条消息**！以下是 https://x.com/home 中的 **10 条消息**：\n\n"
        "### 1. 用户A\n"
        "> 很长的正文\n\n"
        "### 2. 用户B\n"
        "> 另一条正文"
    )

    summary = _summarize_final_outcome(final_result, max_chars=220)

    assert summary == "成功获取了 **11 条消息**！以下是 https:/<path> 中的 **10 条消息**："
    assert "### 1." not in summary


def test_select_relevant_skills_can_still_use_distinctive_url_host_tokens():
    """Distinctive host tokens from a URL should still help site-specific skills match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        github_dir = Path(tmpdir) / "github"
        github_dir.mkdir()
        create_test_skill(
            github_dir,
            "github",
            "GitHub review helper",
            "Use this skill for GitHub pull requests and issues.",
            metadata="tags:\n  - git\n  - review\n",
        )

        generic_dir = Path(tmpdir) / "general"
        generic_dir.mkdir()
        create_test_skill(
            generic_dir,
            "general",
            "General helper",
            "Use this skill for generic writing and task support.",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        selected = loader.select_relevant_skills(
            "Please review https://github.com/example/project/pull/123",
            max_skills=1,
        )

        assert [skill.name for skill in selected] == ["github"]


def test_auto_skill_prompt_uses_short_url_host_tokens_for_section_selection():
    """Short hosts like x.com should still pull in the most relevant extraction section."""
    with tempfile.TemporaryDirectory() as tmpdir:
        browser_dir = Path(tmpdir) / "autobrowser"
        browser_dir.mkdir()
        create_test_skill(
            browser_dir,
            "autobrowser",
            "Autobrowser workflow helper",
            """# Overview

Use this skill to drive autobrowser from the CLI.

## Run

Use autobrowser on Unix and autobrowser.cmd on Windows.

## Feed Extraction

For https://x.com/home, keep a seen map keyed by /status/ links across viewport swaps and accumulate unique posts across scroll iterations.
""",
            metadata="tools:\n  - autobrowser\n",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        prompt = loader.get_auto_skills_prompt(
            "执行 autobrowser help,用 autobrowser 获取https://x.com/home 里的 10 条消息",
            max_skills=1,
            max_content_chars=600,
        )

        assert "https://x.com/home" in prompt
        assert "seen map keyed by /status/ links across viewport swaps" in prompt


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


def test_auto_skill_prompt_prefers_relevant_sections():
    """Auto-loaded skill context should highlight the most relevant sections instead of dumping the whole skill."""
    with tempfile.TemporaryDirectory() as tmpdir:
        browser_dir = Path(tmpdir) / "browser-flow"
        browser_dir.mkdir()
        create_test_skill(
            browser_dir,
            "browser-flow",
            "Autobrowser workflow helper",
            """# Overview

Use this skill when you need to drive autobrowser from the CLI.

## Installation

Install dependencies and prepare the environment.

## Troubleshooting

If `autobrowser help` fails or times out, rerun the command with verbose logging and inspect the browser connection state.

## Release Checklist

Publish packages and update release notes.
""",
            metadata="""tools:\n  - autobrowser\ntriggers:\n  - 执行autobrowser help\n""",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        prompt = loader.get_auto_skills_prompt("执行autobrowser help 失败怎么排查", max_skills=1)

        assert "### Troubleshooting" in prompt
        assert "### Release Checklist" not in prompt
        assert "Skill Root Directory" in prompt


def test_auto_skill_prompt_adds_autobrowser_guardrails():
    """Auto-loaded autobrowser context should include concrete CLI guardrails for common misuse patterns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        browser_dir = Path(tmpdir) / "autobrowser"
        browser_dir.mkdir()
        create_test_skill(
            browser_dir,
            "autobrowser",
            "Autobrowser workflow helper",
            "Use this skill when you need to drive autobrowser from the CLI.",
            metadata="tools:\n  - autobrowser\nplatform: windows\n",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        prompt = loader.get_auto_skills_prompt("用autobrowser打开页面并回答问题", max_skills=1)

        assert "CLI Guardrails" in prompt
        assert "autobrowser` on macOS/Linux and `autobrowser.cmd` on Windows" in prompt
        assert "start --headless" in prompt
        assert "navigate" in prompt
        assert "find text \"我来答\"" in prompt
        assert "click --text" in prompt
        assert "tab select <handle>" in prompt
        assert "--script" in prompt
        assert "wait ms" in prompt
        assert "scroll 500" in prompt
        assert "do not treat a stable visible `article` count (for example 4 or 5) as proof that no more posts exist" in prompt
        assert "`seen` map keyed by the `/status/` link" in prompt
        assert "accumulate unique posts across scroll iterations" in prompt
        assert "clicks `查看新帖子` when present" in prompt
        assert "Do not restart `seen` from scratch in separate eval calls" in prompt
        assert "prefer `write_file` + `autobrowser eval --file <path>` even on macOS/Linux" in prompt
        assert ".edui-editor-iframeholder iframe" in prompt
        assert ".new-editor-deliver-btn" in prompt
        assert "newAnswer=1" in prompt
        assert "[class*=submit]" in prompt
        assert "eval --file" in prompt


def test_auto_skill_prompt_closes_truncated_code_fences():
    """Truncated auto-loaded skill excerpts should not leave fenced code blocks open."""
    with tempfile.TemporaryDirectory() as tmpdir:
        browser_dir = Path(tmpdir) / "autobrowser"
        browser_dir.mkdir()
        create_test_skill(
            browser_dir,
            "autobrowser",
            "Autobrowser workflow helper",
            """## Run

```bash
autobrowser help
autobrowser goto https://x.com/home
"""
            + ("autobrowser eval \"window.scrollBy(0, 500)\"\n" * 120)
            + """
```

## Troubleshooting

If fewer than ten tweets are visible, scroll and retry.
""",
            metadata="tools:\n  - autobrowser\n",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        prompt = loader.get_auto_skills_prompt("用autobrowser抓取x.com里的10条消息", max_skills=1, max_content_chars=400)

        assert "use get_skill for the full version" in prompt.lower()
        assert prompt.count("```") % 2 == 0


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

    assert len(active_messages) == 2
    assert active_messages[0].role == "system"
    assert active_messages[0].content.startswith("system")
    assert "## Current Workspace" in active_messages[0].content
    assert active_messages[0].content.endswith("auto skills")
    assert active_messages[1].role == "user"
    assert len(agent.messages) == 2
    assert agent.messages[0].content.startswith("system")
    assert "auto skills" not in agent.messages[0].content

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


def test_auto_skill_creation_updates_existing_skill_when_content_changes():
    """Changed workflow content should update the existing skill in place and accumulate reusable guidance."""
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
        assert first.skill_path == second.skill_path
        assert first.skill_name == second.skill_name

        loaded_skill = loader.load_skill(second.skill_path)

        assert loaded_skill is not None
        assert loaded_skill.metadata is not None
        assert loaded_skill.metadata["auto_skill"]["run_count"] == 2
        assert loaded_skill.metadata["auto_skill"]["decision_notes"]
        assert loaded_skill.metadata["auto_skill"]["workflow_outline"]
        assert loaded_skill.metadata["auto_skill"]["watchouts"]
        assert "## Decision hints" in loaded_skill.content
        assert "## Watchouts" in loaded_skill.content
        assert "alpha 0" not in loaded_skill.content
        assert "beta 0" not in loaded_skill.content

        fresh_loader = SkillLoader(str(skill_dir))
        discovered = fresh_loader.discover_skills()

        assert [skill.name for skill in discovered] == [first.skill_name]


def test_merge_guidance_lists_prefers_request_specific_x_workflow_entries():
    """Request-aware guidance merging should prefer more specific x.com workflow entries over generic duplicates."""
    merged = _merge_guidance_lists(
        [
            "Inspect the current state, constraints, and true target before committing to any action.",
            "Execute the main task through the verified tool path and keep the successful sequence stable.",
            "Validate the final state with a separate check that proves the user goal was actually achieved.",
        ],
        [
            "Reuse an existing x.com/home tab or open the feed, then confirm the target timeline is active before extracting.",
            "Start with a simple visible-post extraction from `article` nodes to see whether the target count is already available.",
            "Use a single extraction path that collects visible posts, dedupes by stable /status/ links, and returns structured JSON.",
            "Validate that at least the requested number of unique posts were returned before summarizing the result.",
        ],
        "执行 autobrowser help,用 autobrowser 获取https://x.com/home 里的 10 条消息",
        kind="workflow",
        limit=6,
    )

    assert "Reuse an existing x.com/home tab or open the feed, then confirm the target timeline is active before extracting." in merged
    assert "Start with a simple visible-post extraction from `article` nodes to see whether the target count is already available." in merged
    assert "Validate that at least the requested number of unique posts were returned before summarizing the result." in merged
    assert "Use a single extraction path that collects visible posts, dedupes by stable /status/ links, and returns structured JSON." not in merged
    assert "Inspect the current state, constraints, and true target before committing to any action." not in merged
    assert "Execute the main task through the verified tool path and keep the successful sequence stable." not in merged


def test_auto_skill_creation_adds_x_feed_specific_guidance():
    """Generated x.com feed auto skills should include the cheaper visible-extract-first workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "执行 autobrowser help,用 autobrowser 获取https://x.com/home 里的 10 条消息",
            [
                ("bash", {"command": "autobrowser help"}, "Showed help output."),
                ("bash", {"command": "autobrowser server start"}, "Background command started with ID: 123456."),
                ("bash", {"command": "autobrowser status"}, "Connected successfully."),
                ("bash", {"command": "autobrowser tab select t24"}, "Selected x.com/home tab."),
                ("bash", {"command": "autobrowser eval visible-articles"}, "Returned 10 structured feed items."),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "成功获取了 X.com 首页的 **10 条消息**，以下是整理后的内容：\n\n### 1️⃣ @servasyy_ai\n> 最全面的Codex教程！",
            auto_skill_dir=str(skill_dir),
        )

        assert result.created is True
        assert result.skill_path is not None

        loaded_skill = loader.load_skill(result.skill_path)

        assert loaded_skill is not None
        assert "Start with a simple visible-post extraction from `article` nodes" in loaded_skill.content
        assert "If the initial visible extract is short, switch to one file-backed scrolling/dedupe script" in loaded_skill.content
        assert "Prefer write_file plus autobrowser eval --file for any fallback script" in loaded_skill.content


def test_auto_skill_creation_uses_semantic_name_for_new_skill():
    """New auto skills should prefer semantic family names over raw tool names and URL fragments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "在页面回答这个题 https://www.zhihu.com/question/1",
            [
                ("get_skill", {"skill_name": "autobrowser"}, "Loaded autobrowser skill."),
                ("bash", {"command": "autobrowser.cmd open https://www.zhihu.com/question/1"}, "Opened question page."),
                ("bash", {"command": "autobrowser.cmd click 写回答"}, "Opened answer editor."),
                ("bash", {"command": 'autobrowser.cmd type textarea.answer "answer"'}, "Typed answer."),
                ("bash", {"command": "autobrowser.cmd click 发布回答"}, "Submitted answer successfully."),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "已成功回答该问题并提交。",
            auto_skill_dir=str(skill_dir),
            min_tool_calls=5,
        )

        assert result.created is True
        assert result.skill_name.startswith("auto-answer-zhihu")
        assert "https" not in result.skill_name
        assert "get-skill" not in result.skill_name


def test_auto_skill_creation_updates_legacy_named_family_in_place():
    """Legacy auto-generated families should be updated in place instead of spawning a second semantic-name directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_root = Path(tmpdir) / "skills"
        skill_root.mkdir()
        create_generated_auto_skill(
            skill_root,
            "auto-get-skill-bash-https-www-zhihu-com",
            internal_name="auto-get-skill-bash-https-www-zhihu-com",
            family_key="auto-get-skill-bash-https-www-zhihu-com",
            body="Legacy workflow variant.",
        )

        loader = SkillLoader(str(skill_root))
        loader.discover_skills()
        legacy_path = skill_root / "auto-get-skill-bash-https-www-zhihu-com" / "SKILL.md"
        turn_messages = build_turn_messages(
            "在页面回答这个题 https://www.zhihu.com/question/1",
            [
                ("get_skill", {"skill_name": "autobrowser"}, "Loaded autobrowser skill."),
                ("bash", {"command": "autobrowser.cmd open https://www.zhihu.com/question/1"}, "Opened question page."),
                ("bash", {"command": "autobrowser.cmd click 写回答"}, "Opened answer editor."),
                ("bash", {"command": 'autobrowser.cmd type textarea.answer "answer"'}, "Typed answer."),
                ("bash", {"command": "autobrowser.cmd click 发布回答"}, "Submitted answer successfully."),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "已成功回答该问题并提交。",
            auto_skill_dir=str(skill_root),
            min_tool_calls=5,
        )

        assert result.created is True
        assert result.skill_path == legacy_path
        assert result.skill_name == "auto-get-skill-bash-https-www-zhihu-com"
        assert result.write_mode == "updated"

        updated_skill = loader.load_skill(legacy_path)

        assert updated_skill is not None
        assert updated_skill.metadata is not None
        assert updated_skill.metadata["auto_skill"]["family_key"].startswith("auto-answer-zhihu")
        assert "auto-get-skill-bash-https-www-zhihu-com" in updated_skill.metadata["auto_skill"]["legacy_family_keys"]

        fresh_loader = SkillLoader(str(skill_root))
        discovered = fresh_loader.discover_skills()

        assert [skill.name for skill in discovered] == ["auto-get-skill-bash-https-www-zhihu-com"]


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
        assert loaded_skill.metadata["auto_skill"]["family_key"] == result.skill_name
        assert "discovery" in loaded_skill.metadata["auto_skill"]["metrics"]["categories"]
        assert "validation" in loaded_skill.metadata["auto_skill"]["metrics"]["categories"]


def test_cleanup_auto_skills_archives_stale_history_and_repairs_internal_names():
    """Cleanup should archive stale historical variants and repair mismatched frontmatter names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_root = Path(tmpdir) / "skills"
        skill_root.mkdir()
        create_generated_auto_skill(
            skill_root,
            "auto-zhihu-answer",
            internal_name="auto-zhihu-answer",
            score=7,
            generated_at="2026-04-29T08:00:00+00:00",
            body="Older workflow variant.",
        )
        create_generated_auto_skill(
            skill_root,
            "auto-zhihu-answer-2",
            internal_name="auto-zhihu-answer",
            score=9,
            generated_at="2026-04-29T09:00:00+00:00",
            body="Newer workflow variant.",
        )

        report = cleanup_auto_skills(str(skill_root), apply=True)

        assert report.scanned_skills == 2
        assert report.archived_count == 1
        assert report.renamed_count == 1
        assert (skill_root / "auto-zhihu-answer-2" / "SKILL.md").exists()
        assert "name: auto-zhihu-answer-2" in (skill_root / "auto-zhihu-answer-2" / "SKILL.md").read_text(encoding="utf-8")

        archived_skills = list((skill_root / "_archived").rglob("SKILL.md"))
        assert len(archived_skills) == 1
        assert archived_skills[0].parent.name == "auto-zhihu-answer"

        loader = SkillLoader(str(skill_root))
        discovered = loader.discover_skills()

        assert [skill.name for skill in discovered] == ["auto-zhihu-answer-2"]


def test_cleanup_auto_skills_uses_family_key_for_future_variants():
    """Cleanup should also collapse newer suffixed variants that already have unique frontmatter names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_root = Path(tmpdir) / "skills"
        skill_root.mkdir()
        create_generated_auto_skill(
            skill_root,
            "auto-baidu-answer",
            internal_name="auto-baidu-answer",
            family_key="auto-baidu-answer",
            score=8,
            generated_at="2026-04-29T08:00:00+00:00",
            body="Older family member.",
        )
        create_generated_auto_skill(
            skill_root,
            "auto-baidu-answer-2",
            internal_name="auto-baidu-answer-2",
            family_key="auto-baidu-answer",
            score=9,
            generated_at="2026-04-29T09:00:00+00:00",
            body="Newer family member.",
        )

        report = cleanup_auto_skills(str(skill_root), apply=False)

        assert report.archived_count == 1
        assert report.renamed_count == 0
        assert any(action.detail == "historical-version:auto-baidu-answer" for action in report.actions)
        assert (skill_root / "auto-baidu-answer").exists()
        assert not (skill_root / "_archived").exists()


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


def test_auto_skill_creation_rejects_partial_completion_when_requested_count_not_met():
    """Runs that finish cleanly but admit they missed the requested quantity should not become auto-skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "用autobrowser获取x.com里的15条正在关注的新消息",
            [
                ("bash", {"command": "autobrowser.cmd goto https://x.com/home"}, "Opened the feed."),
                ("bash", {"command": "autobrowser.cmd click 查看新帖子"}, "Command failed with exit code 1\nelement not found: 查看新帖子"),
                ("bash", {"command": "autobrowser.cmd eval tweets"}, "SyntaxError: Illegal return statement"),
                ("bash", {"command": "autobrowser.cmd snapshot"}, "Captured snapshot successfully."),
                ("bash", {"command": "autobrowser.cmd eval article-count"}, '{"tweetCount":8,"sample":"example"}'),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "当前页面只显示了 4条推文（包含1条广告），未能达到请求的15条。",
            auto_skill_dir=str(skill_dir),
        )

        assert result.created is False
        assert result.reason == "quality-gate"
        assert "partial-completion-detected" in result.quality_warnings
        assert "requirement-mismatch" in result.quality_warnings


def test_auto_skill_creation_accepts_success_summary_with_numbered_results():
    """Detailed numbered results should not be misread as only one completed item."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "执行 autobrowser help,用 autobrowser 获取https://x.com/home 里的 10 条消息",
            [
                ("bash", {"command": "autobrowser help"}, "Showed help output."),
                ("bash", {"command": "autobrowser server start"}, "Background command started with ID: 123456."),
                ("bash", {"command": "autobrowser status"}, "Connected successfully."),
                ("bash", {"command": "autobrowser tab select t24"}, "Selected x.com/home tab."),
                ("bash", {"command": "autobrowser eval articles"}, "Returned 10 structured feed items."),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "成功获取了 X.com 首页的 **10 条消息**，以下是整理后的内容：\n\n---\n\n### 1️⃣ @servasyy_ai\n**4月28日**\n> 最全面的Codex教程！不到两小时教会你如何使用 Codex App + GPT5.5...\n\n---\n\n### 2️⃣ @PandaTalk8\n**9小时前**\n> 我算是真的孔乙己脱下了长衫...",
            auto_skill_dir=str(skill_dir),
        )

        assert result.created is True
        assert "requirement-mismatch" not in result.quality_warnings


def test_auto_skill_creation_rejects_multilingual_failure_summary():
    """Chinese failure summaries should not be mistaken for successful reusable workflows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "用autobrowser帮我在百度知道里答题",
            [
                ("bash", {"command": "autobrowser.cmd connect"}, "Connected successfully."),
                ("bash", {"command": "autobrowser.cmd goto https://zhidao.baidu.com/question/1.html"}, "Opened the question page."),
                ("bash", {"command": "autobrowser.cmd snapshot"}, "Captured snapshot successfully."),
                ("bash", {"command": "autobrowser.cmd click 我来答"}, "Command failed with exit code 1\nelement not found: 我来答"),
                ("bash", {"command": "autobrowser.cmd screenshot"}, "Captured screenshot successfully."),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "很抱歉，当前页面需要登录，autobrowser 无法处理登录验证流程。",
            auto_skill_dir=str(skill_dir),
        )

        assert result.created is False
        assert result.reason == "quality-gate"
        assert "final-result-not-successful" in result.quality_warnings


def test_auto_skill_creation_rejects_blocked_publish_summary():
    """Answering runs that admit publishing was blocked should never become reusable auto-skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "在页面回答这个题 https://www.zhihu.com/question/1",
            [
                ("bash", {"command": "autobrowser.cmd open https://www.zhihu.com/question/1"}, "Opened the question page."),
                ("bash", {"command": "autobrowser.cmd click 写回答"}, "Opened the answer editor."),
                ("bash", {"command": 'autobrowser.cmd type .public-DraftEditor-content "answer"'}, "Typed answer content into the editor."),
                ("bash", {"command": "autobrowser.cmd click 发布回答"}, '{"found": true, "selector": "发布回答"}'),
                ("bash", {"command": "autobrowser.cmd eval publish-state"}, "Editor still exists"),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "我已经成功在知乎问题页面输入了回答内容，但是由于知乎平台的反自动化机制，发布按钮的点击操作似乎被阻止了。",
            auto_skill_dir=str(skill_dir),
            min_tool_calls=5,
        )

        assert result.created is False
        assert result.reason == "quality-gate"
        assert "final-result-not-successful" in result.quality_warnings


def test_auto_skill_creation_rejects_answering_task_without_submission_steps():
    """Visiting question pages is not enough when the user asked to actually answer them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "用autobrowser帮我在百度知道里答3道题",
            [
                ("bash", {"command": "autobrowser.cmd goto https://zhidao.baidu.com/question/1.html"}, "Opened question 1."),
                ("bash", {"command": "autobrowser.cmd eval \"document.body.innerText.substring(0,500)\""}, "python能做什么？ 1个回答 我来答"),
                ("bash", {"command": "autobrowser.cmd goto https://zhidao.baidu.com/question/2.html"}, "Opened question 2."),
                ("bash", {"command": "autobrowser.cmd eval \"document.body.innerText.substring(0,500)\""}, "pycharm下载速度慢 1个回答 我来答"),
                ("bash", {"command": "autobrowser.cmd goto https://zhidao.baidu.com/question/3.html"}, "Opened question 3."),
                ("bash", {"command": "autobrowser.cmd eval \"document.body.innerText.substring(0,500)\""}, "js返回上一页并刷新的几种方法有哪些？ 1个回答 我来答"),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "我已经成功访问了百度知道推荐问题页面中的3道题，这些题目都已有人回答。",
            auto_skill_dir=str(skill_dir),
        )

        assert result.created is False
        assert result.reason == "quality-gate"
        assert "requirement-mismatch" in result.quality_warnings
        assert "requested-action-not-observed" in result.quality_warnings


def test_auto_skill_creation_accepts_answering_workflow_with_submission_steps():
    """Answering workflows should still be reusable when the trace shows real drafting and submission."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "skills"
        loader = SkillLoader(str(skill_dir))
        turn_messages = build_turn_messages(
            "用autobrowser帮我在百度知道里答3道题",
            [
                ("get_skill", {"skill_name": "autobrowser"}, "Loaded autobrowser skill."),
                ("bash", {"command": "autobrowser.cmd goto https://zhidao.baidu.com/question/1.html"}, "Opened question 1."),
                ("bash", {"command": "autobrowser.cmd click 我来答"}, '{"found": true, "selector": "我来答"}'),
                ("bash", {"command": 'autobrowser.cmd type textarea.answer "answer 1"'}, "Typed answer 1 into the answer editor."),
                ("bash", {"command": "autobrowser.cmd click 提交回答"}, "Submitted answer 1 successfully."),
                ("bash", {"command": "autobrowser.cmd goto https://zhidao.baidu.com/question/2.html"}, "Opened question 2."),
                ("bash", {"command": "autobrowser.cmd click 我来答"}, '{"found": true, "selector": "我来答"}'),
                ("bash", {"command": 'autobrowser.cmd type textarea.answer "answer 2"'}, "Typed answer 2 into the answer editor."),
                ("bash", {"command": "autobrowser.cmd click 提交回答"}, "Submitted answer 2 successfully."),
                ("bash", {"command": "autobrowser.cmd goto https://zhidao.baidu.com/question/3.html"}, "Opened question 3."),
                ("bash", {"command": "autobrowser.cmd click 我来答"}, '{"found": true, "selector": "我来答"}'),
                ("bash", {"command": 'autobrowser.cmd type textarea.answer "answer 3"'}, "Typed answer 3 into the answer editor."),
                ("bash", {"command": "autobrowser.cmd click 提交回答"}, "Submitted answer 3 successfully."),
            ],
        )

        result = maybe_create_auto_skill(
            loader,
            turn_messages,
            "已成功回答3道题，并提交了对应答案。",
            auto_skill_dir=str(skill_dir),
        )

        assert result.created is True
        assert result.tier == "approved"


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


def test_select_relevant_skills_skips_risky_auto_generated_skill():
    """Risky auto-generated skills should stay manually inspectable but not auto-selected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        risky_dir = Path(tmpdir) / "auto-zhidao-answer"
        risky_dir.mkdir()
        create_test_skill(
            risky_dir,
            "auto-zhidao-answer",
            "Auto-generated workflow for 用autobrowser帮我在百度知道里答题",
            "Use this skill to answer Baidu Zhidao questions with autobrowser.",
            metadata=(
                "metadata:\n"
                "  source: mini-agent\n"
                "  auto_skill:\n"
                "    tier: approved\n"
                "    warnings:\n"
                "      - environment-specific-data-detected\n"
                "      - multiple-failed-steps\n"
            ),
        )

        autobrowser_dir = Path(tmpdir) / "autobrowser"
        autobrowser_dir.mkdir()
        create_test_skill(
            autobrowser_dir,
            "autobrowser",
            "Autobrowser workflow helper",
            "Use this skill when you need to drive autobrowser from the CLI.",
            metadata="tools:\n  - autobrowser\ntriggers:\n  - 用autobrowser答题\n",
        )

        loader = SkillLoader(tmpdir)
        loader.discover_skills()

        assert loader.get_skill("auto-zhidao-answer") is not None

        selected = loader.select_relevant_skills("用autobrowser帮我在百度知道里答3道题", max_skills=2)

        assert [skill.name for skill in selected] == ["autobrowser"]


@pytest.mark.parametrize("invalid_value", [0, -1])
def test_tools_config_rejects_non_positive_auto_skill_min_tool_calls(invalid_value):
    """Auto skill creation threshold must stay positive."""
    with pytest.raises(ValidationError):
        ToolsConfig(auto_skill_min_tool_calls=invalid_value)


def test_tools_config_rejects_auto_skill_approved_score_below_candidate_score():
    """Approved quality threshold should never be lower than candidate threshold."""
    with pytest.raises(ValidationError):
        ToolsConfig(auto_skill_candidate_score=6, auto_skill_approved_score=5)
