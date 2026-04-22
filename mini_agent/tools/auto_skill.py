"""Helpers for automatically loading and creating skills."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from mini_agent.schema import Message

from .skill_loader import SkillLoader

_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "how",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass
class AutoSkillCreationResult:
    created: bool
    skill_name: str = ""
    skill_path: Path | None = None
    reason: str = ""


def build_auto_skill_context(
    skill_loader: SkillLoader | None,
    query: str,
    max_skills: int = 2,
) -> list[Message]:
    """Build temporary system messages for auto-selected skills."""
    if skill_loader is None:
        return []

    prompt = skill_loader.get_auto_skills_prompt(query, max_skills=max_skills)
    if not prompt:
        return []

    return [Message(role="system", content=prompt)]


def maybe_create_auto_skill(
    skill_loader: SkillLoader | None,
    turn_messages: Sequence[Message],
    final_result: str,
    *,
    auto_skill_dir: str,
    min_tool_calls: int = 5,
) -> AutoSkillCreationResult:
    """Create a reusable skill when a turn meets the trigger conditions."""
    trace = _collect_turn_trace(turn_messages)
    if not trace:
        return AutoSkillCreationResult(created=False, reason="no-tool-trace")

    tool_call_count = len(trace)
    failed_steps = [step for step in trace if _looks_like_error(step.get("result", ""))]
    successful = _looks_like_success(final_result)
    recovered = bool(failed_steps) and successful
    complex_task = tool_call_count >= min_tool_calls and successful

    if not complex_task and not recovered:
        return AutoSkillCreationResult(created=False, reason="trigger-not-met")

    user_request = _extract_user_request(turn_messages)
    if not user_request:
        return AutoSkillCreationResult(created=False, reason="missing-user-request")

    trigger = "complex-task" if complex_task else "self-recovery"
    skill_name = _build_skill_name(user_request, trace)
    skill_content = _build_skill_markdown(skill_name, user_request, trace, final_result, trigger)
    skill_path = _write_skill(auto_skill_dir, skill_name, skill_content)

    return AutoSkillCreationResult(
        created=True,
        skill_name=skill_name,
        skill_path=skill_path,
        reason=trigger,
    )


def _collect_turn_trace(turn_messages: Sequence[Message]) -> list[dict[str, str]]:
    trace: list[dict[str, str]] = []
    pending_indices: list[int] = []
    for message in turn_messages:
        role = getattr(message, "role", "")
        if role == "assistant":
            tool_calls = getattr(message, "tool_calls", None) or []
            for call in tool_calls:
                trace.append(
                    {
                        "name": _tool_call_name(call),
                        "arguments": _format_arguments(_tool_call_args(call)),
                        "result": "",
                    }
                )
                pending_indices.append(len(trace) - 1)
            continue

        if role == "tool" and trace:
            content = _stringify_message_content(getattr(message, "content", ""))
            if content and pending_indices:
                index = pending_indices.pop(0)
                if not trace[index]["result"]:
                    trace[index]["result"] = content
    return trace


def _extract_user_request(turn_messages: Sequence[Message]) -> str:
    for message in turn_messages:
        if getattr(message, "role", "") == "user":
            return _stringify_message_content(getattr(message, "content", ""))
    return ""


def _looks_like_error(text: str) -> bool:
    lowered = text.lower()
    return "error" in lowered or "failed" in lowered or "cancelled" in lowered


def _looks_like_success(text: str) -> bool:
    lowered = text.lower()
    return bool(text) and not any(token in lowered for token in ("error", "cancelled", "failed", "couldn't be completed"))


def _tool_call_name(call: object) -> str:
    function = getattr(call, "function", None)
    if function is None:
        return "tool"
    return str(getattr(function, "name", "tool"))


def _tool_call_args(call: object) -> dict[str, object]:
    function = getattr(call, "function", None)
    if function is None:
        return {}
    arguments = getattr(function, "arguments", {})
    return arguments if isinstance(arguments, dict) else {}


def _format_arguments(arguments: dict[str, object]) -> str:
    if not arguments:
        return ""
    items = []
    for key, value in list(arguments.items())[:3]:
        value_text = _stringify_message_content(value)
        if len(value_text) > 80:
            value_text = value_text[:77] + "..."
        items.append(f"{key}={value_text}")
    return ", ".join(items)


def _build_skill_name(user_request: str, trace: Sequence[dict[str, str]]) -> str:
    tokens = _tokenize(user_request)
    tool_names = _dedupe([step["name"] for step in trace if step.get("name")])
    parts = ["auto"]
    parts.extend(tool_names[:2])
    parts.extend(tokens[:4])
    slug = _slugify("-".join(parts))
    return slug[:80] or "auto-workflow"


def _build_skill_markdown(
    skill_name: str,
    user_request: str,
    trace: Sequence[dict[str, str]],
    final_result: str,
    trigger: str,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    tool_names = _dedupe([step["name"] for step in trace if step.get("name")])
    steps = "\n".join(
        f"{index + 1}. `{step['name']}`"
        + (f" - `{step['arguments']}`" if step.get("arguments") else "")
        + (f"\n   - result: {step['result']}" if step.get("result") else "")
        for index, step in enumerate(trace)
    )
    tools_summary = ", ".join(tool_names) if tool_names else "workflow tools"
    frontmatter = {
        "name": skill_name,
        "description": f"Auto-generated workflow for {user_request[:120]}",
        "version": "1.0",
        "metadata": {
            "source": "mini-agent",
            "trigger": trigger,
            "generated_at": generated_at,
            "tools": tools_summary,
        },
    }
    frontmatter_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"""---
{frontmatter_yaml}
---

# {skill_name}

Use this skill when the task matches the recorded workflow below.

## When to use

{user_request}

## Procedure

{steps}

## Final outcome

{final_result}

## Notes

- Generated automatically from a successful execution trace.
- Review before reusing if the environment or tool behavior has changed.
"""


def _write_skill(auto_skill_dir: str, skill_name: str, skill_content: str) -> Path:
    root = Path(str(auto_skill_dir).strip()).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    skill_dir = root / skill_name
    suffix = 2
    while skill_dir.exists() and skill_dir.is_dir() and skill_dir.joinpath("SKILL.md").exists():
        existing = skill_dir.joinpath("SKILL.md").read_text(encoding="utf-8")
        if _normalize_skill_content(existing) == _normalize_skill_content(skill_content):
            return skill_dir.joinpath("SKILL.md")
        skill_dir = root / f"{skill_name}-{suffix}"
        suffix += 1

    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(skill_content, encoding="utf-8")
    return skill_file


def _normalize_skill_content(content: str) -> str:
    return "\n".join(
        line for line in content.splitlines() if not line.startswith("  generated_at: ")
    )


def _stringify_message_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if len(token) > 2 and token.lower() not in _STOPWORDS
    ]


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    text = re.sub(r"-{2,}", "-", text)
    return text


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        items.append(value)
    return items
