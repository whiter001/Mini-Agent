"""Skill document loading, rendering, and persistence helpers."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from .auto_skill_cleanup import (
    _CANDIDATE_DIRNAME,
    _AutoSkillRecord,
    _SKILL_FRONTMATTER_PATTERN,
    _normalize_skill_content,
    _rename_generated_skill_content,
)
from .auto_skill_support import (
    _AutoSkillKnowledge,
    _merge_string_lists,
    _normalize_string_list,
    _parse_skill_document,
    _safe_int,
    _sanitize_summary_text,
    _summarize_final_outcome,
)


def _load_existing_auto_skill_knowledge(skill_file: Path | None) -> _AutoSkillKnowledge:
    if skill_file is None or not skill_file.exists():
        return _AutoSkillKnowledge()

    try:
        content = skill_file.read_text(encoding="utf-8")
    except OSError:
        return _AutoSkillKnowledge()

    parsed = _parse_skill_document(content)
    if parsed is None:
        return _AutoSkillKnowledge()

    frontmatter, _ = parsed
    metadata = frontmatter.get("metadata") if isinstance(frontmatter.get("metadata"), dict) else {}
    auto_skill_meta = metadata.get("auto_skill") if isinstance(metadata.get("auto_skill"), dict) else {}

    return _AutoSkillKnowledge(
        observed_requests=_normalize_string_list(auto_skill_meta.get("observed_requests") or frontmatter.get("triggers")),
        tools=_normalize_string_list(frontmatter.get("tools")),
        decision_notes=_normalize_string_list(auto_skill_meta.get("decision_notes")),
        workflow_outline=_normalize_string_list(auto_skill_meta.get("workflow_outline")),
        watchouts=_normalize_string_list(auto_skill_meta.get("watchouts")),
        legacy_family_keys=_normalize_string_list(auto_skill_meta.get("legacy_family_keys")),
        latest_outcome=_summarize_final_outcome(auto_skill_meta.get("latest_outcome") or "", max_chars=220),
        run_count=_safe_int(auto_skill_meta.get("run_count")),
    )


def _build_skill_markdown(
    *,
    skill_name: str,
    user_request: str,
    trigger: str,
    family_key: str,
    legacy_family_keys: Sequence[str],
    resolved_tier: str,
    tool_names: Sequence[str],
    observed_requests: Sequence[str],
    decision_notes: Sequence[str],
    workflow_outline: Sequence[str],
    watchouts: Sequence[str],
    latest_outcome: str,
    run_count: int,
    quality_score: int,
    quality_reasons: Sequence[str],
    quality_warnings: Sequence[str],
    quality_metrics: dict[str, Any],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    tools_summary = ", ".join(tool_names) if tool_names else "workflow tools"
    all_legacy_family_keys = _merge_string_lists(list(legacy_family_keys), limit=10)
    frontmatter = {
        "name": skill_name,
        "description": f"Auto-maintained guidance for {_sanitize_summary_text(user_request, max_chars=120)}",
        "version": "1.0",
        "tools": list(tool_names),
        "triggers": list(observed_requests),
        "metadata": {
            "source": "mini-agent",
            "trigger": trigger,
            "generated_at": generated_at,
            "updated_at": generated_at,
            "tools": tools_summary,
            "auto_skill": {
                "family_key": family_key,
                "legacy_family_keys": all_legacy_family_keys,
                "tier": resolved_tier,
                "score": quality_score,
                "reasons": list(quality_reasons),
                "warnings": list(quality_warnings),
                "metrics": quality_metrics,
                "run_count": run_count,
                "observed_requests": list(observed_requests),
                "decision_notes": list(decision_notes),
                "workflow_outline": list(workflow_outline),
                "watchouts": list(watchouts),
                "latest_outcome": latest_outcome,
            },
        },
    }
    frontmatter_yaml = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    reasons = ", ".join(quality_reasons) if quality_reasons else "n/a"
    warnings = ", ".join(quality_warnings) if quality_warnings else "none"
    return f"""---
{frontmatter_yaml}
---

# {skill_name}

Use this skill when the task matches the recorded workflow below.

## When to use

{_render_bullet_list(observed_requests)}

## Decision hints

{_render_bullet_list(decision_notes)}

## Quality signals

- Tier: {resolved_tier}
- Score: {quality_score}
- Reasons: {reasons}
- Warnings: {warnings}

## Procedure

{_render_numbered_list(workflow_outline)}

## Watchouts

{_render_bullet_list(watchouts)}

## Final outcome

{latest_outcome}

## Notes

- Auto-maintained from successful execution traces.
- This skill is updated in place to preserve reusable guidance instead of accumulating near-duplicate folders.
- Successful observations recorded so far: {run_count}
- Candidate skills stay out of the auto-load pool until reviewed.
- Review before reusing if the environment or tool behavior has changed.
"""


def _validate_generated_skill_content(skill_content: str) -> bool:
    frontmatter_match = _SKILL_FRONTMATTER_PATTERN.match(skill_content)
    if not frontmatter_match:
        return False
    try:
        frontmatter = yaml.safe_load(frontmatter_match.group(1)) or {}
    except yaml.YAMLError:
        return False
    if not frontmatter.get("name") or not frontmatter.get("description"):
        return False
    body = frontmatter_match.group(2)
    return all(section in body for section in ("## When to use", "## Procedure", "## Final outcome"))


def _write_skill(
    auto_skill_dir: str,
    skill_name: str,
    skill_content: str,
    tier: str,
    *,
    existing_record: _AutoSkillRecord | None = None,
) -> tuple[Path, str, str]:
    root = Path(str(auto_skill_dir).strip()).expanduser()
    target_root = root if tier == "approved" else root / _CANDIDATE_DIRNAME
    target_root.mkdir(parents=True, exist_ok=True)

    if existing_record is not None:
        destination_dir = target_root / existing_record.dir_name
        if existing_record.skill_dir != destination_dir:
            # 只有在目标路径空闲时才做跨 tier 迁移，避免误覆盖用户已有目录。
            if not destination_dir.exists():
                destination_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(existing_record.skill_dir), str(destination_dir))
            else:
                destination_dir = existing_record.skill_dir

        destination_dir.mkdir(parents=True, exist_ok=True)
        skill_file = destination_dir / "SKILL.md"
        content_to_write = _rename_generated_skill_content(skill_content, destination_dir.name)
        if skill_file.exists():
            existing = skill_file.read_text(encoding="utf-8")
            if _normalize_skill_content(existing) == _normalize_skill_content(content_to_write):
                return skill_file, destination_dir.name, "reused"
        skill_file.write_text(content_to_write, encoding="utf-8")
        return skill_file, destination_dir.name, "updated"

    candidate_name = skill_name
    skill_dir = target_root / candidate_name
    suffix = 2
    while skill_dir.exists() and skill_dir.is_dir() and skill_dir.joinpath("SKILL.md").exists():
        existing = skill_dir.joinpath("SKILL.md").read_text(encoding="utf-8")
        if _normalize_skill_content(existing) == _normalize_skill_content(skill_content):
            return skill_dir.joinpath("SKILL.md"), candidate_name, "reused"
        candidate_name = f"{skill_name}-{suffix}"
        skill_dir = target_root / candidate_name
        suffix += 1

    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    content_to_write = skill_content if candidate_name == skill_name else _rename_generated_skill_content(skill_content, candidate_name)
    skill_file.write_text(content_to_write, encoding="utf-8")
    return skill_file, candidate_name, "created"


def _render_bullet_list(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _render_numbered_list(items: Sequence[str]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))