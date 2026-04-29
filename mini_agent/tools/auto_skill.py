"""Helpers for automatically loading and creating skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from mini_agent.schema import Message

from .auto_skill_cleanup import (
    AutoSkillCleanupReport,
    _find_matching_auto_skill_record,
    cleanup_auto_skills,
)
from .auto_skill_document import (
    _build_skill_markdown,
    _load_existing_auto_skill_knowledge,
    _validate_generated_skill_content,
    _write_skill,
)
from .auto_skill_guidance import (
    _build_decision_notes,
    _build_legacy_skill_name,
    _build_skill_name,
    _build_watchouts,
    _build_workflow_outline,
    _merge_guidance_lists,
)
from .auto_skill_quality import (
    AutoSkillQualityReport,
    _collect_turn_trace,
    _evaluate_auto_skill_quality,
    _extract_user_request,
)
from .auto_skill_support import (
    _dedupe,
    _merge_string_lists,
    _normalize_string_list,
    _sanitize_summary_text,
    _summarize_final_outcome,
)
from .skill_loader import SkillLoader


@dataclass
class AutoSkillCreationResult:
    created: bool
    skill_name: str = ""
    skill_path: Path | None = None
    reason: str = ""
    write_mode: str = ""
    quality_score: int = 0
    tier: str = ""
    quality_reasons: list[str] = field(default_factory=list)
    quality_warnings: list[str] = field(default_factory=list)


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
    candidate_score_threshold: int = 5,
    approved_score_threshold: int = 8,
) -> AutoSkillCreationResult:
    """Create a reusable skill when a turn meets the quality gates."""
    trace = _collect_turn_trace(turn_messages)
    if not trace:
        return AutoSkillCreationResult(created=False, reason="no-tool-trace")

    user_request = _extract_user_request(turn_messages)
    if not user_request:
        return AutoSkillCreationResult(created=False, reason="missing-user-request")

    quality = _evaluate_auto_skill_quality(
        user_request,
        trace,
        final_result,
        min_tool_calls=min_tool_calls,
        candidate_score_threshold=candidate_score_threshold,
        approved_score_threshold=approved_score_threshold,
    )
    if not quality.should_create:
        return AutoSkillCreationResult(
            created=False,
            reason="quality-gate",
            quality_score=quality.score,
            tier=quality.tier,
            quality_reasons=quality.reasons,
            quality_warnings=quality.warnings,
        )

    trigger = "self-recovery" if quality.metrics.get("recovered") else "complex-task" if quality.metrics.get("meets_step_threshold") else "quality-gate"
    skill_name = _build_skill_name(user_request, trace)
    legacy_family_key = _build_legacy_skill_name(user_request, trace)
    existing_record = _find_matching_auto_skill_record(auto_skill_dir, [skill_name, legacy_family_key])
    resolved_tier = quality.tier
    if existing_record is not None and existing_record.tier == "approved":
        resolved_tier = "approved"

    existing_knowledge = _load_existing_auto_skill_knowledge(existing_record.skill_file if existing_record else None)
    tool_names = _merge_string_lists(
        _dedupe([str(step.get("name", "")) for step in trace if step.get("name")]),
        existing_knowledge.tools,
        limit=8,
    )
    observed_requests = _merge_string_lists(
        [_sanitize_summary_text(user_request, max_chars=160)],
        existing_knowledge.observed_requests,
        limit=5,
    )
    decision_notes = _merge_guidance_lists(
        _build_decision_notes(user_request, trace, quality),
        existing_knowledge.decision_notes,
        user_request,
        kind="decision",
        limit=6,
    )
    workflow_outline = _merge_guidance_lists(
        _build_workflow_outline(user_request, quality),
        existing_knowledge.workflow_outline,
        user_request,
        kind="workflow",
        limit=6,
    )
    watchouts = _merge_string_lists(
        _build_watchouts(user_request, quality),
        existing_knowledge.watchouts,
        limit=6,
    )
    latest_outcome = _summarize_final_outcome(final_result, max_chars=220)
    run_count = max(existing_knowledge.run_count, 0) + 1
    all_legacy_family_keys = _merge_string_lists([legacy_family_key], existing_knowledge.legacy_family_keys, limit=10)

    skill_content = _build_skill_markdown(
        skill_name=skill_name,
        user_request=user_request,
        trigger=trigger,
        family_key=skill_name,
        legacy_family_keys=all_legacy_family_keys,
        resolved_tier=resolved_tier,
        tool_names=tool_names,
        observed_requests=observed_requests,
        decision_notes=decision_notes,
        workflow_outline=workflow_outline,
        watchouts=watchouts,
        latest_outcome=latest_outcome,
        run_count=run_count,
        quality_score=quality.score,
        quality_reasons=quality.reasons,
        quality_warnings=quality.warnings,
        quality_metrics=quality.metrics,
    )
    if not _validate_generated_skill_content(skill_content):
        return AutoSkillCreationResult(
            created=False,
            reason="invalid-skill-content",
            quality_score=quality.score,
            tier=resolved_tier,
            quality_reasons=quality.reasons,
            quality_warnings=quality.warnings,
        )

    skill_path, resolved_skill_name, write_mode = _write_skill(
        auto_skill_dir,
        skill_name,
        skill_content,
        resolved_tier,
        existing_record=existing_record,
    )
    return AutoSkillCreationResult(
        created=True,
        skill_name=resolved_skill_name,
        skill_path=skill_path,
        reason=trigger,
        write_mode=write_mode,
        quality_score=quality.score,
        tier=resolved_tier,
        quality_reasons=quality.reasons,
        quality_warnings=quality.warnings,
    )


