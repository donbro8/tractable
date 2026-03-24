"""AgentContext three-layer assembly function.

TASK-2.3.2: Merges the base template, registration overrides, and pinned
instructions into a final system prompt string.

Three-layer merge (lowest → highest priority)
---------------------------------------------
1. Base template — system_prompt_template from TEMPLATE_REGISTRY with
   ``{repo_name}``, ``{autonomy_level}``, ``{capabilities}`` substituted.
2. Registration overrides — scope restriction ("Your scope is limited to: …")
   and governance policy bullets appended after the template body.
3. Pinned instructions — loaded from AgentContext in the state store,
   appended at the very end with a ``[pinned]`` prefix so they cannot be
   overridden by earlier layers.

Truncation order (when ``max_prompt_chars`` is exceeded)
---------------------------------------------------------
1. Drop the "Recent changes" section (most volatile, least critical).
2. Drop the "Repo summary" section if still over budget.
3. Template body + registration overrides + pinned instructions are
   *never* truncated.

Note on ``RepositoryRegistration.custom_system_prompt``
-------------------------------------------------------
The ``RepositoryRegistration`` model in ``tractable/types/config.py`` does
not currently define a ``custom_system_prompt`` field.  The described
override mechanism is implemented via ``registration.governance_overrides``
and is deferred to a future task when the field is added to the model.

Note on ``get_repo_summary``
----------------------------
``TemporalCodeGraph`` does not expose ``get_repo_summary``; that method
belongs to ``CodeGraph``.  At runtime the concrete FalkorDB implementation
satisfies both protocols.  We cast to ``CodeGraph`` at the call-site so
Pyright is satisfied in strict mode.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import structlog

from tractable.errors import RecoverableError
from tractable.protocols.agent_state_store import AgentStateStore
from tractable.protocols.code_graph import CodeGraph, TemporalCodeGraph
from tractable.types.config import TEMPLATE_REGISTRY, GovernancePolicy, RepositoryRegistration

_log = structlog.get_logger()

_DEFAULT_LOOKBACK_DAYS: int = 7
_DEFAULT_MAX_PROMPT_CHARS: int = 100_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _DefaultDict(dict[str, str]):
    """dict that returns '' for missing keys so .format_map() never raises."""

    def __missing__(self, key: str) -> str:  # noqa: ANN401
        return ""


def _make_template_values(
    repo_name: str,
    autonomy_level: str,
    capabilities: str,
) -> _DefaultDict:
    return _DefaultDict(
        {
            "repo_name": repo_name,
            "autonomy_level": autonomy_level,
            "capabilities": capabilities,
            # These are injected as standalone sections for independent
            # truncation; fill placeholders with empty strings here.
            "repo_architectural_summary": "",
            "cross_repo_digest": "",
            "pinned_instructions": "",
            # Coordinator template uses this placeholder.
            "assigned_repos": repo_name,
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def assemble_context(
    agent_id: str,
    state_store: AgentStateStore,
    graph: TemporalCodeGraph,
    registration: RepositoryRegistration,
    max_prompt_chars: int = _DEFAULT_MAX_PROMPT_CHARS,
) -> str:
    """Assemble a system prompt from the three-layer context model.

    Parameters
    ----------
    agent_id:
        Unique identifier for the agent.  Used to load pinned instructions.
    state_store:
        Provides ``AgentContext`` (including pinned instructions) from the
        state store.
    graph:
        ``TemporalCodeGraph`` used to fetch the repo summary and recent
        changes.  The concrete implementation must also satisfy
        ``CodeGraph`` (for ``get_repo_summary``); the cast is safe because
        the FalkorDB implementation implements both protocols.
    registration:
        Repository registration that identifies the template, autonomy
        level, scope restrictions, and governance policy.
    max_prompt_chars:
        Soft limit on the assembled prompt length.  The "Recent changes"
        section is dropped first, then "Repo summary".  The template
        body, registration overrides, and pinned instructions are
        *never* dropped.

    Returns
    -------
    str
        Assembled system prompt ready to pass to the LLM.
    """
    # ── Layer 1: Base template ────────────────────────────────────────────
    template = TEMPLATE_REGISTRY.get(registration.agent_template)
    if template is None:
        raise RecoverableError(
            f"Unknown agent template: {registration.agent_template!r}. "
            f"Available templates: {list(TEMPLATE_REGISTRY)}"
        )

    caps_str = ", ".join(c.name for c in template.capabilities)
    values = _make_template_values(
        repo_name=registration.name,
        autonomy_level=registration.autonomy_level.value,
        capabilities=caps_str,
    )
    base_section: str = template.system_prompt_template.format_map(values)

    # ── Repo architectural summary (injected as standalone section) ───────
    repo_summary_section: str = ""
    try:
        code_graph = cast(CodeGraph, graph)
        summary = await code_graph.get_repo_summary(registration.name)
        modules_str = (
            ", ".join(summary.key_modules[:5]) if summary.key_modules else "none"
        )
        repo_summary_section = (
            f"\n\nYour domain: {summary.summary_text}"
            f" ({summary.total_entities} entities, key modules: {modules_str})"
        )
    except Exception:
        _log.warning(
            "repo_summary_unavailable",
            agent_id=agent_id,
            repo=registration.name,
            exc_info=True,
        )

    # ── Recent changes digest ─────────────────────────────────────────────
    recent_changes_section: str = ""
    try:
        since: datetime = datetime.now(tz=UTC) - timedelta(days=_DEFAULT_LOOKBACK_DAYS)
        change_set = await graph.get_changes_since(since=since, repo=registration.name)
        if not change_set.is_empty:
            recent_changes_section = f"\n\nRecent changes: {change_set.summary}"
    except Exception:
        _log.warning(
            "recent_changes_unavailable",
            agent_id=agent_id,
            repo=registration.name,
            exc_info=True,
        )

    # ── Layer 2: Registration overrides ──────────────────────────────────
    overrides_parts: list[str] = []

    # Scope restriction — highest-visibility override.
    if registration.scope and registration.scope.allowed_paths:
        paths_str = ", ".join(registration.scope.allowed_paths)
        overrides_parts.append(f"Your scope is limited to: {paths_str}")

    # Governance policy bullets.
    gov_dict = template.governance.model_dump()
    if registration.governance_overrides:
        gov_dict.update(registration.governance_overrides)
    gov = GovernancePolicy.model_validate(gov_dict)
    overrides_parts.append(
        f"Governance: max {gov.max_files_per_change} files per change,"
        f" max {gov.max_lines_per_change} lines per change"
    )

    overrides_section: str = (
        "\n\n" + "\n".join(overrides_parts) if overrides_parts else ""
    )

    # ── Layer 3: Pinned instructions (highest priority, always last) ──────
    agent_context = await state_store.get_agent_context(agent_id)
    pinned_parts = [f"[pinned] {instr}" for instr in agent_context.pinned_instructions]
    pinned_section: str = "\n\n" + "\n".join(pinned_parts) if pinned_parts else ""

    # ── Truncation ────────────────────────────────────────────────────────
    # Core (never truncated): template + overrides + pinned.
    # Mutable sections: repo_summary (2nd) and recent_changes (1st).
    def _assemble(
        include_repo_summary: bool,
        include_recent_changes: bool,
    ) -> str:
        parts = [base_section]
        if include_repo_summary:
            parts.append(repo_summary_section)
        if include_recent_changes:
            parts.append(recent_changes_section)
        parts.append(overrides_section)
        parts.append(pinned_section)
        return "".join(parts)

    assembled = _assemble(
        include_repo_summary=True, include_recent_changes=True
    )
    if len(assembled) <= max_prompt_chars:
        return assembled

    # Drop recent changes first.
    assembled = _assemble(
        include_repo_summary=True, include_recent_changes=False
    )
    if len(assembled) <= max_prompt_chars:
        return assembled

    # Still over budget — drop repo summary too.
    return _assemble(include_repo_summary=False, include_recent_changes=False)
