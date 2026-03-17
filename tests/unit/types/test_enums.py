"""Unit tests for tractable/types/enums.py.

Covers:
- All members present for each enum
- str inheritance
- Round-trip JSON serialization via Pydantic model_dump()
"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from tractable.types.enums import (
    AgentStatus,
    AutonomyLevel,
    ChangeRelevance,
    ChangeRisk,
    ChangeSource,
    EdgeConfidence,
    TaskPhase,
)


# ── AutonomyLevel ──────────────────────────────────────────────────────


def test_autonomy_level_members() -> None:
    members = {e.value for e in AutonomyLevel}
    assert members == {"autonomous", "supervised", "manual"}


def test_autonomy_level_str_inheritance() -> None:
    assert isinstance(AutonomyLevel.SUPERVISED, str)


def test_autonomy_level_json_round_trip() -> None:
    class M(BaseModel):
        level: AutonomyLevel

    obj = M(level=AutonomyLevel.AUTONOMOUS)
    data = obj.model_dump()
    assert data["level"] == "autonomous"
    restored = M.model_validate(data)
    assert restored.level is AutonomyLevel.AUTONOMOUS


# ── ChangeRisk ─────────────────────────────────────────────────────────


def test_change_risk_members() -> None:
    members = {e.value for e in ChangeRisk}
    assert members == {"low", "medium", "high", "critical"}


def test_change_risk_str_inheritance() -> None:
    assert isinstance(ChangeRisk.HIGH, str)


def test_change_risk_json_round_trip() -> None:
    class M(BaseModel):
        risk: ChangeRisk

    obj = M(risk=ChangeRisk.CRITICAL)
    data = obj.model_dump()
    assert data["risk"] == "critical"
    restored = M.model_validate(data)
    assert restored.risk is ChangeRisk.CRITICAL


# ── EdgeConfidence ─────────────────────────────────────────────────────


def test_edge_confidence_members() -> None:
    members = {e.value for e in EdgeConfidence}
    assert members == {"deterministic", "heuristic", "llm_inferred", "declared"}


def test_edge_confidence_str_inheritance() -> None:
    assert isinstance(EdgeConfidence.DETERMINISTIC, str)


def test_edge_confidence_json_round_trip() -> None:
    class M(BaseModel):
        confidence: EdgeConfidence

    obj = M(confidence=EdgeConfidence.LLM_INFERRED)
    data = obj.model_dump()
    assert data["confidence"] == "llm_inferred"
    restored = M.model_validate(data)
    assert restored.confidence is EdgeConfidence.LLM_INFERRED


# ── AgentStatus ────────────────────────────────────────────────────────


def test_agent_status_members() -> None:
    members = {e.value for e in AgentStatus}
    assert members == {
        "idle",
        "working",
        "awaiting_approval",
        "awaiting_coordination",
        "error",
        "dormant",
    }


def test_agent_status_str_inheritance() -> None:
    assert isinstance(AgentStatus.IDLE, str)


def test_agent_status_json_round_trip() -> None:
    class M(BaseModel):
        status: AgentStatus

    obj = M(status=AgentStatus.WORKING)
    data = obj.model_dump()
    assert data["status"] == "working"
    restored = M.model_validate(data)
    assert restored.status is AgentStatus.WORKING


# ── TaskPhase ──────────────────────────────────────────────────────────


def test_task_phase_members() -> None:
    members = {e.value for e in TaskPhase}
    assert members == {
        "submitted",
        "planning",
        "executing",
        "reviewing",
        "coordinating",
        "completed",
        "failed",
    }


def test_task_phase_str_inheritance() -> None:
    assert isinstance(TaskPhase.PLANNING, str)


def test_task_phase_json_round_trip() -> None:
    class M(BaseModel):
        phase: TaskPhase

    obj = M(phase=TaskPhase.COMPLETED)
    data = obj.model_dump()
    assert data["phase"] == "completed"
    restored = M.model_validate(data)
    assert restored.phase is TaskPhase.COMPLETED


# ── ChangeSource ───────────────────────────────────────────────────────


def test_change_source_members() -> None:
    assert set(ChangeSource) == {
        ChangeSource.HUMAN_COMMIT,
        ChangeSource.AGENT_COMMIT,
        ChangeSource.INITIAL_INGESTION,
        ChangeSource.INCREMENTAL_UPDATE,
        ChangeSource.DEPENDENCY_SYNC,
        ChangeSource.MANUAL_DECLARATION,
    }


def test_change_source_str_inheritance() -> None:
    assert isinstance(ChangeSource.HUMAN_COMMIT, str)


def test_change_source_json_round_trip() -> None:
    class M(BaseModel):
        source: ChangeSource

    obj = M(source=ChangeSource.AGENT_COMMIT)
    data = obj.model_dump()
    assert data["source"] == "agent_commit"
    restored = M.model_validate(data)
    assert restored.source is ChangeSource.AGENT_COMMIT


# ── ChangeRelevance ────────────────────────────────────────────────────


def test_change_relevance_members() -> None:
    members = {e.value for e in ChangeRelevance}
    assert members == {"direct", "dependency", "consumer", "transitive"}


def test_change_relevance_str_inheritance() -> None:
    assert isinstance(ChangeRelevance.DIRECT, str)


def test_change_relevance_json_round_trip() -> None:
    class M(BaseModel):
        relevance: ChangeRelevance

    obj = M(relevance=ChangeRelevance.TRANSITIVE)
    data = obj.model_dump()
    assert data["relevance"] == "transitive"
    restored = M.model_validate(data)
    assert restored.relevance is ChangeRelevance.TRANSITIVE


# ── JSON module serialization (not just Pydantic) ──────────────────────


def test_all_enums_json_serializable() -> None:
    """All enum values are plain strings and json.dumps-serializable."""
    for enum_cls in (
        AutonomyLevel,
        ChangeRisk,
        EdgeConfidence,
        AgentStatus,
        TaskPhase,
        ChangeSource,
        ChangeRelevance,
    ):
        for member in enum_cls:
            assert json.dumps(member) == f'"{member.value}"'
