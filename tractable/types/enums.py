"""Enumeration types for the Tractable framework.

All enums inherit from ``enum.StrEnum`` (Python 3.11+), which makes every
member a plain ``str`` subclass — Pydantic JSON serialization works correctly
and values compare equal to string literals.

Sources:
- tech-spec.py §1 — Enumerations & Value Types
- realtime-temporal-spec.py §A — ChangeSource
- realtime-temporal-spec.py §C — ChangeRelevance
"""

from __future__ import annotations

import enum


class AutonomyLevel(enum.StrEnum):
    """How much freedom an agent has before requiring human sign-off."""

    AUTONOMOUS = "autonomous"
    SUPERVISED = "supervised"
    MANUAL = "manual"


class ChangeRisk(enum.StrEnum):
    """Risk classification for a proposed change."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EdgeConfidence(enum.StrEnum):
    """How the edge in the code graph was resolved."""

    DETERMINISTIC = "deterministic"
    HEURISTIC = "heuristic"
    LLM_INFERRED = "llm_inferred"
    DECLARED = "declared"


class AgentStatus(enum.StrEnum):
    """Operational status of an agent."""

    IDLE = "idle"
    WORKING = "working"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_COORDINATION = "awaiting_coordination"
    ERROR = "error"
    DORMANT = "dormant"


class TaskPhase(enum.StrEnum):
    """Phase of a task in the agent workflow."""

    SUBMITTED = "submitted"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    COORDINATING = "coordinating"
    COMPLETED = "completed"
    FAILED = "failed"


class ChangeSource(enum.StrEnum):
    """What caused a new entity version to be created in the graph."""

    HUMAN_COMMIT = "human_commit"
    AGENT_COMMIT = "agent_commit"
    INITIAL_INGESTION = "initial_ingestion"
    INCREMENTAL_UPDATE = "incremental_update"
    DEPENDENCY_SYNC = "dependency_sync"
    MANUAL_DECLARATION = "manual_declaration"


class ChangeRelevance(enum.StrEnum):
    """How relevant a change is to a notified agent."""

    DIRECT = "direct"
    DEPENDENCY = "dependency"
    CONSUMER = "consumer"
    TRANSITIVE = "transitive"
