"""Unit tests for tractable/types/graph.py."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tractable.types.enums import ChangeRisk, EdgeConfidence
from tractable.types.graph import (
    CrossRepoEdge,
    GraphEntity,
    GraphMutation,
    ImpactReport,
    MutationResult,
    RepoGraphSummary,
    Subgraph,
)

_ENTITY = GraphEntity(id="e1", kind="function", name="foo", repo="my-repo", file_path="src/foo.py")
_EDGE = CrossRepoEdge(
    source_entity_id="e1",
    source_repo="repo-a",
    target_entity_id="e2",
    target_repo="repo-b",
    relationship="CALLS",
    confidence=0.95,
    resolution=EdgeConfidence.DETERMINISTIC,
)


# ── GraphEntity ────────────────────────────────────────────────────────


def test_graph_entity_required_fields() -> None:
    e = GraphEntity(id="x", kind="class", name="Bar", repo="repo", file_path="bar.py")
    assert e.id == "x"
    assert e.line is None
    assert e.properties == {}


def test_graph_entity_model_dump_json_serializable() -> None:
    e = GraphEntity(id="x", kind="function", name="foo", repo="my-repo", file_path="src/foo.py")
    data = e.model_dump()
    assert json.dumps(data)  # must not raise
    assert set(data.keys()) == {"id", "kind", "name", "repo", "file_path", "line", "properties"}


def test_graph_entity_properties_default_independent() -> None:
    e1 = GraphEntity(id="a", kind="fn", name="f", repo="r", file_path="f.py")
    e2 = GraphEntity(id="b", kind="fn", name="g", repo="r", file_path="g.py")
    e1.properties["key"] = "val"
    assert e2.properties == {}


def test_graph_entity_missing_required() -> None:
    with pytest.raises(ValidationError):
        GraphEntity(id="x", kind="fn")  # type: ignore[call-arg]


# ── CrossRepoEdge ──────────────────────────────────────────────────────


def test_cross_repo_edge_instantiation() -> None:
    assert _EDGE.resolution is EdgeConfidence.DETERMINISTIC
    assert _EDGE.confidence == 0.95


def test_cross_repo_edge_invalid_resolution() -> None:
    with pytest.raises(ValidationError):
        CrossRepoEdge(
            source_entity_id="a",
            source_repo="r1",
            target_entity_id="b",
            target_repo="r2",
            relationship="CALLS",
            confidence=0.5,
            resolution="invalid",  # type: ignore[arg-type]
        )


def test_cross_repo_edge_all_resolution_values_valid() -> None:
    for conf in EdgeConfidence:
        edge = CrossRepoEdge(
            source_entity_id="a",
            source_repo="r1",
            target_entity_id="b",
            target_repo="r2",
            relationship="CALLS",
            confidence=1.0,
            resolution=conf,
        )
        assert edge.resolution is conf


# ── Subgraph ───────────────────────────────────────────────────────────


def test_subgraph_defaults_empty() -> None:
    sg = Subgraph()
    assert sg.nodes == []
    assert sg.edges == []


def test_subgraph_defaults_independent() -> None:
    sg1 = Subgraph()
    sg2 = Subgraph()
    sg1.nodes.append(_ENTITY)
    assert sg2.nodes == []


def test_subgraph_with_content() -> None:
    sg = Subgraph(nodes=[_ENTITY], edges=[_EDGE])
    assert len(sg.nodes) == 1
    assert len(sg.edges) == 1


# ── ImpactReport ───────────────────────────────────────────────────────


def test_impact_report_defaults() -> None:
    report = ImpactReport(risk_level=ChangeRisk.LOW)
    assert report.directly_affected == []
    assert report.cross_repo_edges == []
    assert report.affected_repos == []


def test_impact_report_all_risk_levels() -> None:
    for risk in ChangeRisk:
        r = ImpactReport(risk_level=risk)
        assert r.risk_level is risk


def test_impact_report_invalid_risk() -> None:
    with pytest.raises(ValidationError):
        ImpactReport(risk_level="extreme")  # type: ignore[arg-type]


# ── RepoGraphSummary ───────────────────────────────────────────────────


def test_repo_graph_summary_instantiation() -> None:
    s = RepoGraphSummary(
        repo_name="my-repo",
        total_entities=100,
        summary_text="A Python API service.",
    )
    assert s.key_modules == []
    assert s.public_interfaces == []
    assert s.cross_repo_dependencies == []


# ── GraphMutation ──────────────────────────────────────────────────────


def test_graph_mutation_valid_operations() -> None:
    for op in ("create_node", "update_node", "delete_node", "create_edge", "update_edge", "delete_edge"):
        m = GraphMutation(operation=op, payload={"id": "x"})  # type: ignore[arg-type]
        assert m.operation == op


def test_graph_mutation_invalid_operation() -> None:
    with pytest.raises(ValidationError):
        GraphMutation(operation="explode_node", payload={})  # type: ignore[arg-type]


def test_graph_mutation_default_payload() -> None:
    m = GraphMutation(operation="delete_node")  # type: ignore[call-arg]
    assert m.payload == {}


# ── MutationResult ─────────────────────────────────────────────────────


def test_mutation_result_defaults() -> None:
    r = MutationResult(applied=5)
    assert r.errors == []


def test_mutation_result_with_errors() -> None:
    r = MutationResult(applied=3, errors=["node x not found"])
    assert len(r.errors) == 1
