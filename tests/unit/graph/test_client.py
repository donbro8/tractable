"""Unit tests for FalkorDBClient.

All tests use mocked redis connections — no live FalkorDB instance required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tractable.graph.client import FalkorDBClient

# ── Initialisation ────────────────────────────────────────────────────────────


class TestFalkorDBClientInit:
    def test_reads_host_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FALKORDB_HOST", "db.internal")
        monkeypatch.setenv("FALKORDB_PORT", "9999")
        monkeypatch.setenv("FALKORDB_GRAPH_NAME", "mygraph")
        monkeypatch.delenv("FALKORDB_PASSWORD", raising=False)
        with patch("tractable.graph.client.aioredis.ConnectionPool"):
            client = FalkorDBClient()
        assert client._host == "db.internal"
        assert client._port == 9999
        assert client._graph_name == "mygraph"

    def test_explicit_args_override_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FALKORDB_HOST", "env-host")
        with patch("tractable.graph.client.aioredis.ConnectionPool"):
            client = FalkorDBClient(host="explicit-host", port=1234)
        assert client._host == "explicit-host"
        assert client._port == 1234

    def test_defaults_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FALKORDB_HOST", raising=False)
        monkeypatch.delenv("FALKORDB_PORT", raising=False)
        monkeypatch.delenv("FALKORDB_GRAPH_NAME", raising=False)
        monkeypatch.delenv("FALKORDB_PASSWORD", raising=False)
        with patch("tractable.graph.client.aioredis.ConnectionPool"):
            client = FalkorDBClient()
        assert client._host == "localhost"
        assert client._port == 6380
        assert client._graph_name == "tractable"

    def test_no_hardcoded_credentials(self) -> None:
        """Credentials must not be hardcoded in the module source."""
        import inspect

        import tractable.graph.client as mod

        source = inspect.getsource(mod)
        # No literal password values — only the env-var name appears
        for literal in ("password123", "admin123", "redis123", "secret"):
            assert literal not in source


# ── _build_query ──────────────────────────────────────────────────────────────


class TestBuildQuery:
    def setup_method(self) -> None:
        with patch("tractable.graph.client.aioredis.ConnectionPool"):
            self.client = FalkorDBClient()

    def test_empty_params_returns_cypher_unchanged(self) -> None:
        cypher = "MATCH (e) RETURN e"
        assert self.client._build_query(cypher, {}) == cypher

    def test_string_param_is_quoted(self) -> None:
        result = self.client._build_query("MATCH (e {repo: $repo}) RETURN e", {"repo": "myrepo"})
        assert result.startswith("CYPHER repo='myrepo'")

    def test_int_param_is_unquoted(self) -> None:
        result = self.client._build_query("MATCH (e) RETURN e LIMIT $n", {"n": 10})
        assert "n=10" in result

    def test_none_param_becomes_null(self) -> None:
        result = self.client._build_query("MATCH (e {x: $x}) RETURN e", {"x": None})
        assert "x=null" in result

    def test_bool_param(self) -> None:
        result = self.client._build_query("RETURN $flag", {"flag": True})
        assert "flag=true" in result

    def test_string_with_single_quote_is_escaped(self) -> None:
        result = self.client._build_query("RETURN $name", {"name": "O'Brien"})
        assert "O\\'Brien" in result


# ── _parse_response ───────────────────────────────────────────────────────────


class TestParseResponse:
    def setup_method(self) -> None:
        with patch("tractable.graph.client.aioredis.ConnectionPool"):
            self.client = FalkorDBClient()

    def test_empty_list_returns_empty_list(self) -> None:
        assert self.client._parse_response([]) == []

    def test_short_list_returns_empty_list(self) -> None:
        assert self.client._parse_response([["cnt"]]) == []

    def test_write_response_no_headers_returns_empty_list(self) -> None:
        # Write queries: [[], [stats]]
        assert self.client._parse_response([[], ["Nodes created: 1"]]) == []

    def test_single_count_row(self) -> None:
        # [headers, [[val]], stats]
        result = self.client._parse_response([["cnt"], [[0]], ["Query internal..."]])
        assert result == [{"cnt": 0}]

    def test_multiple_rows(self) -> None:
        headers = ["name", "age"]
        rows = [["Alice", 30], ["Bob", 25]]
        result = self.client._parse_response([headers, rows, ["stats"]])
        assert result == [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]

    def test_row_length_mismatch_is_skipped(self) -> None:
        result = self.client._parse_response([["a", "b"], [[1]], ["stats"]])
        assert result == []

    def test_two_element_response_without_stats(self) -> None:
        # edge case: response without trailing stats element
        result = self.client._parse_response([["x"], []])
        assert result == []


# ── ping ──────────────────────────────────────────────────────────────────────


class TestPing:
    @pytest.mark.asyncio
    async def test_returns_true_when_redis_pong(self) -> None:
        mock_redis: Any = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("tractable.graph.client.aioredis.ConnectionPool"):
            client = FalkorDBClient()
        with patch("tractable.graph.client.aioredis.Redis", return_value=mock_redis):
            result = await client.ping()

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(self) -> None:
        mock_redis: Any = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("tractable.graph.client.aioredis.ConnectionPool"):
            client = FalkorDBClient()
        with patch("tractable.graph.client.aioredis.Redis", return_value=mock_redis):
            result = await client.ping()

        assert result is False


# ── execute / execute_write ───────────────────────────────────────────────────


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_returns_parsed_rows(self) -> None:
        raw_response: Any = [["cnt"], [[0]], ["Query internal execution time: 0.1 ms"]]

        mock_redis: Any = AsyncMock()
        mock_redis.execute_command = AsyncMock(return_value=raw_response)

        with patch("tractable.graph.client.aioredis.ConnectionPool"):
            client = FalkorDBClient()
        with patch("tractable.graph.client.aioredis.Redis", return_value=mock_redis):
            result = await client.execute("MATCH (e) RETURN count(e) AS cnt", {})

        assert result == [{"cnt": 0}]
        mock_redis.execute_command.assert_called_once_with(
            "GRAPH.QUERY", client._graph_name, "MATCH (e) RETURN count(e) AS cnt"
        )

    @pytest.mark.asyncio
    async def test_execute_write_succeeds_and_returns_empty_for_create(self) -> None:
        # Write: CREATE returns no header rows
        raw_response: Any = [[], ["Nodes created: 1"]]

        mock_redis: Any = AsyncMock()
        mock_redis.execute_command = AsyncMock(return_value=raw_response)

        with patch("tractable.graph.client.aioredis.ConnectionPool"):
            client = FalkorDBClient()
        with patch("tractable.graph.client.aioredis.Redis", return_value=mock_redis):
            result = await client.execute_write(
                "CREATE (:Entity {id: 'test', valid_until: null})", {}
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_execute_delegates_params_to_build_query(self) -> None:
        raw_response: Any = [["e"], [[MagicMock()]], ["stats"]]

        mock_redis: Any = AsyncMock()
        mock_redis.execute_command = AsyncMock(return_value=raw_response)

        with patch("tractable.graph.client.aioredis.ConnectionPool"):
            client = FalkorDBClient()
        with patch("tractable.graph.client.aioredis.Redis", return_value=mock_redis):
            await client.execute("MATCH (e {repo: $repo}) RETURN e", {"repo": "svc"})

        called_query: str = mock_redis.execute_command.call_args[0][2]
        assert "CYPHER repo='svc'" in called_query
