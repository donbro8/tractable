"""FalkorDB async client wrapper.

Uses redis.asyncio to communicate with FalkorDB over the Redis protocol.
All connection parameters come from environment variables — no hardcoded
credentials or host names.

Environment variables:
  FALKORDB_HOST           FalkorDB hostname         (default: localhost)
  FALKORDB_PORT           FalkorDB port             (default: 6380)
  FALKORDB_GRAPH_NAME     Graph name to query       (default: tractable)
  FALKORDB_PASSWORD       Redis AUTH password       (optional)
  FALKORDB_MAX_CONNECTIONS  Connection pool cap     (default: 10)
"""

from __future__ import annotations

import os
from typing import Any, cast

import redis.asyncio as aioredis
import structlog

from tractable.errors import RecoverableError, TransientError

logger = structlog.get_logger()


class FalkorDBClient:
    """Async client for FalkorDB via the Redis protocol.

    All queries are sent as ``GRAPH.QUERY <graph_name> <cypher>`` commands.
    ``execute`` and ``execute_write`` are semantically distinct — write is
    kept separate to allow future read-replica routing — but both use the
    same underlying command today.

    Connection pool size is controlled by ``FALKORDB_MAX_CONNECTIONS``
    (default 10).  A pool acquisition timeout raises :class:`TransientError`
    so the agent retry loop handles it via exponential back-off.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        graph_name: str | None = None,
        password: str | None = None,
    ) -> None:
        self._host: str = host if host is not None else os.environ.get(
            "FALKORDB_HOST", "localhost"
        )
        self._port: int = port if port is not None else int(
            os.environ.get("FALKORDB_PORT", "6380")
        )
        self._graph_name: str = graph_name if graph_name is not None else (
            os.environ.get("FALKORDB_GRAPH_NAME", "tractable")
        )
        _env_password: str | None = os.environ.get("FALKORDB_PASSWORD") or None
        _password: str | None = password if password is not None else _env_password

        max_connections: int = int(
            os.environ.get("FALKORDB_MAX_CONNECTIONS", "10")
        )

        self._pool: aioredis.ConnectionPool = aioredis.ConnectionPool(
            host=self._host,
            port=self._port,
            password=_password,
            decode_responses=True,
            max_connections=max_connections,
        )

    async def ping(self) -> bool:
        """Return ``True`` if FalkorDB is reachable, ``False`` otherwise."""
        client = aioredis.Redis(connection_pool=self._pool)
        try:
            result: Any = await client.ping()  # pyright: ignore[reportUnknownMemberType]
            return bool(result)
        except Exception:  # noqa: BLE001
            return False

    async def execute(
        self, cypher: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute a read Cypher query and return rows as a list of dicts.

        Sends ``GRAPH.QUERY <graph_name> <cypher>`` and parses the
        FalkorDB response format: ``[[headers], [[row...], ...], [stats]]``.

        Raises
        ------
        TransientError
            If the FalkorDB connection is unavailable or the pool is
            exhausted (``ConnectionError``, ``TimeoutError``).
        RecoverableError
            If the server response cannot be parsed into the expected shape.
        """
        client = aioredis.Redis(connection_pool=self._pool)
        query = self._build_query(cypher, params)
        try:
            raw: list[Any] = cast(
                list[Any],
                await client.execute_command(  # pyright: ignore[reportUnknownMemberType]
                    "GRAPH.QUERY", self._graph_name, query
                ),
            )
        except TimeoutError as exc:
            logger.error(
                "falkordb_pool_timeout",
                graph=self._graph_name,
                error=str(exc),
            )
            raise TransientError(
                f"FalkorDB connection pool timed out: {exc}"
            ) from exc
        except ConnectionError as exc:
            logger.error(
                "falkordb_connection_error",
                graph=self._graph_name,
                error=str(exc),
            )
            raise TransientError(
                f"FalkorDB is unreachable: {exc}"
            ) from exc
        except Exception as exc:
            logger.error(
                "falkordb_query_error",
                graph=self._graph_name,
                error=str(exc),
            )
            raise TransientError(
                f"FalkorDB query failed: {exc}"
            ) from exc

        try:
            return self._parse_response(raw)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            logger.error(
                "falkordb_parse_error",
                graph=self._graph_name,
                error=str(exc),
            )
            raise RecoverableError(
                f"FalkorDB response could not be parsed: {exc}"
            ) from exc

    async def execute_write(
        self, cypher: str, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Execute a write Cypher query against FalkorDB.

        Semantically identical to ``execute`` at the protocol level; kept
        separate to allow future read-replica routing without changing call
        sites.
        """
        return await self.execute(cypher, params)

    async def close(self) -> None:
        """Disconnect and release all pooled connections."""
        await self._pool.disconnect()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_query(self, cypher: str, params: dict[str, Any]) -> str:
        """Prepend a ``CYPHER key=val …`` prefix when params are non-empty."""
        if not params:
            return cypher
        parts: list[str] = []
        for key, val in params.items():
            if val is None:
                parts.append(f"{key}=null")
            elif isinstance(val, bool):
                parts.append(f"{key}={'true' if val else 'false'}")
            elif isinstance(val, (int, float)):
                parts.append(f"{key}={val}")
            else:
                escaped = str(val).replace("'", "\\'")
                parts.append(f"{key}='{escaped}'")
        return "CYPHER " + " ".join(parts) + " " + cypher

    def _parse_response(self, result: list[Any]) -> list[dict[str, Any]]:
        """Parse a raw ``GRAPH.QUERY`` response into a list of row dicts.

        FalkorDB returns one of two shapes:
        - Read:  ``[[col_names…], [[val…], …], [stats…]]``
        - Write: ``[[], [stats…]]`` — no headers, no rows
        """
        if len(result) < 3:
            return []
        headers: list[str] = [str(h) for h in cast(list[Any], result[0])]
        if not headers:
            return []
        output: list[dict[str, Any]] = []
        for item in cast(list[Any], result[1]):
            row: list[Any] = list(item)
            if len(row) == len(headers):
                output.append({headers[i]: row[i] for i in range(len(headers))})
        return output
