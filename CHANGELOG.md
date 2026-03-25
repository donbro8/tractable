# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

_Will become `0.3.0` on release._

### Added

- **HCLParser** — tree-sitter HCL → `ParseResult`; enables the `infra_maintainer` template on Terraform repositories.
- **ChangePoller** — polling fallback for repositories registered without a webhook; calls `GitProvider.get_commit_history()` on a configurable interval (default 60 s); stores `last_polled_sha` per repo to skip no-op cycles.
- **Working directory snapshot/restore** — `AgentCheckpoint` now carries a `snapshot_path` pointing to a `.tar.gz` of the working directory at each LangGraph checkpoint transition; crash-resume restores from this snapshot before re-entering any node, eliminating partial-write artefacts.
- **GitHub Actions CI/CD pipeline** — `ci.yml` (lint + type check + unit tests + integration tests on every PR), `release.yml` (PyPI publish on semver tag), `docker.yml` (image build and push on release).
- **Alembic `check` gate in CI** — every PR fails if unapplied migrations are detected; migration authoring guidance added to `README.md`.
- **Full GovernancePolicy enforcement** — `max_lines_per_change` hard stop + re-plan in the REVIEWING node; `max_files_per_change` guard at EXECUTING node start; both violations emit structured log event `governance_violation`.
- **SensitivePathRule enforcement mid-EXECUTING** — file writes matching a sensitive pattern pause EXECUTING, restore working directory from snapshot, write an `AuditEntry` with `event=sensitive_path_blocked`, and notify the human reviewer via a PR comment before resuming.
- **Credential hygiene improvements** — `credentials_secret_ref` resolved at runtime only; no token value appears in logs, subprocess args, or `AgentCheckpoint`; working directories are isolated `tempfile.TemporaryDirectory` instances cleaned up after task completion or failure.
- **AuditEntry append-only enforcement** — removed `delete_audit_entry` / `clear_audit_log` methods from `AgentStateStore` implementation and Protocol; any attempt to delete audit entries now raises `GovernanceError`.
- **E2E integration tests on real GitHub repos** — `tests/e2e/test_phase3_real_repos.py` validates registration, parse, and agent PR open against two fixture-backed real repos (Python API, TypeScript frontend) using `GITHUB_TEST_TOKEN`; skipped cleanly without the token.
- **Test coverage audit** — targeted unit tests added to reach ≥ 80 % coverage across graph mutations, governance enforcement, and webhook ingestion paths; enforced via `coverage report --fail-under=80` in CI.
- **FalkorDB vs Neo4j decision record** — `docs/decisions/001-graph-db.md` documents evaluation results, benchmark data, and recommendation with a forward migration plan.
- **`README.md` quickstart** — 5-minute guide: Docker Compose up → register a repo → submit a task → review agent PR.
- **`CHANGELOG.md`** — this file; semver changelog initialized with Phase 1–3 entries.
- **Example registration files** — `examples/register_typescript_frontend.py` (TypeScript/React, `frontend_maintainer` template) and `examples/register_terraform_infra.py` (Terraform, `infra_maintainer` template, `SUPERVISED` autonomy).
- **`docker-compose.prod.yml`** — production overrides: CPU/memory resource limits, `restart: unless-stopped` policies, named volumes for FalkorDB and PostgreSQL data.
- **CLI `--help` polish** — all `tractable` commands have complete, non-truncated help text.

### Fixed

- **`context.py` error taxonomy gap** — `assemble_agent_context()` raised a bare `ValueError` for an unknown `agent_template`; replaced with `RecoverableError` so the caller can correct `RepositoryRegistration.agent_template` and retry without corrupting state (`tractable/agent/context.py:131`).

---

## [0.2.0] — 2026-02-28

_Phase 2 — Single-Repo Agent Runtime._

### Added

- **LangGraph four-node workflow** — `PLANNING → EXECUTING → REVIEWING → COORDINATING`; checkpoint state saved at each transition so a crashed agent resumes from its last node, not from scratch.
- **Three-layer `AgentContext` assembly** — base template defaults → registration overrides → human-pinned instructions (via `tractable agent edit`) assembled into the system prompt and stored in PostgreSQL.
- **Agent tools** — `code_editor`, `graph_query_mcp`, `git_ops`, `test_runner`, `linter`, `pipeline_watcher`.
- **`GovernancePolicy` enforcement** — scope-violation gate in `code_editor` (write-time); test and lint requirement checks in REVIEWING node.
- **Token budget tracking** — per-task token counter; automatic escalation from Claude Sonnet to Claude Opus when budget is exceeded.
- **`AgentCheckpoint` save/restore** at each LangGraph phase transition for crash recovery.
- **FastAPI webhook endpoint** at `/webhooks/github` with HMAC-SHA256 signature verification.
- **`ChangeIngestionPipeline.ingest_changes()`** — incremental parse of changed files → temporal graph mutations.
- **Redis Pub/Sub `EventBus`** implementation.
- **`AgentLifecycleManager`** — debounced agent wake-up (30 s default); `get_changes_since(last_active)` catch-up mechanism.
- **`pipeline_watcher` tool and `PIPELINE_TRIAGE` sub-workflow** — CI failure classification → fix or escalate.
- **`NotificationRouter`** — determines which agents to notify given a `ChangeIngestionResult`.
- **CLI commands** — `tractable agent context`, `tractable agent edit`, `tractable agent list`, `tractable task submit`, `tractable logs`.
- **Structlog initialization** — context binding, JSON output in production, colored console in dev; every log entry includes `agent_id`, `task_id`, `repo`, `event`, `level`.
- **Error taxonomy retrofit** — `TransientError`, `RecoverableError`, `GovernanceError`, `FatalError` applied across all Phase 1 modules.
- **GitHub write operations** — `create_branch`, `create_pull_request`, `merge_pull_request` with credential injection and URL validation.
- **Docker Compose `tractable` service** — minimal FastAPI server integrated into the stack.
- **Integration tests** for all Phase 2 exit criteria using fixture-based repos (no live GitHub API calls).

### Fixed

- **asyncio event loop conflict** — `AgentLifecycleManager` and `ChangeIngestionPipeline` shared a single event loop across threads, causing `RuntimeError: This event loop is already running`; each worker now creates an isolated loop via `asyncio.new_event_loop()`.
- **Mock patch target mismatch** — integration tests patched `tractable.providers.github.PyGithub` instead of the import path used by the module under test; corrected to `tractable.agent.workflow.GitHubProvider`.
- **Docker networking isolation** — `tractable` service used `localhost` to reach FalkorDB and PostgreSQL inside Docker Compose; corrected to service-name DNS (`falkordb`, `postgres`) matching the Compose network.
- **FalkorDB Cypher syntax incompatibility** — `WHERE` clause used `IS NULL` for optional-match absence, which FalkorDB rejects; replaced with `NOT EXISTS { MATCH ... }` pattern compatible with FalkorDB's Cypher dialect.
- **`RedisError` exception hierarchy** — code caught `redis.exceptions.RedisError` but the installed version raised `redis.RedisError`; unified to catch both via the common base class.

---

## [0.1.0] — 2026-01-31

_Phase 1 — Foundation._

### Added

- **Core type system** — complete `tractable/types/` from `tech-spec.py` and `realtime-temporal-spec.py`; Pydantic v2 models for `RepositoryRegistration`, `AgentContext`, `AgentCheckpoint`, `AuditEntry`, `ParseResult`, `ChangeSet`, and all supporting types.
- **Protocol interfaces** — `tractable/protocols/` defining every integration boundary as a structural Protocol: `GitProvider`, `CodeParser`, `CodeGraph`, `AgentStateStore`, `EventBus`; Pyright runs in strict mode on this package.
- **FalkorDB temporal graph** — full temporal schema with `valid_from`, `valid_until`, `observed_at`, `commit_sha` on every node; modifying a file closes the old entity version and opens a new one; nothing is deleted.
- **`TemporalCodeGraph`** — `query_current()`, `get_entity()`, `apply_mutations()`, `get_changes_since()`, `get_entity_history()`.
- **`GraphConstructionPipeline.ingest_repository()`** — clones repo, parses all files, builds temporal graph in FalkorDB.
- **`GitHubProvider` read operations** — `clone`, `get_file_content`, `list_files`, `get_diff`, `get_commit_history`; token-based rate limit handling with exponential backoff.
- **`PythonParser`** — tree-sitter Python → `ParseResult`; extracts functions, classes, imports.
- **`TypeScriptParser`** (basic) — tree-sitter TypeScript → `ParseResult`; extracts functions, classes, imports.
- **`AgentStateStore` on PostgreSQL** — `AgentContext`, `AgentCheckpoint`, `AuditEntry` tables; Alembic migrations.
- **Docker Compose stack** — FalkorDB, PostgreSQL, Redis services with named volumes and health checks.
- **CLI stubs** — `tractable register` and `tractable status`.
- **`pyproject.toml` toolchain** — `uv`, `ruff`, `pyright` (strict), `pytest`; all quality gates configured from day one.
- **Unit and integration tests** — 308 unit tests passing; integration tests verifying temporal versioning (`get_changes_since()`) and graph ingestion against the live Docker Compose stack.

[Unreleased]: https://github.com/example/tractable/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/example/tractable/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/example/tractable/releases/tag/v0.1.0
