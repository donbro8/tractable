# Tractable

Tractable is an autonomous multi-agent coding framework. Agents act as **external contributors** to your repositories — they clone, branch, write code, run tests, and open pull requests, exactly as a human engineer would. Agent identity and context live in a central state store, not inside any managed repository.

> **Current status:** Phases 1 and 2 complete. The core infrastructure, LangGraph agent runtime, and single-repo agent workflow are fully implemented and tested. Phase 3 (CI/CD, HCL parsing, polling fallback, governance hardening) is next.

---

## Quickstart (5 minutes)

### 1. Start the service stack

```bash
cp .env.example .env          # fill in credentials (see Environment Variables below)
docker compose -f deploy/docker-compose.yml up
```

This starts FalkorDB (graph store), PostgreSQL (agent state), Redis (event bus), and the Tractable registry service.

### 2. Apply database migrations

```bash
uv run alembic upgrade head
```

### 3. Register a repository

Create a registration config (see `examples/register_python_api.py`):

```python
# my_repo.py
from tractable.types.config import RepositoryRegistration, GitProviderConfig

registration = RepositoryRegistration(
    name="my-org/my-api",
    git_url="https://github.com/my-org/my-api.git",
    git_provider=GitProviderConfig(
        provider_type="github",
        credentials_secret_ref="GITHUB_TOKEN",
        default_branch="main",
    ),
    primary_language="python",
    agent_template="api_maintainer",
)
```

Then register it:

```bash
tractable register my_repo.py
```

This validates the config, clones the repository, ingests the codebase into the knowledge graph, creates an agent instance from the template, and registers the GitHub webhook endpoint.

### 4. Submit a task

```bash
tractable task submit --agent <agent-id> --description "Add input validation to the /users endpoint"
```

### 5. Watch it work

```bash
tractable status
tractable logs --task <task-id>
```

The agent plans, writes code, runs tests, and opens a pull request. In `SUPERVISED` mode (the default), it always PRs — you merge.

---

## What It Does

### The agent workflow

Every agent executes four sequential phases managed by LangGraph:

```
PLANNING → EXECUTING → REVIEWING → COORDINATING
```

- **PLANNING** — queries the knowledge graph for relevant context, reads key files, produces a structured plan
- **EXECUTING** — edits files using the `code_editor` tool, iterates based on test output
- **REVIEWING** — runs the test suite and linter; the agent does not proceed until both pass (enforced by `GovernancePolicy`)
- **COORDINATING** — creates a pull request and posts a human-readable summary

Checkpoint state is saved at each phase transition. If an agent crashes, it resumes from the last saved checkpoint, not from scratch.

### The knowledge graph

Tractable builds a **temporal knowledge graph** of your codebase using tree-sitter. Every entity — function, class, import, module — carries version metadata (`valid_from`, `valid_until`, `commit_sha`). When a file changes, the old version is closed and a new version is opened; nothing is ever deleted.

When an agent wakes up after a period of inactivity, it calls `get_changes_since(last_active)` to catch up instantly without re-reading the entire codebase.

### Real-time reactivity

A GitHub webhook endpoint (`/webhooks/github`) receives push and PR events. Changed files are re-parsed incrementally, the graph is updated, and a `ChangeNotification` is published on the Redis event bus. Agents subscribe to their repository's channel and wake up within seconds of a new commit (with a 30-second debounce to batch rapid pushes).

### Governance

Every agent operates under a `GovernancePolicy` that constrains its behaviour:

- **Scope enforcement** — the `code_editor` tool blocks writes outside the agent's `allowed_paths` at write time
- **Change limits** — `max_files_per_change` and `max_lines_per_change` guards prevent runaway edits
- **Sensitive paths** — writes matching a `SensitivePathRule` pattern pause execution, write an audit entry, and notify a human reviewer before resuming
- **Audit log** — every agent action is written to an append-only `AuditEntry` store; `GovernanceError` and above also appear in structured logs

---

## Architecture

```
tractable/
├── types/          # Pydantic v2 core models (agent, config, git, graph, task)
├── protocols/      # Structural Protocol interfaces for every integration point
├── providers/      # Git provider implementations (GitHub; GitLab stub)
├── graph/          # FalkorDB client + temporal graph implementation
├── parsing/        # tree-sitter parsers (Python, TypeScript) + ingestion pipeline
├── state/          # SQLAlchemy async + PostgreSQL (AgentContext, checkpoints, audit)
├── agent/          # LangGraph workflow, nodes, tools, lifecycle manager
├── reactivity/     # FastAPI webhook receiver, change ingestion, notification router
├── registry/       # FastAPI registry service
├── events/         # Redis Pub/Sub event bus
├── cli/            # Typer CLI (register, status, agent, task, logs)
├── errors.py       # Error taxonomy (Transient, Recoverable, Governance, Fatal)
└── logging.py      # structlog configuration
```

### Design rules

**Protocols are the contract boundary.** Every integration point in `tractable/protocols/` is a structural Protocol. Swap an implementation (e.g. FalkorDB → Neo4j, GitHub → GitLab) by providing a class that satisfies the Protocol — agent logic never changes. Pyright runs in strict mode on `protocols/` and `types/`.

**Error taxonomy.** All errors are one of four types:

| Type | Behaviour |
|---|---|
| `TransientError` | Retry with exponential backoff (max 3 attempts) |
| `RecoverableError` | Re-plan the current task phase |
| `GovernanceError` | Halt, write audit entry, notify human |
| `FatalError` | Fail task gracefully, preserve checkpoint |

**Structured logging.** Every log entry includes `agent_id`, `task_id`, `repo`, `event`, and `level`. JSON in production; coloured console in development.

---

## Configuration

### Repository registration options

```python
from tractable.types.config import (
    RepositoryRegistration,
    GitProviderConfig,
    AgentScope,
    AutonomyLevel,
)

registration = RepositoryRegistration(
    name="my-org/my-api",
    git_url="https://github.com/my-org/my-api.git",
    git_provider=GitProviderConfig(
        provider_type="github",               # "github" (full) | "gitlab" (stub)
        credentials_secret_ref="GITHUB_TOKEN", # env var name — never the token itself
        default_branch="main",
    ),
    primary_language="python",                # "python" | "typescript"
    agent_template="api_maintainer",          # built-in template with sensible defaults

    # Optional: restrict the agent to specific paths within the repository.
    # Without a scope the agent has access to the full repo.
    scope=AgentScope(
        allowed_paths=["src/payments/", "src/billing/"],
        allowed_extensions=[".py"],
        deny_paths=["src/payments/legacy/"],   # always excluded, overrides allowed_paths
    ),

    autonomy_level=AutonomyLevel.SUPERVISED,  # agent PRs, human merges (default)
)
```

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL DSN (`postgresql+asyncpg://user:pass@host/db`) |
| `REDIS_URL` | Yes | Redis DSN (`redis://localhost:6379`) |
| `GITHUB_TOKEN` | For GitHub ops | PyGithub authentication |
| `ANTHROPIC_API_KEY` | For agent runs | LLM calls (Claude Sonnet 4.6 by default) |
| `FALKORDB_HOST` | Yes | FalkorDB host (default: `localhost`) |
| `FALKORDB_PORT` | Yes | FalkorDB port (default: `6380`) |

Copy `.env.example` to `.env` and fill in values before running.

---

## CLI Reference

```bash
# Register a repository
tractable register <config.py>

# Show status of all registered repos and their agents
tractable status
tractable status --repo <name>

# Manage agents
tractable agent list
tractable agent context <agent-id>
tractable agent edit <agent-id>          # open agent context in $EDITOR

# Submit and manage tasks
tractable task submit --agent <id> --description "..."
tractable task status <task-id>
tractable task cancel <task-id>

# View logs
tractable logs --agent <agent-id>
tractable logs --task <task-id>
```

---

## Development

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose

### Setup

```bash
uv sync --extra dev
cp .env.example .env   # fill in values
docker compose -f deploy/docker-compose.yml up
uv run alembic upgrade head
```

### Running tests

```bash
# Unit tests — no external services required
uv run pytest tests/unit

# Integration tests — requires Docker Compose stack running
uv run pytest tests/integration

# E2E tests — requires full stack + GITHUB_TEST_TOKEN
uv run pytest tests/e2e

# Single test
uv run pytest tests/unit/path/to/test_file.py::test_name
```

### Linting and type checking

```bash
uv run ruff check          # lint
uv run ruff format         # auto-format
uv run pyright             # strict type check
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Agent runtime | LangGraph |
| LLM | Claude Sonnet 4.6 (default), Claude Opus 4.6 (escalation) |
| Graph database | FalkorDB (Cypher-compatible, Redis-protocol) |
| Agent state store | PostgreSQL via SQLAlchemy async + Alembic |
| API / webhooks | FastAPI + Uvicorn |
| Code parsing | tree-sitter (Python, TypeScript) |
| Event bus | Redis Pub/Sub |
| CLI | Typer + Rich |
| Config / models | Pydantic v2 |
| Packaging | uv + hatchling |

---

## Roadmap

### What's built (Phases 1–2)

- Core type system and Protocol interfaces
- GitHub provider (clone, branch, commit, PR, webhook normalisation)
- Temporal knowledge graph (FalkorDB) with Python and TypeScript parsers
- PostgreSQL agent state store (context, checkpoints, audit log)
- Full LangGraph agent workflow with checkpoint/recovery
- Agent tools: `code_editor`, `graph_query`, `git_ops`, `test_runner`, `linter`, `pipeline_watcher`
- Real-time reactivity: GitHub webhooks → incremental graph updates → agent wake
- Redis Pub/Sub event bus
- Registry service
- CLI: register, status, agent, task, logs
- 501+ unit tests; Docker Compose local stack

### Phase 3 — MVP Hardening (next)

- GitHub Actions CI/CD pipeline (lint, type-check, unit + integration tests on every PR; PyPI publish on tag)
- HCL parser (Terraform support; enables `infra_maintainer` template)
- `ChangePoller` — polling fallback for repos registered without a webhook
- Full `GovernancePolicy` enforcement: `max_lines_per_change`, `max_files_per_change`, `SensitivePathRule`
- Credential hygiene: tokens never appear in logs, audit entries, or committed files
- Append-only audit log enforcement
- E2E tests against real GitHub repos (fixture-backed)
- `CHANGELOG.md` initialized
- Graph database evaluation decision record (FalkorDB vs Neo4j)

### Phase 4+

- MCP server exposing graph queries to agents
- `LLMProvider` abstraction (pluggable; local model support)
- Multi-repo coordinator agent
- GitLab and CodeCommit providers
- LLM fuzzy reference resolver (cross-file entity linking without exact names)

### Phase 5+

- Hierarchical sub-repo agents (folder-level and file-level specialisation)
- Manager/execution agent split

### Phase 6+

- A2A (agent-to-agent) communication protocol
- Cross-cutting meta-agents (code reviewer, security scanner, DevOps monitor)
- Centralised agent card registry

### Phase 7+

- Management UI with full project → repository → agent hierarchy visibility
- Auto-provisioning of optimal agent topology from repository analysis
- Kubernetes deployment manifests
- Scale to thousands of concurrent agents

---

## Contributing

Standard pull-request workflow. The CI pipeline enforces lint, type-check, unit tests, and integration tests on every PR. Run these locally before opening a PR:

```bash
uv run ruff check tractable/ tests/
uv run pyright tractable/
uv run pytest tests/unit
uv run pytest tests/integration  # requires: docker compose -f deploy/docker-compose.yml up
```

Versioning follows [Semantic Versioning](https://semver.org/). Breaking changes to `tractable/protocols/` or `tractable/types/` always increment the major version.

### Branch Protection Settings

Configure the following required status checks on the `main` branch (Settings → Branches → Branch protection rules). These require admin permissions and are not set by the CI workflow files automatically:

| Required Status Check | Workflow |
|---|---|
| `ci / lint` | `.github/workflows/ci.yml` |
| `ci / typecheck` | `.github/workflows/ci.yml` |
| `ci / unit-tests` | `.github/workflows/ci.yml` |
| `ci / integration-tests` | `.github/workflows/ci.yml` |

Recommended additional settings:
- **Require a pull request before merging** — at least 1 approving review
- **Require branches to be up to date before merging**
- **Do not allow bypassing the above settings**
