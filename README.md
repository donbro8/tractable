# Tractable

## What is Tractable

Tractable is an autonomous multi-agent coding framework. Agents act as **external contributors** to your repositories — they clone, branch, write code, run tests, and open pull requests, exactly as a human engineer would. Agent identity and context live in a central state store, not inside any managed repository. The graph is the memory: Tractable builds a live temporal knowledge graph of your codebase so agents never re-read entire repos from scratch.

> **Current status:** Phases 1 and 2 complete. The core infrastructure, LangGraph agent runtime, and single-repo agent workflow are fully implemented and tested. Phase 3 (CI/CD, HCL parsing, polling fallback, governance hardening) is next.

---

## Prerequisites

- **Docker** and **Docker Compose** — for the service stack (FalkorDB, PostgreSQL, Redis)
- **Python 3.11+** with [uv](https://docs.astral.sh/uv/) — install uv with `curl -Lsf https://astral.sh/uv/install.sh | sh`
- **Git** — for cloning the repository
- **GitHub account** with a [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) (repo + webhook scopes) — required for repository operations

---

## Quickstart (5 minutes)

The following commands take you from a fresh clone to a running agent in under 5 minutes. Run them in order; each step's purpose and expected output are shown.

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-org/tractable.git && cd tractable
```

Clones Tractable and enters the project directory. You should see a `tractable/` directory with `pyproject.toml`, `deploy/`, `examples/`, and this README.

### Step 2 — Set up environment variables

```bash
cp .env.example .env          # fill in DATABASE_URL, REDIS_URL, GITHUB_TOKEN
```

Creates your local `.env` from the template. Open `.env` and fill in at minimum `DATABASE_URL`, `REDIS_URL`, `GITHUB_TOKEN`, and `ANTHROPIC_API_KEY`. Do not commit `.env` — it is listed in `.gitignore`.

### Step 3 — Start the service stack

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Starts FalkorDB (graph store), PostgreSQL (agent state), Redis (event bus), and the Tractable registry service in the background. Expected output: containers `tractable-falkordb-1`, `tractable-postgres-1`, `tractable-redis-1`, and `tractable-registry-1` all show `Started`.

### Step 4 — Install Python dependencies

```bash
uv sync --extra dev
```

Creates a virtual environment and installs Tractable and all development tools (ruff, pyright, pytest). Expected output: `Resolved N packages` followed by `Installed N packages`.

### Step 5 — Apply database migrations

```bash
uv run alembic upgrade head
```

Creates all PostgreSQL tables (agent contexts, checkpoints, audit log). Expected output: `Running upgrade -> <revision_id>, <migration description>` for each migration, ending with no error.

### Step 6 — Register a repository

```bash
tractable register examples/register_python_api.py
```

Validates the registration config, clones the repository, ingests the codebase into the knowledge graph, creates an agent instance from the template, and registers the GitHub webhook endpoint. Expected output:
```
Registered agent <agent-id> for repo my-org/my-api
Webhook registered at https://<host>/webhooks/github
```

### Step 7 — Submit a task

```bash
tractable task submit "Fix the failing test" --repo my-api
```

Creates a task record and assigns it to the agent for `my-api`. Expected output: a UUID task ID printed to stdout, e.g. `a3f1b2c4-...`.

### Step 8 — Check status

```bash
tractable status
```

Shows all registered repositories and their agent states. Expected output: a table with your repo listed and agent status `IDLE` (task queued) or `WORKING` (agent executing). Within seconds the agent begins planning.

---

## Core Commands

| Command | Purpose |
|---|---|
| `tractable register <config.py>` | Register a repository: validate config, clone repo, ingest graph, create agent, register webhook |
| `tractable status` | Show all registered repos and the current state of their agents (IDLE, WORKING, BLOCKED) |
| `tractable agent list` | List all agent instances with their IDs, repos, autonomy level, and current status |
| `tractable agent context <agent-id>` | Print the assembled system prompt that will be fed to the LLM for this agent |
| `tractable agent edit <agent-id>` | Open the agent's pinned instructions in `$EDITOR` — changes take effect on the next task |
| `tractable task submit "<description>" --repo <name>` | Submit a natural-language task to the agent for the named registered repo |
| `tractable task status <task-id>` | Show the current phase and status of a specific task (PLANNING, EXECUTING, REVIEWING, DONE, FAILED) |
| `tractable logs` | Stream structured logs; filter by `--agent <id>` or `--task <id>`; follow with `--follow` |

---

## Architecture Overview

Tractable is built in layers: Pydantic v2 core models (`tractable/types/`) define all data structures; structural Protocols (`tractable/protocols/`) define every integration contract; concrete implementations (GitHub provider, FalkorDB graph, PostgreSQL state store) satisfy those Protocols and can be swapped without touching the agent runtime. The LangGraph agent executes four sequential nodes — `PLANNING → EXECUTING → REVIEWING → COORDINATING` — with checkpoint state saved at every transition so crashed agents resume from their last node. A FastAPI webhook receiver keeps the knowledge graph in sync within seconds of every commit, so agents always have current context. See [docs/decisions/](docs/decisions/) for architectural decision records (including the graph database evaluation) and [PLAN.md](.claude/PLAN.md) for the full implementation plan.

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

**Temporal graph.** All graph nodes carry `valid_from`, `valid_until`, `observed_at`, and `commit_sha`. Modifying a file closes the old entity version and opens a new one; nothing is ever deleted. `TemporalCodeGraph.get_changes_since(t)` is the agent's primary catch-up mechanism after dormancy.

**Error taxonomy.** All errors are one of four types:

| Type | Behaviour |
|---|---|
| `TransientError` | Retry with exponential backoff (max 3 attempts) |
| `RecoverableError` | Re-plan the current task phase |
| `GovernanceError` | Halt, write audit entry, notify human |
| `FatalError` | Fail task gracefully, preserve checkpoint |

**Structured logging.** Every log entry includes `agent_id`, `task_id`, `repo`, `event`, and `level`. JSON in production; coloured console in development.

---

## The agent workflow

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

---

## Contributing

Standard pull-request workflow. See [docs/contributing/migrations.md](docs/contributing/migrations.md) for guidance on database migrations. The CI pipeline enforces lint, type-check, unit tests, and integration tests on every PR — all checks must pass before merge. Run these locally before opening a PR:

```bash
uv run ruff check tractable/ tests/
uv run pyright tractable/
uv run pytest tests/unit
uv run pytest tests/integration  # requires: docker compose -f deploy/docker-compose.yml up
```

Versioning follows [Semantic Versioning](https://semver.org/). Breaking changes to `tractable/protocols/` or `tractable/types/` always increment the major version.

### Required secrets for CI

Configure the following secrets in your GitHub repository (Settings → Secrets and variables → Actions):

| Secret | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | LLM calls in integration and E2E tests |
| `GITHUB_TEST_TOKEN` | E2E tests that create branches and PRs on fixture repos |

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

---

## License

MIT License. See [LICENSE](LICENSE) for details.
