FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy dependency files and install deps (without the project itself) for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the package source and install the project
COPY tractable/ ./tractable/
RUN uv sync --frozen --no-dev

# Use uv's managed Python
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "-m", "tractable.cli"]
