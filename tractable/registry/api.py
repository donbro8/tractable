"""FastAPI application factory for the Tractable registry service.

TASK-2.6.1 — mounts the GitHub webhook router.
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from tractable.protocols.reactivity import ChangeIngestionPipeline
from tractable.reactivity.webhook_receiver import create_webhook_router


def create_app(
    pipeline: ChangeIngestionPipeline,
    webhook_secret: str | None = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    pipeline:
        ``ChangeIngestionPipeline`` instance dispatched for every valid webhook.
    webhook_secret:
        HMAC secret for GitHub webhook signature verification.
        Falls back to the ``GITHUB_WEBHOOK_SECRET`` environment variable.
    """
    secret = webhook_secret or os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    app = FastAPI(title="Tractable Registry")
    app.include_router(create_webhook_router(pipeline, secret))
    return app
