# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Firewatch Env Environment.

This module creates an HTTP server that exposes the FirewatchEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import FirewatchAction, SystemObservation
    from .firewatch_env_environment import FirewatchEnvironment, _empty_observation
except (ImportError, SystemError):
    from models import FirewatchAction, SystemObservation
    from server.firewatch_env_environment import FirewatchEnvironment, _empty_observation


# Module-level singleton — ensures /reset and /step share state across HTTP calls.
# openenv-core calls _env_factory() per request; returning the same instance
# preserves episode state between /reset and /step.
_SINGLETON_ENV = FirewatchEnvironment()


def _env_factory() -> FirewatchEnvironment:
    return _SINGLETON_ENV


# Create the app with web interface and README integration
app = create_app(
    _env_factory,
    FirewatchAction,
    SystemObservation,
    env_name="firewatch_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


# Zero-crash policy (CLAUDE.md): invalid requests must return HTTP 200 with error
# in the response body, never HTTP 422 or 500.
@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    obs = _empty_observation(f"Invalid request: {exc.errors()}")
    return JSONResponse(
        status_code=200,
        content=obs.model_dump(),
    )


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        uv run --project . server --host 0.0.0.0 --port 7860
        python -m firewatch_env.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn firewatch_env.server.app:app --workers 4
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="FirewatchEnv server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
