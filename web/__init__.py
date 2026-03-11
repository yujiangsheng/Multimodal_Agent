"""Pinocchio Web interface — FastAPI-based REST & WebSocket API.

Provides a browser-friendly frontend and JSON API for interacting with
the Pinocchio agent.  Key capabilities:

* ``POST /chat``        — synchronous chat with optional file uploads
* ``GET  /chat/stream`` — SSE streaming for real-time token delivery
* ``GET  /sessions``    — list / manage conversation sessions
* ``GET  /memory/*``    — inspect and search the dual-axis memory system

Start the server::

    uvicorn web.app:app --reload

Or via the CLI with ``pinocchio --web``.

See :mod:`web.app` for the full endpoint reference.
"""

from web.app import app

__all__ = ["app"]
