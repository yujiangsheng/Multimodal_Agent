"""Pinocchio Web Demo — FastAPI backend.

Provides a comprehensive REST API and serves the single-page chat UI from
``web/static/index.html``.  The frontend supports text input, multimodal
file uploads (images / audio / video), real-time status monitoring, memory
exploration, multi-session management, and message edit / regenerate.

All 8 sub-systems of the Pinocchio agent are exposed as API endpoints:
tools management, RAG knowledge base, code sandbox, structured tracing,
task planning / ReAct reasoning, multi-agent collaboration, Agent Graph
workflows, and MCP protocol bridging.

Security
--------
Set the ``PINOCCHIO_API_KEY`` environment variable to require an
``X-API-Key`` header on all ``/api/`` endpoints.  When unset, no
authentication is enforced (suitable for local development).

Endpoints
---------
==========================================  ======  ============================================
Path                                        Verb    Description
==========================================  ======  ============================================
``/``                                       GET     Serve the frontend (``index.html``)
``/api/health``                             GET     Quick health / readiness probe
**Chat**
``/api/chat``                               POST    Send a message (form-data with file uploads)
``/api/chat/stream``                        GET     SSE streaming response (text-only)
``/api/chat/progress``                      POST    SSE cognitive loop with phase progress
``/api/chat/regenerate``                    POST    Regenerate the last assistant response
``/api/chat/edit``                          POST    Edit a user message and regenerate
**Sessions**
``/api/sessions``                           GET     List all sessions
``/api/sessions``                           POST    Create a new session and switch to it
``/api/sessions/{id}/switch``               POST    Switch to an existing session
``/api/sessions/{id}``                      PUT     Rename a session
``/api/sessions/{id}``                      DELETE  Delete a session
``/api/sessions/{id}/messages``             GET     Get all messages for a session
**Tools**
``/api/tools``                              GET     List all registered tools
``/api/tools/stats``                        GET     Tool invocation usage metrics
``/api/tools/{name}/enable``                POST    Re-enable a disabled tool
``/api/tools/{name}/disable``               POST    Temporarily disable a tool
**RAG Knowledge Base**
``/api/rag/documents``                      GET     List all ingested documents
``/api/rag/documents``                      POST    Upload & ingest a document file
``/api/rag/documents/text``                 POST    Ingest raw text directly
``/api/rag/documents/{id}``                 DELETE  Delete a document and its chunks
``/api/rag/search``                         POST    Vector/keyword search across chunks
**Code Sandbox**
``/api/sandbox/execute``                    POST    Execute Python code in isolated sandbox
**Tracing**
``/api/tracing``                            GET     Export all traces as JSON
``/api/tracing/summary``                    GET     Human-readable trace summary
``/api/tracing/stats``                      GET     Aggregate trace statistics
``/api/tracing``                            DELETE  Clear all recorded traces
**Planning & ReAct**
``/api/plan``                               POST    Decompose a task into a structured plan
``/api/react``                              POST    Run a ReAct reasoning loop
**Collaboration**
``/api/team/members``                       GET     List team members
``/api/team/members``                       POST    Add a team member
``/api/team/members/{id}``                  DELETE  Remove a team member
``/api/team/collaborate``                   POST    Run a multi-agent collaborative task
**Graph Workflows**
``/api/graph/templates``                    GET     List available graph workflow templates
``/api/graph/run``                          POST    Execute a graph template
**MCP Protocol**
``/api/mcp/connect``                        POST    Connect to an MCP tool server
``/api/mcp/disconnect``                     POST    Disconnect from an MCP server
**Memory**
``/api/memory/episodes``                    GET     All episodic memory records
``/api/memory/knowledge``                   GET     All semantic knowledge entries
``/api/memory/procedures``                  GET     All procedural memory entries
``/api/memory/working``                     GET     Current working-memory items
``/api/memory/trend``                       GET     Improvement trend + per-episode timeline
**System**
``/api/status``                             GET     Agent state summary (JSON)
``/api/reset``                              POST    Reset session (persistent memory kept)
``/api/exit``                               POST    Gracefully shut down the server
``/api/cache/stats``                        GET     Response cache hit/miss statistics
``/api/cache``                              DELETE  Flush the response cache
``/api/config/providers``                   GET     List available LLM provider presets
==========================================  ======  ============================================

Usage
-----
::

    # Install web extras
    pip install -e ".[web]"

    # Start the server (default: http://localhost:8000)
    python -m web.app

    # With authentication
    PINOCCHIO_API_KEY=my-secret python -m web.app
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from config import PinocchioConfig
from pinocchio import Pinocchio
from pinocchio.collaboration.team import TeamMember

# ── Initialise ────────────────────────────────────────────────

app = FastAPI(title="Pinocchio Demo", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Authentication ────────────────────────────────────────────

_API_KEY = os.getenv("PINOCCHIO_API_KEY")

# ── Rate limiting ─────────────────────────────────────────────

_RATE_LIMIT_RPM = int(os.getenv("PINOCCHIO_RATE_LIMIT_RPM", "30"))  # per minute
_RATE_LIMIT_WINDOW = 60  # seconds
_MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB

_rate_tracker: dict[str, list[float]] = defaultdict(list)
_rate_tracker_last_purge: float = 0.0
_RATE_TRACKER_PURGE_INTERVAL = 300  # full purge every 5 minutes


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-IP rate limiting on API endpoints."""
    if request.url.path.startswith("/api/"):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - _RATE_LIMIT_WINDOW

        # Periodic full purge of stale IP keys
        global _rate_tracker_last_purge
        if now - _rate_tracker_last_purge > _RATE_TRACKER_PURGE_INTERVAL:
            stale = [
                ip for ip, ts in _rate_tracker.items()
                if not ts or ts[-1] < window_start
            ]
            for ip in stale:
                del _rate_tracker[ip]
            _rate_tracker_last_purge = now

        # Purge old entries for this IP
        _rate_tracker[client_ip] = [
            t for t in _rate_tracker[client_ip] if t > window_start
        ]
        if len(_rate_tracker[client_ip]) >= _RATE_LIMIT_RPM:
            return JSONResponse(
                {"error": "Rate limit exceeded. Try again later."},
                status_code=429,
                headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
            )
        _rate_tracker[client_ip].append(now)

        # Request size check
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_REQUEST_SIZE:
            return JSONResponse(
                {"error": f"Request too large (max {_MAX_REQUEST_SIZE // 1024 // 1024} MB)"},
                status_code=413,
            )

    return await call_next(request)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Reject unauthenticated API requests when *PINOCCHIO_API_KEY* is set."""
    if _API_KEY and request.url.path.startswith("/api/"):
        key = request.headers.get("X-API-Key", "")
        if key != _API_KEY:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "pinocchio_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

cfg = PinocchioConfig()
agent = Pinocchio(
    model=cfg.model,
    api_key=cfg.api_key,
    base_url=cfg.base_url,
    data_dir=cfg.data_dir,
    verbose=cfg.verbose,
    max_workers=cfg.max_workers,
    parallel_modalities=cfg.parallel_modalities,
    meta_reflect_interval=cfg.meta_reflect_interval,
    num_ctx=cfg.num_ctx,
)


# ── Helpers ───────────────────────────────────────────────────

# Allowed file extensions for uploads (security: prevent arbitrary file types)
_ALLOWED_EXTENSIONS = {
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg",
    # Audio
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
    # Video
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv",
}

# Document file types accepted by the RAG ingest endpoint
_RAG_ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".pdf",
}


def _save_upload(upload: UploadFile, *, allowed: set[str] | None = None) -> str:
    exts = allowed or _ALLOWED_EXTENSIONS
    suffix = Path(upload.filename or "file").suffix.lower()
    if suffix not in exts:
        raise ValueError(
            f"File type '{suffix}' is not allowed. "
            f"Allowed types: {', '.join(sorted(exts))}"
        )
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest)


# ── Chat ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    """Quick readiness probe (no LLM call)."""
    return JSONResponse({
        "status": "ok",
        "version": app.version,
        "model": cfg.model,
        "session_id": agent._current_session_id,
    })


@app.post("/api/chat")
async def chat(
    text: str = Form(default=""),
    images: list[UploadFile] = File(default=[]),
    audios: list[UploadFile] = File(default=[]),
    videos: list[UploadFile] = File(default=[]),
):
    image_paths = [_save_upload(f) for f in images if f.filename and f.size]
    audio_paths = [_save_upload(f) for f in audios if f.filename and f.size]
    video_paths = [_save_upload(f) for f in videos if f.filename and f.size]

    try:
        response = await agent.async_chat(
            text=text or None,
            image_paths=image_paths or None,
            audio_paths=audio_paths or None,
            video_paths=video_paths or None,
        )
        return JSONResponse({
            "response": response,
            "status": "ok",
            "user_message_id": agent._last_user_msg_id,
            "assistant_message_id": agent._last_asst_msg_id,
            "session_id": agent._current_session_id,
        })
    except Exception as exc:
        return JSONResponse(
            {"response": f"处理错误: {exc}", "status": "error"},
            status_code=500,
        )
    finally:
        for p in image_paths + audio_paths + video_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


# ── Streaming Chat (SSE) ─────────────────────────────────────

@app.get("/api/chat/stream")
async def chat_stream(text: str = ""):
    """Server-Sent Events endpoint for streaming text-only responses."""

    def _generate():
        for chunk in agent.chat_stream(text or None):
            # SSE format: each event is "data: <payload>\n\n"
            yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def _async_generate():
        loop = asyncio.get_event_loop()
        gen = _generate()
        while True:
            try:
                event = await loop.run_in_executor(None, next, gen)
                yield event
            except StopIteration:
                break

    return StreamingResponse(
        _async_generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Cognitive Loop with Phase Progress (SSE) ─────────────────

@app.post("/api/chat/progress")
async def chat_progress(
    text: str = Form(default=""),
    images: list[UploadFile] = File(default=[]),
    audios: list[UploadFile] = File(default=[]),
    videos: list[UploadFile] = File(default=[]),
):
    """Full cognitive loop with SSE progress updates per phase.

    Sends phase progress events as they happen, then the final response.
    Event types: ``phase`` (progress updates) and ``result`` (final answer).
    """
    import queue

    image_paths = [_save_upload(f) for f in images if f.filename and f.size]
    audio_paths = [_save_upload(f) for f in audios if f.filename and f.size]
    video_paths = [_save_upload(f) for f in videos if f.filename and f.size]

    progress_queue: queue.Queue[dict] = queue.Queue()

    def _progress_cb(phase: str, status: str, detail: str = "") -> None:
        progress_queue.put({
            "type": "phase",
            "phase": phase,
            "status": status,
            "detail": detail,
        })

    async def _generate():
        loop = asyncio.get_event_loop()

        # Register progress callback
        agent.set_progress_callback(_progress_cb)

        # Run cognitive loop in a thread
        result_container: list[str | Exception] = []

        def _run():
            try:
                r = agent.chat(
                    text=text or None,
                    image_paths=image_paths or None,
                    audio_paths=audio_paths or None,
                    video_paths=video_paths or None,
                )
                result_container.append(r)
            except Exception as exc:
                result_container.append(exc)
            finally:
                progress_queue.put(None)  # sentinel

        thread = __import__("threading").Thread(target=_run, daemon=True)
        thread.start()

        # Stream progress events
        while True:
            try:
                item = await loop.run_in_executor(
                    None, lambda: progress_queue.get(timeout=0.5),
                )
            except Exception:
                if not thread.is_alive():
                    break
                continue

            if item is None:
                break

            payload = json.dumps(item, ensure_ascii=False)
            yield f"data: {payload}\n\n"

        # Clean up callback
        agent.set_progress_callback(None)

        # Send final result
        if result_container:
            val = result_container[0]
            if isinstance(val, Exception):
                result = {
                    "type": "result",
                    "status": "error",
                    "response": f"处理错误: {val}",
                }
            else:
                result = {
                    "type": "result",
                    "status": "ok",
                    "response": val,
                    "user_message_id": agent._last_user_msg_id,
                    "assistant_message_id": agent._last_asst_msg_id,
                    "session_id": agent._current_session_id,
                }
            yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

        # Clean up uploaded files
        for p in image_paths + audio_paths + video_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Session management ────────────────────────────────────────


@app.get("/api/sessions")
async def list_sessions():
    return JSONResponse(agent.list_sessions())


@app.post("/api/sessions")
async def create_session(request: Request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    title = body.get("title", "新对话")
    session = agent.new_session(title)
    return JSONResponse(session)


@app.post("/api/sessions/{session_id}/switch")
async def switch_session(session_id: str):
    result = agent.switch_session(session_id)
    if result is None:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    messages = agent.get_session_messages(session_id)
    return JSONResponse({"session": result, "messages": messages})


@app.put("/api/sessions/{session_id}")
async def rename_session(session_id: str, request: Request):
    body = await request.json()
    title = body.get("title", "")
    if not title:
        return JSONResponse({"error": "title required"}, status_code=400)
    ok = agent.rename_session(session_id, title)
    if not ok:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return JSONResponse({"status": "ok"})


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    ok = agent.delete_session(session_id)
    if not ok:
        return JSONResponse(
            {"error": "Cannot delete active session or session not found"},
            status_code=400,
        )
    return JSONResponse({"status": "ok"})


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    messages = agent.get_session_messages(session_id)
    return JSONResponse(messages)


# ── Regenerate / Edit ─────────────────────────────────────────


@app.post("/api/chat/regenerate")
async def regenerate():
    result = await asyncio.to_thread(agent.regenerate)
    if result is None:
        return JSONResponse({"error": "Nothing to regenerate"}, status_code=400)
    return JSONResponse({
        "response": result,
        "status": "ok",
        "assistant_message_id": agent._last_asst_msg_id,
        "session_id": agent._current_session_id,
    })


@app.post("/api/chat/edit")
async def edit_message(request: Request):
    body = await request.json()
    message_id = body.get("message_id")
    new_text = body.get("text", "")
    if not message_id or not new_text:
        return JSONResponse(
            {"error": "message_id and text required"}, status_code=400,
        )
    result = await asyncio.to_thread(
        agent.edit_and_regenerate, int(message_id), new_text,
    )
    if result is None:
        return JSONResponse({"error": "Message not found"}, status_code=404)
    return JSONResponse({
        "response": result,
        "status": "ok",
        "assistant_message_id": agent._last_asst_msg_id,
        "session_id": agent._current_session_id,
    })


# ── Status & Reset ────────────────────────────────────────────

@app.get("/api/status")
async def status():
    return JSONResponse(agent.status())


@app.post("/api/reset")
async def reset():
    agent.reset()
    return JSONResponse({"status": "ok", "message": "会话已重置"})


@app.post("/api/exit")
async def exit_app():
    """Gracefully shut down the server."""
    import signal

    async def _shutdown():
        await asyncio.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)

    asyncio.ensure_future(_shutdown())
    return JSONResponse({"status": "ok", "message": "服务器正在关闭"})


# ── Memory exploration ────────────────────────────────────────

@app.get("/api/memory/episodes")
async def memory_episodes():
    records = agent.memory.episodic.all()
    return JSONResponse([r.to_dict() for r in records])


@app.get("/api/memory/knowledge")
async def memory_knowledge():
    entries = agent.memory.semantic.all()
    return JSONResponse([e.to_dict() for e in entries])


@app.get("/api/memory/procedures")
async def memory_procedures():
    entries = agent.memory.procedural.all()
    return JSONResponse([e.to_dict() for e in entries])


@app.get("/api/memory/working")
async def memory_working():
    items = agent.memory.working.all_items()
    return JSONResponse([i.to_dict() for i in items])


@app.get("/api/memory/trend")
async def memory_trend():
    """Return improvement trend + per-episode score timeline."""
    trend = agent.memory.improvement_trend()
    episodes = agent.memory.episodic.all()
    timeline = [
        {
            "id": e.episode_id,
            "ts": e.timestamp,
            "score": e.outcome_score,
            "task": e.task_type.value if hasattr(e.task_type, "value") else str(e.task_type),
            "strategy": e.strategy_used,
            "tier": e.memory_tier.value if hasattr(e.memory_tier, "value") else str(e.memory_tier),
        }
        for e in episodes
    ]
    return JSONResponse({"trend": trend, "timeline": timeline})


# ── Tool management ───────────────────────────────────────────


@app.get("/api/tools")
async def list_tools():
    """Return all registered tools with their enabled/disabled status."""
    return JSONResponse(agent.list_tools())


@app.get("/api/tools/stats")
async def tool_stats():
    """Return per-tool invocation counts and timings."""
    return JSONResponse(agent.tool_stats())


@app.post("/api/tools/{tool_name}/enable")
async def enable_tool(tool_name: str):
    ok = agent.enable_tool(tool_name)
    if not ok:
        return JSONResponse({"error": f"Tool '{tool_name}' not found"}, status_code=404)
    return JSONResponse({"status": "ok", "tool": tool_name, "enabled": True})


@app.post("/api/tools/{tool_name}/disable")
async def disable_tool(tool_name: str):
    ok = agent.disable_tool(tool_name)
    if not ok:
        return JSONResponse({"error": f"Tool '{tool_name}' not found"}, status_code=404)
    return JSONResponse({"status": "ok", "tool": tool_name, "enabled": False})


# ── RAG knowledge base ────────────────────────────────────────


@app.get("/api/rag/documents")
async def rag_list_documents():
    """List all ingested documents with metadata."""
    return JSONResponse(agent.document_store.list_documents())


@app.post("/api/rag/documents")
async def rag_ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a document file for RAG retrieval.

    Accepts: .txt, .md, .csv, .json, .pdf
    The file is saved into the agent's data directory so that the
    DocumentStore path-traversal guard is satisfied.
    """
    try:
        # Save into the agent's data_dir (required by DocumentStore security)
        suffix = Path(file.filename or "file").suffix.lower()
        if suffix not in _RAG_ALLOWED_EXTENSIONS:
            return JSONResponse(
                {"error": f"File type '{suffix}' not allowed for RAG. "
                 f"Allowed: {', '.join(sorted(_RAG_ALLOWED_EXTENSIONS))}"},
                status_code=400,
            )
        dest = Path(agent.document_store._data_dir) / f"rag_{uuid.uuid4().hex}{suffix}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        doc_id = await asyncio.to_thread(agent.document_store.ingest, str(dest))
        return JSONResponse({"status": "ok", "doc_id": doc_id, "source": str(dest)})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.post("/api/rag/documents/text")
async def rag_ingest_text(request: Request):
    """Ingest raw text directly into the knowledge base.

    JSON body: ``{"text": "...", "source": "optional label"}``
    """
    body = await request.json()
    text = body.get("text", "")
    source = body.get("source", "inline")
    if not text.strip():
        return JSONResponse({"error": "text is required"}, status_code=400)
    try:
        doc_id = await asyncio.to_thread(
            agent.document_store.ingest_text, text, source,
        )
        return JSONResponse({"status": "ok", "doc_id": doc_id})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.delete("/api/rag/documents/{doc_id}")
async def rag_delete_document(doc_id: str):
    ok = agent.document_store.delete_document(doc_id)
    if not ok:
        return JSONResponse({"error": "Document not found"}, status_code=404)
    return JSONResponse({"status": "ok"})


@app.post("/api/rag/search")
async def rag_search(request: Request):
    """Search the knowledge base.

    JSON body: ``{"query": "...", "top_k": 5, "doc_id": null}``
    """
    body = await request.json()
    query = body.get("query", "")
    top_k = body.get("top_k", 5)
    doc_id = body.get("doc_id")
    if not query.strip():
        return JSONResponse({"error": "query is required"}, status_code=400)
    chunks = await asyncio.to_thread(
        agent.document_store.search, query, top_k=top_k, doc_id=doc_id,
    )
    return JSONResponse([c.to_dict() for c in chunks])


# ── Code sandbox ──────────────────────────────────────────────


@app.post("/api/sandbox/execute")
async def sandbox_execute(request: Request):
    """Execute Python code in the sandboxed subprocess.

    JSON body: ``{"code": "print(1+1)"}``
    """
    body = await request.json()
    code = body.get("code", "")
    if not code.strip():
        return JSONResponse({"error": "code is required"}, status_code=400)
    result = await asyncio.to_thread(agent.sandbox.execute, code)
    return JSONResponse(result.to_dict())


# ── Tracing ───────────────────────────────────────────────────


@app.get("/api/tracing")
async def tracing_export():
    """Export all recorded traces as JSON."""
    return JSONResponse(json.loads(agent.tracer.export_json()))


@app.get("/api/tracing/summary")
async def tracing_summary():
    """Human-readable summary of all traces."""
    return JSONResponse({"summary": agent.tracer.export_summary()})


@app.get("/api/tracing/stats")
async def tracing_stats():
    """Aggregate statistics across all traces."""
    return JSONResponse(agent.tracer.stats())


@app.delete("/api/tracing")
async def tracing_clear():
    agent.tracer.clear()
    return JSONResponse({"status": "ok", "message": "所有追踪记录已清除"})


# ── Planning & ReAct ──────────────────────────────────────────


@app.post("/api/plan")
async def plan_task(request: Request):
    """Decompose a task into a structured multi-step plan.

    JSON body: ``{"text": "...", "context": ""}``
    """
    body = await request.json()
    text = body.get("text", "")
    context = body.get("context", "")
    if not text.strip():
        return JSONResponse({"error": "text is required"}, status_code=400)
    plan = await asyncio.to_thread(agent.planner.decompose, text, context)
    return JSONResponse(plan.to_dict())


@app.post("/api/react")
async def react_run(request: Request):
    """Run a ReAct (Reason + Act) reasoning loop with tool use.

    JSON body: ``{"question": "...", "context": ""}``
    """
    body = await request.json()
    question = body.get("question", "")
    context = body.get("context", "")
    if not question.strip():
        return JSONResponse({"error": "question is required"}, status_code=400)
    trace = await asyncio.to_thread(
        agent.react_executor.run, question, context,
    )
    return JSONResponse(trace.to_dict())


# ── Multi-agent collaboration ─────────────────────────────────


@app.get("/api/team/members")
async def team_list_members():
    """List current team members."""
    members = agent.team.members
    return JSONResponse([
        {"member_id": m.member_id, "role": m.role, "specialty": m.specialty}
        for m in members.values()
    ])


@app.post("/api/team/members")
async def team_add_member(request: Request):
    """Add a team member.

    JSON body: ``{"member_id": "analyst", "role": "analysis", "specialty": "数据分析"}``
    """
    body = await request.json()
    member_id = body.get("member_id", "")
    role = body.get("role", "")
    specialty = body.get("specialty", "")
    if not member_id or not role:
        return JSONResponse(
            {"error": "member_id and role are required"}, status_code=400,
        )
    agent.team.add_member(TeamMember(
        member_id=member_id, role=role, specialty=specialty,
    ))
    return JSONResponse({"status": "ok", "member_id": member_id})


@app.delete("/api/team/members/{member_id}")
async def team_remove_member(member_id: str):
    ok = agent.team.remove_member(member_id)
    if not ok:
        return JSONResponse({"error": "Member not found"}, status_code=404)
    return JSONResponse({"status": "ok"})


@app.post("/api/team/collaborate")
async def team_collaborate(request: Request):
    """Run a multi-agent collaborative task.

    JSON body: ``{"task": "分析 Q3 销售数据并起草报告"}``
    """
    body = await request.json()
    task = body.get("task", "")
    if not task.strip():
        return JSONResponse({"error": "task is required"}, status_code=400)
    result = await asyncio.to_thread(agent.team.collaborate, task)
    return JSONResponse(result.to_dict())


# ── Agent Graph workflows ─────────────────────────────────────


@app.get("/api/graph/templates")
async def graph_list_templates():
    """List available graph workflow templates."""
    return JSONResponse(agent.list_graph_templates())


@app.post("/api/graph/run")
async def graph_run(request: Request):
    """Execute a graph workflow template.

    JSON body: ``{"template": "research", "inputs": {"query": "quantum"}}``
    """
    body = await request.json()
    template = body.get("template", "")
    inputs = body.get("inputs")
    if not template:
        return JSONResponse({"error": "template is required"}, status_code=400)
    result = await asyncio.to_thread(agent.run_graph, template, inputs)
    if "error" in result:
        return JSONResponse(result, status_code=404)
    return JSONResponse(result)


# ── MCP protocol ──────────────────────────────────────────────


@app.post("/api/mcp/connect")
async def mcp_connect(request: Request):
    """Connect to an MCP tool server and register its tools.

    JSON body: ``{"server_url": "http://localhost:8080/mcp"}``
    """
    body = await request.json()
    server_url = body.get("server_url", "")
    if not server_url:
        return JSONResponse({"error": "server_url is required"}, status_code=400)
    try:
        tools = await asyncio.to_thread(agent.connect_mcp, server_url)
        return JSONResponse({"status": "ok", "tools": tools})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)


@app.post("/api/mcp/disconnect")
async def mcp_disconnect(request: Request):
    """Disconnect from an MCP tool server.

    JSON body: ``{"server_url": "http://localhost:8080/mcp"}``
    """
    body = await request.json()
    server_url = body.get("server_url", "")
    if not server_url:
        return JSONResponse({"error": "server_url is required"}, status_code=400)
    try:
        await asyncio.to_thread(agent.disconnect_mcp, server_url)
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)


# ── Cache & Config ────────────────────────────────────────────


@app.get("/api/cache/stats")
async def cache_stats():
    """Response cache hit/miss statistics."""
    return JSONResponse(agent._response_cache.stats())


@app.delete("/api/cache")
async def cache_clear():
    agent._response_cache.clear()
    return JSONResponse({"status": "ok", "message": "响应缓存已清除"})


@app.get("/api/config/providers")
async def config_providers():
    """List available LLM provider presets."""
    return JSONResponse({"providers": PinocchioConfig.available_providers()})


# ── Static files ──────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
