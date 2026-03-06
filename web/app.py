"""Pinocchio Web Demo — FastAPI backend.

Provides a REST API for the frontend to interact with the Pinocchio
multimodal agent.  Serves the single-page frontend from ``web/static/``.

Endpoints
---------
GET  /                  — Serve the frontend
POST /api/chat          — Send a message (with optional file uploads)
GET  /api/status        — Agent state summary
POST /api/reset         — Reset session (persistent memory kept)
GET  /api/memory/episodes   — All episodic memory records
GET  /api/memory/knowledge  — All semantic knowledge entries
GET  /api/memory/procedures — All procedural memory entries
GET  /api/memory/working    — Current working memory items
GET  /api/memory/trend      — Improvement trend over time

Usage
-----
    python -m web.app
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import PinocchioConfig
from pinocchio import Pinocchio

# ── Initialise ────────────────────────────────────────────────

app = FastAPI(title="Pinocchio Demo", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "file").suffix
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest)


# ── Chat ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


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
        return JSONResponse({"response": response, "status": "ok"})
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


# ── Status & Reset ────────────────────────────────────────────

@app.get("/api/status")
async def status():
    return JSONResponse(agent.status())


@app.post("/api/reset")
async def reset():
    agent.reset()
    return JSONResponse({"status": "ok", "message": "会话已重置"})


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


# ── Static files ──────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
