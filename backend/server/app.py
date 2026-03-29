"""FastAPI application — serves the QnA API and React SPA.

Endpoints:
  GET  /api/health           — health check + index status
  POST /api/query            — ask a question (hybrid retrieval + LLM)
  POST /api/ingest           — upload PDF and run ingestion pipeline
  GET  /api/ingest/{task_id} — check ingestion status
  GET  /*                    — serves React SPA (after build)

Run:
    uvicorn backend.server.app:app --reload --port 8000
"""

from __future__ import annotations

import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.logger import get_logger
from backend.server.models import (
    HealthResponse,
    IngestResultResponse,
    IngestStatusResponse,
    QueryRequest,
    QueryResponse,
)

logger = get_logger(__name__)

# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Tech Manual QA",
    description="Production-grade RAG pipeline — PDF ingestion + hybrid retrieval + LLM QnA",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────
_qna_pipeline = None
_executor = ThreadPoolExecutor(max_workers=2)
_ingest_tasks: dict[str, dict[str, Any]] = {}


def _get_qna():
    """Lazy-load QnA pipeline (heavy — loads FAISS + embedding model)."""
    global _qna_pipeline
    if _qna_pipeline is None:
        from backend.qna import QnAPipeline
        _qna_pipeline = QnAPipeline()
    return _qna_pipeline


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health():
    qna = _get_qna()
    return HealthResponse(
        status="ok",
        index_loaded=qna.is_ready,
        n_chunks=len(qna.retriever._chunks) if qna.retriever.is_loaded else 0,
        version="2.0.0",
    )


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest):
    logger.info("POST /api/query — question=%s", request.question[:80])

    qna = _get_qna()
    if not qna.is_ready:
        raise HTTPException(
            status_code=503,
            detail="No ingested content available. Upload a PDF first via /api/ingest.",
        )

    result = qna.ask(
        request.question,
        top_k=request.top_k,
        alpha=request.alpha,
        use_parent=request.use_parent,
    )

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        citations=result.get("citations", []),
        scores=result.get("scores", []),
    )


# ── Streaming Query (SSE) ────────────────────────────────────────────────────

@app.post("/api/query/stream")
def query_stream(request: QueryRequest):
    """Stream answer tokens via Server-Sent Events."""
    logger.info("POST /api/query/stream — question=%s", request.question[:80])

    qna = _get_qna()
    if not qna.is_ready:
        raise HTTPException(status_code=503, detail="No ingested content. Upload a PDF first.")

    # Retrieve first (non-streaming)
    results = qna.retriever.search(
        request.question,
        k=request.top_k,
        alpha=request.alpha,
        use_parent=request.use_parent,
    )

    def event_generator():
        for event in qna.generator.generate_stream(request.question, results):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── PDF Viewer ────────────────────────────────────────────────────────────────

@app.get("/api/pdf/{filename}")
def serve_pdf(filename: str):
    """Serve a PDF from the data directory for in-browser viewing."""
    # Prevent path traversal
    safe_name = Path(filename).name
    pdf_path = settings.DATA_DIR / safe_name
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=safe_name,
        headers={"Content-Disposition": f"inline; filename=\"{safe_name}\""},
    )


# ── Settings API ──────────────────────────────────────────────────────────────

@app.get("/api/settings")
def get_settings():
    """Return current config (no secrets)."""
    return {
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "llm_provider": settings.LLM_PROVIDER,
        "vision_provider": settings.VISION_PROVIDER,
        "groq_llm_model": settings.GROQ_LLM_MODEL,
        "openai_llm_model": settings.OPENAI_LLM_MODEL,
        "openai_embedding_model": settings.OPENAI_EMBEDDING_MODEL,
        "openai_embedding_dim": settings.OPENAI_EMBEDDING_DIM,
        "groq_vision_model": settings.GROQ_VISION_MODEL,
        "openai_vision_model": settings.OPENAI_VISION_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "active_llm_model": settings.active_llm_model,
        "active_embedding_model": settings.active_embedding_model,
        "active_vision_model": settings.active_vision_model,
        "retrieval_top_k": settings.RETRIEVAL_TOP_K,
        "hybrid_alpha": settings.HYBRID_ALPHA,
        "chunk_max_tokens": settings.CHUNK_MAX_TOKENS,
        "llm_temperature": settings.LLM_TEMPERATURE,
        "llm_max_tokens": settings.LLM_MAX_TOKENS,
        "enable_ocr": settings.ENABLE_OCR,
        "enable_vision": settings.ENABLE_VISION,
        "enable_parent_child": settings.ENABLE_PARENT_CHILD,
        "enable_quality_filter": settings.ENABLE_QUALITY_FILTER,
        "enable_dedup": settings.ENABLE_DEDUP,
        "has_groq_key": settings.has_groq_key,
        "has_openai_key": settings.has_openai_key,
        "n_chunks": len(_qna_pipeline.retriever._chunks) if (_qna_pipeline and _qna_pipeline.retriever.is_loaded) else 0,
        "index_loaded": _qna_pipeline.is_ready if _qna_pipeline else False,
    }


@app.put("/api/settings")
def update_settings(body: dict):
    """Update .env file and reload config. Returns updated settings."""
    env_path = settings.PROJECT_ROOT / ".env"
    env_lines = env_path.read_text().splitlines() if env_path.exists() else []

    # Map of frontend key → env var name
    key_map = {
        "embedding_provider": "EMBEDDING_PROVIDER",
        "llm_provider": "LLM_PROVIDER",
        "vision_provider": "VISION_PROVIDER",
        "groq_llm_model": "GROQ_LLM_MODEL",
        "openai_llm_model": "OPENAI_LLM_MODEL",
        "openai_embedding_model": "OPENAI_EMBEDDING_MODEL",
        "openai_embedding_dim": "OPENAI_EMBEDDING_DIM",
        "groq_vision_model": "GROQ_VISION_MODEL",
        "openai_vision_model": "OPENAI_VISION_MODEL",
        "embedding_model": "EMBEDDING_MODEL",
        "retrieval_top_k": "RETRIEVAL_TOP_K",
        "hybrid_alpha": "HYBRID_ALPHA",
        "llm_temperature": "LLM_TEMPERATURE",
        "llm_max_tokens": "LLM_MAX_TOKENS",
    }

    for frontend_key, value in body.items():
        env_var = key_map.get(frontend_key)
        if not env_var:
            continue
        # Update or append in env_lines
        found = False
        for i, line in enumerate(env_lines):
            if line.startswith(f"{env_var}="):
                env_lines[i] = f"{env_var}={value}"
                found = True
                break
        if not found:
            env_lines.append(f"{env_var}={value}")

    env_path.write_text("\n".join(env_lines) + "\n")
    logger.info("Updated .env with keys: %s", list(body.keys()))

    return {"status": "ok", "message": "Settings saved. Restart server to apply model changes."}


# ── Ingestion ─────────────────────────────────────────────────────────────────

@app.post("/api/ingest", status_code=202)
def ingest_pdf(file: UploadFile = File(...)):
    logger.info("POST /api/ingest — file=%s, type=%s", file.filename, file.content_type)

    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    # Save uploaded file
    file_name = f"ingest_{uuid.uuid4().hex[:8]}.pdf"
    pdf_path = settings.UPLOAD_DIR / file_name

    with pdf_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)
    file.file.close()

    task_id = uuid.uuid4().hex[:12]
    _ingest_tasks[task_id] = {
        "status": "queued", "result": None, "error": None,
        "current_step": 0, "total_steps": 5, "step_message": "Queued",
    }

    logger.info("Queued ingest task %s for %s", task_id, pdf_path)
    _executor.submit(_run_ingest, task_id, str(pdf_path))

    return JSONResponse({"task_id": task_id, "status": "queued"}, status_code=202)


@app.get("/api/ingest/{task_id}", response_model=IngestStatusResponse)
def ingest_status(task_id: str):
    if task_id not in _ingest_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = _ingest_tasks[task_id]
    return IngestStatusResponse(
        task_id=task_id,
        status=task["status"],
        error=task.get("error"),
        current_step=task.get("current_step", 0),
        total_steps=task.get("total_steps", 5),
        step_message=task.get("step_message", ""),
        step_detail=task.get("step_detail", ""),
    )


@app.get("/api/ingest/{task_id}/result", response_model=IngestResultResponse)
def ingest_result(task_id: str):
    if task_id not in _ingest_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = _ingest_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Task is {task['status']}")
    r = task["result"]
    return IngestResultResponse(
        task_id=task_id,
        status="completed",
        n_chunks=r.get("n_chunks", 0),
        n_parents=r.get("n_parents", 0),
        n_tables=len(r.get("tables", [])),
        n_images_described=r.get("stats", {}).images_described if hasattr(r.get("stats"), "images_described") else 0,
        embedding_dim=r.get("embedding_dim", 0),
    )


def _run_ingest(task_id: str, pdf_path: str) -> None:
    """Background worker for PDF ingestion."""
    global _qna_pipeline

    def _on_step(current: int, total: int, message: str):
        _ingest_tasks[task_id].update({
            "current_step": current,
            "total_steps": total,
            "step_message": message,
            "step_detail": "",
        })

    def _on_detail(message: str):
        _ingest_tasks[task_id]["step_detail"] = message

    _ingest_tasks[task_id]["status"] = "running"
    _ingest_tasks[task_id]["step_message"] = "Starting..."
    try:
        from backend.ingestion import IngestPipeline
        result = IngestPipeline(on_step=_on_step, on_detail=_on_detail).run(pdf_path)

        if "error" in result:
            _ingest_tasks[task_id].update({"status": "failed", "error": result["error"]})
            return

        _ingest_tasks[task_id].update({"status": "completed", "result": result})

        # Force QnA pipeline to reload artefacts on next query
        _qna_pipeline = None
        logger.info("Ingestion %s completed: %d chunks", task_id, result.get("n_chunks", 0))

    except Exception as exc:
        logger.exception("Ingestion task %s failed", task_id)
        _ingest_tasks[task_id].update({"status": "failed", "error": str(exc)})


# ── React SPA (served AFTER api routes) ───────────────────────────────────────

_frontend_build = settings.FRONTEND_BUILD_DIR
if _frontend_build.exists() and (_frontend_build / "index.html").exists():
    # Serve React build as static files — catches all non-API routes
    app.mount("/", StaticFiles(directory=str(_frontend_build), html=True), name="spa")
    logger.info("Serving React SPA from %s", _frontend_build)
else:
    @app.get("/")
    def root():
        return {
            "message": "Tech Manual QA API v2",
            "docs": "/api/docs",
            "note": "React SPA not built yet. Run: cd frontend && npm run build",
        }
