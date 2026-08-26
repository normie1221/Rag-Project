"""
Audio Intelligence & Evaluation System — FastAPI backend
Fully offline, no external APIs, no cloud calls.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from audio.processor import get_audio_meta
from embeddings.embedder import BGEEmbedder
from evaluation.runner import run_evaluation
from ingest import ingest_audio
from qa_engine.qa import QAEngine
from query import TOP_K, answer_question
from transcriber.whisper_transcriber import WhisperTranscriber
from vector_store.faiss_store import FAISSStore

# ── Global model instances (loaded once at startup) ───────────────────────────
_transcriber: WhisperTranscriber | None = None
_embedder: BGEEmbedder | None = None
_vector_store: FAISSStore | None = None
_qa_engine: QAEngine | None = None
_ready = False

# Source registry  {source_id: metadata_dict}
_sources: dict[str, dict] = {}


def _preload_models():
    global _transcriber, _embedder, _vector_store, _qa_engine, _ready
    try:
        print("[startup] Loading Whisper model (base) …")
        _transcriber = WhisperTranscriber(model_size="base")

        print("[startup] Loading BGE embedding model …")
        _embedder = BGEEmbedder()

        print("[startup] Loading DeBERTa QA model …")
        _qa_engine = QAEngine()

        print("[startup] Initialising FAISS vector store …")
        _vector_store = FAISSStore(dim=_embedder.dim)

        _ready = True
        print("[startup] All models ready.")
    except Exception as exc:
        print(f"[startup] ERROR loading models: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_preload_models, daemon=True).start()
    yield


app = FastAPI(title="Audio Intelligence & Evaluation System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".webm", ".ogg"}


# ── /api/status ───────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    return {"ready": _ready, "sources": len(_sources)}


# ── /api/upload  (SSE streaming) ──────────────────────────────────────────────

@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not _ready:
        return JSONResponse({"error": "Models still loading — please wait."}, status_code=503)

    filename = file.filename or "audio.wav"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return JSONResponse({"error": f"Unsupported format '{ext}'. Allowed: mp3, wav, m4a, flac."}, status_code=400)

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    source_id = uuid.uuid4().hex[:8]

    async def stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def worker():
            try:
                meta = get_audio_meta(tmp_path, filename)
                loop.call_soon_threadsafe(queue.put_nowait,
                    ("meta", {"filename": filename, "duration": meta["duration"]}))

                loop.call_soon_threadsafe(queue.put_nowait, ("status", "Transcribing…"))

                def on_segment(seg: dict):
                    loop.call_soon_threadsafe(queue.put_nowait, ("seg", seg))

                try:
                    result = ingest_audio(
                        tmp_path, filename, source_id,
                        transcriber=_transcriber, embedder=_embedder, vector_store=_vector_store,
                        on_segment=on_segment,
                    )
                except ValueError as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
                    return

                loop.call_soon_threadsafe(queue.put_nowait, ("status", "Embedding transcript…"))

                _sources[source_id] = {
                    "id": source_id,
                    "filename": filename,
                    "duration": meta["duration"],
                    "upload_time": datetime.now().isoformat(),
                    "word_count": result.word_count,
                    "chunk_count": len(result.chunks),
                    "segment_count": len(result.segments),
                }

                loop.call_soon_threadsafe(queue.put_nowait,
                    ("done", {
                        "source_id": source_id,
                        "filename": filename,
                        "chunks": len(result.chunks),
                        "words": result.word_count,
                        "sources": list(_sources.values()),
                    }))

            except Exception as exc:
                import traceback; traceback.print_exc()
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        threading.Thread(target=worker, daemon=True).start()

        while True:
            kind, data = await queue.get()
            yield f"data: {json.dumps({'type': kind, 'data': data})}\n\n"
            if kind in ("done", "error"):
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /api/sources ──────────────────────────────────────────────────────────────

@app.get("/api/sources")
async def list_sources():
    return {"sources": list(_sources.values())}


@app.delete("/api/sources/{source_id}")
async def delete_source(source_id: str):
    if source_id not in _sources:
        raise HTTPException(404, "Source not found.")
    del _sources[source_id]
    if _vector_store:
        _vector_store.remove_source(source_id)
    return {"ok": True, "sources": list(_sources.values())}


# ── /api/chat ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        return JSONResponse({"error": "Empty question."}, status_code=400)

    if not _sources:
        return JSONResponse({
            "refused": True,
            "answer": None,
            "reason": "No audio files loaded. Please upload audio files first.",
            "sources": [],
        })

    if not _ready:
        return JSONResponse({"error": "Models still loading…"}, status_code=503)

    loop = asyncio.get_event_loop()

    def _run_query():
        result = answer_question(question, _embedder, _vector_store, _qa_engine, top_k=TOP_K)
        return {
            "refused": result.refused,
            "answer": result.answer,
            "reason": result.reason,
            "score": result.score,
            "qa_score": result.qa_score,
            "sources": result.sources,
        }

    return await loop.run_in_executor(None, _run_query)


# ── /api/reset ────────────────────────────────────────────────────────────────

@app.post("/api/reset")
async def reset():
    _sources.clear()
    if _vector_store:
        _vector_store.clear()
    return {"ok": True}


# ── /api/evaluate  (SSE streaming) ─────────────────────────────────────────────
# Runs the evaluation harness (evaluation/) against test_audio/ using the same
# models already loaded for the live app. Streams progress the same way
# /api/upload does; the final "done" event carries the full scored report.

@app.post("/api/evaluate")
async def evaluate():
    if not _ready:
        return JSONResponse({"error": "Models still loading — please wait."}, status_code=503)

    async def stream():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def on_progress(msg: str):
            loop.call_soon_threadsafe(queue.put_nowait, ("status", msg))

        def worker():
            try:
                report = run_evaluation(
                    _transcriber, _embedder, _qa_engine,
                    top_k=TOP_K, on_progress=on_progress,
                )
                loop.call_soon_threadsafe(queue.put_nowait, ("done", report.as_dict()))
            except Exception as exc:
                import traceback; traceback.print_exc()
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

        while True:
            kind, data = await queue.get()
            yield f"data: {json.dumps({'type': kind, 'data': data})}\n\n"
            if kind in ("done", "error"):
                break

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Serve frontend ────────────────────────────────────────────────────────────

_HTML = os.path.join(os.path.dirname(__file__), "index.html")


@app.get("/", response_class=HTMLResponse)
async def frontend():
    with open(_HTML, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
