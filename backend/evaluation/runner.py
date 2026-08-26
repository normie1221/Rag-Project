"""
Evaluation harness orchestration.

Ingests every file in `dataset.AUDIO_FILES` into a fresh, isolated vector
store, then runs every question in `dataset.EVAL_ITEMS` through the exact
same `query.answer_question` pipeline that powers `/api/chat`. The result is
a summarized `EvalReport` (see `evaluation.metrics`) — the same object is
used by both the CLI (`evaluate.py`) and the `/api/evaluate` endpoint, so a
score can't drift depending on how it was triggered.

Transcripts are cached to disk (`evaluation/.cache/`) keyed by file mtime,
since re-transcribing the ~22-minute Aurora briefing takes a few minutes on
CPU. Delete that folder (or pass `use_cache=False`) to force a fresh run.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from ingest import ingest_audio
from query import TOP_K, answer_question
from vector_store.faiss_store import FAISSStore

from evaluation.dataset import AUDIO_FILES, EVAL_ITEMS
from evaluation.metrics import EvalReport, score_item

TEST_AUDIO_DIR = Path(__file__).resolve().parent.parent.parent / "test_audio"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


class _CachedTranscriber:
    """Transcribes once per audio file, then serves cached segments."""

    def __init__(self, transcriber, cache_dir: Path = CACHE_DIR):
        self._inner = transcriber
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def transcribe_iter(self, audio_path: str):
        cache_file = self._cache_dir / (Path(audio_path).stem + ".json")
        src_mtime = os.path.getmtime(audio_path)

        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                if cached.get("source_mtime") == src_mtime:
                    yield from cached["segments"]
                    return
            except (json.JSONDecodeError, OSError):
                pass  # fall through and re-transcribe on a corrupt cache

        segments = list(self._inner.transcribe_iter(audio_path))
        cache_file.write_text(json.dumps({"source_mtime": src_mtime, "segments": segments}))
        yield from segments


def run_evaluation(
    transcriber,
    embedder,
    qa_engine,
    top_k: int = TOP_K,
    use_cache: bool = True,
    on_progress: Optional[Callable[[str], None]] = None,
) -> EvalReport:
    def progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    store = FAISSStore(dim=embedder.dim)
    active_transcriber = _CachedTranscriber(transcriber) if use_cache else transcriber

    for i, filename in enumerate(AUDIO_FILES, 1):
        path = TEST_AUDIO_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Evaluation audio file not found: {path}")
        progress(f"Ingesting {filename} ({i}/{len(AUDIO_FILES)})…")
        ingest_audio(
            str(path), filename, source_id=filename,
            transcriber=active_transcriber, embedder=embedder, vector_store=store,
        )

    report = EvalReport()
    for i, item in enumerate(EVAL_ITEMS, 1):
        progress(f"Q{i}/{len(EVAL_ITEMS)}: {item.question}")
        t0 = time.perf_counter()
        result = answer_question(item.question, embedder, store, qa_engine, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        report.items.append(score_item(item, result, latency_ms, top_k))

    progress("Scoring…")
    return report.summarize()
