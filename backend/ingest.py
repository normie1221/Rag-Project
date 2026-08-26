"""
Shared ingestion pipeline: audio -> transcript segments -> overlapping chunks -> embeddings.

This is the single source of truth for turning an audio file into indexed
`Chunk` objects. Both the `/api/upload` endpoint and the evaluation harness
(`evaluation/runner.py`) call into this module so the two code paths can
never silently drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from vector_store.faiss_store import Chunk

CHUNK_WORDS = 120
OVERLAP_WORDS = 30


@dataclass
class IngestResult:
    chunks: list[Chunk]
    segments: list[dict]
    word_count: int
    duration: float


def build_chunks(
    segments: list[dict],
    source_id: str,
    filename: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[Chunk]:
    """Group transcript segments into overlapping chunks, preserving timestamps."""
    chunks: list[Chunk] = []
    buf: list[dict] = []
    buf_wc = 0

    def _flush():
        if not buf:
            return
        text = " ".join(s["text"].strip() for s in buf)
        chunks.append(Chunk(
            text=text,
            source_id=source_id,
            source_name=filename,
            timestamp_start=buf[0].get("start", 0.0),
            timestamp_end=buf[-1].get("end", 0.0),
        ))

    for seg in segments:
        wc = len(seg["text"].split())
        buf.append(seg)
        buf_wc += wc
        if buf_wc >= chunk_words:
            _flush()
            # keep overlap
            kept, keep_wc = [], 0
            for s in reversed(buf):
                sw = len(s["text"].split())
                if keep_wc + sw > overlap_words:
                    break
                kept.insert(0, s)
                keep_wc += sw
            buf = kept
            buf_wc = keep_wc

    _flush()

    # Edge case: single-segment audio with no natural break
    if not chunks and segments:
        all_text = " ".join(s["text"].strip() for s in segments)
        chunks.append(Chunk(
            text=all_text,
            source_id=source_id,
            source_name=filename,
            timestamp_start=segments[0].get("start", 0.0),
            timestamp_end=segments[-1].get("end", 0.0),
        ))

    return chunks


def ingest_audio(
    audio_path: str,
    filename: str,
    source_id: str,
    transcriber,
    embedder,
    vector_store,
    on_segment: Optional[Callable[[dict], None]] = None,
) -> IngestResult:
    """
    Full pipeline for one audio file: transcribe -> chunk -> embed -> index.

    `on_segment`, if given, is called once per transcript segment as it is
    produced (used by the upload endpoint to stream live progress over SSE).
    Returns an `IngestResult`; raises `ValueError` if no speech is detected.
    """
    segments: list[dict] = []
    for seg in transcriber.transcribe_iter(audio_path):
        segments.append(seg)
        if on_segment:
            on_segment(seg)

    if not segments:
        raise ValueError("No speech detected in this audio file.")

    chunks = build_chunks(segments, source_id, filename)
    texts = [c.text for c in chunks]
    vecs = embedder.embed(texts)
    for chunk, vec in zip(chunks, vecs):
        chunk.vector = vec
    vector_store.add_chunks(chunks)

    total_text = " ".join(s["text"] for s in segments)
    return IngestResult(
        chunks=chunks,
        segments=segments,
        word_count=len(total_text.split()),
        duration=segments[-1].get("end", 0.0) if segments else 0.0,
    )
