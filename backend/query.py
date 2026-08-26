"""
Shared retrieval + answer pipeline: question -> embed -> FAISS search ->
similarity gate -> extractive QA.

Single source of truth used by both `/api/chat` and the evaluation harness,
so evaluation results always reflect exactly what a real user would get.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SIMILARITY_THRESHOLD = 0.32   # cosine sim gate (BGE embeddings are L2-normalised)
SOFT_THRESHOLD_RATIO = 0.75   # chunks above (threshold * ratio) are kept as extra context
TOP_K = 6                     # chunks retrieved per query


@dataclass
class AnswerResult:
    refused: bool
    answer: Optional[str] = None
    reason: Optional[str] = None
    score: float = 0.0
    qa_score: float = 0.0
    sources: list[dict] = field(default_factory=list)
    retrieved_source_names: list[str] = field(default_factory=list)


def answer_question(
    question: str,
    embedder,
    vector_store,
    qa_engine,
    top_k: int = TOP_K,
    hard_threshold: float = SIMILARITY_THRESHOLD,
    soft_ratio: float = SOFT_THRESHOLD_RATIO,
) -> AnswerResult:
    """
    Run the full retrieve -> gate -> extract pipeline for one question.

    This never generates text: the answer is always an exact extractive span
    (or a refusal) grounded in the uploaded audio. Refusal happens whenever
    the best-matching chunk's cosine similarity falls below `hard_threshold`,
    which is the anti-hallucination boundary of the whole system.
    """
    question = question.strip()

    q_vec = embedder.embed_query(question)
    results = vector_store.search(q_vec, k=top_k)

    if not results:
        return AnswerResult(
            refused=True,
            reason="No content available in the vector store.",
        )

    best_chunk, best_score = results[0]
    retrieved_source_names = [c.source_name for c, _ in results]

    if best_score < hard_threshold:
        return AnswerResult(
            refused=True,
            reason=(
                "I cannot answer this because the uploaded audio files "
                "do not contain information about this topic."
            ),
            score=round(float(best_score), 3),
            retrieved_source_names=retrieved_source_names,
        )

    soft_thresh = hard_threshold * soft_ratio
    context_chunks = [c for c, s in results if s >= soft_thresh]

    qa_result = qa_engine.answer(question, context_chunks)

    if not qa_result.text:
        # Second, independent refusal gate: the similarity search found
        # plausible-looking chunks, but the QA model itself determined none
        # of them actually answers the question (see qa_engine/qa.py's
        # null-answer detection). Trust that over forcing a guess.
        return AnswerResult(
            refused=True,
            reason=(
                "I cannot answer this because the uploaded audio files "
                "do not contain information about this topic."
            ),
            score=round(float(best_score), 3),
            retrieved_source_names=retrieved_source_names,
        )

    seen, source_refs = set(), []
    for chunk in context_chunks[:3]:
        key = (chunk.source_name, int(chunk.timestamp_start))
        if key not in seen:
            seen.add(key)
            source_refs.append({
                "filename": chunk.source_name,
                "timestamp": _fmt_ts(chunk.timestamp_start),
                "excerpt": chunk.text[:250] + "…" if len(chunk.text) > 250 else chunk.text,
            })

    return AnswerResult(
        refused=False,
        answer=qa_result.text,
        score=round(float(best_score), 3),
        qa_score=round(float(qa_result.score), 3),
        sources=source_refs,
        retrieved_source_names=retrieved_source_names,
    )


def _fmt_ts(seconds: float) -> str:
    s = max(0.0, float(seconds or 0))
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"
