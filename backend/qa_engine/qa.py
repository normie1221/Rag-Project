"""
Extractive QA using DeBERTa-v3 — zero hallucination.
Extracts exact spans from retrieved context; never generates new text.

Implemented directly on AutoModelForQuestionAnswering + AutoTokenizer rather
than the transformers `pipeline("question-answering", ...)` convenience
wrapper: transformers >= 5.0 removed the plain-text "question-answering"
task from the pipeline registry (only document-question-answering and
table-question-answering remain), so the wrapper raises
`Unknown task question-answering` on newer installs. The low-level model +
tokenizer API used here has been stable for years and works the same way
across both the 4.x and 5.x lines.

This also restores SQuAD2-style "no answer" detection: deepset/deberta-v3
-base-squad2 is trained to point at the [CLS] token (index 0) when the
given context doesn't actually answer the question. That comparison — best
real span vs. the [CLS] "null" baseline — is a second, independent
hallucination check, on top of the cosine-similarity gate in `query.py`.
Without it, the model always returns *some* span, confidently, even for
questions the context has no relationship to.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

from vector_store.faiss_store import Chunk

_MODEL_NAME = "deepset/deberta-v3-base-squad2"
_LOCAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "deberta-v3-base-squad2")

# Max context chars passed to the model (DeBERTa max_seq_len = 512 tokens ≈ 2000 chars)
_MAX_CONTEXT_CHARS = 2000
_MAX_SEQ_LEN = 512
_MAX_ANSWER_TOKENS = 40   # reject spans longer than this — extractive answers should be short
_MIN_SCORE = 0.15         # weak-but-nonzero confidence still gets the keyword fallback below this


@dataclass
class QAResult:
    text: str
    score: float


class QAEngine:
    def __init__(self):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self._torch = torch
        model_path = _LOCAL_PATH if os.path.isdir(_LOCAL_PATH) else _MODEL_NAME
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self._model.eval()
        print(f"[qa] Loaded model: {_MODEL_NAME}")

    def answer(self, question: str, chunks: list[Chunk]) -> QAResult:
        if not chunks:
            return QAResult("", 0.0)

        parts = []
        for c in chunks:
            ts = _fmt_ts(c.timestamp_start)
            parts.append(f"[{c.source_name} @ {ts}] {c.text}")
        context = " ".join(parts)[:_MAX_CONTEXT_CHARS]

        answer_text, score = self._extract_span(question, context)

        if not answer_text:
            # The model itself decided [CLS] beats every real span — i.e. it
            # believes this context does not answer the question. Trust that
            # signal; do NOT paper over it with a keyword-overlap guess.
            return QAResult("", 0.0)

        if score < _MIN_SCORE:
            # Model found a real span but isn't confident. Try the softer
            # keyword-overlap heuristic as a second opinion.
            fallback = _extract_best_sentence(question, chunks)
            return QAResult(fallback, score)

        return QAResult(answer_text, score)

    def _extract_span(self, question: str, context: str) -> tuple[str, float]:
        torch = self._torch

        inputs = self._tokenizer(
            question, context,
            return_tensors="pt",
            truncation="only_second",
            max_length=_MAX_SEQ_LEN,
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        sequence_ids = inputs.sequence_ids(0)

        with torch.no_grad():
            outputs = self._model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # SQuAD2 "no answer" baseline: index 0 is [CLS]. A squad2-tuned model
        # predicts [CLS]-[CLS] when it believes the context doesn't answer
        # the question — that raw logit sum is the null score to beat.
        null_score = (start_logits[0] + end_logits[0]).item()

        context_mask = torch.tensor([sid == 1 for sid in sequence_ids])
        num_context_tokens = int(context_mask.sum().item())
        if num_context_tokens == 0:
            return "", 0.0

        masked_start = start_logits.masked_fill(~context_mask, float("-inf"))
        masked_end = end_logits.masked_fill(~context_mask, float("-inf"))

        k = min(20, num_context_tokens)
        top_starts = torch.topk(masked_start, k=k).indices.tolist()
        top_ends = torch.topk(masked_end, k=k).indices.tolist()

        best_raw_score, best_start, best_end = float("-inf"), None, None
        for s in top_starts:
            for e in top_ends:
                if e < s or (e - s + 1) > _MAX_ANSWER_TOKENS:
                    continue
                raw_score = (start_logits[s] + end_logits[e]).item()
                if raw_score > best_raw_score:
                    best_raw_score, best_start, best_end = raw_score, s, e

        if best_start is None or best_raw_score <= null_score:
            return "", 0.0   # no valid span beats the model's own "no answer" baseline

        # Confidence for display / the _MIN_SCORE gate: how far the winning
        # span beats the null baseline, squashed to (0, 1).
        confidence = torch.sigmoid(torch.tensor(best_raw_score - null_score)).item()

        start_char = offset_mapping[best_start][0]
        end_char = offset_mapping[best_end][1]
        return context[start_char:end_char].strip(), confidence


def _extract_best_sentence(question: str, chunks: list[Chunk]) -> str:
    """Keyword-overlap fallback for real-but-low-confidence spans (not for
    cases where the model flagged no answer at all — see `answer()`)."""
    q_words = set(question.lower().split())
    best, best_score = "", 0
    for chunk in chunks:
        sentences = chunk.text.split(".")
        for sent in sentences:
            s_words = set(sent.lower().split())
            overlap = len(q_words & s_words)
            if overlap > best_score:
                best_score = overlap
                best = sent.strip()
    return best or chunks[0].text[:300]


def _fmt_ts(seconds: float) -> str:
    s = max(0.0, float(seconds or 0))
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"
