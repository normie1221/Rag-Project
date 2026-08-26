"""
Scoring functions for the evaluation harness. Pure Python, no ML
dependencies, so this module can be unit-tested in isolation (see
`tests/test_metrics.py`).
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field


# ── Answer-quality metrics (SQuAD-style) ────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def f1_score(prediction: str, reference: str) -> float:
    """Token-overlap F1 between predicted and reference answer spans."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ── Per-item and aggregate result containers ────────────────────────────────

@dataclass
class ItemResult:
    id: str
    question: str
    category: str
    answerable: bool
    expected_source: str | None
    reference_answer: str | None
    refused: bool
    predicted_answer: str | None
    similarity_score: float
    qa_score: float
    retrieved_sources: list[str]
    latency_ms: float
    exact_match: bool = False
    f1: float = 0.0
    retrieval_hit: bool = False
    outcome: str = ""   # one of: correct_answer, false_refusal, hallucination, correct_refusal


def score_item(item, result, latency_ms: float, top_k: int) -> ItemResult:
    """Combine one EvalItem with its AnswerResult into a scored ItemResult."""
    ir = ItemResult(
        id=item.id,
        question=item.question,
        category=item.category,
        answerable=item.answerable,
        expected_source=item.audio_file,
        reference_answer=item.reference_answer,
        refused=result.refused,
        predicted_answer=result.answer,
        similarity_score=result.score,
        qa_score=result.qa_score,
        retrieved_sources=result.retrieved_source_names[:top_k],
        latency_ms=latency_ms,
    )

    if item.answerable:
        ir.retrieval_hit = item.audio_file in ir.retrieved_sources
        if result.refused:
            ir.outcome = "false_refusal"          # bad: should have answered
        else:
            ir.exact_match = exact_match(result.answer or "", item.reference_answer or "")
            ir.f1 = f1_score(result.answer or "", item.reference_answer or "")
            ir.outcome = "correct_answer"
    else:
        ir.outcome = "correct_refusal" if result.refused else "hallucination"

    return ir


@dataclass
class ConfusionMatrix:
    """2x2 matrix over (expected answerable?) x (system answered?)."""
    correct_answer: int = 0    # answerable & answered   -> TP
    false_refusal: int = 0     # answerable & refused    -> FN
    hallucination: int = 0     # unanswerable & answered -> FP
    correct_refusal: int = 0   # unanswerable & refused  -> TN

    def as_dict(self) -> dict:
        return {
            "correct_answer": self.correct_answer,
            "false_refusal": self.false_refusal,
            "hallucination": self.hallucination,
            "correct_refusal": self.correct_refusal,
        }


@dataclass
class EvalReport:
    items: list[ItemResult] = field(default_factory=list)
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)

    # aggregate metrics, filled in by `summarize()`
    total: int = 0
    answerable_count: int = 0
    unanswerable_count: int = 0
    exact_match_rate: float = 0.0
    f1_avg: float = 0.0
    retrieval_recall_at_k: float = 0.0
    refusal_accuracy: float = 0.0      # correct_refusal / unanswerable_count
    hallucination_rate: float = 0.0    # hallucination  / unanswerable_count
    false_refusal_rate: float = 0.0    # false_refusal  / answerable_count
    avg_latency_ms: float = 0.0
    overall_score: float = 0.0

    def summarize(self) -> "EvalReport":
        self.total = len(self.items)
        answerable = [i for i in self.items if i.answerable]
        unanswerable = [i for i in self.items if not i.answerable]
        self.answerable_count = len(answerable)
        self.unanswerable_count = len(unanswerable)

        for i in self.items:
            setattr(self.confusion, i.outcome, getattr(self.confusion, i.outcome) + 1)

        if answerable:
            # Items the system refused count as EM=0 / F1=0 — a false refusal
            # is a failure on this question, not a question we skip scoring.
            self.exact_match_rate = sum(i.exact_match for i in answerable) / len(answerable)
            self.f1_avg = sum(i.f1 for i in answerable) / len(answerable)
            self.retrieval_recall_at_k = sum(i.retrieval_hit for i in answerable) / len(answerable)
            self.false_refusal_rate = self.confusion.false_refusal / len(answerable)

        if unanswerable:
            self.refusal_accuracy = self.confusion.correct_refusal / len(unanswerable)
            self.hallucination_rate = self.confusion.hallucination / len(unanswerable)

        if self.items:
            self.avg_latency_ms = sum(i.latency_ms for i in self.items) / len(self.items)

        # Single headline number: balanced mean of answer quality (F1) and
        # refusal correctness, so a system can't inflate its score by
        # refusing everything (or by never refusing).
        parts = [p for p in (self.f1_avg if answerable else None,
                              self.refusal_accuracy if unanswerable else None) if p is not None]
        self.overall_score = sum(parts) / len(parts) if parts else 0.0
        return self

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "answerable_count": self.answerable_count,
            "unanswerable_count": self.unanswerable_count,
            "exact_match_rate": round(self.exact_match_rate, 4),
            "f1_avg": round(self.f1_avg, 4),
            "retrieval_recall_at_k": round(self.retrieval_recall_at_k, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "false_refusal_rate": round(self.false_refusal_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "overall_score": round(self.overall_score, 4),
            "confusion_matrix": self.confusion.as_dict(),
            "items": [
                {
                    "id": i.id,
                    "question": i.question,
                    "category": i.category,
                    "answerable": i.answerable,
                    "expected_source": i.expected_source,
                    "reference_answer": i.reference_answer,
                    "refused": i.refused,
                    "predicted_answer": i.predicted_answer,
                    "similarity_score": i.similarity_score,
                    "qa_score": i.qa_score,
                    "retrieved_sources": i.retrieved_sources,
                    "latency_ms": round(i.latency_ms, 1),
                    "exact_match": i.exact_match,
                    "f1": round(i.f1, 4),
                    "retrieval_hit": i.retrieval_hit,
                    "outcome": i.outcome,
                }
                for i in self.items
            ],
        }
