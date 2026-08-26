"""
Unit tests for `evaluation/metrics.py`. No ML dependencies required —
run with:  cd backend && python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.dataset import EvalItem
from evaluation.metrics import (
    EvalReport,
    exact_match,
    f1_score,
    normalize_answer,
    score_item,
)


class _FakeAnswerResult:
    """Stand-in for query.AnswerResult, avoids importing the real pipeline."""
    def __init__(self, refused, answer=None, score=0.0, qa_score=0.0,
                 sources=None, retrieved_source_names=None):
        self.refused = refused
        self.answer = answer
        self.score = score
        self.qa_score = qa_score
        self.sources = sources or []
        self.retrieved_source_names = retrieved_source_names or []


# ── normalize_answer / exact_match / f1_score ───────────────────────────────

def test_normalize_answer_strips_articles_punct_case():
    assert normalize_answer("The Biriyani!") == "biriyani"
    assert normalize_answer("  a   Dr. Irvind Mehta ") == "dr irvind mehta"


def test_exact_match_ignores_case_and_articles():
    assert exact_match("2 cups", "2 cups")
    assert exact_match("a Rohan Sharma", "Rohan Sharma")
    assert not exact_match("500 grams", "2 cups")


def test_f1_score_partial_overlap():
    # shares "biriyani" only out of "spicy biriyani dish" vs "biriyani"
    score = f1_score("a spicy biriyani dish", "biriyani")
    assert 0.0 < score < 1.0


def test_f1_score_perfect_and_zero():
    assert f1_score("Rohan Sharma", "Rohan Sharma") == 1.0
    assert f1_score("Google", "Apple") == 0.0


def test_f1_score_empty_prediction():
    assert f1_score("", "biriyani") == 0.0


# ── score_item ───────────────────────────────────────────────────────────────

def test_score_item_correct_answer():
    item = EvalItem("q1", "What dish?", True, "cooking_recipe.wav", "biriyani", "recipe")
    result = _FakeAnswerResult(refused=False, answer="biriyani", score=0.55,
                                retrieved_source_names=["cooking_recipe.wav", "tech_news.wav"])
    ir = score_item(item, result, latency_ms=42.0, top_k=6)
    assert ir.outcome == "correct_answer"
    assert ir.exact_match is True
    assert ir.retrieval_hit is True


def test_score_item_false_refusal():
    item = EvalItem("q2", "What dish?", True, "cooking_recipe.wav", "biriyani", "recipe")
    result = _FakeAnswerResult(refused=True, score=0.1)
    ir = score_item(item, result, latency_ms=10.0, top_k=6)
    assert ir.outcome == "false_refusal"
    assert ir.exact_match is False


def test_score_item_hallucination():
    item = EvalItem("q3", "Capital of France?", False)
    result = _FakeAnswerResult(refused=False, answer="Paris", score=0.4)
    ir = score_item(item, result, latency_ms=10.0, top_k=6)
    assert ir.outcome == "hallucination"


def test_score_item_correct_refusal():
    item = EvalItem("q4", "Capital of France?", False)
    result = _FakeAnswerResult(refused=True, score=0.05)
    ir = score_item(item, result, latency_ms=10.0, top_k=6)
    assert ir.outcome == "correct_refusal"


# ── EvalReport.summarize ─────────────────────────────────────────────────────

def test_summarize_confusion_matrix_and_rates():
    items = [
        EvalItem("a1", "q", True, "f1.wav", "answer one"),
        EvalItem("a2", "q", True, "f1.wav", "answer two"),
        EvalItem("u1", "q", False),
        EvalItem("u2", "q", False),
    ]
    results = [
        _FakeAnswerResult(refused=False, answer="answer one", retrieved_source_names=["f1.wav"]),  # correct
        _FakeAnswerResult(refused=True),                                                            # false refusal
        _FakeAnswerResult(refused=True),                                                            # correct refusal
        _FakeAnswerResult(refused=False, answer="oops"),                                             # hallucination
    ]

    report = EvalReport()
    for item, result in zip(items, results):
        report.items.append(score_item(item, result, latency_ms=5.0, top_k=6))
    report.summarize()

    assert report.confusion.correct_answer == 1
    assert report.confusion.false_refusal == 1
    assert report.confusion.correct_refusal == 1
    assert report.confusion.hallucination == 1

    assert report.answerable_count == 2
    assert report.unanswerable_count == 2
    assert report.exact_match_rate == 0.5      # 1 of 2 answerable correct
    assert report.refusal_accuracy == 0.5      # 1 of 2 unanswerable correctly refused
    assert report.hallucination_rate == 0.5
    assert report.false_refusal_rate == 0.5
    assert 0.0 < report.overall_score < 1.0


def test_summarize_handles_empty_report():
    report = EvalReport().summarize()
    assert report.total == 0
    assert report.overall_score == 0.0


if __name__ == "__main__":
    # Allow `python tests/test_metrics.py` without pytest installed.
    import inspect
    mod = sys.modules[__name__]
    failures = 0
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  ok  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    sys.exit(1 if failures else 0)
