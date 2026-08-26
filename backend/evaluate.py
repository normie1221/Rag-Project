"""
Command-line evaluation runner.

    cd backend
    python evaluate.py

Loads the three models, ingests every file in `test_audio/`, runs the full
question set in `evaluation/dataset.py` through the real retrieval + QA
pipeline, and prints a scored report — the same "evaluation matrix" shown in
the app's Evaluate tab, without needing the server or a browser.

A JSON report is also written to `evaluation/reports/` for diffing between
runs (e.g. after swapping the QA model or changing the similarity threshold).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

from embeddings.embedder import BGEEmbedder
from qa_engine.qa import QAEngine
from transcriber.whisper_transcriber import WhisperTranscriber

from evaluation.metrics import EvalReport
from evaluation.runner import run_evaluation

REPORTS_DIR = Path(__file__).resolve().parent / "evaluation" / "reports"


def _bar(value: float, width: int = 24) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_report(report: EvalReport) -> None:
    d = report.as_dict()
    cm = d["confusion_matrix"]

    print("\n" + "═" * 64)
    print("  EVALUATION REPORT")
    print("═" * 64)
    print(f"  {d['total']} questions  ·  {d['answerable_count']} answerable  ·  "
          f"{d['unanswerable_count']} out-of-scope\n")

    rows = [
        ("Exact Match",        d["exact_match_rate"]),
        ("Answer F1",          d["f1_avg"]),
        ("Retrieval Recall@K", d["retrieval_recall_at_k"]),
        ("Refusal Accuracy",   d["refusal_accuracy"]),
        ("Overall Score",      d["overall_score"]),
    ]
    for label, val in rows:
        print(f"  {label:<20} {_bar(val)}  {val*100:5.1f}%")

    print(f"\n  Hallucination rate   {d['hallucination_rate']*100:5.1f}%   "
          f"(answered when it should have refused)")
    print(f"  False refusal rate   {d['false_refusal_rate']*100:5.1f}%   "
          f"(refused when it had the answer)")
    print(f"  Avg latency          {d['avg_latency_ms']:.0f} ms / question")

    print("\n  Confusion matrix (expected vs. actual behaviour)")
    print("  ┌─────────────────────┬──────────────┬──────────────┐")
    print("  │                     │   Answered   │   Refused    │")
    print("  ├─────────────────────┼──────────────┼──────────────┤")
    print(f"  │ Expected: answerable│ {cm['correct_answer']:^12} │ {cm['false_refusal']:^12} │")
    print(f"  │ Expected: no info   │ {cm['hallucination']:^12} │ {cm['correct_refusal']:^12} │")
    print("  └─────────────────────┴──────────────┴──────────────┘")

    print("\n  Per-question results")
    print("  " + "-" * 60)
    for item in d["items"]:
        if item["outcome"] == "correct_answer":
            mark = "✔" if item["exact_match"] else "~"
        elif item["outcome"] == "correct_refusal":
            mark = "✔"
        else:
            mark = "✘"
        detail = (f"f1={item['f1']:.2f}" if item["answerable"] and not item["refused"]
                   else item["outcome"])
        print(f"  [{mark}] {item['id']:<10} {item['question'][:50]:<50} ({detail})")

    print("\n" + "═" * 64 + "\n")


def save_report(report: EvalReport) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = REPORTS_DIR / f"eval-{stamp}.json"
    path.write_text(json.dumps(report.as_dict(), indent=2))
    return path


def main() -> int:
    print("Loading models (Whisper, BGE, DeBERTa)…")
    t0 = time.time()
    transcriber = WhisperTranscriber(model_size="base")
    embedder = BGEEmbedder()
    qa_engine = QAEngine()
    print(f"Models ready in {time.time() - t0:.1f}s\n")

    report = run_evaluation(
        transcriber, embedder, qa_engine,
        on_progress=lambda msg: print(f"  … {msg}"),
    )

    print_report(report)
    path = save_report(report)
    print(f"Report saved to {path}")

    # Non-zero exit if the system is hallucinating or missing most answers —
    # useful as a CI gate after changing the threshold or swapping a model.
    return 0 if report.hallucination_rate == 0 and report.overall_score >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
