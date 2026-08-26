"""
End-to-end integration test using the *real* Whisper, BGE, and DeBERTa
models (not fakes). Requires `python download_models.py` to have been run
first, so this is skipped automatically in environments where the models
aren't present (e.g. a fresh checkout, or CI without the ~1.2 GB download) —
it will not fail your test run, it will just report as skipped.

Run explicitly with:
    cd backend
    python -m pytest tests/test_integration.py -v -s
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _models_available() -> bool:
    return (
        (MODELS_DIR / "bge-small-en-v1.5").is_dir()
        and (MODELS_DIR / "deberta-v3-base-squad2").is_dir()
        and any(MODELS_DIR.glob("models--Systran--faster-whisper-base"))
    )


pytestmark = pytest.mark.skipif(
    not _models_available(),
    reason="Models not downloaded — run `python download_models.py` first.",
)


@pytest.fixture(scope="module")
def pipeline():
    from embeddings.embedder import BGEEmbedder
    from qa_engine.qa import QAEngine
    from transcriber.whisper_transcriber import WhisperTranscriber
    return {
        "transcriber": WhisperTranscriber(model_size="base"),
        "embedder": BGEEmbedder(),
        "qa_engine": QAEngine(),
    }


def test_full_ingest_and_query(pipeline):
    """Upload one real audio file and ask a question grounded in it."""
    from ingest import ingest_audio
    from query import answer_question
    from vector_store.faiss_store import FAISSStore

    audio_path = Path(__file__).resolve().parent.parent.parent / "test_audio" / "cooking_recipe.wav"
    store = FAISSStore(dim=pipeline["embedder"].dim)

    result = ingest_audio(
        str(audio_path), "cooking_recipe.wav", "test-src",
        transcriber=pipeline["transcriber"], embedder=pipeline["embedder"], vector_store=store,
    )
    assert result.chunks
    assert result.word_count > 0

    answer = answer_question(
        "What dish is being prepared?",
        pipeline["embedder"], store, pipeline["qa_engine"],
    )
    assert not answer.refused
    assert "biriyani" in (answer.answer or "").lower()


def test_out_of_scope_question_is_refused(pipeline):
    """A question unrelated to the loaded audio must be refused, not guessed at."""
    from ingest import ingest_audio
    from query import answer_question
    from vector_store.faiss_store import FAISSStore

    audio_path = Path(__file__).resolve().parent.parent.parent / "test_audio" / "cooking_recipe.wav"
    store = FAISSStore(dim=pipeline["embedder"].dim)
    ingest_audio(
        str(audio_path), "cooking_recipe.wav", "test-src",
        transcriber=pipeline["transcriber"], embedder=pipeline["embedder"], vector_store=store,
    )

    answer = answer_question(
        "What is the capital of France?",
        pipeline["embedder"], store, pipeline["qa_engine"],
    )
    assert answer.refused


def test_evaluation_harness_runs_end_to_end(pipeline):
    """Smoke-test the full evaluation harness with the real models."""
    from evaluation.runner import run_evaluation

    report = run_evaluation(
        pipeline["transcriber"], pipeline["embedder"], pipeline["qa_engine"],
        use_cache=True,
    )
    assert report.total == report.answerable_count + report.unanswerable_count
    assert 0.0 <= report.overall_score <= 1.0
