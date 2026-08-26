"""
Run this ONCE on an internet-connected machine.
All models are saved to backend/models/ — after this the system is fully offline.

Usage:
    python download_models.py
"""
import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "backend", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 60)
print("  Audio RAG System — One-Time Model Downloader")
print("=" * 60)

# ── 1. Whisper (speech-to-text) ───────────────────────────────────────────────
print("\n[1/3] Downloading Whisper 'base' model (~145 MB)…")
try:
    from faster_whisper import WhisperModel
    WhisperModel("base", device="cpu", compute_type="int8", download_root=MODELS_DIR)
    print("  ✓ Whisper base")
except Exception as e:
    print(f"  ✗ Whisper download failed: {e}")
    sys.exit(1)

# ── 2. BGE embeddings (semantic search) ───────────────────────────────────────
print("\n[2/3] Downloading BGE-small-en-v1.5 embeddings (~130 MB)…")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    save_path = os.path.join(MODELS_DIR, "bge-small-en-v1.5")
    model.save(save_path)
    print(f"  ✓ BGE embeddings saved to {save_path}")
except Exception as e:
    print(f"  ✗ BGE download failed: {e}")
    sys.exit(1)

# ── 3. DeBERTa QA (extractive question answering) ────────────────────────────
print("\n[3/3] Downloading DeBERTa-v3-base-squad2 QA model (~900 MB)…")
try:
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    qa_model_name = "deepset/deberta-v3-base-squad2"
    save_path = os.path.join(MODELS_DIR, "deberta-v3-base-squad2")
    AutoTokenizer.from_pretrained(qa_model_name).save_pretrained(save_path)
    AutoModelForQuestionAnswering.from_pretrained(qa_model_name).save_pretrained(save_path)
    print(f"  ✓ DeBERTa QA model saved to {save_path}")
except Exception as e:
    print(f"  ✗ DeBERTa download failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("  All models downloaded successfully!")
print("  Total size: ~1.2 GB")
print("  The system is now fully offline-capable.")
print("=" * 60)
