# Audio Intelligence & Evaluation System
### Local-Only Audio RAG with a Built-In Evaluation Harness

A fully offline system that transcribes audio files, indexes the content, answers
questions using **extractive** AI, and — the flagship piece — **automatically scores
its own answers** against a grounded reference dataset. Zero cloud calls, zero API
keys, zero hallucinated citations.

---

## Why extractive, not generative

The QA model never writes new text. It selects an exact span from the retrieved
transcript, or the system refuses. That's a deliberate trade against a generative
LLM answer layer: extractive spans can be traced back to an exact timestamp in an
exact file, and the accuracy of the answer is trivial to measure automatically — a
predicted string either overlaps the reference span or it doesn't. That's what the
evaluation harness below scores.

## Architecture

```
Audio File
   ↓
Whisper (base, CPU) — speech-to-text with timestamps
   ↓
Text Chunks (120 words, 30-word overlap)             ── ingest.py
   ↓
BGE-small-en-v1.5 — semantic embeddings
   ↓
FAISS (IndexFlatIP) — vector store
   ↓
User Question → BGE embed → FAISS search → top-K chunks
   ↓
Cosine similarity gate (threshold 0.32) → refuse if unrelated
   ↓                                                   ── query.py
DeBERTa-v3-base-squad2 — extractive QA (no generation, no hallucination)
   ↓
Answer + source file + timestamp
```

`ingest.py` and `query.py` are the single source of truth for "audio in →
indexed chunks" and "question in → answer out." Both `/api/upload` +
`/api/chat` (the live app) and the evaluation harness call into these same
two functions, so an evaluation score can never drift from what a real user
actually gets.

## Models Used

| Component | Model | Size | Task |
|---|---|---|---|
| Speech-to-Text | `faster-whisper/base` | 145 MB | Audio transcription |
| Embeddings | `BAAI/bge-small-en-v1.5` | 130 MB | Semantic retrieval |
| QA | `deepset/deberta-v3-base-squad2` | 900 MB | Extractive answering |

Total: ~1.2 GB (one-time download via `download_models.py`, not shipped in the repo).

---

## Setup

### Requirements
- Python 3.10+
- ffmpeg (for audio duration/decoding — `apt install ffmpeg` / `brew install ffmpeg`)
- 4 GB RAM minimum (8 GB recommended)
- ~2 GB disk space for models

### Step 1 — Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Download models (internet required, one time only)

```bash
cd ..   # back to project root
python download_models.py
```

Saves all models to `backend/models/` (~1.2 GB, gitignored). After this the
system runs fully offline.

### Step 3 — Run

```bash
cd backend
python main.py
```

Open `http://127.0.0.1:8000`. The frontend is a single static `index.html`
served directly by FastAPI — no separate frontend build step.

---

## Usage

1. **Upload Audio** tab → drag & drop MP3/WAV/M4A/FLAC files
2. Watch real-time transcription progress
3. Switch to **Ask Questions** tab, ask about the audio content
4. Answers include source file name and timestamp

### Example

Upload `project_meeting.wav`, ask *"What was decided about the project
timeline?"* → `The team decided to extend the deadline by two weeks`
(source: `project_meeting.wav @ 00:14:32`)

---

## Context Boundary (Anti-Hallucination)

If a question is unrelated to any loaded audio, the system responds:

> "I cannot answer this because the uploaded audio files do not contain
> information about this topic."

This is enforced by a cosine similarity gate (threshold: 0.32, in
`query.py`). The system never generates or infers beyond the audio content
— and the **Evaluate** tab exists specifically to prove that boundary holds,
rather than take it on faith.

---

## Evaluation Harness

The **Evaluate** tab (and `evaluate.py` on the command line) runs a fixed set of
33 questions against the 4 files in `test_audio/` through the exact same pipeline
a real user hits, then scores every answer automatically — no manual grading.

### Where the reference answers come from

Every reference answer in `backend/evaluation/dataset.py` was copied from the
**actual Whisper transcript** of the corresponding audio file — not the
"intended" script. Scoring against real transcript text (including its
occasional ASR quirks) means the harness measures the real pipeline,
transcription noise included, instead of an idealized one.

- `about_me.wav`, `cooking_recipe.wav`, `tech_news.wav` — short clips, 6/6/5
  grounded questions on names, quantities, and facts.
- `project_aurora_briefing.mp3` — a 22-minute synthetic briefing document,
  10 grounded questions on numbers, names, and specs spread across the
  transcript (this is what stresses chunking + retrieval over a long,
  multi-topic source).
- 6 out-of-scope questions (capital of France, cricket scores, etc.) with no
  answer in any loaded file — the only correct behaviour is a refusal.

All 4 files are loaded into **one shared index** before the questions run,
so the harness is also testing whether the system retrieves the *correct*
source file out of several candidates, not just whether it finds
*something* plausible.

### What gets measured

| Metric | What it means |
|---|---|
| **Exact Match** | Predicted answer, normalized, equals the reference span exactly |
| **Answer F1** | Token-overlap F1 between predicted and reference spans (SQuAD-style) |
| **Retrieval Recall@K** | Was the correct source file among the top-K retrieved chunks? |
| **Refusal Accuracy** | Of the out-of-scope questions, how many were correctly refused? |
| **Hallucination Rate** | Of the out-of-scope questions, how many got an answer anyway? |
| **False Refusal Rate** | Of the answerable questions, how many were wrongly refused? |
| **Overall Score** | Mean of Answer F1 and Refusal Accuracy — a system can't inflate this by refusing everything or never refusing |

Every score is also broken down into a **confusion matrix**:

|                          | Answered              | Refused               |
|--------------------------|------------------------|------------------------|
| **Expected: answerable** | ✅ correct answer       | ⚠️ false refusal       |
| **Expected: no info**    | 🚨 hallucination       | ✅ correct refusal     |

A false refusal is a usability problem. A hallucination is a trust problem —
the harness reports them separately because they're not equally bad.

### Running it

**In the app:** open the **Evaluate** tab → *Run Evaluation*. First run
transcribes and caches the 4 audio files (a couple of minutes, mostly spent
on the 22-minute briefing); subsequent runs reuse the cache and finish in
seconds.

**From the command line:**

```bash
cd backend
python evaluate.py
```

Prints the same report as the UI (metric bars, confusion matrix, per-question
pass/fail) and writes a timestamped JSON snapshot to
`backend/evaluation/reports/` — useful for diffing scores after changing the
similarity threshold, swapping the QA model, or editing the chunking
parameters. Exits non-zero if the hallucination rate is above 0% or the
overall score drops below 0.5, so it can be used as a CI gate.

### Extending the dataset

Add a case to `EVAL_ITEMS` in `backend/evaluation/dataset.py`: transcribe
the audio, copy the real transcript text as `reference_answer`, and point
`audio_file` at the exact filename in `test_audio/`. For a refusal case,
just set `answerable=False` and leave `audio_file`/`reference_answer` unset.

---

## Testing

```bash
cd backend
python -m pytest tests/ -v
```

- `tests/test_metrics.py` — unit tests for the scoring functions (Exact
  Match, F1, confusion-matrix bucketing). Pure Python, no model downloads
  needed.
- `tests/test_integration.py` — end-to-end tests using the real Whisper /
  BGE / DeBERTa models. Skipped automatically until you've run
  `python download_models.py`.

---

## Security

- No internet calls during operation
- No telemetry, no logging to external systems
- All files remain on the local machine
- Models loaded from `backend/models/` only
- No API keys required

---

## Project Structure

```
Adi-rag/
├── README.md
├── download_models.py            # one-time model download
├── .gitignore
├── test_audio/                   # sample clips used by the app and the eval harness
│   ├── about_me.wav
│   ├── cooking_recipe.wav
│   ├── tech_news.wav
│   └── project_aurora_briefing.mp3
└── backend/
    ├── main.py                   # FastAPI app: upload / chat / evaluate endpoints
    ├── ingest.py                 # shared: audio -> transcript -> chunks -> embeddings
    ├── query.py                  # shared: question -> retrieval -> gate -> extractive QA
    ├── index.html                # frontend (Upload / Ask / Evaluate tabs, no build step)
    ├── evaluate.py                # CLI entrypoint for the evaluation harness
    ├── requirements.txt
    ├── requirements-dev.txt      # pytest, for `tests/`
    ├── audio/processor.py        # audio duration/metadata (ffprobe, mutagen fallback)
    ├── transcriber/               # Whisper transcription
    ├── embeddings/                # BGE embedding wrapper
    ├── vector_store/              # FAISS store + Chunk dataclass
    ├── qa_engine/                 # DeBERTa extractive QA
    ├── evaluation/
    │   ├── dataset.py            # 33 grounded Q&A pairs (real transcript answers)
    │   ├── metrics.py            # EM / F1 / confusion matrix scoring
    │   ├── runner.py             # orchestrates ingest + query + score, with caching
    │   └── reports/              # JSON snapshots from `evaluate.py` (gitignored)
    ├── tests/
    │   ├── test_metrics.py       # pure-python scoring tests
    │   └── test_integration.py   # real-model end-to-end tests (skips if not downloaded)
    └── models/                   # downloaded weights (gitignored, run download_models.py)
```
# Rag-Project
# Rag-Project
