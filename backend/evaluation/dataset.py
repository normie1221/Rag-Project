"""
Evaluation dataset for the Audio RAG pipeline.

Every reference answer below was taken verbatim from the actual Whisper
transcript of the corresponding file in `test_audio/` (not hand-written from
the "intended" script), so scoring a prediction against it tells you how the
*real* pipeline performs end-to-end — transcription noise included.

Each `EvalItem` is either:
  - answerable: the question is grounded in one specific audio file. The
    harness checks both whether the model answered (vs. refused) and how
    close the extracted answer is to the reference span.
  - not answerable: the question is unrelated to every audio file in the
    eval set. The only correct behaviour is a refusal — the harness treats
    any answer here as a hallucination.

To add a case: transcribe the audio, quote the real transcript text as the
reference answer, and set `audio_file` to the exact filename in
`test_audio/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# All audio files loaded into the vector store before the eval questions run.
# Using every source at once (instead of one file per question) is what makes
# Retrieval Recall@K meaningful — the system has to pick the right document
# out of several, not just find *something* in a single-source index.
AUDIO_FILES: list[str] = [
    "about_me.wav",
    "cooking_recipe.wav",
    "tech_news.wav",
    "project_aurora_briefing.mp3",
]


@dataclass
class EvalItem:
    id: str
    question: str
    answerable: bool
    audio_file: Optional[str] = None       # expected source file, if answerable
    reference_answer: Optional[str] = None  # ground-truth span, if answerable
    category: str = "general"


EVAL_ITEMS: list[EvalItem] = [
    # ── about_me.wav ──────────────────────────────────────────────────────
    EvalItem("bio-1", "What is the person's name?", True,
             "about_me.wav", "Rohan Sharma", "bio"),
    EvalItem("bio-2", "In which city is he working as a software engineer?", True,
             "about_me.wav", "Bangalore", "bio"),
    EvalItem("bio-3", "Which university did he graduate from?", True,
             "about_me.wav", "IIT Delhi", "bio"),
    EvalItem("bio-4", "In what year did he complete his degree?", True,
             "about_me.wav", "2020", "bio"),
    EvalItem("bio-5", "What are his favorite programming languages?", True,
             "about_me.wav", "Python and Kotlin", "bio"),
    EvalItem("bio-6", "What does he enjoy doing in his free time?", True,
             "about_me.wav", "playing chess and reading science fiction books", "bio"),

    # ── cooking_recipe.wav ───────────────────────────────────────────────
    EvalItem("recipe-1", "What dish is being prepared in this recipe?", True,
             "cooking_recipe.wav", "biriyani", "recipe"),
    EvalItem("recipe-2", "How many cups of basmati rice are needed?", True,
             "cooking_recipe.wav", "2 cups", "recipe"),
    EvalItem("recipe-3", "How much chicken is required?", True,
             "cooking_recipe.wav", "500 grams", "recipe"),
    EvalItem("recipe-4", "How long should the chicken be marinated?", True,
             "cooking_recipe.wav", "30 minutes", "recipe"),
    EvalItem("recipe-5", "What should the biriyani be served with?", True,
             "cooking_recipe.wav", "raita", "recipe"),
    EvalItem("recipe-6", "How long should the biriyani cook on low flame?", True,
             "cooking_recipe.wav", "45 minutes", "recipe"),

    # ── tech_news.wav ─────────────────────────────────────────────────────
    EvalItem("news-1", "Which company released a new version of ChatGPT?", True,
             "tech_news.wav", "Open AI", "news"),
    EvalItem("news-2", "Which company announced an update to Gemini?", True,
             "tech_news.wav", "Google", "news"),
    EvalItem("news-3", "Which company is integrating AI into the next version of iOS?", True,
             "tech_news.wav", "Apple", "news"),
    EvalItem("news-4", "How much money has flowed into AI startups this year?", True,
             "tech_news.wav", "over $50 billion", "news"),
    EvalItem("news-5", "In which industries do experts predict AI will transform jobs?", True,
             "tech_news.wav", "healthcare, education, and finance", "news"),

    # ── project_aurora_briefing.mp3 (long, multi-topic document) ─────────
    EvalItem("aurora-1", "What is the total allocated budget for Project Aurora?", True,
             "project_aurora_briefing.mp3", "1,200 crore rupees", "aurora"),
    EvalItem("aurora-2", "Who is the project director of Project Aurora?", True,
             "project_aurora_briefing.mp3", "Dr. Irvind Mehta", "aurora"),
    EvalItem("aurora-3", "How many main divisions does the project have?", True,
             "project_aurora_briefing.mp3", "three", "aurora"),
    EvalItem("aurora-4", "What is the maximum human target detection range in clear weather?", True,
             "project_aurora_briefing.mp3", "5 km", "aurora"),
    EvalItem("aurora-5", "What is the power consumption of the full sensor suite at peak load?", True,
             "project_aurora_briefing.mp3", "240 watts", "aurora"),
    EvalItem("aurora-6", "How many engineers are on Suresh Ramakrishnan's hardware team?", True,
             "project_aurora_briefing.mp3", "42 engineers", "aurora"),
    EvalItem("aurora-7", "Where was the first field trial conducted?", True,
             "project_aurora_briefing.mp3", "Lungewala in the Rajasthan Desert", "aurora"),
    EvalItem("aurora-8", "What classification accuracy does the AI model achieve on the standard test dataset?", True,
             "project_aurora_briefing.mp3", "94.7%", "aurora"),
    EvalItem("aurora-9", "How many novel technologies has the project filed patents for?", True,
             "project_aurora_briefing.mp3", "7", "aurora"),
    EvalItem("aurora-10", "How many years does the post-deployment support plan cover?", True,
             "project_aurora_briefing.mp3", "five-year", "aurora"),

    # ── out-of-scope questions (correct behaviour: refuse) ────────────────
    EvalItem("refuse-1", "What is the capital city of France?", False, category="refusal"),
    EvalItem("refuse-2", "Who won the cricket world cup in 2023?", False, category="refusal"),
    EvalItem("refuse-3", "What is the boiling point of water in Celsius?", False, category="refusal"),
    EvalItem("refuse-4", "How do I reset a wifi router?", False, category="refusal"),
    EvalItem("refuse-5", "Explain Einstein's theory of relativity.", False, category="refusal"),
    EvalItem("refuse-6", "What is the current stock price of Apple?", False, category="refusal"),
]
