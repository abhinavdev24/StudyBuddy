"""Central configuration for StudyBuddy.

All environment variables are read **here and nowhere else** (ARCHITECTURE §4.3).
Downstream modules import `settings`, `ROUTING`, and the Chroma/embedding/db
constants from this module so there is a single source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load `.env` from the project root (searches parent dirs). Real values stay
# gitignored; `.env.example` documents the contract.
load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    return val if val not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # ── Gateway / provider credentials ──────────────────────────────────
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", "") or "")
    # None => the OpenAI SDK uses its default base url (api.openai.com).
    openai_base_url: str | None = field(default_factory=lambda: _env("OPENAI_BASE_URL"))

    # ── Per-task chat models (see ROUTING below) ────────────────────────
    extraction_model: str = field(default_factory=lambda: _env("EXTRACTION_MODEL", "gpt-4o-mini"))
    quiz_model: str = field(default_factory=lambda: _env("QUIZ_MODEL", "gpt-4o"))
    eval_model: str = field(default_factory=lambda: _env("EVAL_MODEL", "gpt-4o-mini"))
    adaptive_model: str = field(default_factory=lambda: _env("ADAPTIVE_MODEL", "gpt-4o-mini"))
    tutor_model: str = field(default_factory=lambda: _env("TUTOR_MODEL", "gpt-4o"))
    summary_model: str = field(default_factory=lambda: _env("SUMMARY_MODEL", "gpt-4o-mini"))
    # Defaults to the quiz model if FLASHCARDS_MODEL is unset.
    flashcards_model: str = field(
        default_factory=lambda: _env("FLASHCARDS_MODEL", _env("QUIZ_MODEL", "gpt-4o"))
    )

    def require_api_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.openai_api_key


settings = Settings()

# task → model name. The single routing map consumed by studybuddy/llm.py.
ROUTING: dict[str, str] = {
    "extraction": settings.extraction_model,
    "quiz": settings.quiz_model,
    "evaluation": settings.eval_model,
    "adaptation": settings.adaptive_model,
    "tutor": settings.tutor_model,
    "summary": settings.summary_model,
    "flashcards": settings.flashcards_model,
}

# ── Vector store / embeddings (local, no API) ───────────────────────────
CHROMA_DIR: str = _env("CHROMA_DIR", "data/chroma")
EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RETRIEVAL_K: int = int(_env("RETRIEVAL_K", "4"))

# ── Weak-spot tracker ───────────────────────────────────────────────────
STUDYBUDDY_DB: str = _env("STUDYBUDDY_DB", "data/studybuddy.db")
