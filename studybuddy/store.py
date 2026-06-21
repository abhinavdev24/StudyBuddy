"""Session store for concepts and the latest quiz.

The LangGraph orchestrator (Phase 6) and the Gradio UI (Phase 7) populate this
so the Tutor's tools and the deterministic pipeline share one source of truth.

As of Phase 9 this is **SQLite-backed** (via `studybuddy.tracker`) so concepts
and quizzes survive a process restart. The function names/signatures are
unchanged, so existing callers don't need edits. Vector chunks still live in
Chroma; graded performance lives in the tracker's `concept_scores`.
"""
from __future__ import annotations

from typing import Optional

from studybuddy import tracker


def set_concepts(session_id: str, concepts: list[dict], doc_id: str = "doc1") -> None:
    """Persist concepts for a session (replaces the given document's concepts)."""
    tracker.save_concepts(session_id, concepts, doc_id=doc_id, replace=True)


def add_concepts(session_id: str, concepts: list[dict], doc_id: str) -> None:
    """Append concepts for an additional document (multi-doc; Phase 15)."""
    tracker.save_concepts(session_id, concepts, doc_id=doc_id, replace=False)


def get_concepts(session_id: str, doc_id: Optional[str] = None) -> list[dict]:
    return tracker.load_concepts(session_id, doc_id=doc_id)


def set_latest_quiz(session_id: str, quiz: list[dict], kind: str = "round") -> None:
    tracker.save_quiz(session_id, quiz, kind=kind)


def get_latest_quiz(session_id: str) -> list[dict]:
    return tracker.load_latest_quiz(session_id)
