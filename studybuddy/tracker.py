"""Weak-Spot Tracker (ARCHITECTURE §3.5 / §5). No LLM.

SQLite-backed per-concept performance store. Records every answer and exposes
mastery scores and weak concepts to drive the adaptive loop and dashboard.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

from studybuddy import scheduler

from studybuddy.config import STUDYBUDDY_DB


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    os.makedirs(os.path.dirname(STUDYBUDDY_DB) or ".", exist_ok=True)
    conn = sqlite3.connect(STUDYBUDDY_DB)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist (idempotent)."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id     TEXT PRIMARY KEY,
                created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
                material_name  TEXT
            );

            CREATE TABLE IF NOT EXISTS concept_scores (
                session_id   TEXT,
                concept_id   TEXT,
                concept_name TEXT,
                attempts     INTEGER DEFAULT 0,
                correct      INTEGER DEFAULT 0,
                streak       INTEGER DEFAULT 0,
                last_seen    TEXT,
                difficulty   TEXT DEFAULT 'recall',
                PRIMARY KEY (session_id, concept_id)
            );

            CREATE TABLE IF NOT EXISTS quiz_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT,
                question_id   TEXT,
                concept_id    TEXT,
                question_type TEXT,
                difficulty    TEXT,
                user_answer   TEXT,
                correct       INTEGER,
                answered_at   TEXT DEFAULT CURRENT_TIMESTAMP
            );

            -- ── Phase 9: durable session artifacts ──────────────────────
            CREATE TABLE IF NOT EXISTS documents (
                session_id TEXT,
                doc_id     TEXT,
                name       TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, doc_id)
            );

            -- Full concept dict stored as JSON (lossless); indexed columns for querying.
            CREATE TABLE IF NOT EXISTS concepts (
                session_id   TEXT,
                doc_id       TEXT,
                concept_id   TEXT,
                concept_json TEXT,
                PRIMARY KEY (session_id, doc_id, concept_id)
            );

            CREATE TABLE IF NOT EXISTS quizzes (
                quiz_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                kind       TEXT DEFAULT 'round',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS questions (
                quiz_id       INTEGER,
                question_id   TEXT,
                concept_id    TEXT,
                position      INTEGER,
                question_json TEXT
            );
            """
        )
        _ensure_columns(conn)


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add Phase-12 columns to pre-existing tables (idempotent migration)."""
    def cols(table: str) -> set:
        return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    if "confidence" not in cols("quiz_history"):
        conn.execute("ALTER TABLE quiz_history ADD COLUMN confidence TEXT DEFAULT 'med'")
    cs = cols("concept_scores")
    if "mastery_points" not in cs:
        conn.execute("ALTER TABLE concept_scores ADD COLUMN mastery_points REAL DEFAULT 0")
    # Phase 13: spaced-repetition schedule columns.
    if "reps" not in cs:
        conn.execute("ALTER TABLE concept_scores ADD COLUMN reps INTEGER DEFAULT 0")
    if "interval" not in cs:
        conn.execute("ALTER TABLE concept_scores ADD COLUMN interval INTEGER DEFAULT 0")
    if "ease" not in cs:
        conn.execute("ALTER TABLE concept_scores ADD COLUMN ease REAL DEFAULT 2.5")
    if "due_at" not in cs:
        conn.execute("ALTER TABLE concept_scores ADD COLUMN due_at TEXT")


# Mastery credit for a correct answer, weighted by stated confidence
# (lucky-guess guard: low-confidence-correct earns less). Wrong answers earn 0.
_MASTERY_CREDIT = {"high": 1.0, "med": 0.85, "low": 0.6}


def record_answer(
    session_id: str,
    concept_id: str,
    concept_name: str,
    question_type: str,
    difficulty: str,
    user_answer: str,
    correct: bool,
    question_id: Optional[str] = None,
    confidence: str = "med",
) -> None:
    """Persist one graded answer and update per-concept aggregates.

    `confidence` (low|med|high) weights mastery credit for correct answers so a
    low-confidence correct answer counts for less than a confident one.
    """
    init_db()
    correct_int = 1 if correct else 0
    confidence = confidence if confidence in _MASTERY_CREDIT else "med"
    mastery_delta = _MASTERY_CREDIT[confidence] if correct else 0.0
    with _connect() as conn:
        conn.execute(
            """INSERT INTO quiz_history
               (session_id, question_id, concept_id, question_type, difficulty,
                user_answer, correct, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, question_id, concept_id, question_type, difficulty,
             user_answer, correct_int, confidence),
        )

        row = conn.execute(
            "SELECT attempts, correct, streak, reps, interval, ease FROM concept_scores "
            "WHERE session_id=? AND concept_id=?",
            (session_id, concept_id),
        ).fetchone()

        if row is None:
            reps, interval, ease = scheduler.schedule_after(0, 0, 2.5, correct, confidence)
            conn.execute(
                """INSERT INTO concept_scores
                   (session_id, concept_id, concept_name, attempts, correct, streak,
                    last_seen, difficulty, mastery_points, reps, interval, ease, due_at)
                   VALUES (?, ?, ?, 1, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)""",
                (session_id, concept_id, concept_name, correct_int, correct_int,
                 difficulty, mastery_delta, reps, interval, ease,
                 scheduler.next_due(interval)),
            )
        else:
            streak = (row["streak"] + 1) if correct else 0
            reps, interval, ease = scheduler.schedule_after(
                row["reps"] or 0, row["interval"] or 0, row["ease"] or 2.5, correct, confidence
            )
            conn.execute(
                """UPDATE concept_scores
                   SET attempts = attempts + 1,
                       correct  = correct + ?,
                       streak   = ?,
                       last_seen = CURRENT_TIMESTAMP,
                       difficulty = ?,
                       concept_name = ?,
                       mastery_points = mastery_points + ?,
                       reps = ?, interval = ?, ease = ?, due_at = ?
                   WHERE session_id=? AND concept_id=?""",
                (correct_int, streak, difficulty, concept_name, mastery_delta,
                 reps, interval, ease, scheduler.next_due(interval),
                 session_id, concept_id),
            )


def concept_scores(session_id: str) -> list[dict]:
    """Return per-concept aggregates with a derived `accuracy` (0-1)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM concept_scores WHERE session_id=? ORDER BY concept_id",
            (session_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["accuracy"] = (d["correct"] / d["attempts"]) if d["attempts"] else 0.0
        # Confidence-weighted mastery (<= accuracy): downweights lucky guesses.
        d["mastery"] = (d.get("mastery_points", 0.0) / d["attempts"]) if d["attempts"] else 0.0
        out.append(d)
    return out


def weak_concepts(session_id: str, threshold: float = 0.6) -> list[dict]:
    """Concepts whose accuracy is below `threshold` (attempted at least once)."""
    return [
        c for c in concept_scores(session_id)
        if c["attempts"] > 0 and c["accuracy"] < threshold
    ]


def due_concepts(session_id: str) -> list[dict]:
    """Concepts whose spaced-repetition `due_at` has arrived (Phase 13)."""
    now = scheduler.now_iso()
    return [
        c for c in concept_scores(session_id)
        if c.get("due_at") and c["due_at"] <= now
    ]


def answer_history(session_id: str) -> list[dict]:
    """Every graded answer in order (Phase 16 — powers the progress trend)."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT answered_at, concept_id, correct FROM quiz_history "
            "WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Phase 9: durable concept / document / quiz persistence ───────────────


def upsert_document(session_id: str, doc_id: str, name: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (session_id, doc_id, name) VALUES (?, ?, ?)",
            (session_id, doc_id, name),
        )


def list_documents(session_id: str) -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT doc_id, name, created_at FROM documents WHERE session_id=? ORDER BY rowid",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_concepts(
    session_id: str, concepts: list[dict], doc_id: str = "doc1", replace: bool = True
) -> None:
    """Persist concepts for a session/document (full dict stored as JSON)."""
    init_db()
    with _connect() as conn:
        if replace:
            conn.execute(
                "DELETE FROM concepts WHERE session_id=? AND doc_id=?", (session_id, doc_id)
            )
        for c in concepts:
            conn.execute(
                "INSERT OR REPLACE INTO concepts "
                "(session_id, doc_id, concept_id, concept_json) VALUES (?, ?, ?, ?)",
                (session_id, doc_id, c.get("concept_id"), json.dumps(c)),
            )


def load_concepts(session_id: str, doc_id: Optional[str] = None) -> list[dict]:
    init_db()
    with _connect() as conn:
        if doc_id is None:
            rows = conn.execute(
                "SELECT concept_json FROM concepts WHERE session_id=? ORDER BY rowid",
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT concept_json FROM concepts WHERE session_id=? AND doc_id=? ORDER BY rowid",
                (session_id, doc_id),
            ).fetchall()
    return [json.loads(r["concept_json"]) for r in rows]


def save_quiz(session_id: str, quiz: list[dict], kind: str = "round") -> int:
    """Persist a quiz and its questions; return the new quiz_id."""
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO quizzes (session_id, kind) VALUES (?, ?)", (session_id, kind)
        )
        quiz_id = cur.lastrowid
        for pos, q in enumerate(quiz):
            conn.execute(
                "INSERT INTO questions "
                "(quiz_id, question_id, concept_id, position, question_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (quiz_id, q.get("question_id"), q.get("concept_id"), pos, json.dumps(q)),
            )
    return quiz_id


def load_latest_quiz(session_id: str) -> list[dict]:
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT quiz_id FROM quizzes WHERE session_id=? ORDER BY quiz_id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if row is None:
            return []
        rows = conn.execute(
            "SELECT question_json FROM questions WHERE quiz_id=? ORDER BY position",
            (row["quiz_id"],),
        ).fetchall()
    return [json.loads(r["question_json"]) for r in rows]
