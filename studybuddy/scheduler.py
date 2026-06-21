"""Spaced-repetition scheduling (ARCHITECTURE §10, Phase 13).

A compact SM-2 implementation (no LLM). Given a concept's current schedule state
and the latest answer, compute the next review interval and due date. Correct
answers grow the interval (scaled by an ease factor); a wrong answer resets it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

DEFAULT_EASE = 2.5
MIN_EASE = 1.3

# Map a binary correct/incorrect + confidence onto an SM-2 quality grade (0-5).
_QUALITY = {"high": 5, "med": 4, "low": 3}


def schedule_after(
    reps: int, interval: int, ease: float, correct: bool, confidence: str = "med"
) -> tuple[int, int, float]:
    """Return the new `(reps, interval_days, ease)` after one graded answer."""
    ease = ease or DEFAULT_EASE
    quality = _QUALITY.get(confidence, 4) if correct else 1

    if quality < 3:  # failed — relearn from the start
        reps = 0
        interval = 1
    else:
        if reps <= 0:
            interval = 1
        elif reps == 1:
            interval = 6
        else:
            interval = max(1, round(interval * ease))
        reps += 1

    ease = max(MIN_EASE, ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    return reps, interval, round(ease, 3)


def next_due(interval_days: int, now: datetime | None = None) -> str:
    """ISO timestamp `interval_days` from now (UTC)."""
    now = now or datetime.now(timezone.utc)
    return (now + timedelta(days=interval_days)).isoformat()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
