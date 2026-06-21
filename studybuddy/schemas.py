"""Pydantic schemas for structured agent output (ARCHITECTURE §3.2 / §3.3).

These models are the contract used with
`get_chat_model(task).with_structured_output(Model)`, so the LLM is forced to
return validated data the rest of the pipeline can rely on.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ── Concept extraction ──────────────────────────────────────────────────


class Concept(BaseModel):
    """A single key idea extracted from a chunk of study material."""

    concept_id: str = Field(description="Short stable id, e.g. 'c1', 'photosynthesis'.")
    name: str = Field(description="Concise concept name.")
    definition: str = Field(description="One- to two-sentence definition in plain language.")
    key_terms: list[str] = Field(
        default_factory=list, description="Important terms/vocabulary for this concept."
    )
    relationships: list[str] = Field(
        default_factory=list,
        description="How this concept relates to others in the material.",
    )
    importance: Optional[str] = Field(
        default=None, description="Why this concept matters / its role."
    )
    chunk_ref: int = Field(
        description="Order index of the source chunk this concept came from (for traceback)."
    )


class ConceptList(BaseModel):
    """Wrapper so structured output returns a top-level object, not a bare list."""

    concepts: list[Concept] = Field(default_factory=list)


# ── Quiz generation ─────────────────────────────────────────────────────

QuestionType = Literal["multiple_choice", "true_false", "short_answer"]
Difficulty = Literal["recall", "application", "analysis"]


class Question(BaseModel):
    """A single quiz item, grounded in the source material."""

    question_id: str = Field(description="Short stable id, e.g. 'q1'.")
    concept_id: str = Field(description="The concept this question assesses.")
    question_type: QuestionType
    difficulty: Difficulty
    prompt: str = Field(description="The question text shown to the student.")
    options: list[str] = Field(
        default_factory=list,
        description="Choices for multiple_choice (empty for other types).",
    )
    answer: str = Field(
        description="Correct answer. For MCQ the exact option text; for TF 'True'/'False'."
    )
    explanation: str = Field(description="Why the answer is correct, grounded in the material.")


class QuizList(BaseModel):
    questions: list[Question] = Field(default_factory=list)


# ── Flashcards (Phase 11) ────────────────────────────────────────────────


class Flashcard(BaseModel):
    """A study flashcard grounded in the material."""

    concept_id: str = Field(description="The concept this card reviews.")
    front: str = Field(description="Prompt side: a term or question.")
    back: str = Field(description="Answer side: the definition or answer.")


class FlashcardList(BaseModel):
    cards: list[Flashcard] = Field(default_factory=list)
