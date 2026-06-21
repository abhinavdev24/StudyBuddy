"""Evaluator Agent (ARCHITECTURE §3.4).

Grades a student's answer to a question. MCQ/TF are graded deterministically in
code; `short_answer` is graded by an LLM **against the source passage retrieved
from Chroma** for the question's concept, then the result is recorded in the
weak-spot tracker.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from studybuddy.llm import get_chat_model
from studybuddy.tracker import record_answer
from studybuddy.vectorstore import get_retriever


class _ShortAnswerVerdict(BaseModel):
    correct: bool = Field(description="True if the student's answer is essentially correct.")
    explanation: str = Field(description="Brief justification grounded in the source passage.")


_SA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are grading a student's short-answer response against the course "
            "material. Judge correctness on meaning, not exact wording. Be fair but "
            "rigorous: a vague or partially-wrong answer is incorrect.\n"
            "Ground your judgement ONLY in the provided source passage and the reference "
            "answer. Return a verdict and a brief explanation.",
        ),
        (
            "human",
            "Question: {question}\n\nReference answer: {reference}\n\n"
            "Source passage:\n{passage}\n\nStudent answer: {user_answer}",
        ),
    ]
)


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def grade(session_id: str, question: dict, user_answer: str, confidence: str = "med") -> dict:
    """Grade `user_answer` for `question`; record it; return the verdict.

    `confidence` (low|med|high) is forwarded to the tracker for mastery weighting.
    Returns `{question_id, correct, explanation}`.
    """
    qtype = question.get("question_type")
    reference = question.get("answer", "")
    concept_id = question.get("concept_id", "")

    if qtype in ("multiple_choice", "true_false"):
        correct = _norm(user_answer) == _norm(reference)
        explanation = question.get("explanation", "") or (
            f"Correct answer: {reference}."
        )
    elif qtype == "short_answer":
        # Retrieve the source passage for this concept to ground the grading.
        passage = ""
        docs = get_retriever(session_id).invoke(
            f"{concept_id} {question.get('prompt', '')}"
        )
        if docs:
            passage = "\n\n".join(d.page_content.strip() for d in docs)

        chain = _SA_PROMPT | get_chat_model("evaluation").with_structured_output(
            _ShortAnswerVerdict
        )
        verdict: _ShortAnswerVerdict = chain.invoke(
            {
                "question": question.get("prompt", ""),
                "reference": reference,
                "passage": passage or "(no passage retrieved)",
                "user_answer": user_answer,
            }
        )
        correct = verdict.correct
        explanation = verdict.explanation
    else:
        raise ValueError(f"Unknown question_type: {qtype!r}")

    record_answer(
        session_id=session_id,
        concept_id=concept_id,
        concept_name=question.get("concept_name", concept_id),
        question_type=qtype,
        difficulty=question.get("difficulty", "recall"),
        user_answer=user_answer,
        correct=correct,
        question_id=question.get("question_id"),
        confidence=confidence,
    )

    return {
        "question_id": question.get("question_id"),
        "correct": correct,
        "explanation": explanation,
    }


_EXPLAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "A student answered a question. Using ONLY the source passage, explain clearly "
            "whether their answer is right or wrong and WHY, then state the correct answer and "
            "the key idea they should remember. Be encouraging and concise. If the passage "
            "doesn't cover it, say so.",
        ),
        (
            "human",
            "Question: {question}\nReference answer: {reference}\nStudent answer: {user_answer}\n\n"
            "Source passage:\n{passage}",
        ),
    ]
)


def explain_answer(session_id: str, question: dict, user_answer: str) -> str:
    """Grounded "why is my answer wrong/right" follow-up explanation."""
    docs = get_retriever(session_id).invoke(
        f"{question.get('concept_id', '')} {question.get('prompt', '')}"
    )
    passage = "\n\n".join(d.page_content.strip() for d in docs) if docs else "(no passage found)"
    chain = _EXPLAIN_PROMPT | get_chat_model("evaluation")
    return chain.invoke(
        {
            "question": question.get("prompt", ""),
            "reference": question.get("answer", ""),
            "user_answer": user_answer,
            "passage": passage,
        }
    ).content
