"""Adaptive Agent (ARCHITECTURE §3.6).

Looks at per-concept performance and plans the next round: which concepts to
re-drill and at what difficulty. Thresholds: accuracy `<0.60` -> re-drill,
`>0.80` -> escalate difficulty, otherwise maintain. The LLM applies these rules
and returns validated structured output.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from studybuddy.llm import get_chat_model

_DIFFICULTY_ORDER = ["recall", "application", "analysis"]


class _ConceptPlan(BaseModel):
    concept_id: str
    action: Literal["redrill", "escalate", "maintain"] = Field(
        description="redrill if <60% accuracy, escalate if >80%, else maintain."
    )
    difficulty: Literal["recall", "application", "analysis"] = Field(
        description="Difficulty to use next for this concept."
    )


class _RedrillPlan(BaseModel):
    plans: list[_ConceptPlan] = Field(default_factory=list)
    rationale: str = Field(default="", description="Brief explanation of the plan.")


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an adaptive learning planner. Given per-concept performance, decide "
            "what to do next for EACH concept using these rules:\n"
            "- accuracy < 0.60  -> action 'redrill', keep difficulty at 'recall' (or current).\n"
            "- accuracy > 0.80  -> action 'escalate', raise difficulty one step "
            "(recall->application->analysis; analysis stays analysis).\n"
            "- otherwise          -> action 'maintain', keep the current difficulty.\n"
            "Return a plan entry for every concept provided, plus a short rationale.",
        ),
        ("human", "Per-concept performance:\n{scores}"),
    ]
)


def _escalate(difficulty: str) -> str:
    try:
        i = _DIFFICULTY_ORDER.index(difficulty)
    except ValueError:
        return "application"
    return _DIFFICULTY_ORDER[min(i + 1, len(_DIFFICULTY_ORDER) - 1)]


def plan_redrill(scores: list[dict]) -> dict:
    """Plan the next round from `concept_scores` rows.

    Returns `{redrill_concepts: [concept_id], difficulty_by_concept: {id: difficulty}}`.
    """
    if not scores:
        return {"redrill_concepts": [], "difficulty_by_concept": {}, "rationale": "No data yet."}

    lines = [
        f"- {s.get('concept_id')} ({s.get('concept_name')}): "
        f"{s.get('correct', 0)}/{s.get('attempts', 0)} = {s.get('accuracy', 0):.0%}, "
        f"current difficulty {s.get('difficulty', 'recall')}"
        for s in scores
    ]
    chain = _PROMPT | get_chat_model("adaptation").with_structured_output(_RedrillPlan)
    plan: _RedrillPlan = chain.invoke({"scores": "\n".join(lines)})

    redrill_concepts: list[str] = []
    difficulty_by_concept: dict[str, str] = {}
    for p in plan.plans:
        difficulty_by_concept[p.concept_id] = p.difficulty
        if p.action == "redrill":
            redrill_concepts.append(p.concept_id)

    return {
        "redrill_concepts": redrill_concepts,
        "difficulty_by_concept": difficulty_by_concept,
        "rationale": plan.rationale,
    }
