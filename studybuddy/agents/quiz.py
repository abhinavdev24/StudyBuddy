"""Quiz Generation Agent (ARCHITECTURE §3.3 + §10.2).

Generates questions across 3 formats x 3 difficulties from extracted concepts,
**grounded in source passages retrieved from Chroma** so items stay faithful to
the material.

Phase 10 makes generation flexible: callers can fix the number of questions,
restrict the allowed question types, pin a difficulty, target specific concepts,
and pass previous prompts to avoid (powering regenerate / add-more).
"""
from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from studybuddy.llm import get_chat_model
from studybuddy.schemas import QuizList
from studybuddy.vectorstore import get_retriever

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert assessment writer. Generate quiz questions that test a "
            "student's understanding of the given concepts, grounded ONLY in the provided "
            "source passages.\n"
            "Rules:\n"
            "- {count_rule}\n"
            "- Allowed question types: {types_rule}.\n"
            "- Difficulty: {difficulty_rule}.\n"
            "- multiple_choice: provide exactly 4 plausible options; `answer` must equal one "
            "option's text verbatim.\n"
            "- true_false: `answer` is 'True' or 'False'; leave options empty.\n"
            "- short_answer: leave options empty; `answer` is a concise model answer.\n"
            "- Every question needs a clear `explanation` justified by the source.\n"
            "- Avoid trivial or trick questions. Set concept_id to the assessed concept.\n"
            "{avoid_rule}",
        ),
        (
            "human",
            "Concepts to assess:\n{concepts}\n\n"
            "Source passages (ground all questions in these):\n{passages}",
        ),
    ]
)

_ALL_TYPES = ["multiple_choice", "true_false", "short_answer"]


def _format_concepts(concepts: list[dict]) -> str:
    lines = []
    for c in concepts:
        terms = ", ".join(c.get("key_terms", []) or [])
        lines.append(
            f"- [{c.get('concept_id')}] {c.get('name')}: {c.get('definition', '')}"
            + (f" (terms: {terms})" if terms else "")
        )
    return "\n".join(lines)


def generate_quiz(
    session_id: str,
    concepts: list[dict],
    target_concepts: Optional[list[str]] = None,
    difficulty: Optional[str] = None,
    question_types: Optional[list[str]] = None,
    num_questions: Optional[int] = None,
    avoid_questions: Optional[list[str]] = None,
    passages_override: Optional[str] = None,
) -> list[dict]:
    """Generate grounded quiz questions for `concepts`.

    Args:
        session_id: study session whose Chroma index supplies grounding passages.
        concepts: concept dicts (from the Concept Extraction Agent).
        target_concepts: optional concept_ids to focus on (weakness rounds).
        difficulty: optional fixed difficulty (recall|application|analysis).
        question_types: optional whitelist of allowed types (subset of
            multiple_choice|true_false|short_answer).
        num_questions: optional exact total number of questions.
        avoid_questions: optional list of prior question prompts not to repeat.
    """
    if not concepts:
        return []

    if target_concepts:
        focused = [c for c in concepts if c.get("concept_id") in set(target_concepts)]
        concepts = focused or concepts

    # Grounding passages: either an explicit passage (quiz-this-paragraph) or
    # passages retrieved from Chroma for the concepts being assessed.
    if passages_override is not None:
        passages_text = passages_override.strip() or "(no passage provided)"
    else:
        retriever = get_retriever(session_id)
        seen: set[str] = set()
        passages: list[str] = []
        for c in concepts:
            query = f"{c.get('name', '')}: {c.get('definition', '')}"
            for doc in retriever.invoke(query):
                snippet = doc.page_content.strip()
                if snippet and snippet not in seen:
                    seen.add(snippet)
                    passages.append(snippet)
        passages_text = "\n\n---\n\n".join(passages) if passages else "(no passages retrieved)"

    # Build the flexible rule clauses.
    types = [t for t in (question_types or _ALL_TYPES) if t in _ALL_TYPES] or _ALL_TYPES
    types_rule = ", ".join(types)
    if num_questions:
        count_rule = f"Generate EXACTLY {num_questions} question(s) total."
    else:
        count_rule = "Generate roughly 2 questions per concept."
    if difficulty:
        difficulty_rule = f"Use difficulty '{difficulty}' for every question."
    else:
        difficulty_rule = "Vary difficulty across recall, application, and analysis."
    if avoid_questions:
        avoid_rule = "Do NOT repeat or closely paraphrase any of these previous questions:\n" + \
            "\n".join(f"- {p}" for p in avoid_questions)
    else:
        avoid_rule = ""

    chain = _PROMPT | get_chat_model("quiz").with_structured_output(QuizList)
    result: QuizList = chain.invoke(
        {
            "concepts": _format_concepts(concepts),
            "passages": passages_text,
            "count_rule": count_rule,
            "types_rule": types_rule,
            "difficulty_rule": difficulty_rule,
            "avoid_rule": avoid_rule,
        }
    )
    quiz = [q.model_dump() for q in result.questions]

    # Enforce the constraints the model may have loosely followed.
    quiz = [q for q in quiz if q.get("question_type") in types] or quiz
    if num_questions:
        quiz = quiz[:num_questions]
    return quiz
