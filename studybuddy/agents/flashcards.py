"""Flashcard Agent (ARCHITECTURE §10.2).

Generates front/back study cards from a session's concepts, grounded in the
source passages retrieved from Chroma. Uses the `flashcards` routing task
(defaults to the quiz model).
"""
from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from studybuddy import store
from studybuddy.llm import get_chat_model
from studybuddy.schemas import FlashcardList
from studybuddy.vectorstore import get_retriever

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You create concise study flashcards from course material. For each concept, "
            "produce 1-2 cards: `front` is a term or question, `back` is the answer/definition. "
            "Ground every card ONLY in the provided source passages — do not invent facts. "
            "Set concept_id to the concept the card reviews.",
        ),
        ("human", "Concepts:\n{concepts}\n\nSource passages:\n{passages}"),
    ]
)


def make_flashcards(session_id: str, target_concepts: Optional[list[str]] = None) -> list[dict]:
    """Generate grounded flashcards for a session's concepts."""
    concepts = store.get_concepts(session_id)
    if target_concepts:
        focused = [c for c in concepts if c.get("concept_id") in set(target_concepts)]
        concepts = focused or concepts
    if not concepts:
        return []

    retriever = get_retriever(session_id)
    seen: set[str] = set()
    passages: list[str] = []
    lines: list[str] = []
    for c in concepts:
        lines.append(f"- [{c.get('concept_id')}] {c.get('name')}: {c.get('definition', '')}")
        for doc in retriever.invoke(f"{c.get('name', '')}: {c.get('definition', '')}"):
            snippet = doc.page_content.strip()
            if snippet and snippet not in seen:
                seen.add(snippet)
                passages.append(snippet)

    chain = _PROMPT | get_chat_model("flashcards").with_structured_output(FlashcardList)
    result: FlashcardList = chain.invoke(
        {
            "concepts": "\n".join(lines),
            "passages": "\n\n---\n\n".join(passages) if passages else "(no passages retrieved)",
        }
    )
    return [card.model_dump() for card in result.cards]
