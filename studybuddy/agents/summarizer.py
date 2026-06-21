"""Summarizer Agent (ARCHITECTURE §10.2, Phase 14).

Generates a grounded, condensed cheat-sheet from a session's concepts using the
`summary` routing task. Output is Markdown suitable for study or export.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from studybuddy import store
from studybuddy.llm import get_chat_model
from studybuddy.vectorstore import get_retriever

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write concise study cheat-sheets. Using ONLY the provided source passages, "
            "produce a clean Markdown summary: a short intro, then one bullet section per "
            "concept with its key idea and must-know terms. Do not invent facts beyond the "
            "passages. Keep it tight and scannable.",
        ),
        ("human", "Concepts:\n{concepts}\n\nSource passages:\n{passages}"),
    ]
)


def make_summary(
    session_id: str, target_concepts: Optional[list[str]] = None, scope: str = "doc"
) -> str:
    """Return a grounded Markdown cheat-sheet for the session's concepts."""
    concepts = store.get_concepts(session_id)
    if target_concepts:
        focused = [c for c in concepts if c.get("concept_id") in set(target_concepts)]
        concepts = focused or concepts
    if not concepts:
        return "_No material to summarize yet._"

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

    chain = _PROMPT | get_chat_model("summary")
    return chain.invoke(
        {
            "concepts": "\n".join(lines),
            "passages": "\n\n---\n\n".join(passages) if passages else "(no passages retrieved)",
        }
    ).content
