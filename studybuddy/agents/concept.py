"""Concept Extraction Agent (ARCHITECTURE §3.2).

For each chunk, extract 3-5 key concepts as validated structured output via
`prompt | get_chat_model('extraction').with_structured_output(ConceptList)`.
Each concept carries `chunk_ref` so quizzes and grading can trace back to the
source text.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from studybuddy.llm import get_chat_model
from studybuddy.schemas import ConceptList

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a study assistant that extracts the key concepts a student must "
            "learn from a passage of course material.\n"
            "Extract 3-5 of the MOST IMPORTANT concepts (fewer if the passage is short).\n"
            "For each concept provide: a short stable concept_id, a clear name, a concise "
            "1-2 sentence definition, the key terms/vocabulary, how it relates to other "
            "concepts in the passage, and why it matters.\n"
            "Stay faithful to the passage — do not invent facts not supported by it.\n"
            "Set chunk_ref to the provided chunk order index for every concept.",
        ),
        (
            "human",
            "Chunk order index: {order}\nTopic: {topic}\n\nPassage:\n{text}",
        ),
    ]
)


def extract_concepts(chunks: list[dict]) -> list[dict]:
    """Extract concepts from `chunks` (`[{topic, text, order}]`).

    Returns a flat list of concept dicts (validated against `Concept`).
    """
    chain = _PROMPT | get_chat_model("extraction").with_structured_output(ConceptList)

    concepts: list[dict] = []
    for chunk in chunks:
        order = chunk.get("order", 0)
        result: ConceptList = chain.invoke(
            {
                "order": order,
                "topic": chunk.get("topic", "General"),
                "text": chunk["text"],
            }
        )
        for c in result.concepts:
            # Trust the chunk's true order over whatever the model echoed.
            c.chunk_ref = order
            concepts.append(c.model_dump())
    return concepts
