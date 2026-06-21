"""Tutor Agent — tool-calling, with per-session memory (ARCHITECTURE §3.7).

A `create_tool_calling_agent` + `AgentExecutor` over `get_chat_model('tutor')`
and the StudyBuddy tools, wrapped in `RunnableWithMessageHistory` for per-session
chat memory. It answers open-ended questions (RAG via `search_material`) and
drives the app (quiz, weak spots, explanations) by choosing tools.
"""
from __future__ import annotations

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from studybuddy.llm import get_chat_model
from studybuddy.tools import TOOLS, set_turn_context

_SYSTEM = (
    "You are StudyBuddy, a friendly and rigorous study tutor. You help a student "
    "learn from material they have uploaded.\n"
    "- For any question about the material, call `search_material` and ground your "
    "answer in the returned passages. Do not invent facts.\n"
    "- If the student asks to be quizzed/tested, call `quiz_me` (pass a topic and/or "
    "difficulty if they specified one).\n"
    "- To explain a concept, call `explain_concept`. To report struggles, call "
    "`get_weak_concepts`. To enumerate topics, call `list_concepts`.\n"
    "- Always be clear that this is an educational aid and may contain errors.\n"
    "- If the material doesn't cover something, say so rather than guessing."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Per-session chat histories (in-process memory).
_HISTORIES: dict[str, ChatMessageHistory] = {}
_TUTOR = None  # built lazily so importing this module needs no API key


def _get_history(session_id: str) -> ChatMessageHistory:
    return _HISTORIES.setdefault(session_id, ChatMessageHistory())


def build_tutor():
    """Build (once) the tool-calling agent wrapped with per-session memory."""
    global _TUTOR
    if _TUTOR is None:
        agent = create_tool_calling_agent(get_chat_model("tutor"), TOOLS, _PROMPT)
        executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=False)
        _TUTOR = RunnableWithMessageHistory(
            executor,
            _get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="output",
        )
    return _TUTOR


def chat(session_id: str, message: str) -> dict:
    """Send `message` to the tutor for `session_id`.

    Returns `{answer, sources, actions}`. `sources` are passages the tutor cited;
    `actions` are app actions it triggered (e.g. starting a quiz).
    """
    sources, actions = set_turn_context(session_id)
    result = build_tutor().invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}},
    )
    return {"answer": result["output"], "sources": sources, "actions": actions}
