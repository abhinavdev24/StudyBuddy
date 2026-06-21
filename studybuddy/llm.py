"""The single choke point for every chat model in StudyBuddy.

Agents must call `get_chat_model(task)` rather than constructing `ChatOpenAI`
or any provider SDK directly. Model name comes from the task routing map and
credentials come from `settings` — both defined in studybuddy/config.py.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from studybuddy.config import ROUTING, settings


def get_chat_model(task: str, **kw) -> ChatOpenAI:
    """Return a configured `ChatOpenAI` for the given routing `task`.

    Args:
        task: one of the keys in `config.ROUTING`
            (extraction | quiz | evaluation | adaptation | tutor | summary).
        **kw: overrides passed through to `ChatOpenAI`
            (e.g. `temperature`, `max_tokens`).
    """
    if task not in ROUTING:
        raise KeyError(f"Unknown task '{task}'. Known tasks: {sorted(ROUTING)}")

    return ChatOpenAI(
        model=ROUTING[task],
        base_url=settings.openai_base_url,
        api_key=settings.require_api_key(),
        temperature=kw.pop("temperature", 0.2),
        **kw,
    )
