"""LangChain `@tool`s for the Tutor Agent (ARCHITECTURE §3.7).

These wrap the same retrieval / concept store / tracker / quiz functions the UI
and graph use, so the agent and the deterministic pipeline share one source of
truth. The active session id and per-turn `sources`/`actions` collectors are
passed via contextvars (set by `tutor.chat`) since `@tool` signatures are fixed
by what the LLM sees.
"""
from __future__ import annotations

import contextvars

from langchain_core.tools import tool

from studybuddy import store
from studybuddy.llm import get_chat_model
from studybuddy.tracker import weak_concepts
from studybuddy.vectorstore import get_retriever

# Set by tutor.chat() for the duration of one turn.
_session_var: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")
_sources_var: contextvars.ContextVar[list] = contextvars.ContextVar("sources", default=[])
_actions_var: contextvars.ContextVar[list] = contextvars.ContextVar("actions", default=[])


def set_turn_context(session_id: str) -> tuple[list, list]:
    """Begin a tutor turn: bind the session and fresh source/action collectors."""
    sources: list = []
    actions: list = []
    _session_var.set(session_id)
    _sources_var.set(sources)
    _actions_var.set(actions)
    return sources, actions


def _session() -> str:
    sid = _session_var.get()
    if not sid:
        raise RuntimeError("No active session — call tutor.chat(session_id, message).")
    return sid


def _cite(doc) -> dict:
    md = doc.metadata or {}
    return {
        "topic": md.get("topic", "General"),
        "order": md.get("order"),
        "snippet": doc.page_content.strip()[:240],
    }


@tool
def search_material(query: str) -> str:
    """Search the uploaded study material for passages relevant to a question.
    Use this to answer any open-ended question about the material. Returns source
    passages you must base your answer on; cite them."""
    docs = get_retriever(_session()).invoke(query)
    if not docs:
        return "No relevant passages found in the uploaded material."
    for d in docs:
        _sources_var.get().append(_cite(d))
    return "\n\n---\n\n".join(d.page_content.strip() for d in docs)


@tool
def list_concepts() -> str:
    """List the key concepts extracted from the uploaded material."""
    concepts = store.get_concepts(_session())
    if not concepts:
        return "No concepts have been extracted yet for this session."
    return "\n".join(
        f"- {c.get('name')} [{c.get('concept_id')}]: {c.get('definition', '')}"
        for c in concepts
    )


@tool
def get_weak_concepts() -> str:
    """Report which concepts the student is currently struggling with."""
    weak = weak_concepts(_session())
    if not weak:
        return "No weak concepts recorded yet — the student is doing well or hasn't been quizzed."
    return "\n".join(
        f"- {w['concept_name']} [{w['concept_id']}]: "
        f"{w['correct']}/{w['attempts']} correct ({w['accuracy']:.0%})"
        for w in weak
    )


def _parse_types(question_types: str) -> list | None:
    if not question_types:
        return None
    valid = {"multiple_choice", "true_false", "short_answer"}
    picked = [t.strip() for t in question_types.replace(";", ",").split(",")]
    picked = [t for t in picked if t in valid]
    return picked or None


@tool
def quiz_me(
    topic: str = "", difficulty: str = "", num_questions: int = 0, question_types: str = ""
) -> str:
    """Start a quiz round for the student. Optionally focus on a topic/concept, pin a
    difficulty (recall, application, analysis), set the number of questions, and/or
    restrict question types (comma-separated: multiple_choice, true_false, short_answer).
    Use when the student asks to be quizzed or tested."""
    session_id = _session()
    concepts = store.get_concepts(session_id)

    # Fallback: if concepts haven't been extracted yet, extract them on the fly
    # from passages retrieved for the requested topic so the tool still works.
    if not concepts:
        from studybuddy.agents.concept import extract_concepts

        docs = get_retriever(session_id).invoke(topic or "key concepts")
        pseudo = [
            {"topic": (d.metadata or {}).get("topic", "General"),
             "text": d.page_content, "order": i}
            for i, d in enumerate(docs)
        ]
        concepts = extract_concepts(pseudo) if pseudo else []
        if concepts:
            store.set_concepts(session_id, concepts)

    if not concepts:
        return "There's no material to quiz on yet. Upload or paste study material first."

    target = None
    if topic:
        t = topic.lower()
        matches = [
            c["concept_id"] for c in concepts
            if t in c.get("name", "").lower() or t in c.get("concept_id", "").lower()
        ]
        target = matches or None

    # Route through the LangGraph entry function so the Tutor and the
    # deterministic pipeline start quizzes via one shared path.
    from studybuddy.graph import start_quiz

    quiz = start_quiz(
        session_id,
        target_concepts=target,
        difficulty=difficulty or None,
        question_types=_parse_types(question_types),
        num_questions=num_questions or None,
    )
    _actions_var.get().append(
        {"type": "quiz", "topic": topic or None,
         "difficulty": difficulty or None, "num_questions": len(quiz)}
    )
    return (
        f"Started a quiz with {len(quiz)} question(s)"
        + (f" on '{topic}'" if topic else "")
        + (f" at {difficulty} difficulty" if difficulty else "")
        + ". The questions are now available in the Quiz tab."
    )


@tool
def regenerate_quiz(difficulty: str = "", num_questions: int = 0, question_types: str = "") -> str:
    """Replace the current quiz with a fresh set of DIFFERENT questions on the same
    material (avoids repeating the previous prompts). Use when the student wants new
    or different questions."""
    session_id = _session()
    from studybuddy.graph import regenerate_quiz as _regen

    quiz = _regen(
        session_id,
        difficulty=difficulty or None,
        question_types=_parse_types(question_types),
        num_questions=num_questions or None,
    )
    if not quiz:
        return "There's no quiz to regenerate yet — start one first."
    _actions_var.get().append({"type": "regenerate", "num_questions": len(quiz)})
    return f"Regenerated the quiz with {len(quiz)} fresh question(s) in the Quiz tab."


@tool
def make_flashcards(topic: str = "") -> str:
    """Create study flashcards from the material, optionally focused on a topic/concept.
    Use when the student asks for flashcards or cards to review."""
    session_id = _session()
    concepts = store.get_concepts(session_id)
    target = None
    if topic and concepts:
        t = topic.lower()
        target = [c["concept_id"] for c in concepts
                  if t in c.get("name", "").lower() or t in c.get("concept_id", "").lower()] or None

    from studybuddy.graph import make_flashcards as _mk

    cards = _mk(session_id, target_concepts=target)
    if not cards:
        return "There's no material to make flashcards from yet. Upload or paste study material first."
    _actions_var.get().append({"type": "flashcards", "num_cards": len(cards)})
    preview = "; ".join(f"{c['front']} → {c['back']}" for c in cards[:3])
    return f"Made {len(cards)} flashcard(s). Examples: {preview}"


@tool
def quiz_passage(passage: str, num_questions: int = 3) -> str:
    """Quiz the student on a specific passage of text they provide (not the whole
    document). Use when the student pastes a paragraph and asks to be quizzed on it."""
    session_id = _session()
    from studybuddy.graph import quiz_passage as _qp

    quiz = _qp(session_id, passage, num_questions=num_questions or 3)
    if not quiz:
        return "Please provide a passage to quiz on."
    _actions_var.get().append({"type": "quiz", "source": "passage", "num_questions": len(quiz)})
    return f"Started a {len(quiz)}-question quiz on your passage. See the Quiz tab."


@tool
def make_summary(topic: str = "") -> str:
    """Produce a concise, grounded cheat-sheet of the material, optionally focused on a
    topic/concept. Use when the student asks for a summary, notes, or a cheat sheet."""
    session_id = _session()
    concepts = store.get_concepts(session_id)
    target = None
    if topic and concepts:
        t = topic.lower()
        target = [c["concept_id"] for c in concepts
                  if t in c.get("name", "").lower() or t in c.get("concept_id", "").lower()] or None

    from studybuddy.agents.summarizer import make_summary as _summ

    _actions_var.get().append({"type": "summary", "topic": topic or None})
    return _summ(session_id, target_concepts=target)


@tool
def explain_answer(question: str, user_answer: str) -> str:
    """Explain whether the student's answer to a question is right or wrong and why,
    grounded in the material. Use when the student asks why they got something wrong
    or wants their answer checked."""
    session_id = _session()
    from studybuddy.agents.evaluator import explain_answer as _explain

    return _explain(session_id, {"prompt": question, "answer": "", "concept_id": ""}, user_answer)


@tool
def explain_concept(name: str) -> str:
    """Explain a concept from the uploaded material in clear, simple terms."""
    session_id = _session()
    docs = get_retriever(session_id).invoke(name)
    if not docs:
        return f"I couldn't find material about '{name}'."
    for d in docs:
        _sources_var.get().append(_cite(d))
    passage = "\n\n".join(d.page_content.strip() for d in docs)

    msg = (
        "Explain the concept clearly and concisely for a student, using ONLY the "
        f"source material below. If it isn't covered, say so.\n\n"
        f"Concept: {name}\n\nSource material:\n{passage}"
    )
    return get_chat_model("summary").invoke(msg).content


TOOLS = [
    search_material,
    list_concepts,
    get_weak_concepts,
    quiz_me,
    regenerate_quiz,
    make_flashcards,
    quiz_passage,
    make_summary,
    explain_answer,
    explain_concept,
]
