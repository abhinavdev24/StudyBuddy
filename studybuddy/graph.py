"""LangGraph orchestrator (ARCHITECTURE §3.8 / §6).

Wires the agents into a deterministic state machine and exposes the entry
functions the UI and the Tutor's tools both call:
`ingest_material`, `start_quiz`, `submit_answers`, `next_round`.

The "new material" pipeline (ingest -> embed -> extract -> quiz) is a compiled
`StateGraph`; the human-in-the-loop boundary (the student answering) is handled
by `submit_answers`, and the adaptive weakness round by `next_round`, which runs
the adapt -> quiz sub-pipeline. LLM nodes are wrapped in a retry guard that
re-runs the node up to N times on malformed/structured-output failure.
"""
from __future__ import annotations

from typing import Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from studybuddy import store
from studybuddy.agents.adaptive import plan_redrill
from studybuddy.agents.concept import extract_concepts
from studybuddy.agents.evaluator import grade
from studybuddy.agents.ingest import chunk_text
from studybuddy.agents.quiz import generate_quiz
from studybuddy.tracker import (
    concept_scores,
    due_concepts,
    init_db,
    list_documents,
    upsert_document,
)
from studybuddy.vectorstore import index_chunks

_MAX_RETRIES = 2

# Per-session live state, so entry functions can be called independently across
# the human-in-the-loop boundary (start a round, wait, submit, adapt).
_SESSIONS: dict[str, dict] = {}


class StudyState(TypedDict, total=False):
    session_id: str
    doc_id: str
    raw_text: str
    chunks: list
    concepts: list
    quiz: list
    user_answers: dict
    results: list
    round: int
    adaptive_plan: dict
    error: Optional[str]


# ── Retry guard ─────────────────────────────────────────────────────────


def _guard(fn: Callable[[StudyState], dict]) -> Callable[[StudyState], dict]:
    """Wrap an LLM node so it re-runs up to N times on failure (§6 guard)."""

    def wrapped(state: StudyState) -> dict:
        last_err: Optional[Exception] = None
        for _ in range(_MAX_RETRIES + 1):
            try:
                out = fn(state)
                out.setdefault("error", None)
                return out
            except Exception as e:  # structured-output validation, API hiccups, etc.
                last_err = e
        return {"error": f"{fn.__name__} failed after retries: {last_err}"}

    return wrapped


# ── Nodes ───────────────────────────────────────────────────────────────


def _ingest_node(state: StudyState) -> dict:
    return {"chunks": chunk_text(state["raw_text"])}


def _embed_node(state: StudyState) -> dict:
    index_chunks(state["session_id"], state["chunks"], doc_id=state.get("doc_id", "doc1"))
    return {}


def _extract_node(state: StudyState) -> dict:
    concepts = extract_concepts(state["chunks"])
    # set_concepts replaces only this doc's concepts, so multiple docs accumulate.
    store.set_concepts(state["session_id"], concepts, doc_id=state.get("doc_id", "doc1"))
    return {"concepts": concepts}


def _quiz_node(state: StudyState) -> dict:
    quiz = generate_quiz(state["session_id"], state["concepts"])
    store.set_latest_quiz(state["session_id"], quiz)
    return {"quiz": quiz, "round": state.get("round", 0) + 1}


def _build_ingest_pipeline():
    g = StateGraph(StudyState)
    g.add_node("ingest", _ingest_node)
    g.add_node("embed", _embed_node)
    g.add_node("extract_concepts", _guard(_extract_node))
    g.add_node("generate_quiz", _guard(_quiz_node))

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "embed")
    g.add_edge("embed", "extract_concepts")

    # Conditional edge: only proceed to quiz if extraction succeeded.
    def _after_extract(state: StudyState) -> str:
        return "generate_quiz" if not state.get("error") else END

    g.add_conditional_edges("extract_concepts", _after_extract, ["generate_quiz", END])
    g.add_edge("generate_quiz", END)
    return g.compile()


_PIPELINE = _build_ingest_pipeline()


# ── Weakness-round quiz (groups targets by planned difficulty) ───────────


def _weakness_quiz(
    session_id: str, concepts: list, scores: list, plan: dict, due_ids: Optional[set] = None
) -> list:
    diffmap = plan.get("difficulty_by_concept", {})
    redrill = set(plan.get("redrill_concepts", []))
    acc = {s["concept_id"]: s.get("accuracy", 0.0) for s in scores}
    escalate = {cid for cid, a in acc.items() if a > 0.8}

    # Target re-drill (weak) + escalate (mastered) + spaced-repetition due concepts.
    targets = redrill | escalate | (due_ids or set())
    if not targets:
        targets = set(acc) or {c["concept_id"] for c in concepts}

    # Group targeted concepts by the difficulty the adaptive plan assigned them.
    groups: dict[str, list[str]] = {}
    for cid in targets:
        groups.setdefault(diffmap.get(cid, "recall"), []).append(cid)

    quiz: list = []
    for difficulty, ids in groups.items():
        quiz.extend(
            generate_quiz(session_id, concepts, target_concepts=ids, difficulty=difficulty)
        )
    return quiz


# ── Entry functions (shared by UI and Tutor tools) ───────────────────────


def ingest_material(
    session_id: str,
    material: str,
    material_name: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> dict:
    """Ingest + embed + extract concepts + generate round-1 quiz. Returns state.

    Each call **adds** a document to the session (multi-doc, Phase 15): concepts
    from earlier documents are preserved and retrieval spans all of them.
    """
    init_db()
    if doc_id is None:
        doc_id = f"doc{len(list_documents(session_id)) + 1}"
    upsert_document(session_id, doc_id, material_name or doc_id)
    state: StudyState = {
        "session_id": session_id,
        "doc_id": doc_id,
        "raw_text": material,
        "round": 0,
        "error": None,
    }
    result = _PIPELINE.invoke(state)
    _SESSIONS[session_id] = dict(result)
    return _SESSIONS[session_id]


def start_quiz(
    session_id: str,
    target_concepts: Optional[list[str]] = None,
    difficulty: Optional[str] = None,
    question_types: Optional[list[str]] = None,
    num_questions: Optional[int] = None,
) -> list:
    """Generate a quiz from the session's stored concepts (used by `quiz_me`)."""
    concepts = store.get_concepts(session_id)
    if not concepts:
        return []
    quiz = generate_quiz(
        session_id,
        concepts,
        target_concepts=target_concepts,
        difficulty=difficulty,
        question_types=question_types,
        num_questions=num_questions,
    )
    store.set_latest_quiz(session_id, quiz)
    state = _SESSIONS.setdefault(session_id, {"session_id": session_id, "round": 0})
    state["quiz"] = quiz
    state["round"] = state.get("round", 0) + 1
    return quiz


def _renumber(quiz: list[dict], start: int = 1) -> list[dict]:
    """Give questions stable, unique ids (q1, q2, …) so rounds don't collide."""
    for i, q in enumerate(quiz, start):
        q["question_id"] = f"q{i}"
    return quiz


def regenerate_quiz(
    session_id: str,
    target_concepts: Optional[list[str]] = None,
    difficulty: Optional[str] = None,
    question_types: Optional[list[str]] = None,
    num_questions: Optional[int] = None,
) -> list:
    """Re-roll the current round with *different* questions (avoids prior prompts)."""
    concepts = store.get_concepts(session_id)
    if not concepts:
        return []
    avoid = [q.get("prompt", "") for q in store.get_latest_quiz(session_id)]
    quiz = generate_quiz(
        session_id,
        concepts,
        target_concepts=target_concepts,
        difficulty=difficulty,
        question_types=question_types,
        num_questions=num_questions or (len(avoid) or None),
        avoid_questions=avoid,
    )
    quiz = _renumber(quiz)
    store.set_latest_quiz(session_id, quiz, kind="regenerate")
    return quiz


def add_questions(
    session_id: str,
    n: int = 3,
    target_concepts: Optional[list[str]] = None,
    difficulty: Optional[str] = None,
    question_types: Optional[list[str]] = None,
) -> list:
    """Append `n` fresh questions to the current quiz (no duplicates)."""
    concepts = store.get_concepts(session_id)
    if not concepts:
        return []
    existing = store.get_latest_quiz(session_id)
    avoid = [q.get("prompt", "") for q in existing]
    fresh = generate_quiz(
        session_id,
        concepts,
        target_concepts=target_concepts,
        difficulty=difficulty,
        question_types=question_types,
        num_questions=n,
        avoid_questions=avoid,
    )
    combined = _renumber(existing + fresh)
    store.set_latest_quiz(session_id, combined, kind="round")
    return combined


# ── Phase 11: alternate study modes ──────────────────────────────────────


def make_flashcards(session_id: str, target_concepts: Optional[list[str]] = None) -> list[dict]:
    """Generate grounded flashcards for the session (see agents/flashcards.py)."""
    from studybuddy.agents.flashcards import make_flashcards as _mk

    return _mk(session_id, target_concepts=target_concepts)


def next_question(session_id: str, target_concepts: Optional[list[str]] = None) -> Optional[dict]:
    """Practice mode: produce a single grounded question to answer immediately."""
    concepts = store.get_concepts(session_id)
    if not concepts:
        return None
    avoid = [q.get("prompt", "") for q in store.get_latest_quiz(session_id)]
    quiz = generate_quiz(
        session_id,
        concepts,
        target_concepts=target_concepts,
        num_questions=1,
        avoid_questions=avoid[-20:],  # discourage immediate repeats
    )
    if not quiz:
        return None
    q = _renumber(quiz)[0]
    store.set_latest_quiz(session_id, [q], kind="practice")
    return q


def quiz_passage(
    session_id: str,
    passage: str,
    num_questions: Optional[int] = 5,
    difficulty: Optional[str] = None,
    question_types: Optional[list[str]] = None,
) -> list[dict]:
    """Generate questions grounded directly on a pasted passage (no extraction)."""
    if not passage or not passage.strip():
        return []
    pseudo = [{"concept_id": "passage", "name": "Passage", "definition": passage[:200],
               "key_terms": [], "chunk_ref": 0}]
    quiz = generate_quiz(
        session_id,
        pseudo,
        difficulty=difficulty,
        question_types=question_types,
        num_questions=num_questions,
        passages_override=passage,
    )
    quiz = _renumber(quiz)
    store.set_latest_quiz(session_id, quiz, kind="passage")
    return quiz


def submit_answers(session_id: str, answers: dict, confidence: str = "med") -> list:
    """Grade the student's answers to the latest quiz; record + return results."""
    quiz = store.get_latest_quiz(session_id)
    results = []
    for q in quiz:
        ua = answers.get(q.get("question_id"))
        if ua is None or ua == "":
            continue
        results.append(grade(session_id, q, ua, confidence=confidence))
    state = _SESSIONS.setdefault(session_id, {"session_id": session_id})
    state["user_answers"] = answers
    state["results"] = results
    return results


def next_round(session_id: str) -> dict:
    """Plan from current scores and generate the adaptive weakness round."""
    scores = concept_scores(session_id)
    plan = plan_redrill(scores)
    concepts = store.get_concepts(session_id)
    due_ids = {d["concept_id"] for d in due_concepts(session_id)}
    quiz = _weakness_quiz(session_id, concepts, scores, plan, due_ids=due_ids)
    store.set_latest_quiz(session_id, quiz)
    state = _SESSIONS.setdefault(session_id, {"session_id": session_id, "round": 0})
    state["adaptive_plan"] = plan
    state["quiz"] = quiz
    state["round"] = state.get("round", 0) + 1
    return {"adaptive_plan": plan, "quiz": quiz}


# ── Smoke driver ──────────────────────────────────────────────────────────

_DEMO = """1. Cells
Cells are the basic structural and functional units of all living organisms.
2. Mitochondria
Mitochondria are the powerhouse of the cell and produce ATP through respiration.
3. Photosynthesis
Photosynthesis converts light energy into chemical energy in the chloroplasts of plants.
"""


_DEMO2 = """1. Gravity
Gravity is a force of attraction between masses, described by Newton.
2. Inertia
Inertia is the tendency of an object to resist changes in its motion.
"""


def _smoke() -> None:
    import uuid
    from studybuddy.tracker import due_concepts

    sid = "smoke_" + uuid.uuid4().hex[:8]  # fresh session so persisted state is clean

    print("== Phase 0-6: ingest + adaptive round ==")
    st = ingest_material(sid, _DEMO, "biology")
    print(f"concepts={len(st['concepts'])} round1_quiz={len(st['quiz'])} error={st.get('error')}")

    concepts = store.get_concepts(sid)
    quiz = store.get_latest_quiz(sid)
    strong = concepts[0]["concept_id"]  # answer concept #1 right, the rest wrong
    answers = {q["question_id"]: (q["answer"] if q["concept_id"] == strong else "wrong")
               for q in quiz}
    submit_answers(sid, answers, confidence="high")
    print("scores:", [(s["concept_id"], f"{s['accuracy']:.0%}", f"m{s['mastery']:.0%}")
                      for s in concept_scores(sid)])

    nr = next_round(sid)
    plan = nr["adaptive_plan"]
    print("redrill:", plan["redrill_concepts"],
          "| targets:", sorted({q["concept_id"] for q in nr["quiz"]}),
          "| difficulties:", sorted({q["difficulty"] for q in nr["quiz"]}))

    print("== Phase 10: regenerate + add-more ==")
    q_re = regenerate_quiz(sid, num_questions=3)
    q_add = add_questions(sid, n=2)
    print(f"regenerated={len(q_re)} then added -> {len(q_add)} (unique ids: "
          f"{len({q['question_id'] for q in q_add}) == len(q_add)})")

    print("== Phase 11: flashcards + practice + passage ==")
    print("flashcards:", len(make_flashcards(sid)),
          "| practice_one:", bool(next_question(sid)),
          "| passage_quiz:", len(quiz_passage(sid, "Mitosis divides one cell into two.", num_questions=2)))

    print("== Phase 13: spaced repetition ==")
    print("due concepts now:", [d["concept_id"] for d in due_concepts(sid)])

    print("== Phase 15: multi-document ==")
    ingest_material(sid, _DEMO2, "physics")
    from studybuddy.vectorstore import get_retriever
    print("docs:", [d["doc_id"] for d in list_documents(sid)],
          "| concepts across docs:", len(store.get_concepts(sid)),
          "| retrieval spans:", len(get_retriever(sid).invoke("force and motion")))

    print("== Phase 14 + 16: summary + export ==")
    from studybuddy.agents.summarizer import make_summary
    from studybuddy.export import export_markdown
    print("summary chars:", len(make_summary(sid)))
    print("exported:", export_markdown(sid))


if __name__ == "__main__":
    _smoke()
