"""Gradio UI (ARCHITECTURE §7).

Four tabs over the LangGraph entry functions and the Tutor agent:
  1. Upload / Paste — ingest a PDF or pasted text (embed + extract concepts).
  2. Quiz          — answer the current round; get per-question feedback.
  3. Dashboard     — concept mastery map (red -> yellow -> green) + weakness round.
  4. Tutor         — chat with the tool-calling tutor (RAG, quizzing, weak spots).
"""
from __future__ import annotations

import uuid

import gradio as gr
import pandas as pd

from studybuddy import store, tracker
from studybuddy.export import export_markdown
from studybuddy.agents.ingest import ingest_pdf, ingest_text
from studybuddy.agents.summarizer import make_summary
from studybuddy.agents.tutor import chat as tutor_chat
from studybuddy.graph import (
    add_questions,
    ingest_material,
    make_flashcards,
    next_question,
    next_round,
    quiz_passage,
    regenerate_quiz,
    start_quiz,
    submit_answers,
)
from studybuddy.tracker import concept_scores

_DISCLAIMER = (
    "_StudyBuddy is an educational aid. Generated concepts, questions, grades, and "
    "tutor answers can contain errors — verify against your source material._"
)


def _concepts_md(session_id: str) -> str:
    concepts = store.get_concepts(session_id)
    if not concepts:
        return "_No concepts extracted yet._"
    lines = ["### Extracted concepts"]
    for c in concepts:
        lines.append(f"- **{c.get('name')}** — {c.get('definition', '')}")
    return "\n".join(lines)


def _mastery_html(session_id: str) -> str:
    scores = concept_scores(session_id)
    if not scores:
        return "<p><em>No quiz attempts yet. Take a quiz to populate your mastery map.</em></p>"

    def color(acc: float) -> str:
        if acc < 0.5:
            return "#e7484f"  # red
        if acc < 0.8:
            return "#f5a623"  # yellow
        return "#34a853"      # green

    rows = []
    for s in scores:
        acc = s["accuracy"]
        rows.append(
            f"<div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>"
            f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
            f"background:{color(acc)};'></span>"
            f"<span style='min-width:200px;'>{s['concept_name']}</span>"
            f"<span style='color:#666;'>{s['correct']}/{s['attempts']} "
            f"({acc:.0%}) · {s.get('difficulty', 'recall')}</span></div>"
        )
    return "<div>" + "".join(rows) + "</div>"


# ── Tab callbacks ─────────────────────────────────────────────────────────


def _concept_choices(session_id: str):
    return [(c.get("name", c.get("concept_id")), c.get("concept_id")) for c in store.get_concepts(session_id)]


def _doc_choices(session_id: str):
    return [("All documents", "")] + [
        (d["name"], d["doc_id"]) for d in tracker.list_documents(session_id)
    ]


def _documents_md(session_id: str) -> str:
    docs = tracker.list_documents(session_id)
    if not docs:
        return "_No documents ingested yet._"
    return "### Documents in this session\n" + "\n".join(
        f"- **{d['name']}** (`{d['doc_id']}`)" for d in docs
    )


def _do_ingest(session_id: str, pdf_file, pasted: str):
    def _result(status, quiz):
        return (status, _concepts_md(session_id), quiz,
                gr.update(choices=_concept_choices(session_id)),
                _documents_md(session_id),
                gr.update(choices=_doc_choices(session_id)))
    try:
        if pdf_file is not None:
            text = ingest_pdf(pdf_file if isinstance(pdf_file, str) else pdf_file.name)
            name = "uploaded.pdf"
        elif pasted and pasted.strip():
            text = ingest_text(pasted)
            name = "pasted text"
        else:
            return _result("⚠️ Upload a PDF or paste some text first.", [])
        state = ingest_material(session_id, text, name)
        if state.get("error"):
            return _result(f"❌ {state['error']}", [])
        status = (
            f"✅ Ingested **{name}** — {len(state['concepts'])} concepts, "
            f"{len(state['quiz'])} questions in round 1. Go to the **Quiz** tab."
        )
        return _result(status, state["quiz"])
    except Exception as e:  # surface ingest/extract failures to the user
        return _result(f"❌ {e}", [])


# ── Quiz-control callbacks (Phase 10) ─────────────────────────────────────


def _quiz_opts(sid, concepts_sel, num, diff, types, doc_scope):
    # Explicit concept selection wins; otherwise scope to a chosen document.
    if concepts_sel:
        target = list(concepts_sel)
    elif doc_scope:
        target = [c.get("concept_id") for c in store.get_concepts(sid, doc_id=doc_scope)] or None
    else:
        target = None
    difficulty = None if (not diff or diff == "mixed") else diff
    qtypes = list(types) if types else None
    return target, difficulty, qtypes, (int(num) if num else None)


def _do_generate(sid, concepts_sel, num, diff, types, doc_scope):
    if not store.get_concepts(sid):
        return []
    target, difficulty, qtypes, n = _quiz_opts(sid, concepts_sel, num, diff, types, doc_scope)
    return start_quiz(sid, target_concepts=target, difficulty=difficulty,
                      question_types=qtypes, num_questions=n)


def _do_regenerate(sid, concepts_sel, num, diff, types, doc_scope):
    target, difficulty, qtypes, n = _quiz_opts(sid, concepts_sel, num, diff, types, doc_scope)
    return regenerate_quiz(sid, target_concepts=target, difficulty=difficulty,
                           question_types=qtypes, num_questions=n)


def _do_add(sid, concepts_sel, num, diff, types, doc_scope):
    target, difficulty, qtypes, _ = _quiz_opts(sid, concepts_sel, num, diff, types, doc_scope)
    return add_questions(sid, n=3, target_concepts=target, difficulty=difficulty,
                         question_types=qtypes)


def _do_practice(sid, concepts_sel):
    q = next_question(sid, target_concepts=list(concepts_sel) if concepts_sel else None)
    return [q] if q else []


def _do_passage_quiz(sid, passage, num):
    return quiz_passage(sid, passage, num_questions=int(num) if num else 3)


# ── Flashcards (Phase 11) ─────────────────────────────────────────────────


def _do_flashcards(sid, concepts_sel):
    if not store.get_concepts(sid):
        return []
    return make_flashcards(sid, target_concepts=list(concepts_sel) if concepts_sel else None)


# ── Progress chart + export (Phase 16) ─────────────────────────────────────


def _progress_df(sid):
    """Cumulative per-concept accuracy over successive attempts."""
    counts: dict = {}
    correct: dict = {}
    rows = []
    for h in tracker.answer_history(sid):
        cid = h["concept_id"] or "?"
        counts[cid] = counts.get(cid, 0) + 1
        correct[cid] = correct.get(cid, 0) + (h["correct"] or 0)
        rows.append({"attempt": counts[cid], "concept": cid,
                     "accuracy": correct[cid] / counts[cid]})
    return pd.DataFrame(rows or {"attempt": [], "concept": [], "accuracy": []})


def _do_weakness_round(session_id: str):
    scores = concept_scores(session_id)
    if not scores:
        return "⚠️ Take a quiz first so I know what to re-drill.", [], _mastery_html(session_id)
    result = next_round(session_id)
    plan = result["adaptive_plan"]
    msg = (
        f"🔁 Weakness round ready — {len(result['quiz'])} questions. "
        f"Re-drilling: {', '.join(plan['redrill_concepts']) or 'none'}. "
        f"Go to the **Quiz** tab."
    )
    return msg, result["quiz"], _mastery_html(session_id)


def _tutor_respond(session_id: str, message: str, history: list):
    history = history or []
    if not message or not message.strip():
        return history, ""
    result = tutor_chat(session_id, message)
    answer = result["answer"]
    if result["sources"]:
        cites = "; ".join(
            f"{s['topic']}" + (f" (#{s['order']})" if s.get("order") is not None else "")
            for s in result["sources"]
        )
        answer += f"\n\n_Sources: {cites}_"
    if result["actions"]:
        answer += f"\n\n_Actions: {result['actions']}_"
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, ""


# ── App ─────────────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    with gr.Blocks(title="StudyBuddy") as demo:
        session_state = gr.State(value="")
        quiz_state = gr.State(value=[])
        flashcards_state = gr.State(value=[])

        gr.Markdown("# 📚 StudyBuddy\nAdaptive, agentic study assistant.")
        gr.Markdown(_DISCLAIMER)

        with gr.Tab("Upload / Paste"):
            pdf_in = gr.File(label="Upload a PDF (with a text layer)", file_types=[".pdf"])
            text_in = gr.Textbox(label="…or paste study material", lines=8)
            ingest_btn = gr.Button("Ingest material", variant="primary")
            gr.Markdown("_Each ingest **adds** a document — your session can hold several._")
            ingest_status = gr.Markdown()
            documents_md = gr.Markdown()
            concepts_md = gr.Markdown()

        with gr.Tab("Quiz"):
            gr.Markdown("Answer the current round, then submit for feedback.")

            with gr.Accordion("Quiz options", open=False):
                doc_scope = gr.Dropdown(
                    label="Scope to document", choices=[("All documents", "")], value=""
                )
                concept_select = gr.Dropdown(
                    label="Focus concepts (blank = all)", multiselect=True, choices=[]
                )
                with gr.Row():
                    num_slider = gr.Slider(1, 15, value=6, step=1, label="Number of questions")
                    difficulty_sel = gr.Radio(
                        ["mixed", "recall", "application", "analysis"],
                        value="mixed", label="Difficulty",
                    )
                type_sel = gr.CheckboxGroup(
                    ["multiple_choice", "true_false", "short_answer"],
                    value=["multiple_choice", "true_false", "short_answer"],
                    label="Question types",
                )
                with gr.Row():
                    gen_btn = gr.Button("Generate quiz", variant="primary")
                    regen_btn = gr.Button("Regenerate (different questions)")
                    add_btn = gr.Button("Add 3 more")
                    practice_btn = gr.Button("Practice: one question")

            with gr.Accordion("Quiz a specific passage", open=False):
                passage_in = gr.Textbox(label="Paste a passage to be quizzed on", lines=4)
                passage_num = gr.Slider(1, 10, value=3, step=1, label="Number of questions")
                passage_btn = gr.Button("Quiz this passage")

            @gr.render(inputs=[quiz_state, session_state])
            def render_quiz(quiz, sid):
                if not quiz:
                    gr.Markdown("_No quiz yet. Ingest material or ask the tutor to quiz you._")
                    return
                comps = []
                for i, q in enumerate(quiz, 1):
                    gr.Markdown(f"**Q{i} ({q['difficulty']}).** {q['prompt']}")
                    if q["question_type"] == "multiple_choice":
                        c = gr.Radio(choices=q["options"], label="Your answer")
                    elif q["question_type"] == "true_false":
                        c = gr.Radio(choices=["True", "False"], label="Your answer")
                    else:
                        c = gr.Textbox(label="Your answer")
                    comps.append(c)
                confidence_sel = gr.Radio(
                    ["low", "med", "high"], value="med",
                    label="How confident were you? (weights your mastery score)",
                )
                feedback = gr.Markdown()
                submit_btn = gr.Button("Submit answers", variant="primary")

                def _submit(sid_, confidence, *vals):
                    answers = {q["question_id"]: v for q, v in zip(quiz, vals)}
                    results = submit_answers(sid_, answers, confidence=confidence)
                    by_id = {r["question_id"]: r for r in results}
                    lines = ["### Results"]
                    correct = sum(1 for r in results if r["correct"])
                    lines.append(f"**Score: {correct}/{len(results)}**\n")
                    for i, q in enumerate(quiz, 1):
                        r = by_id.get(q["question_id"])
                        if not r:
                            continue
                        mark = "✅" if r["correct"] else "❌"
                        lines.append(f"{mark} **Q{i}.** {r['explanation']}")
                    lines.append("\n_Check the **Dashboard** tab for your updated mastery map._")
                    return "\n\n".join(lines)

                submit_btn.click(
                    _submit, inputs=[session_state, confidence_sel, *comps], outputs=feedback
                )

        with gr.Tab("Flashcards"):
            gr.Markdown("Generate flashcards from your material; expand a card to reveal the back.")
            fc_btn = gr.Button("Generate flashcards", variant="primary")

            @gr.render(inputs=[flashcards_state])
            def render_cards(cards):
                if not cards:
                    gr.Markdown("_No flashcards yet. Click **Generate flashcards**._")
                    return
                for i, c in enumerate(cards, 1):
                    with gr.Accordion(f"Card {i}: {c['front']}", open=False):
                        gr.Markdown(c["back"])

        with gr.Tab("Dashboard"):
            gr.Markdown("### Concept mastery\nRed (<50%) · Yellow (<80%) · Green (≥80%)")
            mastery_html = gr.HTML()
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                weakness_btn = gr.Button("Start weakness round", variant="primary")
            weakness_status = gr.Markdown()

            gr.Markdown("### Progress over time")
            progress_plot = gr.LinePlot(
                x="attempt", y="accuracy", color="concept",
                title="Cumulative accuracy per concept", height=300,
            )
            progress_btn = gr.Button("Refresh progress")

            gr.Markdown("### Cheat sheet")
            cheatsheet_btn = gr.Button("Generate cheat sheet")
            cheatsheet_md = gr.Markdown()

            gr.Markdown("### Export")
            export_btn = gr.Button("Export study pack (Markdown)")
            export_file = gr.File(label="Download")

        with gr.Tab("Tutor"):
            gr.Markdown(
                "Ask anything about your material, or say *“quiz me on X”* / "
                "*“what am I weak on?”*"
            )
            chatbot = gr.Chatbot(height=420)
            msg_in = gr.Textbox(label="Message", placeholder="Ask the tutor…")
            send_btn = gr.Button("Send", variant="primary")

        # ── Wiring ───────────────────────────────────────────────────────
        demo.load(lambda: str(uuid.uuid4()), outputs=session_state)

        ingest_btn.click(
            _do_ingest,
            inputs=[session_state, pdf_in, text_in],
            outputs=[ingest_status, concepts_md, quiz_state, concept_select,
                     documents_md, doc_scope],
        )

        _quiz_ctrl_inputs = [
            session_state, concept_select, num_slider, difficulty_sel, type_sel, doc_scope
        ]
        gen_btn.click(_do_generate, inputs=_quiz_ctrl_inputs, outputs=quiz_state)
        regen_btn.click(_do_regenerate, inputs=_quiz_ctrl_inputs, outputs=quiz_state)
        add_btn.click(_do_add, inputs=_quiz_ctrl_inputs, outputs=quiz_state)
        practice_btn.click(_do_practice, inputs=[session_state, concept_select], outputs=quiz_state)
        passage_btn.click(
            _do_passage_quiz, inputs=[session_state, passage_in, passage_num], outputs=quiz_state
        )

        fc_btn.click(_do_flashcards, inputs=[session_state, concept_select], outputs=flashcards_state)

        cheatsheet_btn.click(
            lambda sid: make_summary(sid) if store.get_concepts(sid) else "_No material yet._",
            inputs=session_state, outputs=cheatsheet_md,
        )
        progress_btn.click(_progress_df, inputs=session_state, outputs=progress_plot)
        export_btn.click(export_markdown, inputs=session_state, outputs=export_file)

        refresh_btn.click(
            lambda sid: _mastery_html(sid), inputs=session_state, outputs=mastery_html
        )
        weakness_btn.click(
            _do_weakness_round,
            inputs=session_state,
            outputs=[weakness_status, quiz_state, mastery_html],
        )

        send_btn.click(
            _tutor_respond,
            inputs=[session_state, msg_in, chatbot],
            outputs=[chatbot, msg_in],
        )
        msg_in.submit(
            _tutor_respond,
            inputs=[session_state, msg_in, chatbot],
            outputs=[chatbot, msg_in],
        )

    return demo


if __name__ == "__main__":
    build_app().launch(share=True)
