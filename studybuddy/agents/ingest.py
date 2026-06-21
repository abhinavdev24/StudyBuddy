"""Ingest Agent (no LLM).

Turns a PDF or pasted text into ordered topic chunks ready for embedding.
Text extraction uses PyMuPDF; splitting uses LangChain's
RecursiveCharacterTextSplitter. A best-effort `topic` is attached to each
chunk by detecting heading-like lines (e.g. "1. Cells", "DNA").
"""
from __future__ import annotations

import re

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ~1500 chars per chunk with ~150 overlap keeps related sentences together
# while staying small enough for cheap embedding and grounded retrieval.
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# A heading-ish line: "1. Cells", "1.2 Mitosis", "DNA", "The Cell Cycle".
# Short (<= 8 words), no trailing period, optionally numbered.
_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][^.\n]{0,60})\s*$")


def ingest_pdf(path: str) -> str:
    """Extract the full text layer of a PDF as a single string."""
    parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    text = "\n".join(parts).strip()
    if not text:
        raise ValueError(
            f"No extractable text in {path!r} (scanned/image-only PDFs need OCR, out of scope)."
        )
    return text


def ingest_text(text: str) -> str:
    """Normalize pasted text (trim + collapse blank-line runs)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty material: nothing to ingest.")
    return re.sub(r"\n{3,}", "\n\n", text)


def _looks_like_heading(line: str) -> str | None:
    line = line.rstrip()
    if not line or len(line.split()) > 8:
        return None
    m = _HEADING.match(line)
    return m.group(1).strip() if m else None


def chunk_text(text: str) -> list[dict]:
    """Split `text` into ordered chunks: `[{topic, text, order}]`.

    Topic is the most recent heading-like line seen before the chunk's content,
    falling back to "General" when no heading has appeared yet.
    """
    text = ingest_text(text)

    # Track the running topic by scanning headings line-by-line, then map each
    # produced chunk back to whichever heading most recently preceded it.
    pieces = _SPLITTER.split_text(text)
    chunks: list[dict] = []
    current_topic = "General"
    for order, piece in enumerate(pieces):
        # Prefer a heading found inside this piece (its leading lines).
        for line in piece.splitlines()[:3]:
            heading = _looks_like_heading(line)
            if heading:
                current_topic = heading
                break
        chunks.append({"topic": current_topic, "text": piece.strip(), "order": order})
    return chunks
