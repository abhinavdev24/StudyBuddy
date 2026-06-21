"""Chroma vector store wrapper with **local** embeddings (no API).

Chunks from the Ingest Agent are embedded with a sentence-transformers model
and persisted to `CHROMA_DIR`. Each chunk carries `{session_id, topic, order}`
metadata so a retriever can be scoped to a single study session.
"""
from __future__ import annotations

import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from studybuddy.config import CHROMA_DIR, EMBEDDING_MODEL, RETRIEVAL_K

_COLLECTION = "studybuddy"

# A single embeddings instance is reused — loading the model is the slow part.
_embeddings: HuggingFaceEmbeddings | None = None
_store: Chroma | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_store() -> Chroma:
    global _store
    if _store is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _store = Chroma(
            collection_name=_COLLECTION,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
    return _store


def index_chunks(session_id: str, chunks: list[dict], doc_id: str = "doc1") -> int:
    """Embed and persist `chunks` for `session_id`/`doc_id`. Returns count indexed.

    Each chunk is a dict `{topic, text, order}` (from `ingest.chunk_text`).
    Ids are scoped to session + document so multiple documents coexist and
    re-indexing a document is idempotent per chunk order.
    """
    if not chunks:
        return 0

    store = _get_store()
    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "session_id": session_id,
                "doc_id": doc_id,
                "topic": c.get("topic", "General"),
                "order": c.get("order", i),
            },
        )
        for i, c in enumerate(chunks)
    ]
    ids = [f"{session_id}:{doc_id}:{c.get('order', i)}" for i, c in enumerate(chunks)]
    store.add_documents(docs, ids=ids)
    return len(docs)


def get_retriever(session_id: str, k: int = RETRIEVAL_K, doc_id: Optional[str] = None):
    """Return a retriever scoped to a `session_id`, optionally a single `doc_id`."""
    if doc_id is None:
        flt: dict = {"session_id": session_id}
    else:
        flt = {"$and": [{"session_id": session_id}, {"doc_id": doc_id}]}
    return _get_store().as_retriever(search_kwargs={"k": k, "filter": flt})
