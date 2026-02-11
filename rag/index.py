from __future__ import annotations

import json
from pathlib import Path

from rag.config import settings
from rag.data.documents import CorpusConfig, build_documents, chunk_documents
from rag.retrievers.persistent_bm25 import PersistentBM25

def _paths(storage_dir: str):
    base = Path(storage_dir)
    return {
        "faiss": base / "faiss",
        "bm25": base / "bm25.pkl",
        "meta": base / "meta.json",
    }

def build_all(data_jsonl: str, backend: str, storage_dir: str, chunk_size: int, chunk_overlap: int):
    p = _paths(storage_dir)
    Path(storage_dir).mkdir(parents=True, exist_ok=True)

    docs = build_documents(data_jsonl)
    chunks = chunk_documents(docs, CorpusConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    bm25 = PersistentBM25.build(chunks)
    bm25.save(str(p["bm25"]))

    if backend == "faiss":
        from rag.vectorstores.faiss_store import build_faiss
        build_faiss(chunks, str(p["faiss"]))
    elif backend == "milvus":
        from rag.vectorstores.milvus_store import build_milvus
        build_milvus(chunks)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    meta = {
        "backend": backend,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "embedding_model": settings.embedding_model,
    }
    p["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta

def load_vectorstore(storage_dir: str):
    backend = settings.vector_backend
    p = _paths(storage_dir)
    if backend == "faiss":
        from rag.vectorstores.faiss_store import load_faiss
        return load_faiss(str(p["faiss"]))
    if backend == "milvus":
        from rag.vectorstores.milvus_store import load_milvus
        return load_milvus()
    raise ValueError(f"Unknown backend: {backend}")

def load_bm25(storage_dir: str) -> PersistentBM25:
    p = _paths(storage_dir)
    return PersistentBM25.load(str(p["bm25"]))
