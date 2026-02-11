from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from rag.embeddings import get_embeddings

def build_faiss(docs: List[Document], path: str):
    from langchain_community.vectorstores.faiss import FAISS
    embeddings = get_embeddings()
    vs = FAISS.from_documents(docs, embedding=embeddings)
    Path(path).mkdir(parents=True, exist_ok=True)
    vs.save_local(path)
    return vs

def load_faiss(path: str):
    from langchain_community.vectorstores.faiss import FAISS
    embeddings = get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
