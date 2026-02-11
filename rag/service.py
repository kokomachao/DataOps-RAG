from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from diskcache import Cache
from langchain_core.documents import Document

from rag.config import settings
from rag.index import load_bm25, load_vectorstore
from rag.retrievers.hybrid_rrf import HybridRetriever
from rag.utils import normalize_query, sha1_json
from rag.chains.sop_chain import build_sop_answer

@dataclass
class AskResponse:
    answer_md: str
    sop: Dict[str, Any]
    sources: List[Dict[str, Any]]
    debug: Dict[str, Any]

class RAGService:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.vs = load_vectorstore(storage_dir)
        self.bm25 = load_bm25(storage_dir)
        self.retriever = HybridRetriever(self.vs, self.bm25)
        self.cache = Cache(settings.cache_dir)
        self.ttl = settings.cache_ttl_seconds

    def _docs_to_sources(self, docs: List[Document]) -> List[Dict[str, Any]]:
        out = []
        for d in docs:
            md = d.metadata or {}
            out.append({
                "qid": md.get("qid"),
                "component": md.get("component"),
                "tags": md.get("tags"),
                "score": md.get("score"),
                "accepted": md.get("accepted"),
                "title": md.get("title"),
                "chunk_id": md.get("chunk_id"),
                "snippet": d.page_content[:600],
            })
        return out

    def ask(self, question: str,
            components: Optional[List[str]] = None,
            tags: Optional[List[str]] = None,
            top_k: Optional[int] = None,
            fetch_k: Optional[int] = None,
            debug: bool = True) -> AskResponse:

        q = normalize_query(question)
        top_k = top_k or settings.top_k
        fetch_k = fetch_k or settings.fetch_k

        cache_key = sha1_json({"q": q, "components": components, "tags": tags, "k": top_k, "fk": fetch_k})
        cached = self.cache.get(cache_key, default=None)
        if cached is not None:
            return cached

        fused, r_debug = self.retriever.retrieve(q, top_k=top_k, fetch_k=fetch_k, components=components, tags=tags)
        docs = [d for d, _ in fused]
        sop = build_sop_answer(q, docs)

        answer_md = (
            f"### 结论摘要\n{sop.get('summary','')}\n\n"
            f"### 可能原因\n" + "\n".join([f"- {x}" for x in sop.get("possible_causes", [])]) + "\n\n"
            f"### 快速检查\n" + "\n".join([f"- {x}" for x in sop.get("checks", [])]) + "\n\n"
            f"### SOP\n" + "\n".join([f"{i+1}. {x}" for i, x in enumerate(sop.get("step_by_step_sop", []))]) + "\n"
        )

        resp = AskResponse(
            answer_md=answer_md,
            sop=sop,
            sources=self._docs_to_sources(docs),
            debug=r_debug if debug else {},
        )
        self.cache.set(cache_key, resp, expire=self.ttl)
        return resp
