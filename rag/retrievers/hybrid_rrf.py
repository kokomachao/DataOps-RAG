from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from rag.config import settings
from rag.retrievers.persistent_bm25 import PersistentBM25

def _doc_key(d: Document) -> str:
    md = d.metadata or {}
    return f"{md.get('qid','')}-{md.get('chunk_id','')}-{md.get('title','')[:80]}"

def rrf_fuse(
    dense: List[Tuple[Document, float]],
    sparse: List[Tuple[Document, float]],
    k: int,
    c: int,
) -> Tuple[List[Tuple[Document, float]], Dict]:
    # Reciprocal Rank Fusion: score(d)=sum_i 1/(rank_i(d)+c)
    rrf: Dict[str, float] = {}
    doc_by_id: Dict[str, Document] = {}

    for rank, (d, _) in enumerate(dense, start=1):
        key = _doc_key(d)
        doc_by_id[key] = d
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (rank + c)

    for rank, (d, _) in enumerate(sparse, start=1):
        key = _doc_key(d)
        doc_by_id[key] = d
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (rank + c)

    fused_items = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    fused = [(doc_by_id[key], score) for key, score in fused_items]

    debug = {
        "dense_n": len(dense),
        "bm25_n": len(sparse),
        "rrf_c": c,
        "rrf_top": [{"key": key, "rrf": float(score)} for key, score in fused_items],
    }
    return fused, debug

class HybridRetriever:
    def __init__(self, vectorstore, bm25: PersistentBM25):
        self.vectorstore = vectorstore
        self.bm25 = bm25

    def dense_search(
        self,
        query: str,
        k: int,
        fetch_k: int,
        components: Optional[List[str]],
        tags: Optional[List[str]],
    ) -> List[Tuple[Document, float]]:
        backend = settings.vector_backend

        # Milvus: try expr for component filter, tags post-filter (expr syntax differs across versions)
        if backend == "milvus":
            expr_parts = []
            if components:
                comps = ",".join([f'"{c}"' for c in components])
                expr_parts.append(f'component in [{comps}]')
            expr = " and ".join(expr_parts) if expr_parts else None

            try:
                kwargs = {"k": fetch_k}
                if expr:
                    kwargs["expr"] = expr
                docs_scores = self.vectorstore.similarity_search_with_score(query, **kwargs)
            except TypeError:
                docs_scores = self.vectorstore.similarity_search_with_score(
                    query, k=fetch_k, search_kwargs={"expr": expr} if expr else {}
                )

            dense = [(d, float(s)) for d, s in docs_scores]
        else:
            docs_scores = self.vectorstore.similarity_search_with_score(query, k=fetch_k)
            dense = [(d, float(s)) for d, s in docs_scores]

        # Post-filter
        if components:
            cset = set([c.lower() for c in components])
            dense = [(d, s) for d, s in dense if str(d.metadata.get("component","")).lower() in cset]
        if tags:
            tset = set([t.lower() for t in tags])
            dense = [(d, s) for d, s in dense if tset.intersection(set([x.lower() for x in (d.metadata.get("tags") or [])]))]

        return dense[:k]

    def retrieve(
        self,
        query: str,
        top_k: int,
        fetch_k: int,
        components: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Tuple[List[Tuple[Document, float]], Dict]:
        dense = self.dense_search(query, k=top_k, fetch_k=fetch_k, components=components, tags=tags)
        sparse = self.bm25.search(query, k=top_k, components=components, tags=tags)
        fused, debug = rrf_fuse(dense, sparse, k=top_k, c=settings.rrf_c)

        debug.update({
            "dense_preview": [{"title": d.metadata.get("title",""), "component": d.metadata.get("component","")} for d, _ in dense[:3]],
            "bm25_preview": [{"title": d.metadata.get("title",""), "component": d.metadata.get("component","")} for d, _ in sparse[:3]],
        })
        return fused, debug
