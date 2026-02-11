from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9_\.\#/-]+|[\u4e00-\u9fff]+")

def default_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

@dataclass
class BM25Index:
    docs: List[Document]
    tokenized_corpus: List[List[str]]
    bm25: BM25Okapi
    by_component: Dict[str, List[int]]
    by_tag: Dict[str, List[int]]

class PersistentBM25:
    def __init__(self, index: BM25Index):
        self.index = index

    @classmethod
    def build(cls, docs: List[Document], tokenize_fn=default_tokenize) -> "PersistentBM25":
        tokenized = [tokenize_fn(d.page_content) for d in docs]
        bm25 = BM25Okapi(tokenized)

        by_component: Dict[str, List[int]] = {}
        by_tag: Dict[str, List[int]] = {}
        for i, d in enumerate(docs):
            comp = str(d.metadata.get("component", "")).lower()
            if comp:
                by_component.setdefault(comp, []).append(i)
            for t in (d.metadata.get("tags") or []):
                tl = str(t).lower()
                by_tag.setdefault(tl, []).append(i)

        return cls(BM25Index(docs=docs, tokenized_corpus=tokenized, bm25=bm25,
                            by_component=by_component, by_tag=by_tag))

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self.index, f)

    @classmethod
    def load(cls, path: str) -> "PersistentBM25":
        with open(path, "rb") as f:
            idx = pickle.load(f)
        return cls(idx)

    def _subset_indices(self, components: Optional[List[str]], tags: Optional[List[str]]) -> Optional[List[int]]:
        sets: List[set] = []
        if components:
            cset = set()
            for c in components:
                cset |= set(self.index.by_component.get(c.lower(), []))
            sets.append(cset)
        if tags:
            tset = set()
            for t in tags:
                tset |= set(self.index.by_tag.get(t.lower(), []))
            sets.append(tset)
        if not sets:
            return None
        inter = sets[0]
        for s in sets[1:]:
            inter &= s
        return sorted(inter)

    def search(
        self,
        query: str,
        k: int = 8,
        components: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        tokenize_fn=default_tokenize,
    ) -> List[Tuple[Document, float]]:
        qtok = tokenize_fn(query)
        subset = self._subset_indices(components, tags)
        scores = self.index.bm25.get_scores(qtok)

        if subset is None:
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        else:
            top_idx = sorted(subset, key=lambda i: scores[i], reverse=True)[:k]

        return [(self.index.docs[i], float(scores[i])) for i in top_idx if scores[i] > 0]
