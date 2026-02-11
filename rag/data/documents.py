from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class CorpusConfig:
    chunk_size: int = 900
    chunk_overlap: int = 150

def load_records(jsonl_path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def record_to_document(rec: Dict) -> Document:
    content = (
        f"[Component] {rec.get('component')}\n"
        f"[Tags] {', '.join(rec.get('tags', []))}\n"
        f"[Title] {rec.get('title')}\n\n"
        f"Question:\n{rec.get('question')}\n\n"
        f"Answer:\n{rec.get('answer')}\n"
    )
    metadata = {
        "qid": rec.get("qid"),
        "component": rec.get("component"),
        "tags": rec.get("tags", []),
        "score": rec.get("score", 0),
        "accepted": rec.get("accepted", False),
        "title": rec.get("title", ""),
    }
    return Document(page_content=content, metadata=metadata)

def build_documents(jsonl_path: str) -> List[Document]:
    return [record_to_document(r) for r in load_records(jsonl_path)]

def chunk_documents(docs: List[Document], cfg: CorpusConfig) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: List[Document] = []
    for d in docs:
        splitted = splitter.split_documents([d])
        for i, c in enumerate(splitted):
            c.metadata = dict(c.metadata)
            c.metadata["chunk_id"] = i
            chunks.append(c)
    return chunks
