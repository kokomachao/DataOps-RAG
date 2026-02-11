from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from rag.config import settings
from rag.service import RAGService

app = FastAPI(title="Data Platform RAG Troubleshooting Assistant", version="1.0.0")

svc: Optional[RAGService] = None

class AskRequest(BaseModel):
    question: str = Field(..., description="User question")
    components: Optional[List[str]] = Field(default=None, description="Filter by component e.g. ['spark']")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags e.g. ['apache-spark']")
    top_k: Optional[int] = Field(default=None, description="Top k after fusion")
    fetch_k: Optional[int] = Field(default=None, description="Candidate k for dense retrieval before filtering")
    debug: bool = Field(default=True, description="Return debug details")

class AskResult(BaseModel):
    answer_md: str
    sop: Dict[str, Any]
    sources: List[Dict[str, Any]]
    debug: Dict[str, Any]

@app.on_event("startup")
def _startup():
    global svc
    svc = RAGService(settings.storage_dir)

@app.get("/health")
def health():
    return {"ok": True, "backend": settings.vector_backend, "collection": settings.milvus_collection}

@app.post("/ask", response_model=AskResult)
def ask(req: AskRequest):
    assert svc is not None
    resp = svc.ask(
        question=req.question,
        components=req.components,
        tags=req.tags,
        top_k=req.top_k,
        fetch_k=req.fetch_k,
        debug=req.debug,
    )
    return AskResult(answer_md=resp.answer_md, sop=resp.sop, sources=resp.sources, debug=resp.debug)
