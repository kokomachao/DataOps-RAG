from __future__ import annotations
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

def _get(name: str, default: str | None = None) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else (default or "")

@dataclass(frozen=True)
class Settings:
    # LLM
    openai_api_key: str = _get("OPENAI_API_KEY", "")
    openai_base_url: str = _get("OPENAI_BASE_URL", "")
    openai_model: str = _get("OPENAI_MODEL", "gpt-4o-mini")
    llm_temperature: float = float(_get("LLM_TEMPERATURE", "0.2"))

    # Embeddings
    embedding_model: str = _get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    embedding_device: str = _get("EMBEDDING_DEVICE", "cpu")

    # Storage
    storage_dir: str = _get("STORAGE_DIR", "storage")
    vector_backend: str = _get("VECTOR_BACKEND", "faiss").lower()  # faiss|milvus

    # Milvus
    milvus_uri: str = _get("MILVUS_URI", "http://localhost:19530")
    milvus_collection: str = _get("MILVUS_COLLECTION", "stack_rag")

    # Retrieval
    top_k: int = int(_get("TOP_K", "8"))
    fetch_k: int = int(_get("FETCH_K", "40"))
    rrf_c: int = int(_get("RRF_C", "60"))

    # Cache
    cache_dir: str = _get("CACHE_DIR", "storage/cache")
    cache_ttl_seconds: int = int(_get("CACHE_TTL_SECONDS", "3600"))

settings = Settings()
