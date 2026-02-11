from __future__ import annotations
from typing import List
from langchain_core.documents import Document
from rag.config import settings
from rag.embeddings import get_embeddings

def build_milvus(docs: List[Document]):
    from langchain_milvus import Milvus
    embeddings = get_embeddings()
    return Milvus.from_documents(
        docs,
        embedding=embeddings,
        collection_name=settings.milvus_collection,
        connection_args={"uri": settings.milvus_uri},
        drop_old=True,
    )

def load_milvus():
    from langchain_milvus import Milvus
    embeddings = get_embeddings()
    return Milvus(
        embedding_function=embeddings,
        collection_name=settings.milvus_collection,
        connection_args={"uri": settings.milvus_uri},
    )
