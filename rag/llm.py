from __future__ import annotations
from rag.config import settings

def get_llm():
    from langchain_openai import ChatOpenAI
    kwargs = {"model": settings.openai_model, "temperature": settings.llm_temperature}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return ChatOpenAI(api_key=settings.openai_api_key, **kwargs)
