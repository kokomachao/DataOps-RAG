from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.llm import get_llm

SOP_SCHEMA = {
  "type": "object",
  "properties": {
    "summary": {"type": "string"},
    "possible_causes": {"type": "array", "items": {"type": "string"}},
    "checks": {"type": "array", "items": {"type": "string"}},
    "step_by_step_sop": {"type": "array", "items": {"type": "string"}},
    "mitigations": {"type": "array", "items": {"type": "string"}},
    "rollback_plan": {"type": "array", "items": {"type": "string"}},
    "when_to_escalate": {"type": "array", "items": {"type": "string"}},
    "references": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["summary","possible_causes","checks","step_by_step_sop","mitigations","rollback_plan","when_to_escalate","references"]
}

def _format_context(docs: List[Document], max_chars: int = 12000) -> str:
    parts, total = [], 0
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        header = f"## Source {i} | {md.get('component')} | score={md.get('score')} | title={md.get('title')}\n"
        chunk = header + d.page_content + "\n\n"
        total += len(chunk)
        if total > max_chars:
            break
        parts.append(chunk)
    return "".join(parts)

def build_sop_answer(question: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
    llm = get_llm()
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是资深数据平台排障助手（Spark/Flink/Kafka/Hadoop/Hive）。"
         "你必须基于给定的检索上下文回答，不要胡编；如果上下文不足，请明确说明缺口与下一步获取信息的方法。"
         "输出必须是严格 JSON（不要 markdown 代码块）。"),
        ("human",
         "用户问题：\n{question}\n\n"
         "检索上下文：\n{context}\n\n"
         "请生成一个标准化排障结果 JSON，字段遵循以下 JSON Schema（仅用作结构参考，不要输出 schema）：\n"
         f"{json.dumps(SOP_SCHEMA, ensure_ascii=False)}\n\n"
         "要求：\n"
         "1) checks/步骤尽量可操作（命令/配置项/日志路径）\n"
         "2) references 里给出你引用的 Source 编号（如 'Source 2'）与其标题\n")
    ])

    context = _format_context(retrieved_docs)
    out = llm.invoke(prompt.invoke({"question": question, "context": context}))

    try:
        return json.loads(out.content)
    except Exception:
        return parser.parse(out.content)
