from __future__ import annotations
import hashlib, json, re
from typing import Any

def sha1_json(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

def normalize_query(q: str) -> str:
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    return q
