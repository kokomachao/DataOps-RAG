from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup
from tqdm import tqdm

# Optional: use pandas for speed if installed
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

@dataclass
class QARecord:
    qid: int
    title: str
    question: str
    answer: str
    tags: List[str]
    component: str
    score: int
    accepted: bool

def _html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for pre in soup.find_all("pre"):
        pre_text = pre.get_text("\n", strip=False)
        pre.clear()
        pre.append("\n```text\n" + pre_text + "\n```\n")
    text = soup.get_text("\n")
    text = "\n".join([line.rstrip() for line in text.splitlines()])
    text = "\n".join([line for line in text.splitlines() if line.strip() != ""])
    return text.strip()

def _choose_component(tags: List[str], components: List[str]) -> Optional[str]:
    low = [t.lower() for t in tags]
    for c in components:
        cl = c.lower()
        for t in low:
            if cl in t or (cl == "spark" and "apache-spark" in t):
                return c
    return None

def _load_tags(tags_csv: str) -> Dict[int, List[str]]:
    """Tags.csv: Id=QuestionId, Tag=tag string; one question may have multiple rows."""
    tags_map: Dict[int, List[str]] = {}

    if pd is not None:
        for chunk in pd.read_csv(tags_csv, chunksize=1_000_000):
            if "Id" not in chunk.columns or "Tag" not in chunk.columns:
                raise ValueError("Tags.csv must have columns: Id, Tag")
            for qid, tag in zip(chunk["Id"].tolist(), chunk["Tag"].tolist()):
                if pd.isna(tag) or pd.isna(qid):
                    continue
                qid_i = int(qid)
                tags_map.setdefault(qid_i, []).append(str(tag))
        return tags_map

    with open(tags_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "Id" not in reader.fieldnames or "Tag" not in reader.fieldnames:
            raise ValueError("Tags.csv must have columns: Id, Tag")
        for row in reader:
            qid = row.get("Id")
            tag = row.get("Tag")
            if not qid or not tag:
                continue
            qid_i = int(float(qid))
            tags_map.setdefault(qid_i, []).append(tag)
    return tags_map

def build_dataset_from_csv_tables(
    questions_csv: str,
    answers_csv: str,
    tags_csv: str,
    out_jsonl: str,
    components: List[str],
    min_q_score: int = -5,
    min_a_score: int = -5,
    max_questions: Optional[int] = None,
) -> int:
    """
    Kaggle StackSample CSV version:
      - Questions.csv: Id, Title, Body, Score...
      - Answers.csv: ParentId -> Questions.Id
      - Tags.csv: (Id=QuestionId, Tag)

    Output JSONL per question with one chosen answer:
      - Choose answer with max score (Kaggle CSV has no AcceptedAnswerId).

    Designed to stream big files:
      - Load Tags.csv in memory (smallest table ~65MB)
      - Stream Questions/Answers in chunks (pandas if available, else csv)
    """
    tags_map = _load_tags(tags_csv)

    # Candidate questions: by component mapping from tags
    candidates: Dict[int, Tuple[str, List[str]]] = {}
    for qid, tags in tags_map.items():
        comp = _choose_component(tags, components)
        if comp is not None:
            candidates[qid] = (comp, tags)

    if max_questions is not None:
        keep_ids = sorted(list(candidates.keys()))[:max_questions]
        candidates = {qid: candidates[qid] for qid in keep_ids}

    cand_ids: Set[int] = set(candidates.keys())

    # Read Questions.csv (filtered)
    questions: Dict[int, Dict] = {}
    if pd is not None:
        for chunk in tqdm(pd.read_csv(questions_csv, chunksize=200_000), desc="Reading Questions.csv", unit="chunk"):
            if "Id" not in chunk.columns:
                raise ValueError("Questions.csv must have column: Id")
            chunk = chunk[chunk["Id"].isin(cand_ids)]
            for _, r in chunk.iterrows():
                qid = int(r["Id"])
                score = 0 if pd.isna(r.get("Score")) else int(r.get("Score", 0))
                if score < min_q_score:
                    continue
                title = "" if pd.isna(r.get("Title")) else str(r.get("Title"))
                body = "" if pd.isna(r.get("Body")) else str(r.get("Body"))
                questions[qid] = {"title": title, "question": _html_to_text(body), "q_score": score}
    else:
        with open(questions_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "Id" not in reader.fieldnames:
                raise ValueError("Questions.csv must have column: Id")
            for row in tqdm(reader, desc="Reading Questions.csv", unit="row"):
                try:
                    qid = int(float(row.get("Id", "0")))
                except Exception:
                    continue
                if qid not in cand_ids:
                    continue
                try:
                    score = int(float(row.get("Score", "0") or 0))
                except Exception:
                    score = 0
                if score < min_q_score:
                    continue
                questions[qid] = {
                    "title": row.get("Title", "") or "",
                    "question": _html_to_text(row.get("Body", "") or ""),
                    "q_score": score,
                }

    # Stream Answers.csv and keep best answer per ParentId
    best_answer: Dict[int, Tuple[int, str]] = {}
    if pd is not None:
        for chunk in tqdm(pd.read_csv(answers_csv, chunksize=300_000), desc="Reading Answers.csv", unit="chunk"):
            if "ParentId" not in chunk.columns:
                raise ValueError("Answers.csv must have column: ParentId")
            chunk = chunk[chunk["ParentId"].isin(cand_ids)]
            for _, r in chunk.iterrows():
                pid = int(r["ParentId"])
                a_score = 0 if pd.isna(r.get("Score")) else int(r.get("Score", 0))
                if a_score < min_a_score:
                    continue
                body = "" if pd.isna(r.get("Body")) else str(r.get("Body"))
                body_text = _html_to_text(body)
                prev = best_answer.get(pid)
                if prev is None or a_score > prev[0]:
                    best_answer[pid] = (a_score, body_text)
    else:
        with open(answers_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if "ParentId" not in reader.fieldnames:
                raise ValueError("Answers.csv must have column: ParentId")
            for row in tqdm(reader, desc="Reading Answers.csv", unit="row"):
                try:
                    pid = int(float(row.get("ParentId", "0")))
                except Exception:
                    continue
                if pid not in cand_ids:
                    continue
                try:
                    a_score = int(float(row.get("Score", "0") or 0))
                except Exception:
                    a_score = 0
                if a_score < min_a_score:
                    continue
                body_text = _html_to_text(row.get("Body", "") or "")
                prev = best_answer.get(pid)
                if prev is None or a_score > prev[0]:
                    best_answer[pid] = (a_score, body_text)

    # Emit JSONL
    written = 0
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for qid, (comp, tags) in tqdm(list(candidates.items()), desc="Writing JSONL", unit="q"):
            q = questions.get(qid)
            a = best_answer.get(qid)
            if not q or not a:
                continue
            rec = QARecord(
                qid=qid,
                title=q["title"],
                question=q["question"],
                answer=a[1],
                tags=tags,
                component=comp,
                score=int(q.get("q_score", 0)) + int(a[0]),
                accepted=False,
            )
            out.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
            written += 1

    return written
