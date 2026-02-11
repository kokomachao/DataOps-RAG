from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from lxml import etree
from tqdm import tqdm

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
    # keep code blocks with fences for better retrieval
    for pre in soup.find_all("pre"):
        pre_text = pre.get_text("\n", strip=False)
        pre.clear()
        pre.append("\n```text\n" + pre_text + "\n```\n")
    text = soup.get_text("\n")
    text = "\n".join([line.rstrip() for line in text.splitlines()])
    text = "\n".join([line for line in text.splitlines() if line.strip() != ""])
    return text.strip()

def _parse_tags(tag_field: str) -> List[str]:
    if not tag_field:
        return []
    return [t for t in tag_field.replace("><", " ").replace("<", "").replace(">", "").split() if t]

def _choose_component(tags: List[str], components: List[str]) -> Optional[str]:
    low = [t.lower() for t in tags]
    for c in components:
        cl = c.lower()
        for t in low:
            if cl in t or (cl == "spark" and "apache-spark" in t):
                return c
    return None

def build_dataset_from_posts_xml(
    posts_xml: str,
    out_jsonl: str,
    components: List[str],
    max_questions: int = 200_000,
    min_score: int = -5,
) -> int:
    # First pass: collect questions and answers (grouped)
    questions: Dict[int, Dict] = {}
    answers_by_parent: Dict[int, List[Dict]] = {}

    context = etree.iterparse(posts_xml, events=("end",), tag="row", recover=True, huge_tree=True)
    pbar = tqdm(desc="Parsing Posts.xml", unit="row")

    for _, elem in context:
        pbar.update(1)
        try:
            post_type = int(elem.get("PostTypeId", "0"))
        except ValueError:
            elem.clear()
            continue

        if post_type == 1:  # question
            qid = int(elem.get("Id"))
            tags = _parse_tags(elem.get("Tags", ""))
            comp = _choose_component(tags, components)
            if comp is None:
                elem.clear()
                continue
            score = int(elem.get("Score", "0"))
            if score < min_score:
                elem.clear()
                continue

            questions[qid] = {
                "Id": qid,
                "Title": elem.get("Title", "") or "",
                "Body": elem.get("Body", "") or "",
                "Tags": tags,
                "Component": comp,
                "Score": score,
                "AcceptedAnswerId": int(elem.get("AcceptedAnswerId")) if elem.get("AcceptedAnswerId") else None,
            }
        elif post_type == 2:  # answer
            parent_id = elem.get("ParentId")
            if parent_id:
                pid = int(parent_id)
                answers_by_parent.setdefault(pid, []).append({
                    "Id": int(elem.get("Id")),
                    "Body": elem.get("Body", "") or "",
                    "Score": int(elem.get("Score", "0")),
                })

        elem.clear()

        # soft stop for quick experiments
        if len(questions) >= max_questions and pbar.n > 1_000_000:
            break

    pbar.close()

    written = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for qid, q in tqdm(list(questions.items()), desc="Building QA records", unit="q"):
            cand = answers_by_parent.get(qid, [])
            if not cand:
                continue
            accepted_id = q.get("AcceptedAnswerId")
            chosen = None
            if accepted_id is not None:
                for a in cand:
                    if a["Id"] == accepted_id:
                        chosen = a
                        break
            if chosen is None:
                chosen = max(cand, key=lambda x: x.get("Score", 0))

            rec = QARecord(
                qid=qid,
                title=q["Title"],
                question=_html_to_text(q["Body"]),
                answer=_html_to_text(chosen["Body"]),
                tags=q["Tags"],
                component=q["Component"],
                score=int(q.get("Score", 0)) + int(chosen.get("Score", 0)),
                accepted=(chosen["Id"] == accepted_id) if accepted_id is not None else False,
            )
            f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
            written += 1

    return written
