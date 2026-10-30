"""Pattern 09: permission-aware retrieval.

Access control is applied *before* ranking: a document the
user may not read is never scored, never placed in the prompt
and never cited. Stale or unreviewed documents are excluded by
policy, and the answer must cite the documents it used.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from adp.core.llm import LLM, Message

WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    groups: frozenset[str]  # who may read it
    reviewed: bool = True
    age_days: int = 0


@dataclass(frozen=True)
class User:
    id: str
    groups: frozenset[str]


@dataclass
class Retrieval:
    docs: list[Doc]
    scores: list[float]
    filtered_out: dict[str, str] = field(default_factory=dict)


class PermissionAwareRetriever:
    def __init__(self, docs: list[Doc],
                 max_age_days: int = 365) -> None:
        self.docs = docs
        self.max_age = max_age_days
        n = len(docs)
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(tokens(d.text)))
        self.idf = {t: math.log((n + 1) / (c + 0.5))
                    for t, c in df.items()}

    def allowed(self, user: User, doc: Doc) -> str | None:
        """Return a reason if the doc must be excluded."""
        if not (user.groups & doc.groups):
            return "no_access"
        if not doc.reviewed:
            return "unreviewed"
        if doc.age_days > self.max_age:
            return "stale"
        return None

    def search(self, user: User, query: str,
               k: int = 3) -> Retrieval:
        q = tokens(query)
        out: dict[str, str] = {}
        scored: list[tuple[float, Doc]] = []
        for d in self.docs:
            reason = self.allowed(user, d)
            if reason:
                out[d.id] = reason
                continue
            tf = Counter(tokens(d.text))
            s = sum(tf[t] * self.idf.get(t, 0.0) for t in q)
            if s > 0:
                scored.append((s, d))
        scored.sort(key=lambda x: (-x[0], x[1].id))
        top = scored[:k]
        return Retrieval([d for _, d in top],
                         [round(s, 3) for s, _ in top], out)


ANSWER_PROMPT = (
    "Answer only from the sources. Cite each fact as "
    "[doc:ID]. If the sources do not answer, say so."
)


def answer(llm: LLM, retriever: PermissionAwareRetriever,
           user: User, question: str) -> tuple[str, list[str]]:
    r = retriever.search(user, question)
    if not r.docs:
        return "I don't have a source I can share for that.", []
    context = "\n".join(f"[doc:{d.id}] {d.text}"
                        for d in r.docs)
    resp = llm.complete([
        Message("system", ANSWER_PROMPT),
        Message("user", f"Sources:\n{context}\n\nQ: {question}"),
    ])
    cited = re.findall(r"\[doc:([\w-]+)\]", resp.text)
    allowed_ids = {d.id for d in r.docs}
    if not cited or not set(cited) <= allowed_ids:
        return "I couldn't produce a sourced answer.", []
    return resp.text, sorted(set(cited))
