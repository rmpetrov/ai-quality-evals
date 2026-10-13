"""Real lexical BM25 retrieval. Input is documents plus a query, never gold labels."""

import math
import re
from collections import Counter
from collections.abc import Iterable

from ai_quality_evals.models import Document, RetrievalHit, RetrievalResult


def tokenize(text: str) -> tuple[str, ...]:
    """Case-folded ASCII word/number tokens; no stemming or query expansion."""
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


class BM25Retriever:
    def __init__(self, documents: Iterable[Document], *, k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        ordered = sorted(documents, key=lambda document: document.document_id)
        if not ordered or len({doc.document_id for doc in ordered}) != len(ordered):
            raise ValueError("index requires nonempty documents with unique IDs")
        # Index body (including H1 title) once. Metadata, IDs, and labels are not indexed.
        self._terms = {doc.document_id: Counter(tokenize(doc.content)) for doc in ordered}
        self._lengths = {doc_id: sum(terms.values()) for doc_id, terms in self._terms.items()}
        self._average_length = sum(self._lengths.values()) / len(ordered)
        if self._average_length == 0:
            raise ValueError("index requires at least one token")
        frequency: Counter[str] = Counter()
        for terms in self._terms.values():
            frequency.update(terms.keys())
        self._idf = {
            term: math.log1p((len(ordered) - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }

    def retrieve(self, query: str, *, top_k: int = 3) -> RetrievalResult:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        # Each distinct query term contributes once, in sorted order for hash-seed stability.
        terms = sorted(set(tokenize(query)))
        hits: list[RetrievalHit] = []
        for doc_id, frequencies in self._terms.items():
            normalization = self.k1 * (
                1 - self.b + self.b * self._lengths[doc_id] / self._average_length
            )
            score = math.fsum(
                self._idf[term]
                * frequencies[term]
                * (self.k1 + 1)
                / (frequencies[term] + normalization)
                for term in terms
                if frequencies[term] > 0
            )
            if score > 0:
                hits.append(RetrievalHit(doc_id, score))
        # Sort on full precision, then round only the public score.
        hits.sort(key=lambda hit: (-hit.score, hit.document_id))
        return RetrievalResult(
            query,
            top_k,
            tuple(RetrievalHit(hit.document_id, round(hit.score, 12)) for hit in hits[:top_k]),
        )
