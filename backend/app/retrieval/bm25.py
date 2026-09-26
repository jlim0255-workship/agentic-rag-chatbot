"""Okapi BM25 (real BM25, so it may be called that; PRD 9). Standard library only."""
from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Score every doc against the query. Built per call: the corpus is ~100 chunks, so this
    takes milliseconds and is never stale after re-ingestion."""
    tfs = [Counter(tokenize(d)) for d in docs]
    if not tfs:
        return []
    lengths = [sum(tf.values()) for tf in tfs]
    avg_len = sum(lengths) / len(lengths)
    n = len(tfs)
    df = Counter(term for tf in tfs for term in tf)
    # "+1" inside the log keeps IDF positive for terms that appear in most docs.
    idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
    terms = tokenize(query)
    return [
        sum(idf[t] * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * length / avg_len))
            for t in terms if t in tf)
        for tf, length in zip(tfs, lengths)
    ]
