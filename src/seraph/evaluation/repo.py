"""Small, versioned relevance sets for evaluating code search on Git repositories."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def is_relevant(result: Mapping, expected: Sequence[Mapping]) -> bool:
    """Match a result to a labeled path, optionally narrowed to exact symbols."""
    for target in expected:
        if result.get("path") != target["path"]:
            continue
        symbols = target.get("symbols", [])
        if not symbols or result.get("symbol") in symbols:
            return True
    return False


def score_case(results: Sequence[Mapping], expected: Sequence[Mapping], k: int = 5) -> dict:
    """Score one query with binary relevance; missing evidence receives zero."""
    relevance = [is_relevant(result, expected) for result in results]
    first = next((i for i, hit in enumerate(relevance, 1) if hit), None)
    dcg = sum(1 / math.log2(i + 2) for i, hit in enumerate(relevance[:k]) if hit)
    ideal_count = min(len(expected), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_count))
    returned = relevance[:k]
    return {
        "hit@1": float(bool(relevance and relevance[0])),
        f"hit@{k}": float(first is not None and first <= k),
        "mrr": 1 / first if first is not None else 0.0,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        f"noise@{k}": (sum(not hit for hit in returned) / len(returned)) if returned else 0.0,
        "first_relevant_rank": first,
    }


def aggregate(per_query: Mapping[str, Mapping]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = [key for key in next(iter(per_query.values())) if key != "first_relevant_rank"]
    return {key: sum(float(row[key]) for row in per_query.values()) / len(per_query) for key in keys}
