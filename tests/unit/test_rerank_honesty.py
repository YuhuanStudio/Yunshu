"""a grounded rerank/scoring hunt found the subsystem CORRECT (dot/euclidean,
NaN-sort, fallback backbone, model-isolation all genuinely hold;
ranking sort/index/top_n correct). The one real issue was an honesty overclaim: the module
docstring advertised /v1/rerank as "cross-encoder reranking", but the implementation is a
BI-ENCODER (embeds query and each document separately, ranks by cosine) — there is no joint
(query, document) cross-encoder forward pass. Corrected the label so the code doesn't
misrepresent the method (the project's documented anti-overclaim discipline).
"""
from __future__ import annotations

import inspect

from yunshu_gateway.routers import scoring


def test_rerank_docstring_does_not_overclaim_cross_encoder():
    src = inspect.getsource(scoring)
    # the false "cross-encoder reranking" claim must be gone
    assert "cross-encoder reranking" not in src.lower()
    # and the honest method must be named
    assert "BI-ENCODER" in src or "bi-encoder" in src.lower()


def test_rerank_still_ranks_by_cosine_in_code():
    # guard that the description still matches the implementation (cosine similarity)
    src = inspect.getsource(scoring.create_rerank)
    assert '_compute_similarity(query_emb, doc_emb, "cosine")' in src
