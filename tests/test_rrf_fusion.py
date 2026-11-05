"""
Unit Test Suite for Phase 5: Reciprocal Rank Fusion (RRF).

Validates:
1. Two ranked lists containing completely different chunks.
2. Same chunk appearing in both lists.
3. Correct addition of reciprocal rank contributions (formula accuracy).
4. Correct ranking based on fused scores.
5. Duplicate chunk IDs within an individual list.
6. Missing or invalid chunk IDs.
7. Empty FAISS result list.
8. Empty BM25 result list.
9. Both lists empty.
10. Configurable RRF constant.
11. Original document metadata remains unchanged.
12. Deterministic ordering when scores are tied.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_doc_chat.src.document_chat.rrf_fusion import (
    RRFResult,
    reciprocal_rank_fusion,
    rrf_fuse_documents,
)


def make_doc(chunk_id: str, text: str = "sample text", extra_meta: dict | None = None) -> Document:
    """Helper to create a mock Document with stable metadata."""
    metadata = {"chunk_id": chunk_id, "source": "test_doc.txt"}
    if extra_meta:
        metadata.update(extra_meta)
    return Document(page_content=text, metadata=metadata)


def test_01_two_ranked_lists_completely_different_chunks():
    """1. Two ranked lists containing completely different chunks."""
    list1 = [make_doc("chunk_a"), make_doc("chunk_b")]
    list2 = [make_doc("chunk_c"), make_doc("chunk_d")]

    results = reciprocal_rank_fusion([list1, list2], k=60)

    # 4 distinct chunks should all be present
    assert len(results) == 4
    retrieved_cids = [r.document.metadata["chunk_id"] for r in results]
    assert set(retrieved_cids) == {"chunk_a", "chunk_b", "chunk_c", "chunk_d"}

    # Rank 1 chunks (a and c) score 1/(60+1) = 1/61
    # Rank 2 chunks (b and d) score 1/(60+2) = 1/62
    score_rank1 = 1.0 / 61.0
    score_rank2 = 1.0 / 62.0

    scores_by_cid = {r.document.metadata["chunk_id"]: r.score for r in results}
    assert math.isclose(scores_by_cid["chunk_a"], score_rank1, rel_tol=1e-9)
    assert math.isclose(scores_by_cid["chunk_c"], score_rank1, rel_tol=1e-9)
    assert math.isclose(scores_by_cid["chunk_b"], score_rank2, rel_tol=1e-9)
    assert math.isclose(scores_by_cid["chunk_d"], score_rank2, rel_tol=1e-9)


def test_02_same_chunk_appearing_in_both_lists():
    """2. Same chunk appearing in both lists is deduplicated to a single output entry."""
    doc_shared = make_doc("chunk_shared", text="Shared chunk text")
    list1 = [doc_shared, make_doc("chunk_unique_1")]
    list2 = [make_doc("chunk_unique_2"), doc_shared]

    results = reciprocal_rank_fusion([list1, list2], k=60)

    # 3 total distinct chunks, shared chunk returned exactly once
    assert len(results) == 3
    retrieved_cids = [r.document.metadata["chunk_id"] for r in results]
    assert retrieved_cids.count("chunk_shared") == 1
    assert set(retrieved_cids) == {"chunk_shared", "chunk_unique_1", "chunk_unique_2"}


def test_03_correct_addition_of_reciprocal_rank_contributions():
    """3. Verify mathematical formula: RRF(d) = sum_{r in R} 1 / (k + rank_r(d))."""
    # chunk_shared is at rank 1 in list 1, rank 2 in list 2
    shared = make_doc("chunk_shared")
    unique_1 = make_doc("chunk_unique_1")
    unique_2 = make_doc("chunk_unique_2")

    list1 = [shared, unique_1]        # shared at rank 1, unique_1 at rank 2
    list2 = [unique_2, shared]        # unique_2 at rank 1, shared at rank 2

    results = reciprocal_rank_fusion([list1, list2], k=60)
    scores_by_cid = {r.document.metadata["chunk_id"]: r.score for r in results}

    # chunk_shared: 1/(60+1) + 1/(60+2)
    expected_shared_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert math.isclose(scores_by_cid["chunk_shared"], expected_shared_score, rel_tol=1e-9)

    # unique_1: 1/(60+2)
    expected_u1_score = 1.0 / 62.0
    assert math.isclose(scores_by_cid["chunk_unique_1"], expected_u1_score, rel_tol=1e-9)

    # unique_2: 1/(60+1)
    expected_u2_score = 1.0 / 61.0
    assert math.isclose(scores_by_cid["chunk_unique_2"], expected_u2_score, rel_tol=1e-9)


def test_04_correct_ranking_based_on_fused_scores():
    """4. Correct descending ranking based on fused scores."""
    chunk_shared = make_doc("chunk_shared")
    chunk_top_sparse = make_doc("chunk_top_sparse")
    chunk_low_dense = make_doc("chunk_low_dense")

    list_dense = [chunk_shared, chunk_low_dense]       # ranks: 1, 2
    list_sparse = [chunk_top_sparse, chunk_shared]     # ranks: 1, 2

    results = reciprocal_rank_fusion([list_dense, list_sparse], k=60)

    # chunk_shared has combined rank 1+2 score (approx 0.0325) -> Must be Rank 1
    assert results[0].document.metadata["chunk_id"] == "chunk_shared"
    assert results[0].rank == 1

    # Verify strictly descending scores
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score
        assert results[i].rank == i + 1


def test_05_duplicate_chunk_ids_within_individual_list():
    """5. Duplicate chunk IDs within an individual list count only the best occurrence."""
    doc_a = make_doc("chunk_a", text="original")
    doc_a_dup = make_doc("chunk_a", text="duplicate occurrence")
    doc_b = make_doc("chunk_b")

    # list contains chunk_a at rank 1 and again at rank 3
    list_with_dups = [doc_a, doc_b, doc_a_dup]

    results = reciprocal_rank_fusion([list_with_dups], k=60)

    assert len(results) == 2
    assert results[0].document.metadata["chunk_id"] == "chunk_a"
    # Score should be strictly 1/(60+1), not (1/61 + 1/63)
    assert math.isclose(results[0].score, 1.0 / 61.0, rel_tol=1e-9)
    # The Document object preserved must be the first one encountered
    assert results[0].document.page_content == "original"


def test_06_missing_or_invalid_chunk_ids():
    """6. Missing or invalid chunk IDs are rejected explicitly."""
    # Missing chunk_id in metadata
    with pytest.raises(ValueError, match="missing or invalid 'chunk_id'"):
        doc_no_id = Document(page_content="text", metadata={"source": "test.txt"})
        reciprocal_rank_fusion([[doc_no_id]])

    # Empty string chunk_id
    with pytest.raises(ValueError, match="missing or invalid 'chunk_id'"):
        doc_empty_id = Document(page_content="text", metadata={"chunk_id": "   "})
        reciprocal_rank_fusion([[doc_empty_id]])

    # None chunk_id
    with pytest.raises(ValueError, match="missing or invalid 'chunk_id'"):
        doc_none_id = Document(page_content="text", metadata={"chunk_id": None})
        reciprocal_rank_fusion([[doc_none_id]])

    # Non-string chunk_id
    with pytest.raises(ValueError, match="missing or invalid 'chunk_id'"):
        doc_int_id = Document(page_content="text", metadata={"chunk_id": 12345})
        reciprocal_rank_fusion([[doc_int_id]])

    # Non-Document object
    with pytest.raises(TypeError, match="Expected Document object"):
        reciprocal_rank_fusion([[{"chunk_id": "chunk_dict"}]])  # type: ignore


def test_07_empty_faiss_result_list():
    """7. Empty FAISS result list handled gracefully; scores derived from BM25 only."""
    dense_list = []
    sparse_list = [make_doc("chunk_b1"), make_doc("chunk_b2")]

    results = reciprocal_rank_fusion([dense_list, sparse_list], k=60)

    assert len(results) == 2
    assert results[0].document.metadata["chunk_id"] == "chunk_b1"
    assert math.isclose(results[0].score, 1.0 / 61.0, rel_tol=1e-9)
    assert results[1].document.metadata["chunk_id"] == "chunk_b2"
    assert math.isclose(results[1].score, 1.0 / 62.0, rel_tol=1e-9)


def test_08_empty_bm25_result_list():
    """8. Empty BM25 result list handled gracefully; scores derived from FAISS only."""
    dense_list = [make_doc("chunk_d1"), make_doc("chunk_d2")]
    sparse_list = []

    results = reciprocal_rank_fusion([dense_list, sparse_list], k=60)

    assert len(results) == 2
    assert results[0].document.metadata["chunk_id"] == "chunk_d1"
    assert math.isclose(results[0].score, 1.0 / 61.0, rel_tol=1e-9)
    assert results[1].document.metadata["chunk_id"] == "chunk_d2"
    assert math.isclose(results[1].score, 1.0 / 62.0, rel_tol=1e-9)


def test_09_both_lists_empty():
    """9. Both input lists empty returns an empty list without error."""
    assert reciprocal_rank_fusion([[], []], k=60) == []
    assert reciprocal_rank_fusion([], k=60) == []
    assert rrf_fuse_documents([[], []], k=60) == []


def test_10_configurable_rrf_constant():
    """10. RRF constant k is configurable, modifying score denominators accordingly."""
    doc = make_doc("chunk_test")

    # With default k=60: score = 1 / 61
    res_60 = reciprocal_rank_fusion([[doc]], k=60)
    assert math.isclose(res_60[0].score, 1.0 / 61.0, rel_tol=1e-9)

    # With k=10: score = 1 / 11
    res_10 = reciprocal_rank_fusion([[doc]], k=10)
    assert math.isclose(res_10[0].score, 1.0 / 11.0, rel_tol=1e-9)

    # With k=100: score = 1 / 101
    res_100 = reciprocal_rank_fusion([[doc]], k=100)
    assert math.isclose(res_100[0].score, 1.0 / 101.0, rel_tol=1e-9)

    # Invalid k <= 0 raises ValueError
    with pytest.raises(ValueError, match="smoothing parameter k must be positive"):
        reciprocal_rank_fusion([[doc]], k=0)

    with pytest.raises(ValueError, match="smoothing parameter k must be positive"):
        reciprocal_rank_fusion([[doc]], k=-10)


def test_11_original_document_metadata_remains_unchanged():
    """11. Original document metadata is not modified or polluted with RRF internal state."""
    original_meta = {
        "chunk_id": "chunk_meta_test",
        "source": "DOC-SEC-601.docx",
        "page": 4,
        "content_hash": "abc123hash",
    }
    doc = Document(page_content="Sensitive content", metadata=dict(original_meta))

    results = reciprocal_rank_fusion([[doc]], k=60)

    # Verify metadata content and keys remain identical
    assert doc.metadata == original_meta
    assert results[0].document.metadata == original_meta
    assert "rrf_score" not in results[0].document.metadata
    assert "score" not in results[0].document.metadata
    assert "rank" not in results[0].document.metadata


def test_12_deterministic_ordering_on_tied_scores():
    """12. Deterministic tie-breaking on chunk_id ensures stable ordering across runs."""
    doc_b = make_doc("chunk_bravo")
    doc_a = make_doc("chunk_alpha")
    doc_c = make_doc("chunk_charlie")

    # All three documents appear at rank 1 in their respective lists -> All have score 1/61
    list1 = [doc_b]
    list2 = [doc_a]
    list3 = [doc_c]

    results1 = reciprocal_rank_fusion([list1, list2, list3], k=60)
    cids1 = [r.document.metadata["chunk_id"] for r in results1]

    # Different list input order:
    results2 = reciprocal_rank_fusion([list3, list1, list2], k=60)
    cids2 = [r.document.metadata["chunk_id"] for r in results2]

    # In both cases, alphabetical tie-breaking on chunk_id guarantees: alpha, bravo, charlie
    assert cids1 == ["chunk_alpha", "chunk_bravo", "chunk_charlie"]
    assert cids2 == ["chunk_alpha", "chunk_bravo", "chunk_charlie"]
    assert cids1 == cids2

