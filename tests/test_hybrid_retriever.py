"""
Unit Test Suite for Phase 6: Hybrid Retrieval Pipeline.

Validates:
1. Both retrievers are called with the same user query.
2. FAISS retrieves no more than Top-10.
3. BM25 retrieves no more than Top-10.
4. Both ranked lists are passed to RRF.
5. Final output contains no more than five chunks.
6. A chunk appearing in both lists is returned only once.
7. Chunks appearing in both lists receive combined RRF scores.
8. A chunk ranked highly by both systems is ranked appropriately.
9. Empty FAISS results handled gracefully.
10. Empty BM25 results handled gracefully.
11. Both retrievers returning empty results (and empty query handling).
12. Original metadata is preserved without mutation.
13. Original document content is unchanged.
14. Invalid retrieval results produce useful, explicit errors.
15. Verification that no new index files or directories are created on disk.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, List, Optional

import pytest
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_doc_chat.src.document_chat.hybrid_retriever import HybridRetriever


class MockDenseRetriever:
    """Mock dense FAISS retriever recording queries without calling APIs."""
    def __init__(self, docs: Optional[List[Document]] = None):
        self.docs = docs if docs is not None else []
        self.queries: List[str] = []

    def invoke(self, query: str) -> List[Document]:
        self.queries.append(query)
        return list(self.docs)


class MockSparseRetriever:
    """Mock sparse BM25 retriever recording queries and returning mock documents."""
    def __init__(self, docs: Optional[List[Document]] = None):
        self.docs = docs if docs is not None else []
        self.queries: List[tuple[str, int]] = []

    def retrieve_documents(self, query: str, top_k: int = 10) -> List[Document]:
        self.queries.append((query, top_k))
        return list(self.docs)


def make_doc(cid: str, text: str = "content", meta: dict | None = None) -> Document:
    metadata = {"chunk_id": cid, "source": f"{cid}.txt", "page": 1}
    if meta:
        metadata.update(meta)
    return Document(page_content=text, metadata=metadata)


def test_01_both_retrievers_called_with_same_user_query():
    """1. Verify both retrievers receive the exact same user search query."""
    dense = MockDenseRetriever([make_doc("d1")])
    sparse = MockSparseRetriever([make_doc("s1")])
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse)

    query = "kubernetes horizontal pod autoscaler metrics"
    retriever.retrieve(query)

    assert dense.queries == [query]
    assert len(sparse.queries) == 1
    assert sparse.queries[0][0] == query


def test_02_faiss_retrieves_no_more_than_top_10():
    """2. Dense retriever output is bounded to no more than Top-10 candidates."""
    # 15 candidate documents returned by FAISS mock
    dense_15 = [make_doc(f"dense_{i}") for i in range(15)]
    dense = MockDenseRetriever(dense_15)
    sparse = MockSparseRetriever([])

    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse, dense_k=10)
    fetched_dense = retriever._fetch_dense("test query")

    assert len(fetched_dense) == 10
    assert [d.metadata["chunk_id"] for d in fetched_dense] == [f"dense_{i}" for i in range(10)]


def test_03_bm25_retrieves_no_more_than_top_10():
    """3. Sparse retriever output is bounded to no more than Top-10 candidates."""
    # 15 candidate documents returned by BM25 mock
    sparse_15 = [make_doc(f"sparse_{i}") for i in range(15)]
    dense = MockDenseRetriever([])
    sparse = MockSparseRetriever(sparse_15)

    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse, sparse_k=10)
    fetched_sparse = retriever._fetch_sparse("test query")

    assert len(fetched_sparse) == 10
    assert [d.metadata["chunk_id"] for d in fetched_sparse] == [f"sparse_{i}" for i in range(10)]


def test_04_both_ranked_lists_passed_to_rrf():
    """4. Both dense and sparse result lists are passed to RRF fusion."""
    dense_docs = [make_doc("d1"), make_doc("d2")]
    sparse_docs = [make_doc("s1"), make_doc("s2")]

    dense = MockDenseRetriever(dense_docs)
    sparse = MockSparseRetriever(sparse_docs)
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse, final_k=5)

    fused_results = retriever.retrieve_with_scores("query")
    fused_cids = {r.document.metadata["chunk_id"] for r in fused_results}

    # All chunks from both lists should be accounted for
    assert fused_cids == {"d1", "d2", "s1", "s2"}


def test_05_final_output_contains_no_more_than_five_chunks():
    """5. Final hybrid output contains at most 5 chunks (final_k=5)."""
    dense_10 = [make_doc(f"d_{i}") for i in range(10)]
    sparse_10 = [make_doc(f"s_{i}") for i in range(10)]

    dense = MockDenseRetriever(dense_10)
    sparse = MockSparseRetriever(sparse_10)
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse, final_k=5)

    docs = retriever.retrieve("query")
    assert len(docs) == 5


def test_06_chunk_appearing_in_both_lists_returned_only_once():
    """6. A chunk appearing in both dense and sparse lists is deduplicated."""
    shared_chunk = make_doc("shared_chunk", text="Common knowledge")
    dense_docs = [shared_chunk, make_doc("d_unique")]
    sparse_docs = [make_doc("s_unique"), shared_chunk]

    dense = MockDenseRetriever(dense_docs)
    sparse = MockSparseRetriever(sparse_docs)
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse)

    docs = retriever.retrieve("query")
    cids = [d.metadata["chunk_id"] for d in docs]

    assert cids.count("shared_chunk") == 1
    assert len(cids) == 3


def test_07_chunks_in_both_lists_receive_combined_rrf_scores():
    """7. Shared chunks accumulate RRF reciprocal scores from both systems."""
    shared = make_doc("shared_chunk")
    d_only = make_doc("dense_only")
    s_only = make_doc("sparse_only")

    # shared is rank 1 in dense (1/61), rank 2 in sparse (1/62)
    dense = MockDenseRetriever([shared, d_only])
    sparse = MockSparseRetriever([s_only, shared])
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse, rrf_k=60)

    fused = retriever.retrieve_with_scores("query")
    scores = {r.document.metadata["chunk_id"]: r.score for r in fused}

    expected_shared_score = (1.0 / 61.0) + (1.0 / 62.0)
    expected_s_only_score = 1.0 / 61.0
    expected_d_only_score = 1.0 / 62.0

    assert math.isclose(scores["shared_chunk"], expected_shared_score, rel_tol=1e-9)
    assert math.isclose(scores["sparse_only"], expected_s_only_score, rel_tol=1e-9)
    assert math.isclose(scores["dense_only"], expected_d_only_score, rel_tol=1e-9)


def test_08_chunk_ranked_highly_by_both_systems_ranks_top():
    """8. A chunk supported by both dense and sparse retrieval beats single-modality chunks."""
    shared = make_doc("shared_evidence")
    d_top = make_doc("dense_top")
    s_top = make_doc("sparse_top")

    # Dense: [d_top (rank 1), shared (rank 2)]
    # Sparse: [s_top (rank 1), shared (rank 2)]
    dense = MockDenseRetriever([d_top, shared])
    sparse = MockSparseRetriever([s_top, shared])
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse, rrf_k=60)

    fused = retriever.retrieve_with_scores("query")

    # shared has 1/62 + 1/62 = 2/62 ≈ 0.032258
    # d_top has 1/61 ≈ 0.016393
    # s_top has 1/61 ≈ 0.016393
    # Therefore shared MUST rank #1!
    assert fused[0].document.metadata["chunk_id"] == "shared_evidence"
    assert fused[0].rank == 1


def test_09_empty_faiss_results():
    """9. Empty FAISS results handled gracefully without breaking sparse retrieval."""
    dense = MockDenseRetriever([])
    sparse = MockSparseRetriever([make_doc("s1"), make_doc("s2")])
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse)

    docs = retriever.retrieve("query")
    assert len(docs) == 2
    assert [d.metadata["chunk_id"] for d in docs] == ["s1", "s2"]


def test_10_empty_bm25_results():
    """10. Empty BM25 results handled gracefully without breaking dense retrieval."""
    dense = MockDenseRetriever([make_doc("d1"), make_doc("d2")])
    sparse = MockSparseRetriever([])
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse)

    docs = retriever.retrieve("query")
    assert len(docs) == 2
    assert [d.metadata["chunk_id"] for d in docs] == ["d1", "d2"]


def test_11_both_retrievers_returning_empty_results_and_empty_query():
    """11. Both retrievers returning empty, or empty/whitespace query returns empty list."""
    dense = MockDenseRetriever([])
    sparse = MockSparseRetriever([])
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse)

    assert retriever.retrieve("some query") == []
    assert retriever.retrieve_with_scores("some query") == []
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_12_original_metadata_is_preserved():
    """12. Original Document metadata dictionary is preserved without alteration."""
    original_meta = {
        "chunk_id": "meta_chunk_99",
        "source": "DOC-NET-802.docx",
        "page": 7,
        "content_hash": "sha256_mock_hash",
        "custom_tag": "critical_network_spec",
    }
    doc = Document(page_content="eBGP peering configuration", metadata=dict(original_meta))

    dense = MockDenseRetriever([doc])
    sparse = MockSparseRetriever([])
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse)

    retrieved = retriever.retrieve("eBGP")[0]

    # Verify exact match of all metadata keys and values
    assert retrieved.metadata == original_meta
    assert "score" not in retrieved.metadata
    assert "rrf_score" not in retrieved.metadata


def test_13_original_document_content_is_unchanged():
    """13. Original page_content is preserved identically byte-for-byte."""
    special_text = (
        "Multi-line technical specifications:\n"
        "- Equation: E = mc^2\n"
        "- Acronym: CRC16 & SHA-256\n"
        "- Code snippet: `if (x < 10) return true;`\n"
        "- Quotes: 'single' and \"double\""
    )
    doc = Document(page_content=special_text, metadata={"chunk_id": "content_chunk_1"})

    dense = MockDenseRetriever([doc])
    sparse = MockSparseRetriever([])
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse)

    retrieved = retriever.retrieve("technical specs")[0]
    assert retrieved.page_content == special_text


def test_14_invalid_retrieval_results_produce_useful_errors():
    """14. Invalid return types from retrievers produce explicit, actionable errors."""
    # Case A: Dense retriever returns non-list
    class BadDenseRetriever:
        def invoke(self, q):
            return "not a list"

    bad_retriever = HybridRetriever(BadDenseRetriever(), MockSparseRetriever([]))
    with pytest.raises(TypeError, match="expected list of Documents"):
        bad_retriever.retrieve("query")

    # Case B: Sparse retriever returns non-list
    class BadSparseRetriever:
        def retrieve_documents(self, q, top_k):
            return 12345

    bad_retriever_2 = HybridRetriever(MockDenseRetriever([]), BadSparseRetriever())
    with pytest.raises(TypeError, match="expected list of Documents"):
        bad_retriever_2.retrieve("query")

    # Case C: Retriever returns list containing non-Document items
    bad_retriever_3 = HybridRetriever(
        MockDenseRetriever(["string_not_doc"]),  # type: ignore
        MockSparseRetriever([]),
    )
    with pytest.raises(TypeError, match="Expected Document object"):
        bad_retriever_3.retrieve("query")

    # Case D: Document missing chunk_id
    bad_doc = Document(page_content="No ID", metadata={"source": "test.txt"})
    bad_retriever_4 = HybridRetriever(
        MockDenseRetriever([bad_doc]),
        MockSparseRetriever([]),
    )
    with pytest.raises(ValueError, match="missing or invalid 'chunk_id'"):
        bad_retriever_4.retrieve("query")


def test_15_no_second_index_created():
    """15. Verify that HybridRetriever reuses components without writing new indexes to disk."""
    import os
    faiss_mtime_before = os.path.getmtime(PROJECT_ROOT / "faiss_index" / "benchmark_70docs_v1" / "index.faiss")
    bm25_mtime_before = os.path.getmtime(PROJECT_ROOT / "data" / "bm25_index" / "bm25_index.pkl")

    dense = MockDenseRetriever([make_doc("c1")])
    sparse = MockSparseRetriever([make_doc("c2")])
    retriever = HybridRetriever(faiss_retriever=dense, bm25_retriever=sparse)

    # Execute multiple queries
    for q in ["query 1", "query 2", "query 3"]:
        retriever.retrieve(q)

    # Check modification times did not change
    faiss_mtime_after = os.path.getmtime(PROJECT_ROOT / "faiss_index" / "benchmark_70docs_v1" / "index.faiss")
    bm25_mtime_after = os.path.getmtime(PROJECT_ROOT / "data" / "bm25_index" / "bm25_index.pkl")

    assert faiss_mtime_before == faiss_mtime_after
    assert bm25_mtime_before == bm25_mtime_after

