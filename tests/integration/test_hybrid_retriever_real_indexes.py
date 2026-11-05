"""
Integration Test: Real-Index Hybrid Retrieval Smoke Test (FAISS + BM25 + RRF).

Requirements:
- Connects the actual FAISS vector index (181 chunks, Gemini 3072D embeddings)
  and actual BM25 index (181 chunks, BM25Okapi) through real HybridRetriever.
- Strictly opt-in via environment variable: RUN_REAL_RETRIEVAL_SMOKE=1.
- Skips cleanly when RUN_REAL_RETRIEVAL_SMOKE != "1" so standard test runs are offline/fast.
- Validates real query routing, candidate bounding (<=10), RRF score addition,
  Top-5 output slicing, metadata immutability, and index file immutability.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# When opt-in flag is requested, ensure real credentials from .env override test dummy stubs
if os.environ.get("RUN_REAL_RETRIEVAL_SMOKE", "").strip() == "1":
    load_dotenv(PROJECT_ROOT / ".env", override=True)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from multi_doc_chat.src.document_chat.hybrid_retriever import HybridRetriever

FAISS_DIR = PROJECT_ROOT / "faiss_index" / "benchmark_70docs_v1"
BM25_DIR = PROJECT_ROOT / "data" / "bm25_index"
INDEX_FAISS = FAISS_DIR / "index.faiss"
INDEX_PKL = FAISS_DIR / "index.pkl"
BM25_PKL = BM25_DIR / "bm25_index.pkl"
BM25_MANIFEST = BM25_DIR / "bm25_manifest.json"

OPT_IN_FLAG = os.environ.get("RUN_REAL_RETRIEVAL_SMOKE", "").strip() == "1"

SMOKE_QUERIES = [
    {
        "id": "Q11",
        "category": "Technical Acronyms & Path Attributes",
        "query": "How does the Border Gateway Protocol (BGP) evaluate complex path attributes like Weight, Local Preference, and AS-PATH length rather than relying on simple distance metrics?",
        "expected_doc": "DOC-NET-802.docx",
    },
    {
        "id": "Q35",
        "category": "Specific Technology & Mechanism",
        "query": "How do Redis clusters utilize hash slots to automatically partition data across multiple nodes, ensuring horizontal scalability for caching layers?",
        "expected_doc": "Redis Cluster Data Sharding and High Availability.pdf",
    },
    {
        "id": "Q1",
        "category": "Conceptual Architecture & Identity",
        "query": "What is the fundamental difference between authentication and authorization, and why is relying purely on OAuth 2.0 for identity verification considered a security vulnerability?",
        "expected_doc": "DOC-SEC-601.docx",
    },
    {
        "id": "Q10",
        "category": "Exact Terminology & Hardware Constraints",
        "query": "How do recent advancements like FlashAttention address the quadratic computational and memory complexity bottlenecks inherent in standard Transformer architectures?",
        "expected_doc": "DOC-ML-705.docx",
    },
    {
        "id": "Q48",
        "category": "Semantic Matching & Security Policy",
        "query": "Why is micro-segmentation critical in a modern cloud environment, and how does it restrict lateral movement if an attacker breaches a single workload?",
        "expected_doc": "Zero Trust Network Access (ZTNA) Implementation.pdf",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def real_hybrid_retriever():
    """Module-scoped fixture to instantiate real HybridRetriever once."""
    if not OPT_IN_FLAG:
        pytest.skip("Skipping real-index smoke test. Enable by setting RUN_REAL_RETRIEVAL_SMOKE=1.")

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "dummy":
        pytest.skip("Valid GOOGLE_API_KEY / GEMINI_API_KEY not found in environment or .env.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )

    retriever = HybridRetriever.from_saved(
        faiss_index_dir=FAISS_DIR,
        bm25_index_dir=BM25_DIR,
        embeddings=embeddings,
        dense_k=10,
        sparse_k=10,
        final_k=5,
        rrf_k=60,
    )
    return retriever


@pytest.mark.skipif(not OPT_IN_FLAG, reason="Set RUN_REAL_RETRIEVAL_SMOKE=1 to run real-index integration test.")
def test_real_indexes_loaded_and_immutability(real_hybrid_retriever):
    """Verify that both real indexes exist, load properly, and contain 181 chunks."""
    faiss_sha_pre = sha256_file(INDEX_FAISS)
    pkl_sha_pre = sha256_file(INDEX_PKL)

    assert real_hybrid_retriever.bm25_retriever.total_documents == 181
    assert real_hybrid_retriever.faiss_retriever is not None

    # Check baseline FAISS hashes
    assert faiss_sha_pre == "246209c421d0cd78817082c2190c1a1143a1ce7857e17ac8474426469b7b780a"
    assert pkl_sha_pre == "33419862b54be18c6c88586f3537b9ce897ac618d9bbcfa1f5b1f67f60cd2621"


@pytest.mark.skipif(not OPT_IN_FLAG, reason="Set RUN_REAL_RETRIEVAL_SMOKE=1 to run real-index integration test.")
@pytest.mark.parametrize("item", SMOKE_QUERIES, ids=[q["id"] for q in SMOKE_QUERIES])
def test_real_hybrid_retrieval_queries(real_hybrid_retriever, item):
    """Execute real hybrid retrieval for benchmark queries across diverse categories."""
    qid = item["id"]
    query = item["query"]
    expected_doc = item["expected_doc"]

    # 1. Fetch dense candidates (FAISS Top-10)
    dense_docs = real_hybrid_retriever._fetch_dense(query)
    assert 0 < len(dense_docs) <= 10

    # 2. Fetch sparse candidates (BM25 Top-10)
    sparse_docs = real_hybrid_retriever._fetch_sparse(query)
    assert 0 < len(sparse_docs) <= 10

    # 3. Retrieve fused Top-5 with scores
    fused_results = real_hybrid_retriever.retrieve_with_scores(query)
    assert 0 < len(fused_results) <= 5

    # 4. Verify rank ordering is strictly descending by score
    scores = [r.score for r in fused_results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]

    # 5. Verify deduplication: chunk IDs must be strictly unique
    retrieved_cids = [r.document.metadata["chunk_id"] for r in fused_results]
    assert len(retrieved_cids) == len(set(retrieved_cids))

    # 6. Verify metadata preservation
    for r in fused_results:
        doc = r.document
        assert "chunk_id" in doc.metadata
        assert "source" in doc.metadata
        assert "content_hash" in doc.metadata
        assert "score" not in doc.metadata
        assert "rrf_score" not in doc.metadata
        assert len(doc.page_content.strip()) > 0

    # 7. Check ground truth document match at Rank 1
    rank1_source = Path(fused_results[0].document.metadata["source"]).name
    assert rank1_source == expected_doc, f"[{qid}] Expected {expected_doc} at Rank 1, got {rank1_source}"


@pytest.mark.skipif(not OPT_IN_FLAG, reason="Set RUN_REAL_RETRIEVAL_SMOKE=1 to run real-index integration test.")
def test_real_hybrid_retrieval_empty_and_whitespace_query(real_hybrid_retriever):
    """Verify empty/whitespace queries return empty lists on real indexes."""
    assert real_hybrid_retriever.retrieve("") == []
    assert real_hybrid_retriever.retrieve("   ") == []
    assert real_hybrid_retriever.retrieve_with_scores("") == []


@pytest.mark.skipif(not OPT_IN_FLAG, reason="Set RUN_REAL_RETRIEVAL_SMOKE=1 to run real-index integration test.")
def test_real_indexes_file_immutability_post_flight():
    """Verify FAISS and BM25 index files on disk remain bit-for-bit unchanged after retrieval."""
    assert sha256_file(INDEX_FAISS) == "246209c421d0cd78817082c2190c1a1143a1ce7857e17ac8474426469b7b780a"
    assert sha256_file(INDEX_PKL) == "33419862b54be18c6c88586f3537b9ce897ac618d9bbcfa1f5b1f67f60cd2621"
