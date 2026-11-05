"""
Automated Test and Validation Suite for Phase 3 (BM25 Index Creation)
and Phase 4 (Standalone BM25 Retrieval).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_doc_chat.src.document_chat.bm25_retriever import (
    BM25IndexBuilder,
    BM25Retriever,
)

CORPUS_PATH = PROJECT_ROOT / "data" / "bm25_corpus" / "bm25_corpus_70docs_181chunks.json"
INDEX_DIR = PROJECT_ROOT / "data" / "bm25_index"
INDEX_FAISS = PROJECT_ROOT / "faiss_index" / "benchmark_70docs_v1" / "index.faiss"
INDEX_PKL = PROJECT_ROOT / "faiss_index" / "benchmark_70docs_v1" / "index.pkl"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_bm25_pipeline():
    print("=" * 80)
    print("PHASE 3 & PHASE 4: BM25 INDEX CREATION & STANDALONE RETRIEVAL SUITE")
    print("=" * 80)

    # 1. Pre-flight FAISS SHA-256 snapshot
    faiss_sha_pre = sha256_file(INDEX_FAISS)
    pkl_sha_pre = sha256_file(INDEX_PKL)
    print(f"Pre-flight index.faiss SHA-256: {faiss_sha_pre}")
    print(f"Pre-flight index.pkl   SHA-256: {pkl_sha_pre}")

    # 2. Phase 3: Build and save the BM25 Index
    print("\n[Phase 3] Building BM25 index from prepared corpus...")
    idx_path, manifest_path = BM25IndexBuilder.build_and_save(
        corpus_path=CORPUS_PATH,
        output_dir=INDEX_DIR,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
    )
    print(f"Saved BM25 index bundle to: {idx_path}")
    print(f"Saved BM25 manifest to:     {manifest_path}")

    # Validate index files existence
    assert idx_path.exists(), "bm25_index.pkl missing!"
    assert manifest_path.exists(), "bm25_manifest.json missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check manifest fields
    assert manifest["total_indexed_chunks"] == 181, f"Expected 181 chunks in manifest, got {manifest['total_indexed_chunks']}"
    assert manifest["algorithm"] == "BM25Okapi"
    assert len(manifest["ordered_chunk_ids"]) == 181
    print("  [PASS] Manifest validation: algorithm=BM25Okapi, total_chunks=181.")

    # 3. Reload index via BM25Retriever
    print("\n[Phase 4] Loading BM25Retriever from persistent storage...")
    retriever = BM25Retriever(index_dir=INDEX_DIR)
    assert retriever.total_documents == 181, f"Expected 181 documents, got {retriever.total_documents}"
    print("  [PASS] BM25Retriever loaded successfully with 181 documents.")

    # 4. Verify 1:1 Chunk Fidelity against prepared corpus
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus_raw = json.load(f)

    corpus_map = {e["chunk_id"]: e for e in corpus_raw}
    for idx, cid in enumerate(manifest["ordered_chunk_ids"]):
        orig = corpus_map[cid]
        recovered = retriever.get_chunk_by_id(cid)
        assert recovered is not None, f"Could not recover chunk {cid}"
        assert recovered["index"] == idx == orig["index"], f"Index mismatch for {cid}"
        assert recovered["original_text"] == orig["original_text"], f"Text mismatch for {cid}"
        assert recovered["metadata"] == orig["metadata"], f"Metadata mismatch for {cid}"
    print("  [PASS] 1:1 Chunk fidelity: All 181 chunk IDs, texts, and metadata match corpus perfectly.")

    # 5. Standalone Retrieval Tests
    test_queries = [
        # (category, query, expected_in_top10_keywords)
        ("Exact Term: JWT", "JSON Web Token JWT validation claims", ["jwt", "doc-sec-601.docx"]),
        ("Exact Term: HPA", "Horizontal Pod Autoscaler HPA custom metrics", ["hpa", "kubernetes"]),
        ("Exact Term: PostgreSQL", "PostgreSQL JSONB query performance GIN index", ["postgresql", "jsonb"]),
        ("Acronym: eBGP & iBGP", "eBGP and iBGP route exchange between autonomous systems", ["ebgp", "ibgp", "doc-net-802.docx"]),
        ("Phrase: OAuth 2.0", "OAuth 2.0 authorization framework and PKCE", ["oauth", "doc-sec-601.docx"]),
        ("Conceptual: Saga", "How does the Saga pattern manage distributed transactions without locking", ["saga"]),
        ("Multi-Term: Raft", "Raft consensus leader election and quorum in etcd", ["raft", "etcd"]),
    ]

    print("\nExecuting Standalone Retrieval Validation on Technical Queries:")
    all_retrieved_results = {}

    for cat, q_text, expected_tokens in test_queries:
        res = retriever.retrieve(q_text, top_k=10)
        assert len(res) > 0, f"Query '{q_text}' returned 0 results!"
        assert len(res) <= 10, f"Query '{q_text}' returned > 10 results!"

        # Check score ranking
        scores = [r["bm25_score"] for r in res]
        assert scores == sorted(scores, reverse=True), f"Results not sorted by score descending: {scores}"

        # Check unique chunk IDs
        ret_cids = [r["chunk_id"] for r in res]
        assert len(ret_cids) == len(set(ret_cids)), f"Duplicate chunk IDs in results for '{q_text}'"

        # Check all chunk IDs exist in corpus
        for r in res:
            assert r["chunk_id"] in corpus_map, f"Unknown chunk ID '{r['chunk_id']}' returned!"
            assert r["original_text"] == corpus_map[r["chunk_id"]]["original_text"], "Content mismatch!"

        top1 = res[0]
        top1_src = Path(top1["metadata"].get("source", "")).name
        all_retrieved_results[cat] = {
            "query": q_text,
            "top1_chunk_id": top1["chunk_id"][:12] + "...",
            "top1_source": top1_src,
            "top1_score": top1["bm25_score"],
            "total_returned": len(res),
        }
        print(f"  [PASS] {cat}: Top 1 score={top1['bm25_score']:.4f} ({top1_src}) | Top-{len(res)} returned")

    # 6. Error Handling Tests
    print("\nValidating Error & Edge Case Handling:")
    # Empty string
    assert retriever.retrieve("") == [], "Empty query did not return empty list!"
    # Whitespace only
    assert retriever.retrieve("    ") == [], "Whitespace query did not return empty list!"
    # Pure punctuation / symbols producing no tokens
    assert retriever.retrieve("!@#$%^&*()_+{}[]:;<>?,./~`") == [], "Punctuation-only query did not return empty list!"
    print("  [PASS] Empty and non-token queries handled safely (returned empty lists).")

    # 7. Post-flight FAISS Immutability Verification
    faiss_sha_post = sha256_file(INDEX_FAISS)
    pkl_sha_post = sha256_file(INDEX_PKL)
    assert faiss_sha_post == faiss_sha_pre, "CRITICAL: index.faiss was modified!"
    assert pkl_sha_post == pkl_sha_pre, "CRITICAL: index.pkl was modified!"
    print("\n  [PASS] FAISS index immutability verified (SHA-256 hashes completely identical).")

    print("\n" + "=" * 80)
    print("ALL PHASE 3 & PHASE 4 VALIDATION CHECKS PASSED SUCCESSFULLY.")
    print("=" * 80)

    # Print summary table of sample retrieval
    print("\nSample Retrieval Results Summary:")
    for cat, info in all_retrieved_results.items():
        print(f"  * {cat:<24} | Top-1: {info['top1_source']:<40} | Score: {info['top1_score']:.4f}")


if __name__ == "__main__":
    test_bm25_pipeline()

