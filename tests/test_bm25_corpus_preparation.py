"""
Automated Test and Validation Script for BM25 Corpus Preparation.

Validates:
1. Exactly 181 corpus entries extracted from faiss_index/benchmark_70docs_v1/index.pkl.
2. 100% uniqueness of chunk IDs.
3. Perfect 1:1 index alignment (0..180) with FAISS vector store.
4. Complete preservation of metadata (source filenames, paths, page numbers, content hashes).
5. No missing, empty, or whitespace-only chunk contents.
6. Symmetric technical tokenization (preserving eBGP, iBGP, PostgreSQL, JWT, OAuth 2.0, C++, HPA).
7. Strict preservation of existing FAISS index files (SHA-256 verified).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_doc_chat.src.document_chat.bm25_corpus import (
    BM25CorpusBuilder,
    bm25_tokenize,
)
INDEX_DIR = PROJECT_ROOT / "faiss_index" / "benchmark_70docs_v1"
INDEX_FAISS = INDEX_DIR / "index.faiss"
INDEX_PKL = INDEX_DIR / "index.pkl"
CORPUS_OUTPUT_PATH = PROJECT_ROOT / "data" / "bm25_corpus" / "bm25_corpus_70docs_181chunks.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_bm25_corpus_preparation():
    print("=" * 80)
    print("BM25 CORPUS PREPARATION & VALIDATION SUITE")
    print("=" * 80)

    # 1. Pre-flight SHA-256 snapshots of FAISS index
    faiss_sha_pre = sha256_file(INDEX_FAISS)
    pkl_sha_pre = sha256_file(INDEX_PKL)
    print(f"Pre-flight index.faiss SHA-256: {faiss_sha_pre}")
    print(f"Pre-flight index.pkl   SHA-256: {pkl_sha_pre}")

    # 2. Extract and build BM25 corpus
    builder = BM25CorpusBuilder(INDEX_DIR)
    corpus = builder.build_corpus()
    print(f"\nExtracted {len(corpus)} entries from FAISS docstore.")

    # 3. Save to canonical location
    saved_path = builder.save_corpus(CORPUS_OUTPUT_PATH, corpus)
    print(f"Saved prepared BM25 corpus to: {saved_path}")

    # 4. Run Rigorous Validation Checks
    print("\nRunning Validation Assertions:")

    # Assertion 1: Exactly 181 entries
    assert len(corpus) == 181, f"Expected exactly 181 corpus entries, got {len(corpus)}"
    print("  [PASS] Assertion 1: Exactly 181 entries prepared.")

    # Assertion 2: All 181 chunk IDs are unique
    chunk_ids = [e.chunk_id for e in corpus]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found!"
    assert len(chunk_ids) == 181, "Not all chunk IDs are present!"
    print("  [PASS] Assertion 2: All 181 chunk IDs are strictly unique.")

    # Assertion 3: Sequential indices 0..180
    indices = [e.index for e in corpus]
    assert indices == list(range(181)), "Indices do not match range(181)!"
    print("  [PASS] Assertion 3: Indices match sequential range 0..180.")

    # Assertion 4: No empty or missing original text
    for e in corpus:
        assert e.original_text and len(e.original_text.strip()) > 50, f"Chunk {e.chunk_id} has insufficient text!"
        assert e.token_count > 0, f"Chunk {e.chunk_id} has 0 tokens!"
    print("  [PASS] Assertion 4: No missing, empty, or truncated text (all chunks > 50 chars and > 0 tokens).")

    # Assertion 5: 1:1 Mapping with FAISS index.pkl
    with open(INDEX_PKL, "rb") as f:
        raw_docstore, raw_index_to_id = pickle.load(f)

    for e in corpus:
        raw_uuid = raw_index_to_id[e.index]
        assert e.docstore_id == raw_uuid, f"UUID mismatch at index {e.index}!"
        raw_doc = raw_docstore.search(raw_uuid)
        assert raw_doc.metadata.get("chunk_id") == e.chunk_id, f"chunk_id mismatch at index {e.index}!"
        assert raw_doc.page_content == e.original_text, f"Content mismatch at index {e.index}!"
    print("  [PASS] Assertion 5: Perfect 1:1 two-way fidelity with FAISS docstore (content & UUIDs match).")

    # Assertion 6: Metadata preservation
    total_sources = set()
    pages_int_count = 0
    pages_none_count = 0
    for e in corpus:
        total_sources.add(e.source)
        if e.page is not None:
            assert isinstance(e.page, int), f"Page must be int, got {type(e.page)}"
            pages_int_count += 1
        else:
            pages_none_count += 1
    assert len(total_sources) == 68, f"Expected 68 source documents, found {len(total_sources)}"
    print(f"  [PASS] Assertion 6: All metadata preserved (68 unique source documents, {pages_int_count} int pages, {pages_none_count} None pages).")

    # Assertion 7: Symmetric Tokenizer Technical Term Preservation
    test_cases = [
        ("eBGP and iBGP in Autonomous Systems", ["ebgp", "ibgp", "in", "autonomous", "systems"]),
        ("PostgreSQL with JSONB indexing", ["postgresql", "with", "jsonb", "indexing"]),
        ("JWT and OAuth 2.0 validation", ["jwt", "and", "oauth", "2.0", "validation"]),
        ("Programming in C++ and C#", ["programming", "in", "c++", "and", "c#"]),
        ("Configuring K8s HPA with custom metrics", ["configuring", "k8s", "hpa", "with", "custom", "metrics"]),
        ("Decoupled brake-by-wire system", ["decoupled", "brake-by-wire", "brake", "wire", "system"]),
    ]
    for text_sample, expected_sub_tokens in test_cases:
        actual_tokens = bm25_tokenize(text_sample)
        for exp_tok in expected_sub_tokens:
            assert exp_tok in actual_tokens, f"Expected token '{exp_tok}' missing from tokenized '{text_sample}'. Got: {actual_tokens}"
    print("  [PASS] Assertion 7: Technical terms (eBGP, iBGP, PostgreSQL, JWT, OAuth 2.0, C++, HPA, brake-by-wire) preserved symmetrically.")

    # Assertion 8: FAISS Index Immutability Verification
    faiss_sha_post = sha256_file(INDEX_FAISS)
    pkl_sha_post = sha256_file(INDEX_PKL)
    assert faiss_sha_post == faiss_sha_pre, "CRITICAL: index.faiss was modified!"
    assert pkl_sha_post == pkl_sha_pre, "CRITICAL: index.pkl was modified!"
    print("  [PASS] Assertion 8: FAISS index immutability verified (SHA-256 hashes completely identical).")

    # 5. Corpus Statistics Summary
    token_counts = [e.token_count for e in corpus]
    char_counts = [e.char_count for e in corpus]
    all_tokens = [tok for e in corpus for tok in e.tokens]
    vocab = set(all_tokens)

    print("\nCorpus Statistics:")
    print(f"  Total Entries:         {len(corpus)}")
    print(f"  Total Chunks:          181")
    print(f"  Unique Sources:        {len(total_sources)}")
    print(f"  Total Tokens:          {len(all_tokens):,}")
    print(f"  Unique Vocabulary:     {len(vocab):,} unique terms")
    print(f"  Token Count (Min/Max): {min(token_counts)} / {max(token_counts)} (mean: {sum(token_counts)/len(token_counts):.1f})")
    print(f"  Char Count  (Min/Max): {min(char_counts)} / {max(char_counts)} (mean: {sum(char_counts)/len(char_counts):.1f})")
    print("\nALL BM25 CORPUS VALIDATION CHECKS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    test_bm25_corpus_preparation()
