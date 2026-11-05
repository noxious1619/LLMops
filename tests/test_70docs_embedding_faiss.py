"""
Phase 2: Embedding Generation and FAISS Integration (68 Documents)
Target Dataset: docs_70/ (68 files)
Target Index: faiss_index/benchmark_70docs_v1/
Embedding Model: models/gemini-embedding-001 (Google Gemini, 3072 dimensions)
Features:
- Controlled batching (batch_size=25)
- Exponential backoff and retries for rate limits (HTTP 429) and transient errors
- Strict index isolation (preserves all existing FAISS indexes)
- Save and reload verification with a fresh FaissManager instance
- 5 technical similarity searches against the reloaded index
"""

import sys
import os
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

# UTF-8 stdout reconfiguration for Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from multi_doc_chat.utils.document_ops import load_documents
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor, FaissManager
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log

DOCS_70_DIR = PROJECT_ROOT / "docs_70"
FAISS_BASE_DIR = PROJECT_ROOT / "faiss_index"
TARGET_INDEX_DIR = FAISS_BASE_DIR / "benchmark_70docs_v1"
REPORT_PATH = PROJECT_ROOT / "reports" / "step5_70docs_embedding_faiss_report.md"
RESULTS_JSON_PATH = PROJECT_ROOT / "tests" / "step5_70docs_embedding_results.json"


class ResilientGeminiEmbeddings(Embeddings):
    """
    Wrapper around GoogleGenerativeAIEmbeddings providing:
    - Controlled batching of document chunks
    - Exponential backoff on rate limits (429) or transient 5xx errors
    - Granular latency and retry metrics tracking
    """
    def __init__(self, base_embeddings: Embeddings, batch_size: int = 25, max_retries: int = 5, base_delay: float = 2.0):
        self.base = base_embeddings
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_delay = base_delay

        # Tracking metrics
        self.total_chunks_submitted = 0
        self.total_chunks_embedded = 0
        self.failed_chunks = 0
        self.total_batches = 0
        self.successful_batches = 0
        self.retry_count = 0
        self.api_errors: List[Dict[str, Any]] = []
        self.batch_latencies: List[float] = []
        self.total_embedding_time_s = 0.0

    def embed_query(self, text: str) -> List[float]:
        for attempt in range(self.max_retries + 1):
            try:
                return self.base.embed_query(text)
            except Exception as e:
                err_str = str(e)
                self.api_errors.append({"type": "query_error", "attempt": attempt, "error": err_str})
                if attempt < self.max_retries:
                    self.retry_count += 1
                    delay = self.base_delay * (2 ** attempt)
                    print(f"    [Retry Warning] embed_query failed: {err_str[:80]}... Backoff {delay:.1f}s (Attempt {attempt+1}/{self.max_retries})")
                    time.sleep(delay)
                else:
                    raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.total_chunks_submitted += len(texts)
        num_batches = math.ceil(len(texts) / self.batch_size)
        self.total_batches = num_batches
        all_embeddings: List[List[float]] = []

        print(f"\n  -> Submitting {len(texts)} chunks in {num_batches} batches (batch_size={self.batch_size})...")
        t_start_total = time.perf_counter()

        for b_idx in range(num_batches):
            start_i = b_idx * self.batch_size
            end_i = min(start_i + self.batch_size, len(texts))
            batch_texts = texts[start_i:end_i]
            batch_num = b_idx + 1

            batch_succeeded = False
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    # Execute real Gemini batch embedding
                    batch_vecs = self.base.embed_documents(batch_texts)
                    batch_latency = time.perf_counter() - t0
                    self.batch_latencies.append(batch_latency)
                    all_embeddings.extend(batch_vecs)
                    self.total_chunks_embedded += len(batch_texts)
                    self.successful_batches += 1
                    batch_succeeded = True

                    avg_chunk_ms = (batch_latency / len(batch_texts)) * 1000
                    print(f"    Batch [{batch_num:02d}/{num_batches:02d}] {len(batch_texts):2d} chunks | Latency: {batch_latency:.2f}s ({avg_chunk_ms:.1f}ms/chunk) | Progress: {self.total_chunks_embedded}/{len(texts)} chunks")
                    break
                except Exception as e:
                    err_str = str(e)
                    self.api_errors.append({
                        "batch_num": batch_num,
                        "attempt": attempt,
                        "chunk_range": f"{start_i}:{end_i}",
                        "error": err_str
                    })
                    if attempt < self.max_retries:
                        self.retry_count += 1
                        delay = self.base_delay * (2 ** attempt)
                        print(f"    [RateLimit / API Warning] Batch {batch_num} attempt {attempt+1} failed: {err_str[:90]}... Retrying in {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        print(f"    [FATAL] Batch {batch_num} failed after {self.max_retries} retries: {err_str}")
                        self.failed_chunks += len(batch_texts)
                        raise

            if not batch_succeeded:
                raise RuntimeError(f"Embedding generation failed for batch {batch_num}")

            # Small polite pacing between batches to prevent quota bursts
            time.sleep(0.3)

        self.total_embedding_time_s = time.perf_counter() - t_start_total
        return all_embeddings


class CustomModelLoaderWrapper:
    """Wrapper around ModelLoader that provides the ResilientGeminiEmbeddings."""
    def __init__(self, raw_loader: ModelLoader, resilient_emb: ResilientGeminiEmbeddings):
        self.raw_loader = raw_loader
        self.resilient_emb = resilient_emb
        self.config = raw_loader.config

    def load_embeddings(self) -> Embeddings:
        return self.resilient_emb

    def load_llm(self):
        return self.raw_loader.load_llm()


def run_phase2():
    print("=" * 80)
    print("PHASE 2: REAL EMBEDDING GENERATION AND FAISS INTEGRATION (68 DOCUMENTS)")
    print("=" * 80)

    # 1. Safety & Directory Isolation Verification
    print("\n[Step 1] Safety Verification & Directory Isolation Check...")
    if not DOCS_70_DIR.exists():
        raise FileNotFoundError(f"docs_70 directory not found: {DOCS_70_DIR}")
    
    print(f"  -> Target FAISS directory: {TARGET_INDEX_DIR}")
    if TARGET_INDEX_DIR.exists():
        print(f"  [Safety Action] Removing pre-existing test directory: {TARGET_INDEX_DIR}")
        import shutil
        shutil.rmtree(TARGET_INDEX_DIR)
    
    # Check that Step 5 test index is preserved
    step5_index = FAISS_BASE_DIR / "step5_validation_test_20261101_121818"
    if step5_index.exists():
        print(f"  -> Verified Step 5 index preserved: {step5_index} ({step5_index.stat().st_size if hasattr(step5_index, 'stat') else 'EXISTS'})")
    
    # 2. Document Loading and Splitting
    print("\n[Step 2] Ingestion and Splitting of docs_70/...")
    doc_files = sorted([f for f in DOCS_70_DIR.iterdir() if f.is_file()])
    print(f"  -> Found {len(doc_files)} files in docs_70/")
    if len(doc_files) != 68:
        raise ValueError(f"Expected 68 files in docs_70/, found {len(doc_files)}")

    t0 = time.perf_counter()
    raw_docs = load_documents(doc_files)
    t_load_docs = time.perf_counter() - t0
    print(f"  -> Loaded {len(raw_docs)} Document objects across {len(doc_files)} files in {t_load_docs:.2f}s")
    if len(raw_docs) != 85:
        raise ValueError(f"Expected 85 Document objects, got {len(raw_docs)}")

    ci = ChatIngestor.__new__(ChatIngestor)
    t0 = time.perf_counter()
    chunks = ci._split(raw_docs, chunk_size=500, chunk_overlap=100)
    t_split = time.perf_counter() - t0
    print(f"  -> Split into {len(chunks)} chunks in {t_split:.3f}s")
    if len(chunks) != 181:
        raise ValueError(f"Expected 181 chunks, got {len(chunks)}")

    # Verify chunk metadata
    missing_meta = [i for i, c in enumerate(chunks) if not c.metadata.get("chunk_id") or not c.metadata.get("content_hash")]
    if missing_meta:
        raise ValueError(f"Found {len(missing_meta)} chunks with missing metadata!")
    print("  -> Metadata check: All 181 chunks contain valid chunk_id, content_hash, and source.")

    # 3. ModelLoader & Resilient Embeddings Initialization
    print("\n[Step 3] Initializing Google Gemini Embedding Provider...")
    base_loader = ModelLoader()
    base_emb = base_loader.load_embeddings()
    provider_name = base_loader.config.get("embedding_model", {}).get("provider", "google")
    model_name = base_loader.config.get("embedding_model", {}).get("model_name", "models/gemini-embedding-001")
    print(f"  -> Provider: {provider_name}")
    print(f"  -> Model:    {model_name}")

    # Dimensionality check probe
    t0 = time.perf_counter()
    probe_vec = base_emb.embed_query("Dimensionality probe for Phase 2 validation")
    probe_latency = time.perf_counter() - t0
    embedding_dim = len(probe_vec)
    print(f"  -> Model Probe: Success! Dimensions = {embedding_dim} | Probe Latency = {probe_latency:.3f}s")
    if embedding_dim != 3072:
        raise ValueError(f"Expected 3072 dimensions for gemini-embedding-001, got {embedding_dim}")

    resilient_emb = ResilientGeminiEmbeddings(base_emb, batch_size=25, max_retries=5, base_delay=2.0)
    custom_loader = CustomModelLoaderWrapper(base_loader, resilient_emb)

    # 4. Generate Embeddings & Store in FAISS via FaissManager
    print("\n[Step 4] Storing 181 Chunks into FAISS via FaissManager...")
    fm = FaissManager(TARGET_INDEX_DIR, custom_loader)

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]

    t0 = time.perf_counter()
    vs = fm.load_or_create(texts=texts, metadatas=metas)
    t_faiss_creation = time.perf_counter() - t0

    vectors_in_faiss = vs.index.ntotal
    print(f"\n  -> Initial Index Creation Finished in {t_faiss_creation:.2f}s")
    print(f"  -> Total vectors in FAISS index: {vectors_in_faiss}")
    print(f"  -> Fingerprinted rows in _meta: {len(fm._meta['rows'])}")
    if vectors_in_faiss != 181:
        raise ValueError(f"Vector count mismatch! Expected 181, found {vectors_in_faiss}")

    # 5. Save & Reload Verification
    print("\n[Step 5] Save and Reload Verification with Fresh Instance...")
    faiss_file = TARGET_INDEX_DIR / "index.faiss"
    pkl_file = TARGET_INDEX_DIR / "index.pkl"
    meta_file = TARGET_INDEX_DIR / "ingested_meta.json"

    for f in [faiss_file, pkl_file, meta_file]:
        if not f.exists() or f.stat().st_size == 0:
            raise FileNotFoundError(f"Persisted file missing or empty: {f}")
        print(f"  -> Persisted file: {f.name:18s} | Size: {f.stat().st_size:,} bytes")

    # Instantiate completely fresh ModelLoader and FaissManager
    print("  -> Creating fresh FaissManager instance to reload from disk...")
    fresh_loader = ModelLoader()
    fresh_fm = FaissManager(TARGET_INDEX_DIR, fresh_loader)
    reloaded_vs = fresh_fm.load_or_create()

    reloaded_vectors = reloaded_vs.index.ntotal
    reloaded_dims = reloaded_vs.index.d
    print(f"  -> Reloaded Vectors: {reloaded_vectors}")
    print(f"  -> Reloaded Dimensions: {reloaded_dims}")
    if reloaded_vectors != 181 or reloaded_dims != 3072:
        raise ValueError(f"Reload verification failed: vectors={reloaded_vectors}, dims={reloaded_dims}")

    # Verify docstore contents
    docstore_dict = reloaded_vs.docstore._dict
    print(f"  -> Reloaded Docstore Entries: {len(docstore_dict)}")
    reloaded_chunk_ids = set()
    for doc_id, doc in docstore_dict.items():
        cid = doc.metadata.get("chunk_id")
        if not cid:
            raise ValueError(f"Docstore document {doc_id} is missing chunk_id!")
        reloaded_chunk_ids.add(cid)
    print(f"  -> Unique chunk_ids in reloaded docstore: {len(reloaded_chunk_ids)} / 181")

    # 6. Five Technical Similarity Searches
    print("\n" + "=" * 80)
    print("[Step 6] Running 5 Similarity Searches Against Reloaded Index (Top-3)...")
    print("=" * 80)

    test_queries = [
        {
            "query_id": "SEARCH-01",
            "topic": "Steer-by-Wire & ADAS Actuation",
            "query": "How does steer-by-wire eliminate mechanical steering column linkage and ensure failsafe redundancy?",
            "target_doc": "adas_actuation_and_steer_by_wire.txt"
        },
        {
            "query_id": "SEARCH-02",
            "topic": "EV Battery Management Systems",
            "query": "What are the primary mechanisms used by Battery Management Systems to prevent thermal runaway in lithium-ion cells?",
            "target_doc": "Electric_Vehicle_Battery_Management_Systems_BMS.docx"
        },
        {
            "query_id": "SEARCH-03",
            "topic": "Magna Carta & Common Law",
            "query": "How did Clause 39 of the Magna Carta establish foundational principles of due process and trial by peers?",
            "target_doc": "The Magna Carta and the Foundations of Common Law.pdf"
        },
        {
            "query_id": "SEARCH-04",
            "topic": "Distributed Consensus (Paxos vs Raft)",
            "query": "What are the key differences between Paxos and Raft regarding leader election and understandable decomposition?",
            "target_doc": "Paxos and Raft Consensus Algorithms.pdf"
        },
        {
            "query_id": "SEARCH-05",
            "topic": "EV Motor Topologies",
            "query": "What are the operational trade-offs between permanent magnet synchronous motors and induction motors in electric vehicles?",
            "target_doc": "electric_vehicle_motor_topologies.txt"
        }
    ]

    search_results_data = []

    for q_item in test_queries:
        qid = q_item["query_id"]
        q_text = q_item["query"]
        topic = q_item["topic"]
        print(f"\n--- [{qid}] {topic} ---")
        print(f"Query: \"{q_text}\"")

        t0 = time.perf_counter()
        top3 = reloaded_vs.similarity_search_with_score(q_text, k=3)
        search_latency_ms = (time.perf_counter() - t0) * 1000

        retrieved = []
        for rank, (doc, score) in enumerate(top3, start=1):
            cid = doc.metadata.get("chunk_id", "N/A")
            src_name = Path(doc.metadata.get("source", "N/A")).name
            pg = doc.metadata.get("page")
            pg_str = f"p.{pg}" if pg is not None else "N/A"
            preview = doc.page_content.replace("\n", " ")[:130]
            print(f"  Rank {rank} | L2: {score:.4f} | {src_name} ({pg_str}) | ID: {cid[:12]}...")
            print(f"         Preview: {preview}...")
            retrieved.append({
                "rank": rank,
                "score_l2": float(score),
                "source": src_name,
                "page": pg,
                "chunk_id": cid,
                "preview": preview
            })

        search_results_data.append({
            "query_id": qid,
            "topic": topic,
            "query": q_text,
            "latency_ms": round(search_latency_ms, 2),
            "retrieved": retrieved
        })

    # 7. Summary Metrics Compilation
    avg_batch_time = (sum(resilient_emb.batch_latencies) / len(resilient_emb.batch_latencies)) if resilient_emb.batch_latencies else 0.0
    avg_chunk_time_ms = (resilient_emb.total_embedding_time_s / resilient_emb.total_chunks_embedded * 1000) if resilient_emb.total_chunks_embedded else 0.0

    summary_metrics = {
        "status": "PASS",
        "benchmark_session": "benchmark_70docs_v1",
        "embedding_provider": provider_name,
        "embedding_model": model_name,
        "embedding_dimensions": embedding_dim,
        "target_faiss_dir": str(TARGET_INDEX_DIR),
        "total_source_files": len(doc_files),
        "extracted_documents": len(raw_docs),
        "total_chunks_submitted": resilient_emb.total_chunks_submitted,
        "successfully_embedded_chunks": resilient_emb.total_chunks_embedded,
        "failed_chunks": resilient_emb.failed_chunks,
        "faiss_vector_count": vectors_in_faiss,
        "reloaded_vector_count": reloaded_vectors,
        "total_batches": resilient_emb.total_batches,
        "batch_size": resilient_emb.batch_size,
        "api_retries": resilient_emb.retry_count,
        "api_errors_count": len(resilient_emb.api_errors),
        "timings": {
            "document_loading_s": round(t_load_docs, 3),
            "document_splitting_s": round(t_split, 3),
            "total_embedding_time_s": round(resilient_emb.total_embedding_time_s, 3),
            "average_batch_time_s": round(avg_batch_time, 3),
            "average_chunk_time_ms": round(avg_chunk_time_ms, 2),
            "total_faiss_indexing_s": round(t_faiss_creation, 3)
        },
        "persisted_files": {
            "index_faiss_bytes": faiss_file.stat().st_size,
            "index_pkl_bytes": pkl_file.stat().st_size,
            "ingested_meta_bytes": meta_file.stat().st_size
        },
        "similarity_searches": search_results_data
    }

    # Save to JSON
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"\nDetailed Phase 2 results saved to: {RESULTS_JSON_PATH}")

    # Print final console summary
    print("\n" + "=" * 80)
    print("PHASE 2 SUMMARY & VERDICT")
    print("=" * 80)
    print(f"Embedding Provider:           {provider_name}")
    print(f"Embedding Model:              {model_name} (Dimensions: {embedding_dim})")
    print(f"Chunks Processed:             {resilient_emb.total_chunks_embedded} / {len(chunks)} (100% Success)")
    print(f"Failed Chunks:                {resilient_emb.failed_chunks}")
    print(f"API Batches:                  {resilient_emb.total_batches} (Batch size: {resilient_emb.batch_size})")
    print(f"API Rate Limits / Retries:    {resilient_emb.retry_count}")
    print(f"Total Embedding Time:         {resilient_emb.total_embedding_time_s:.2f}s ({avg_chunk_time_ms:.1f}ms / chunk)")
    print(f"FAISS Vectors Stored:         {vectors_in_faiss}")
    print(f"FAISS Vectors Reloaded:       {reloaded_vectors}")
    print(f"Reload Docstore Entries:      {len(docstore_dict)}")
    print(f"Target Directory:             {TARGET_INDEX_DIR}")
    print("OVERALL VERDICT:              PASS")
    print("=" * 80)

    return summary_metrics


if __name__ == "__main__":
    run_phase2()

