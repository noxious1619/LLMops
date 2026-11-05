"""
Phase 8 Functional Validation Script for ConversationalRAG with Hybrid Retrieval.

Executes end-to-end queries across 5 target categories, multi-turn history,
and dense fallback scenarios, verifying index immutability throughout.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Ensure .env is loaded before imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.src.document_chat.hybrid_retriever import HybridRetriever

FAISS_DIR = PROJECT_ROOT / "faiss_index" / "benchmark_70docs_v1"
FAISS_FILE = FAISS_DIR / "index.faiss"
PKL_FILE = FAISS_DIR / "index.pkl"

EXPECTED_FAISS_HASH = "246209c421d0cd78817082c2190c1a1143a1ce7857e17ac8474426469b7b780a"
EXPECTED_PKL_HASH = "33419862b54be18c6c88586f3537b9ce897ac618d9bbcfa1f5b1f67f60cd2621"


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_index_integrity(stage: str) -> Dict[str, Any]:
    faiss_hash = get_sha256(FAISS_FILE)
    pkl_hash = get_sha256(PKL_FILE)
    faiss_match = faiss_hash == EXPECTED_FAISS_HASH
    pkl_match = pkl_hash == EXPECTED_PKL_HASH
    print(f"[{stage}] FAISS index SHA-256: {faiss_hash} (Match: {faiss_match})")
    print(f"[{stage}] FAISS pkl SHA-256:   {pkl_hash} (Match: {pkl_match})")
    assert faiss_match and pkl_match, f"Index integrity violation at stage {stage}!"
    return {
        "faiss_hash": faiss_hash,
        "pkl_hash": pkl_hash,
        "verified": True,
    }


def main():
    print("=" * 70)
    print("PHASE 8: FULL CONVERSATIONAL RAG FUNCTIONAL VALIDATION")
    print("=" * 70)

    # 1. Pre-validation integrity check
    pre_integrity = verify_index_integrity("PRE-VALIDATION")

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pre_integrity": pre_integrity,
        "categories": {},
        "conversational_history": {},
        "fallback_validation": {},
    }

    # Initialize ConversationalRAG with hybrid mode on benchmark index
    rag = ConversationalRAG(session_id="phase8_eval_session")
    rag.load_retriever(index_path=str(FAISS_DIR), mode="hybrid")

    test_queries = [
        {
            "category": "Category A: Exact Technical Specifications",
            "id": "Q_SPEC_REDIS",
            "question": "How many hash slots are used in a Redis Cluster and what function is used to calculate the slot for a key?",
            "expectation": "16,384 slots (0 to 16383), CRC16 modulo 16384.",
        },
        {
            "category": "Category B: Conceptual and Comparative",
            "id": "Q1",
            "question": "What is the fundamental difference between authentication and authorization, and why is relying purely on OAuth 2.0 for identity verification considered a security vulnerability?",
            "expectation": "Authn identifies identity, authz grants permissions; OAuth 2.0 is an authorization framework, not an authentication protocol.",
        },
        {
            "category": "Category C: Technical Acronyms and Abbreviations",
            "id": "Q11",
            "question": "How does the Border Gateway Protocol (BGP) evaluate complex path attributes like Weight, Local Preference, and AS-PATH length rather than relying on simple distance metrics?",
            "expectation": "BGP path selection order: highest Weight, highest Local Preference, shortest AS-PATH, etc.",
        },
        {
            "category": "Category D: Multi-Chunk Evidence Retrieval",
            "id": "Q42",
            "question": "What is the function of the Raft consensus algorithm within etcd, and why is a quorum required for Kubernetes to maintain reliable cluster configuration data?",
            "expectation": "Consensus on state/log replication; quorum (n/2 + 1) prevents split-brain and ensures majority agreement on configuration.",
        },
        {
            "category": "Category E: Unanswerable / Out-of-Domain (Negative Test)",
            "id": "Q_NEGATIVE",
            "question": "What is the exact rotational period, surface gravity, and atmospheric chemical composition of exoplanet Kepler-186f?",
            "expectation": "System must state the provided documents/context do not contain this information.",
        },
    ]

    for item in test_queries:
        cat = item["category"]
        q_id = item["id"]
        q = item["question"]
        print(f"\nEvaluating {cat} [{q_id}]:\n  Q: {q}")

        # Retrieve documents directly from hybrid retriever to inspect fusion candidates
        t0 = time.time()
        docs = rag.retriever.invoke(q) if hasattr(rag.retriever, "invoke") else rag.retriever.retrieve(q)
        retrieval_latency = round(time.time() - t0, 3)

        chunk_ids = [d.metadata.get("chunk_id", "unknown") for d in docs]
        sources = [d.metadata.get("source", "unknown") for d in docs]
        print(f"  Retrieved {len(docs)} chunks in {retrieval_latency}s: {chunk_ids}")

        # Execute end-to-end answer generation
        t1 = time.time()
        answer = rag.invoke(q)
        total_latency = round(time.time() - t1 + retrieval_latency, 3)

        print(f"  Answer preview: {answer[:180]}...")

        results["categories"][q_id] = {
            "category": cat,
            "question": q,
            "expectation": item["expectation"],
            "retrieved_chunk_count": len(docs),
            "retrieved_chunk_ids": chunk_ids,
            "retrieved_sources": sources,
            "retrieval_latency_sec": retrieval_latency,
            "total_latency_sec": total_latency,
            "answer": answer,
        }

    # 2. Multi-turn dialogue / conversational history
    print("\nEvaluating Multi-turn Dialogue (Follow-up Contextualization):")
    from langchain_core.messages import AIMessage, HumanMessage

    conv_rag = ConversationalRAG(session_id="multi_turn_session")
    conv_rag.load_retriever(index_path=str(FAISS_DIR), mode="hybrid")
    history: List[Any] = []

    turn1_q = "What is the primary role of the Leader node in the Raft consensus algorithm used by etcd?"
    turn1_ans = conv_rag.invoke(turn1_q, chat_history=history)
    history.append(HumanMessage(content=turn1_q))
    history.append(AIMessage(content=turn1_ans))
    print(f"  Turn 1 Q: {turn1_q}")
    print(f"  Turn 1 Ans: {turn1_ans[:140]}...")

    turn2_q = "What happens if it becomes partitioned from the rest of the cluster?"
    turn2_ans = conv_rag.invoke(turn2_q, chat_history=history)
    history.append(HumanMessage(content=turn2_q))
    history.append(AIMessage(content=turn2_ans))
    print(f"  Turn 2 Q: {turn2_q}")
    print(f"  Turn 2 Ans: {turn2_ans[:140]}...")

    results["conversational_history"] = {
        "turn_1": {"question": turn1_q, "answer": turn1_ans},
        "turn_2": {"question": turn2_q, "answer": turn2_ans},
        "history_length": len(history),
    }

    # 3. Fallback validation:
    # 3a. Explicit dense mode
    print("\nEvaluating Fallback Scenario 3a (Explicit dense mode):")
    dense_rag = ConversationalRAG(session_id="dense_fallback_session")
    dense_rag.load_retriever(index_path=str(FAISS_DIR), mode="dense")
    dense_ans = dense_rag.invoke("How many hash slots are in a Redis Cluster?")
    print(f"  Mode: {dense_rag.retrieval_mode}")
    print(f"  Answer: {dense_ans[:140]}...")

    # 3b. Simulated BM25 failure with fallback_to_dense=True
    print("\nEvaluating Fallback Scenario 3b (Simulated BM25 runtime failure):")
    from unittest.mock import patch

    class BrokenBM25:
        def retrieve_documents(self, q, top_k):
            raise RuntimeError("Simulated BM25 memory corruption error!")

    # Instantiate hybrid retriever with broken BM25 and fallback_to_dense=True
    from langchain_community.vectorstores import FAISS
    from multi_doc_chat.utils.model_loader import ModelLoader

    embeddings = ModelLoader().load_embeddings()
    vs = FAISS.load_local(str(FAISS_DIR), embeddings, index_name="index", allow_dangerous_deserialization=True)
    dense_retriever = vs.as_retriever(search_kwargs={"k": 10})
    fault_tolerant_retriever = HybridRetriever(
        faiss_retriever=dense_retriever,
        bm25_retriever=BrokenBM25(),
        fallback_to_dense=True,
    )

    ft_docs = fault_tolerant_retriever.invoke("How many hash slots are in a Redis Cluster?")
    print(f"  Retrieved {len(ft_docs)} chunks via dense fallback successfully.")
    assert len(ft_docs) > 0, "Dense fallback failed to retrieve any chunks!"

    results["fallback_validation"] = {
        "dense_mode_success": "16,384" in dense_ans or "16384" in dense_ans,
        "dense_mode_answer": dense_ans,
        "bm25_failure_recovered_docs": len(ft_docs),
        "bm25_fallback_chunk_ids": [d.metadata.get("chunk_id") for d in ft_docs],
    }

    # 4. Post-validation integrity check
    post_integrity = verify_index_integrity("POST-VALIDATION")
    results["post_integrity"] = post_integrity

    # Save output
    out_path = PROJECT_ROOT / "reports" / "phase8_functional_validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll Phase 8 validation checks completed successfully! Saved to {out_path}")


if __name__ == "__main__":
    main()
