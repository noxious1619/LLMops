# Phase 11: Baseline Record — Hybrid RAG Architecture & Performance

## 1. Executive Overview
This document records the established **Hybrid RAG Baseline** performance and cryptographic index fingerprints before introducing Cross-Encoder reranking. 
All data, indexes, configuration, and historical evaluation artifacts are preserved in their exact state without modification.

---

## 2. Established Group A Benchmark Results (178 Verified Questions)
Source: [`reports/phase10_hybrid_200q_results.json`](file:///c:/Users/Dishant%20Pandey/Desktop/Resume/LLM_OPS/reports/phase10_hybrid_200q_results.json) and [`reports/phase10_hybrid_200q_evaluation_report.md`](file:///c:/Users/Dishant%20Pandey/Desktop/Resume/LLM_OPS/reports/phase10_hybrid_200q_evaluation_report.md)

| Metric | Baseline Value | Denominator / Evaluation Basis |
|---|---:|---|
| **Hit Rate@5** | **100.00%** | 178 / 178 queries with $\ge 1$ relevant chunk in Top-5 |
| **MRR** | **0.9522** | Mean Reciprocal Rank over 178 queries |
| **Precision@5** | **22.36%** | 199 relevant retrieved / 890 total retrieved slots |
| **Recall@5 (Macro)** | **99.72%** | Average per-query Recall@5 across all 178 queries |
| **Recall@5 (Micro)** | **99.50%** | 199 relevant retrieved / 200 total ground-truth chunks |
| **nDCG@5** | **96.29%** | Normalized Discounted Cumulative Gain at Top-5 |
| **Rank-1 Accuracy** | **90.45%** | 161 / 178 queries where Rank 1 chunk is relevant |
| **Average Latency** | **0.5325s** | Mean end-to-end hybrid retrieval time per query |
| **P50 Latency** | **0.5133s** | Median retrieval latency |
| **P95 Latency** | **0.5770s** | 95th-percentile retrieval latency |

---

## 3. Cryptographic Index Hashes (SHA-256)
Verified prior to commencing Phase 1:

| Artifact | File Path | Verified SHA-256 Checksum |
|---|---|---|
| **FAISS Vector Index** | `faiss_index/benchmark_70docs_v1/index.faiss` | `246209c421d0cd78817082c2190c1a1143a1ce7857e17ac8474426469b7b780a` |
| **FAISS Pickle Store** | `faiss_index/benchmark_70docs_v1/index.pkl` | `33419862b54be18c6c88586f3537b9ce897ac618d9bbcfa1f5b1f67f60cd2621` |
| **BM25 Inverted Index** | `data/bm25_index/bm25_index.pkl` | `9bf695fafd18f60555439acc1ab6795cf715e51b45094762896f10d87aca707d` |
| **BM25 Tokenized Corpus** | `data/bm25_corpus/bm25_corpus_70docs_181chunks.json` | `fc4bd84f95adf9f6e1e718bf768e2b1d2d47597db07dd0dc76e309a97b316e14` |
| **Ground-Truth Benchmark** | `tests/retrieval_ground_truth_200_corrected_v1.json` | `0b0cd9d97eb7f16d7b90fc9d0db274472114de3f23d6c2e9667671c7778876a2` |

---

## 4. Preservation Commitments
1. Zero changes to existing FAISS vector embeddings or index files.
2. Zero changes to BM25 index and corpus structure.
3. The existing `.retrieve()` method on `HybridRetriever` will continue to return Top-5 chunks preserving 100% backward compatibility.
4. RRF constant $k=60$ remains unchanged.
5. No historical evaluation results or files will be overwritten or altered.

