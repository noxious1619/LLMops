# Step 11: Corrected 200-Question Retrieval Benchmark Evaluation Report

## Executive Summary

This report presents the consolidated retrieval performance of the **Dense Retrieval Baseline** evaluated across the complete **200-question benchmark** using the newly validated and canonical ground-truth dataset (`tests/retrieval_ground_truth_200_corrected_v1.json`).

### Benchmark Architecture & Configuration
- **Embedding Model**: Google Gemini `models/gemini-embedding-001` (3072 dimensions)
- **Vector Database**: FAISS (IndexFlatL2, 181 vectors)
- **Corpus**: 68 documents (181 chunks, chunk size 500 tokens, overlap 100 tokens)
- **Retrieval Mode**: Dense semantic vector similarity, Top-K = 5
- **Evaluated Questions**: 200 total (178 Verified Primary, 5 Partially Supported, 17 Negative Controls)

---

## 1. Three-Way Metric Comparison

This table isolates the effect of ground-truth corrections from changes in retrieval rankings:

| Metric | Step 9 Original GT | Corrected GT, Same Retrieval | Fresh Run, Corrected GT | Key Takeaway |
|---|---:|---:|---:|---|
| **Hit Rate@5** | 100.0% | 100.0% | 100.0% | Remains 100% across all runs due to small corpus clustering. |
| **Macro Recall@5** | 100.0% | 100.0% | 100.0% | Remains 100% (87.6% single-chunk degeneracy + 2-chunk doc clustering). |
| **Micro Recall@5** | 100.0% | 100.0% | 100.0% | All 200 ground-truth chunks retrieved in top 5. |
| **Rank 1 Accuracy** | 78.65% (140/178) | 79.78% (142/178) | 79.78% (142/178) | +1.13% increase (+2 hits) from correcting Q12 and Q15 mapping shift. |
| **MRR** | 0.8834 | 0.889 | 0.889 | +0.0056 improvement due to Q12 and Q15 moving from Rank 2 to Rank 1. |
| **Precision@5** | 22.47% | 22.47% | 22.47% | Identical precision: 200 relevant chunks retrieved across 890 positions. |
| **nDCG@5** | 91.34% | 91.75% | 91.75% | +0.86% improvement in ranking quality from correcting Q12 and Q15 to Rank 1. |
| **Min-Sufficient Coverage** | N/A | 100.0% | 100.0% | 100% of questions retrieved their minimum sufficient evidence set. |

---

## 2. Cohort-Level Performance Breakdown

| Evaluation Cohort | Question Count | Hit Rate@5 | Rank 1 Accuracy | MRR | nDCG@5 | Macro Recall@5 | Precision@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Group A: Verified Primary** | **178** | **100.0%** | **79.78%** | **0.889** | **91.75%** | **100.0%** | **22.47%** |
| *• Original 128 (Q1–Q150)* | 128 | 100.0% | 76.56% | 0.8704 | 90.42% | 100.0% | 21.88% |
| *• New 50 (Q151–Q200)* | 50 | 100.0% | 88.0% | 0.9367 | 95.15% | 100.0% | 24.0% |
| *• Single-Chunk (N=156)* | 156 | 100.0% | 77.56% | 0.8766 | 90.84% | 100.0% | 20.0% |
| *• Multi-Chunk (N=22)* | 22 | 100.0% | 95.45% | 0.9773 | 98.24% | 100.0% | 40.0% |
| **Group B: Partially Supported** | **5** | **100.0%** | **80.0%** | **0.9** | **92.62%** | **100.0%** | **32.0%** |
| **Group C: Negative Controls** | **17** | **0.0%** | **0.0%** | **0.0** | **0.0%** | **N/A** | **0.0%** |

---

## 3. Latency Profile

| Operation | Mean Latency | Median (P50) | P95 Latency |
|---|---:|---:|---:|
| **Query Embedding (Gemini API)** | 0.60367s | — | — |
| **FAISS Vector Search (L2 Flat)** | 0.00031s | — | — |
| **Total Pipeline Latency** | **0.60398s** | **0.51279s** | **1.22869s** |

- Total benchmark execution time: **120.8s** across 200 queries.
- Failed queries: **0 / 200 (100% success rate)**.

---

## 4. Analysis of Negative Controls and Abstention

- **Negative Controls ($N=17$)**: All 17 questions represent out-of-domain topics (e.g. quantum computing, CRISPR, gravitational waves) with zero support in the corpus.
- **Abstention Behavior**: Because standard FAISS dense retrieval computes $K$-nearest neighbors without a distance rejection threshold, the system unconditionally returns 5 chunks for every negative query (average L2 distance 0.72–0.98).
- **Production Implication**: An abstention mechanism (e.g. distance thresholding $>0.75$ or a confidence judge) must be incorporated in production to prevent hallucinations on out-of-domain queries.

---

## 5. Readiness for Step 12 (Hybrid Retrieval BM25 + FAISS)

With the corrected ground truth and a validated baseline:
1. **Baseline Frozen**: Rank 1 Accuracy is locked at **79.78%**, MRR at **0.8890**, and nDCG@5 at **91.75%**.
2. **Clear Target for Step 12**: The 36 non-Rank-1 queries in Group A suffer from dense vector vocabulary dispersion (where semantically related chunks from other sections scored marginally closer than the exact factual passage). Hybrid BM25 lexical keyword scoring is specifically positioned to boost these exact-keyword matches directly into Rank 1.
