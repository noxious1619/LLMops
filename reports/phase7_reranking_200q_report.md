# Phase 7: Full 200-Question Evaluation Report — Hybrid Retrieval + Cross-Encoder Reranking

## Executive Summary

This report documents the standalone quantitative evaluation of the **production two-stage retrieval pipeline** (FAISS Dense Top-10 + BM25 Sparse Top-10 fused via Reciprocal Rank Fusion $k=60$ into a 20-candidate pool, followed by Cross-Encoder reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2` with batch size 5 on NVIDIA RTX 3050 GPU) across all **200 questions** in `tests/retrieval_ground_truth_200_corrected_v1.json`.

### Benchmark Architecture & Configuration
- **First-Stage Dense Retriever**: FAISS CPU (`IndexFlatL2`, 181 vectors, 3072 dims via Google Gemini `models/gemini-embedding-001`, Top-10)
- **First-Stage Sparse Retriever**: BM25Okapi over canonical 181 chunks (Top-10)
- **Fusion Algorithm**: Reciprocal Rank Fusion (RRF, constant $k=60$)
- **Candidate Pool Depth**: Top-20 unique chunks after deduplication
- **Second-Stage Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Reranker Execution Device**: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (CUDA active, RTX 3050 6GB Laptop GPU)
- **Reranker Batch Size**: 5
- **Final Retrieved Depth**: Top-5 chunks passed to prompt context
- **Corpus**: 68 documents (181 total chunks)
- **Evaluated Questions**: 200 total (178 Verified Primary, 5 Partially Supported, 17 Negative Controls)
- **Total Benchmark Wall-Clock Duration**: 159.67s

---

## 1. Dataset & Execution Cohorts

| Evaluation Cohort | Total Questions | Evaluated Successfully | Skipped / Failed | Ground-Truth Status |
|---|---:|---:|---:|---|
| **Group A: Verified Primary** | 178 | 178 | 0 | Fully verified against corpus evidence |
| **Group B: Partially Supported** | 5 | 5 | 0 | Corpus contains partial evidence |
| **Group C: Negative Controls** | 17 | 17 | 0 | Out-of-domain (expected: 0 relevant chunks) |
| **Total** | **200** | **200** | **0** | **200 questions executed** |

> [!NOTE]
> 100% of the 200 benchmark questions were executed against the Two-Stage Hybrid + Cross-Encoder pipeline without any failures, exceptions, or CPU fallback events.

---

## 2. Primary Performance Metrics (Group A: Verified Primary, N=178)

The canonical performance metrics for the **178 Verified Primary Questions (Group A)** reflect the full two-stage pipeline effectiveness:

| Metric | Result | Denominator / Definition |
|---|---:|---|
| **Candidate Recall@20 (Macro)** | **100.0%** | Average per-query recall in the RRF 20-candidate pool prior to reranking |
| **Candidate Recall@20 (Micro)** | **100.0%** | Relevant chunks in candidate pool / Total ground-truth chunks (200) |
| **Candidate Hit@20** | **100.0%** | 178 / 178 queries with $\ge 1$ relevant chunk in candidate pool |
| **Hit@5** | **100.0%** | 178 / 178 queries with $\ge 1$ relevant chunk in Top-5 |
| **Rank-1 Accuracy** | **91.01%** | 162 / 178 queries where the Rank 1 chunk is relevant |
| **MRR (Mean Reciprocal Rank)** | **0.9537** | Mean Reciprocal Rank ($1/\text{rank}$ of first hit) over 178 queries |
| **Precision@5** | **22.36%** | 199 relevant chunks retrieved / (178 $\times$ 5) slots |
| **Recall@5 (Macro)** | **99.72%** | Average per-query Recall@5 across all 178 queries |
| **Recall@5 (Micro)** | **99.5%** | 199 relevant retrieved / 200 total ground-truth chunks |
| **nDCG@5** | **96.25%** | Normalized Discounted Cumulative Gain at Top-5 across 178 queries |
| **Min-Sufficient Coverage** | **99.44%** | Queries retrieving all designated minimum sufficient chunks in Top-5 |

---

## 3. Category-Wise Cohort Results

| Metric | Group A (Verified, N=178) | Group B (Partial, N=5) | Group C (Negative, N=17) |
|---|---:|---:|---:|
| **Candidate Hit@20** | 100.0% | 100.0% | 0.0% |
| **Candidate Recall@20 (Macro)** | 100.0% | 100.0% | N/A |
| **Candidate Recall@20 (Micro)** | 100.0% | 100.0% | N/A |
| **Hit@5** | 100.0% | 100.0% | 0.0% |
| **Rank-1 Accuracy** | 91.01% | 100.0% | 0.0% |
| **MRR** | 0.9537 | 1.0 | 0.0 |
| **Recall@5 (Macro)** | 99.72% | 100.0% | N/A |
| **Recall@5 (Micro)** | 99.5% | 100.0% | N/A |
| **Precision@5** | 22.36% | 32.0% | 0.0% |
| **nDCG@5** | 96.25% | 100.0% | 0.0% |
| **Min-Sufficient Coverage** | 99.44% | 100.0% | N/A |

### Negative Control Audit (Group C)
- **Total Negative Queries Evaluated**: 17
- **Total Candidate Slots Checked**: 340
- **Total Final Slots Retrieved**: 85
- **Corpus Chunks Matching Any Known Ground-Truth Chunk**: 81
- **False Positive Ground-Truth Retrieval Rate**: 0.0%
- **Negative Control Rerank Score Distribution**:
  - Mean Score: `-5.9182`
  - Median Score (P50): `-5.8884`
  - Minimum Score: `-10.1865`
  - Maximum Score: `0.1006`

> [!NOTE]
> Negative queries consistently receive low Cross-Encoder scores (mean rerank score is heavily negative), providing strong discriminative signal for downstream hallucination suppression.

---

## 4. Latency & Resource Profiling

All measurements reflect dedicated inference on NVIDIA GeForce RTX 3050 Laptop GPU with PyTorch CUDA acceleration.

### Per-Stage Latency Breakdown (Across All 200 Questions)

| Pipeline Stage | Mean (ms) | P50 / Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|---:|
| **1. Dense Retrieval (FAISS + Embedding)** | 545.47 | 536.72 | 615.06 | 483.5 | 1175.93 |
| **2. Sparse Retrieval (BM25Okapi)** | 1.53 | 1.38 | 2.37 | 0.91 | 2.99 |
| **3. RRF Fusion & Candidate Dedup** | 0.12 | 0.1 | 0.22 | 0.07 | 0.31 |
| **Candidate Retrieval Subtotal (Stage 1)** | 547.11 | 538.28 | 616.21 | 484.63 | 1178.39 |
| **4. Cross-Encoder Reranking (Stage 2)** | 100.57 | 96.35 | 146.75 | 68.27 | 236.24 |
| **Total Two-Stage Retrieval Latency** | **647.68** | **637.03** | **724.41** | **563.92** | **1289.16** |

### Hardware & VRAM Profiling
- **GPU Hardware**: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Dedicated VRAM**: 6143.5 MB (6.0 GB)
- **Initial Allocated VRAM**: 86.65 MB
- **Peak Allocated VRAM**: 140.06 MB
- **Final Allocated VRAM**: 94.78 MB
- **Net Memory Delta**: +8.12 MB
- **Zero Memory Leak Verification**: Confirmed. Net VRAM delta after 200 consecutive evaluations is negligible (+8.12 MB).

---

## 5. Non-Rank-1 Retrieval Analysis (Group A)

In Group A (Verified Primary), **162 out of 178 queries (91.01%)** retrieved a verified relevant chunk directly at **Rank 1**.

There were **16 queries** where the first relevant chunk appeared at Rank 2 or below:

| Question ID | First Relevant Rank | Hit@5 | Ground Truth Chunks | Retrieved Top-1 Chunk | Top-1 Score | Top-1 Source |
|---|:---:|:---:|---|---|---:|---|
| `Q8` | Rank 2 | True | `690a06854d38...` | `21880320ccda...` | 7.456496 | DOC-ML-705.docx |
| `Q48` | Rank 2 | True | `6896e97a707d...` | `7cea7ee15c77...` | -4.093485 | DOC-DEV-1003.docx |
| `Q52` | Rank 2 | True | `1b0f1a5005ed...` | `da5fcfcd4528...` | 3.23398 | british_industrial_revolution.txt |
| `Q53` | Rank 2 | True | `1b0f1a5005ed...` | `da5fcfcd4528...` | 3.461075 | british_industrial_revolution.txt |
| `Q55` | Rank 2 | True | `1b0f1a5005ed...` | `ae41f29d9176...` | 0.000464 | british_industrial_revolution.txt |
| `Q65` | Rank 2 | True | `aeb5d6a85cfb..., 2a08c3f6222f...` | `50c8846d1f39...` | 4.284386 | the_meiji_restoration.txt |
| `Q75` | Rank 4 | True | `81d2cff6ed56...` | `df9d9f5f76c7...` | 5.407926 | silk_road_trade_networks.txt |
| `Q85` | Rank 2 | True | `a2205e36bef0...` | `27bd1db7c0d8...` | -0.264847 | thirty_years_war_and_westphalia.txt |
| `Q99` | Rank 2 | True | `8816857cf5bb...` | `dda33a0bc078...` | -0.452657 | soviet_command_economy_and_five_year_plans.txt |
| `Q100` | Rank 2 | True | `8816857cf5bb...` | `dda33a0bc078...` | -0.415427 | soviet_command_economy_and_five_year_plans.txt |
| `Q103` | Rank 2 | True | `5e15540954da...` | `0e811e581ced...` | 3.967373 | Automotive_ECU_Architecture_and_Microcontrollers.docx |
| `Q174` | Rank 2 | True | `5ec3013d18d3...` | `0e811e581ced...` | 2.23738 | Automotive_ECU_Architecture_and_Microcontrollers.docx |
| `Q178` | Rank 2 | True | `0a69fdba5be1...` | `02239555f4b0...` | 4.752998 | electric_vehicle_motor_topologies.txt |
| `Q181` | Rank 2 | True | `226acd19eb61...` | `7b0b026d058f...` | 0.394594 | The Russian Revolution and the Rise of the Bolsheviks.pdf |
| `Q182` | Rank 2 | True | `7cc4c4e46579...` | `8094f9bdc311...` | 3.891288 | The Age of Discovery and Maritime Trade Routes.pdf |
| `Q198` | Rank 2 | True | `ea5fb4b74238...` | `cbea36887cdd...` | 2.977013 | Cold War Geopolitics_ The Strategy of Containment.pdf |

---

## 6. Cryptographic Index Integrity Verification

SHA-256 hashes of all input indexes, corpora, and benchmark datasets were recorded pre-flight and verified immediately following evaluation completion:

| File / Artifact | Expected Baseline SHA-256 | Pre-Flight Hash | Post-Flight Hash | Match Status |
|---|---|---|---|:---:|
| `faiss_index\benchmark_70docs_v1\index.faiss` | `246209c421d0cd78817082c2190c1a1143a1ce7857e17ac8474426469b7b780a` | `246209c421d0cd78817082c2190c1a1143a1ce7857e17ac8474426469b7b780a` | `246209c421d0cd78817082c2190c1a1143a1ce7857e17ac8474426469b7b780a` | **MATCH (Identical)** |
| `faiss_index\benchmark_70docs_v1\index.pkl` | `33419862b54be18c6c88586f3537b9ce897ac618d9bbcfa1f5b1f67f60cd2621` | `33419862b54be18c6c88586f3537b9ce897ac618d9bbcfa1f5b1f67f60cd2621` | `33419862b54be18c6c88586f3537b9ce897ac618d9bbcfa1f5b1f67f60cd2621` | **MATCH (Identical)** |
| `data\bm25_index\bm25_index.pkl` | `9bf695fafd18f60555439acc1ab6795cf715e51b45094762896f10d87aca707d` | `9bf695fafd18f60555439acc1ab6795cf715e51b45094762896f10d87aca707d` | `9bf695fafd18f60555439acc1ab6795cf715e51b45094762896f10d87aca707d` | **MATCH (Identical)** |
| `data\bm25_index\bm25_manifest.json` | `85f28f36887af22dd5320c7f84d7964333d18de02c86a57ee6861ba3e3a2590d` | `85f28f36887af22dd5320c7f84d7964333d18de02c86a57ee6861ba3e3a2590d` | `85f28f36887af22dd5320c7f84d7964333d18de02c86a57ee6861ba3e3a2590d` | **MATCH (Identical)** |
| `data\bm25_corpus\bm25_corpus_70docs_181chunks.json` | `fc4bd84f95adf9f6e1e718bf768e2b1d2d47597db07dd0dc76e309a97b316e14` | `fc4bd84f95adf9f6e1e718bf768e2b1d2d47597db07dd0dc76e309a97b316e14` | `fc4bd84f95adf9f6e1e718bf768e2b1d2d47597db07dd0dc76e309a97b316e14` | **MATCH (Identical)** |
| `tests\retrieval_ground_truth_200_corrected_v1.json` | `0b0cd9d97eb7f16d7b90fc9d0db274472114de3f23d6c2e9667671c7778876a2` | `0b0cd9d97eb7f16d7b90fc9d0db274472114de3f23d6c2e9667671c7778876a2` | `0b0cd9d97eb7f16d7b90fc9d0db274472114de3f23d6c2e9667671c7778876a2` | **MATCH (Identical)** |

> [!IMPORTANT]
> 100% Cryptographic immutability maintained. Zero files were modified during Phase 7 evaluation.

---

## 7. Phase 7 Quantitative Summary

- **Total Benchmark Questions**: 200
- **Primary Evaluation Cohort (Group A)**: 178 questions
- **Candidate Recall@20 (Macro)**: 100.0%
- **Candidate Recall@20 (Micro)**: 100.0%
- **Hit@5**: 100.0%
- **Rank-1 Accuracy**: 91.01%
- **MRR**: 0.9537
- **Precision@5**: 22.36%
- **Macro Recall@5**: 99.72%
- **Micro Recall@5**: 99.5%
- **nDCG@5**: 96.25%
- **Minimum Sufficient Coverage**: 99.44%
- **Average End-to-End Retrieval Latency**: 647.68 ms (P50: 637.03 ms, P95: 724.41 ms)
- **Cross-Encoder Inference Latency**: 100.57 ms (P50: 96.35 ms, P95: 146.75 ms)

This empirical dataset forms the rigorous factual foundation for the Phase 8 Production Decision.