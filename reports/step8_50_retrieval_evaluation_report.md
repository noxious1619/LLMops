# Step 8 — Independent Retrieval Evaluation Report (50 New Questions)

> **Evaluation Session:** `benchmark_70docs_v1` (50 Additional Questions Evaluation)  
> **Questions Dataset:** `tests/retrieval_questions_50.json` (Q151–Q200)  
> **Ground-Truth Dataset:** `tests/retrieval_ground_truth_50.json`  
> **FAISS Index:** `faiss_index/benchmark_70docs_v1/` (Read-Only)  
> **Embedding Model:** `models/gemini-embedding-001` (3072 Dimensions)  
> **Top-K Retrieved:** 5  
> **Evaluation Date:** 2026-11-03 11:50:50  

---

## 1. Executive Summary

An independent evaluation of the **50 new evaluation questions (Q151–Q200)** was executed against the existing 181-vector FAISS index.

| Evaluation Metric | Measured Value | Benchmark Description |
| :--- | :---: | :--- |
| **Total Questions Attempted** | **50** | Complete evaluation set (Q151–Q200) |
| **Successfully Executed** | **50** | 100% API calls completed without errors or timeouts |
| **Failed Query Calls** | **0** | Zero API or FAISS runtime exceptions |
| **Hit Rate@5** | **100.0%** | Percentage of questions with relevant chunk in top-5 (50/50) |
| **Precision@5** | **24.0%** | Total relevant chunks in top-5 / (50 × 5 total retrieved slots) |
| **Mean Reciprocal Rank (MRR)** | **0.9367** | Average reciprocal rank across all 50 questions |
| **Rank 1 Accuracy** | **88.0%** | Relevant chunk retrieved at position 1 (44/50) |
| **Total Wall-Clock Time** | **28.7s** | Total end-to-end evaluation execution duration |

---

## 2. Integrity and Safety Verification

All evaluation and index constraints were strictly verified:

| File / Component | Check | Result | Status |
| :--- | :--- | :--- | :---: |
| `faiss_index/benchmark_70docs_v1/index.faiss` | Vector count & read-only hash | 181 vectors, size/mtime unchanged | ✅ PASS |
| `faiss_index/benchmark_70docs_v1/index.pkl` | Docstore chunk records | 181 chunks loaded and verified | ✅ PASS |
| `tests/retrieval_questions_50.json` | Dataset non-modification | File size and mtime intact | ✅ PASS |
| `tests/retrieval_ground_truth_50.json` | Ground-truth non-modification | File size and mtime intact | ✅ PASS |
| Original 150-Question Benchmark | Step 6 results/report isolation | Unmodified and completely isolated | ✅ PASS |
| Embedding Provider | Google Gemini API (gemini-embedding-001) | 3072-dimensional real query embeddings | ✅ PASS |
| Search Configuration | FAISS L2 Euclidean distance | k=5 nearest-neighbor search | ✅ PASS |

---

## 3. Category-Wise Performance Breakdown

The 50 questions span 5 distinct categories created during Step 7:

| Question Category | Count | Hit Rate@5 | Rank 1 Accuracy | MRR | Precision@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Factual (Q151–Q165)** | 15 | **100.0%** | **100.0%** | **1.0** | 20.0% |
| **Conceptual (Q166–Q175)** | 10 | **100.0%** | **80.0%** | **0.8833** | 20.0% |
| **Comparison (Q176–Q185)** | 10 | **100.0%** | **80.0%** | **0.9** | 20.0% |
| **Multi-Chunk (Q186–Q195)** | 10 | **100.0%** | **100.0%** | **1.0** | 40.0% |
| **Difficult Reasoning (Q196–Q200)** | 5 | **100.0%** | **60.0%** | **0.8** | 20.0% |

---

## 4. Difficulty Tier Breakdown

| Difficulty Tier | Count | Hit Rate@5 | Rank 1 Accuracy | MRR | Precision@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Easy** | 7 | **100.0%** | **100.0%** | **1.0** | 20.0% |
| **Medium** | 28 | **100.0%** | **85.71%** | **0.9226** | 20.0% |
| **Hard** | 15 | **100.0%** | **86.67%** | **0.9333** | 33.33% |

---

## 5. Single-Chunk vs. Multi-Chunk Analysis

| Query Complexity | Count | Hit Rate@5 | Rank 1 Accuracy | MRR | Precision@5 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Single-Chunk Queries** | 40 | **100.0%** | **85.0%** | **0.9208** | 20.0% |
| **Multi-Chunk Queries** | 10 | **100.0%** | **100.0%** | **1.0** | 40.0% |

> [!NOTE]
> For multi-chunk queries, Precision@5 is naturally higher (40.0%) because each query has 2 target chunks eligible for retrieval in the top-5 window. Hit Rate@5 is evaluated on whether at least one relevant chunk was retrieved in the top-5.

---

## 6. Latency Performance

| Latency Component | Minimum | Maximum | Average |
| :--- | :---: | :---: | :---: |
| **Query Embedding (Gemini API)** | 0.4989s | 0.61602s | **0.57366s** |
| **FAISS Vector Search** | 0.00025s | 0.00054s | **0.0003s** |
| **Total Query Latency** | 0.49919s | 0.61639s | **0.57396s** |

---

## 7. Retrieval Outliers and Misses

### 7.1 Retrieval Misses (0 in Top-5)

**Zero misses occurred.** All 50 questions retrieved at least one relevant chunk within the top-5 results.

### 7.2 Non-Rank-1 Retrievals (Hits at Rank 2–5)
Questions where the nearest neighbor was not the primary relevant chunk (demonstrating vector competition across documents):

| Question ID | Rank | Score | Question Snippet | Top Retrieved Document |
| :---: | :---: | :---: | :--- | :--- |
| **Q174** | **3** | 0.4567 | How does transitioning to a centralized zonal compute archit… | `Automotive_ECU_Architecture_and_Microcontrollers.docx` |
| **Q170** | **2** | 0.4859 | How does a Steer-by-Wire (SbW) system eliminate mechanical l… | `adas_actuation_and_steer_by_wire.txt` |
| **Q178** | **2** | 0.4446 | How do Permanent Magnet Synchronous Motors (PMSM) and AC Ind… | `electric_vehicle_motor_topologies.txt` |
| **Q182** | **2** | 0.5702 | How did Portuguese maritime exploration along the African co… | `The Age of Discovery and Maritime Trade Routes.pdf` |
| **Q196** | **2** | 0.4700 | In extreme freezing winter conditions, why does a vehicle he… | `automotive_hvac_and_heat_pump_systems.txt` |
| **Q198** | **2** | 0.6296 | How did the strategic premise of American containment shift … | `Cold War Geopolitics_ The Strategy of Containment.pdf` |

### 7.3 Top-10 Most Confident Rank 1 Retrievals (Lowest Distance Scores)

| Question ID | Rank | Distance Score | Category | Question Snippet |
| :---: | :---: | :---: | :--- | :--- |
| **Q195** | 1 | 0.3954 | multi_chunk | How does an Electric Vehicle Battery Management System (BMS) comb… |
| **Q193** | 1 | 0.4145 | multi_chunk | How do understeer and oversteer dynamics originate from tire slip… |
| **Q171** | 1 | 0.4236 | conceptual | What is tire slip angle in vehicle dynamics, and how does it gene… |
| **Q191** | 1 | 0.4257 | multi_chunk | How did vehicle on-board diagnostics evolve from standardized phy… |
| **Q156** | 1 | 0.4347 | factual | What HTTP status code should an API server return when receiving … |
| **Q179** | 1 | 0.4379 | comparison | What are the mechanical distinctions between Dual-Clutch Transmis… |
| **Q177** | 1 | 0.4469 | comparison | What architectural and infrastructure differences distinguish Ded… |
| **Q197** | 1 | 0.4497 | difficult_reasoning | Under what high-volume write conditions should a database archite… |
| **Q157** | 1 | 0.4590 | factual | What SQL command syntax creates a path-optimized GIN index on a P… |
| **Q192** | 1 | 0.4681 | multi_chunk | What factors limit electric motor regenerative braking, and how d… |

---

## 8. Limitations and Observations

1. **In-Distribution Domain Specificity:**  
   The questions target specific facts and concepts documented in `docs_70/`. Because the 68 documents represent distinct domains (classical mechanics, ancient history, automotive control theory, REST API architecture), the 3072-dimensional Gemini embeddings cleanly cluster domain-specific terms into separated manifolds.
2. **Rank 1 Competition:**  
   The questions that landed at Rank 2 or 3 reveal genuine vector competition, typically where adjacent background chunks or closely related technical topics (e.g. steer-by-wire vs EPS, or related history documents) were retrieved slightly closer.
3. **Execution vs Retrieval Distinction:**  
   100% of API calls completed without errors (100% execution success). Retrieval effectiveness is reflected by the **Rank 1 Accuracy** and **MRR**, which quantify actual rank positioning rather than binary API availability.