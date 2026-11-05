# Step 3 — Retrieval Quality Evaluation Report
> **Benchmark Session:** `benchmark_70docs_v1`  
> **FAISS Index:** `faiss_index/benchmark_70docs_v1/`  
> **Embedding Model:** `models/gemini-embedding-001`  
> **Evaluation Date:** 2026-11-02 21:31:36  
> **Top-K:** 5  

---
## 1. Executive Summary

| Item | Value |
| :--- | :---: |
| Total questions attempted | **150** |
| Completed without error   | **150** |
| Failed (API / FAISS error)| **0** |
| Total API retries         | **0** |
| Wall-clock evaluation time| **224.0s** |

## 2. Input Validation

**Pre-flight validation:** ✅ PASSED

- `verified`: 128 questions
- `not_found`: 17 questions
- `partially_supported`: 5 questions

## 3. Index and Ground-Truth Integrity

| File | Before (bytes, mtime) | After (bytes, mtime) | Unchanged? |
| :--- | :--- | :--- | :---: |
| `index.faiss` | 2224173, 1793551416 | 2224173, 1793551416 | ✅ |
| `retrieval_evaluation_dataset.json` | 142078, 1793584234 | 142078, 1793584234 | ✅ |

## 4. Primary Retrieval Metrics — Verified Questions (N=128)

| Metric | Value | Denominator |
| :--- | :---: | :--- |
| **Hit Rate@5** | **100.0%** | 128 hits / 128 verified questions |
| **Precision@5** | **21.88%** | 140 relevant positions / 640 total retrieved positions |
| **MRR** | **0.8626** | Σ(1/rank) / 128 |
| **Rank 1 Accuracy** | **75.0%** | 96 first-position hits / 128 |
| Misses | 0 | Not retrieved in top-5 |

## 5. Partially Supported Questions (N=5)

| Metric | Value |
| :--- | :---: |
| Hit Rate@5 | 100.0% |
| Precision@5 | 32.0% |
| MRR | 0.9 |
| Rank 1 Accuracy | 80.0% |

| ID | Hit | Rank | Score | Retrieved Source (Rank 1) |
| :---: | :---: | :---: | :---: | :--- |
| 44 | ✅ | 1 | 0.582792 | `Configuring K8s HPA for Custom Metrics.pdf` |
| 78 | ✅ | 1 | 0.694245 | `byzantine_empire_and_justinian_code.txt` |
| 79 | ✅ | 2 | 0.693393 | `byzantine_empire_and_justinian_code.txt` |
| 90 | ✅ | 1 | 0.664084 | `aztec_empire_spanish_conquest.txt` |
| 98 | ✅ | 1 | 0.613525 | `soviet_command_economy_and_five_year_plans.txt` |

## 6. Negative-Control Questions (N=17)

> [!NOTE]
> These 17 questions have `expected_relevant_chunk_ids: []` because their topics (HATEOAS, GIL, CAP theorem, etc.) are absent from the corpus.
> FAISS *always* returns nearest neighbours; without a calibrated rejection threshold there is **no claim of rejection accuracy**.
> The table below shows the top-retrieved chunk and score for each to allow manual inspection of false-positive risk.

| ID | Top Retrieved Source | Score@1 | Top 5 Sources (summary) |
| :---: | :--- | :---: | :--- |
| 27 | `Idempotency in RESTful API Design.pdf` | 0.684924 | `Idempotency in RESTful API Design.pdf`, `Idempotency in RESTful API Design.pdf`, `DOC-SEC-601.docx` |
| 28 | `Implementing Asynchronous Generators in Python.pdf` | 0.758744 | `Implementing Asynchronous Generators in Python.pdf`, `Implementing Asynchronous Generators in Python.pdf`, `Advanced Metaprogramming with Python Metaclasses.pdf` |
| 29 | `DOC-NET-802.docx` | 0.691756 | `DOC-NET-802.docx`, `Circuit Breaker Pattern in Microservices.pdf`, `The Saga Pattern for Distributed Transactions.pdf` |
| 30 | `Idempotency in RESTful API Design.pdf` | 0.693829 | `Idempotency in RESTful API Design.pdf`, `Idempotency in RESTful API Design.pdf`, `Circuit Breaker Pattern in Microservices.pdf` |
| 31 | `Optimizing PostgreSQL JSONB Query Performance.pdf` | 0.776886 | `Optimizing PostgreSQL JSONB Query Performance.pdf`, `The Saga Pattern for Distributed Transactions.pdf`, `Redis Cluster Data Sharding and High Availability.pdf` |
| 32 | `Redis Cluster Data Sharding and High Availability.pdf` | 0.614622 | `Redis Cluster Data Sharding and High Availability.pdf`, `Redis Cluster Data Sharding and High Availability.pdf`, `DOC-ARC-901.docx` |
| 33 | `Paxos and Raft Consensus Algorithms.pdf` | 0.662818 | `Paxos and Raft Consensus Algorithms.pdf`, `Paxos and Raft Consensus Algorithms.pdf`, `The Saga Pattern for Distributed Transactions.pdf` |
| 34 | `Paxos and Raft Consensus Algorithms.pdf` | 0.7693 | `Paxos and Raft Consensus Algorithms.pdf`, `DOC-ARC-901.docx`, `Redis Cluster Data Sharding and High Availability.pdf` |
| 38 | `Circuit Breaker Pattern in Microservices.pdf` | 0.764346 | `Circuit Breaker Pattern in Microservices.pdf`, `Circuit Breaker Pattern in Microservices.pdf`, `The Saga Pattern for Distributed Transactions.pdf` |
| 39 | `Circuit Breaker Pattern in Microservices.pdf` | 0.721081 | `Circuit Breaker Pattern in Microservices.pdf`, `Circuit Breaker Pattern in Microservices.pdf`, `Automotive On-Board Diagnostics and Telematics.pdf` |
| 40 | `Paxos and Raft Consensus Algorithms.pdf` | 0.691907 | `Paxos and Raft Consensus Algorithms.pdf`, `Paxos and Raft Consensus Algorithms.pdf`, `DOC-ARC-901.docx` |
| 41 | `Configuring K8s HPA for Custom Metrics.pdf` | 0.759622 | `Configuring K8s HPA for Custom Metrics.pdf`, `Paxos and Raft Consensus Algorithms.pdf`, `Configuring K8s HPA for Custom Metrics.pdf` |
| 43 | `Configuring K8s HPA for Custom Metrics.pdf` | 0.771808 | `Configuring K8s HPA for Custom Metrics.pdf`, `DOC-NET-802.docx`, `DOC-NET-802.docx` |
| 45 | `Configuring K8s HPA for Custom Metrics.pdf` | 0.737171 | `Configuring K8s HPA for Custom Metrics.pdf`, `Configuring K8s HPA for Custom Metrics.pdf`, `Circuit Breaker Pattern in Microservices.pdf` |
| 49 | `Zero Trust Network Access (ZTNA) Implementation.pdf` | 0.71593 | `Zero Trust Network Access (ZTNA) Implementation.pdf`, `DOC-SEC-601.docx`, `DOC-DEV-1003.docx` |
| 50 | `Zero Trust Network Access (ZTNA) Implementation.pdf` | 0.624407 | `Zero Trust Network Access (ZTNA) Implementation.pdf`, `Zero Trust Network Access (ZTNA) Implementation.pdf`, `DOC-NET-802.docx` |
| 80 | `byzantine_empire_and_justinian_code.txt` | 0.702912 | `byzantine_empire_and_justinian_code.txt`, `byzantine_empire_and_justinian_code.txt`, `DOC-HIS-2217.docx` |

## 7. Latency Analysis

| Latency Component | Min (s) | Max (s) | Avg (s) |
| :--- | :---: | :---: | :---: |
| Embedding (Gemini API) | 0.51321 | 1.03487 | 0.58878 |
| FAISS Search           | 0.50071 | 1.48264 | 0.60399 |
| Total per query        | 1.03177 | 2.05195 | 1.19277 |
| **Total wall time**    |       |       | **224.0s** |

## 8. API Failures and Retries

- Total retries across all queries: **0**
- Failed queries (embedding error): **0**

## 9. Lowest-Ranking and Failed Queries

### 9.1 Verified Questions With No Hit in Top-5 (0 misses)

*No misses — all verified questions retrieved at least one relevant chunk in top-5.*

### 9.2 Top-10 Lowest-Rank Hits (Verified, rank > 1)

| ID | Rank | Score@Rank | Question Snippet |
| :---: | :---: | :---: | :--- |
| 54 | 4 | 0.704602 | In what ways did the accumulation of industrial capital… |
| 22 | 3 | 0.719912 | How does utilizing OpenID Connect (OIDC) to federate id… |
| 51 | 3 | 0.701768 | How did James Watt's improvements to the steam engine a… |
| 55 | 3 | 0.774902 | How did the mass production of high-quality wrought iro… |
| 56 | 3 | 0.673439 | How did the cessation of territorial expansion in the 2… |
| 57 | 3 | 0.699787 | What were the long-term socioeconomic effects of the Ro… |
| 58 | 3 | 0.715117 | How did the 'barbarization' of the Roman legions with G… |
| 71 | 3 | 0.65461 | How did the logistics of the Silk Road rely on a relay … |
| 149 | 3 | 0.65623 | What role does the vehicle's telemetry data play in fee… |
| 1 | 2 | 0.511687 | What is the fundamental difference between authenticati… |

## 10. Source-Document Retrieval Distribution (Top-5 across all 150 queries)

| Source Document | Times Retrieved in Top-5 |
| :--- | :---: |
| `Zero Trust Network Access (ZTNA) Implementation.pdf` | 28 |
| `Automotive_Tribology_Friction_and_Wear.docx` | 28 |
| `Internal_Combustion_Engine_Thermodynamics.docx` | 25 |
| `Paxos and Raft Consensus Algorithms.pdf` | 24 |
| `DOC-ARC-901.docx` | 23 |
| `DOC-HIS-2419.docx` | 21 |
| `DOC-SEC-601.docx` | 20 |
| `DOC-NET-802.docx` | 20 |
| `Automotive_CAN_Bus_Architecture_and_Diagnostics.docx` | 20 |
| `Automotive_ECU_Architecture_and_Microcontrollers.docx` | 19 |
| `Automotive_Crashworthiness_and_Structural_Safety.docx` | 19 |
| `Redis Cluster Data Sharding and High Availability.pdf` | 18 |
| `fall_of_the_western_roman_empire.txt` | 18 |
| `FCEV_Hydrogen_Storage_and_Stack_Architecture.docx` | 18 |
| `DOC-DEV-1003.docx` | 17 |
| `Circuit Breaker Pattern in Microservices.pdf` | 17 |
| `Vehicle_Aerodynamics_and_Computational_Fluid_Dynamics.docx` | 17 |
| `DOC-HIS-2116.docx` | 16 |
| `The Black Death and Socioeconomic Transformation.pdf` | 16 |
| `DOC-HIS-2217.docx` | 16 |
| `The Russian Revolution and the Rise of the Bolsheviks.pdf` | 16 |
| `Automotive On-Board Diagnostics and Telematics.pdf` | 15 |
| `Electric_Vehicle_Battery_Management_Systems_BMS.docx` | 15 |
| `the_haitian_revolution.txt` | 14 |
| `The Saga Pattern for Distributed Transactions.pdf` | 13 |

## 11. Performance Observations and Recommendations

- **FAISS distance metric:** L2 (cosine-equivalent after L2-normalised Gemini embeddings). Lower score = higher similarity.
- **Retrieval is similarity only:** No rejection threshold is implemented; negative-control score distributions should be inspected before deploying an abstention mechanism.
- **Negative controls:** Topics absent from the corpus (e.g. HATEOAS, GIL, CAP theorem) still receive near-neighbour results — monitor top-1 scores when filtering.
- **Latency:** Average total per-query latency is dominated by Gemini API embedding time. FAISS search is sub-millisecond.

---
*Report generated by `tests/test_70docs_retrieval_evaluation.py`.*