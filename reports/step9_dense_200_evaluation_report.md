# Step 9 — Complete Dense Retrieval Baseline Evaluation (200 Questions)

> **Evaluation Session:** `benchmark_70docs_v1`  
> **Retrieval Architecture:** Dense Semantic Retrieval (FAISS L2 + Google Gemini `models/gemini-embedding-001`, 3072 dims)  
> **Evaluation Scope:** Consolidated 200 Questions (150 Original Benchmark + 50 Dataset Expansion)  
> **Corpus:** 68 documents (`docs_70/`), 181 indexed vectors / chunks  
> **Execution Date:** 2026-11-03 13:57:37  

---

## 1. Executive Summary

This report establishes the **official Dense Retrieval Baseline** across the consolidated **200-question benchmark**. All 200 questions were embedded on-the-fly using the real Google Gemini Embedding 001 API and queried against the 181-vector FAISS index in strict read-only mode.

### 1.1 Consolidated Results Across Evaluation Groups

| Evaluation Group | Question Count | Execution Status | Hit Rate@5 | Rank 1 Accuracy | MRR | Macro Recall@5 | Micro Recall@5 | Precision@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Group A — Verified Questions (Primary Benchmark)** | **178** | 178 / 178 (100%) | **100.0%** | **78.65%** | **0.8834** | **100.0%** | **100.0%** | **22.47%** |
| *— Original Verified (Q1–Q150)* | 128 | 128 / 128 (100%) | 100.0% | 75.0% | 0.8626 | 100.0% | 100.0% | 21.88% |
| *— New Verified (Q151–Q200)* | 50 | 50 / 50 (100%) | 100.0% | 88.0% | 0.9367 | 100.0% | 100.0% | 24.0% |
| **Group B — Partially Supported Questions** | **5** | 5 / 5 (100%) | **100.0%** | **80.0%** | **0.9** | **100.0%** | **100.0%** | **32.0%** |
| **Group C — Negative Controls (Out-of-Domain)** | **17** | 17 / 17 (100%) | *N/A (0 GT)* | *N/A* | *0.0000* | *N/A* | *N/A* | *0.0%* |
| **Total Cohort (All Queries)** | **200** | **200 / 200 (100%)** | — | — | — | — | — | — |

*Wall-clock execution time for all 200 questions: **107.6s** (Average end-to-end latency per query: **0.538s**).*

---

## 2. Integrity and Reproducibility Audit

### 2.1 File Hashes and Immutability Verification
Pre- and post-flight cryptographic hashes confirm that zero data files, vector stores, or ground-truth files were modified or rebuilt during evaluation:

| Resource Path | Description | Pre-Flight SHA-256 | Post-Flight SHA-256 | Status |
| :--- | :--- | :--- | :--- | :---: |
| `faiss_index/benchmark_70docs_v1/index.faiss` | Vector index binary | `246209c421d0cd78…` | `246209c421d0cd78…` | ✅ IDENTICAL |
| `faiss_index/benchmark_70docs_v1/index.pkl` | Docstore records | `33419862b54be18c…` | `33419862b54be18c…` | ✅ IDENTICAL |
| `tests/retrieval_evaluation_dataset.json` | Original 150 questions | `bc383d31227b63bd…` | `bc383d31227b63bd…` | ✅ IDENTICAL |
| `tests/retrieval_questions_50.json` | 50 Questions | `a69c6246613e113a…` | `a69c6246613e113a…` | ✅ IDENTICAL |
| `tests/retrieval_ground_truth_50.json` | 50 Ground Truth | `963e99db50ed421d…` | `963e99db50ed421d…` | ✅ IDENTICAL |

### 2.2 System & Runtime Environment
- **Python Version:** 3.14.3
- **LangChain:** 1.4.3
- **LangChain Community:** 0.4.2
- **LangChain Google GenAI:** 4.4.0
- **FAISS Distance Metric:** L2 Euclidean Distance (Ranked ascending: lower distance = higher similarity)
- **Top-K Retrieved:** 5 chunks per query

---

## 3. Primary Retrieval Benchmark: Group A Verified Questions (N=178)

Group A comprises the 178 fully verified, answerable questions in the corpus (128 from the initial benchmark and 50 from the dataset expansion).

### 3.1 Core Metrics Summary

| Metric | Measured Value | Theoretical Maximum | Mathematical Definition / Explanation |
| :--- | :---: | :---: | :--- |
| **Hit Rate@5** | **100.0%** | 100.0% | **178/178** questions had ≥1 relevant chunk in top 5. |
| **Rank 1 Accuracy** | **78.65%** | 100.0% | **140/178** questions returned a target chunk as the absolute nearest neighbor. |
| **Mean Reciprocal Rank (MRR)** | **0.8834** | 1.0000 | Average reciprocal rank of the first relevant retrieved chunk across all 178 questions. |
| **Macro Recall@5** | **100.0%** | 100.0% | Unweighted mean of individual question recall values. |
| **Micro Recall@5** | **100.0%** | 100.0% | **200/200** total ground-truth chunks retrieved in top 5. |
| **Precision@5** | **22.47%** | 100.0% | **200/890** retrieved positions holding verified relevant chunks. |

### 3.2 Single-Chunk vs. Multi-Chunk Retrieval Performance

| Complexity Tier | Question Count | Hit Rate@5 | Rank 1 Accuracy | MRR | Macro Recall@5 | Micro Recall@5 | Precision@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Chunk Queries** | 156 | **100.0%** | **76.28%** | **0.8702** | **100.0%** | **100.0%** | **20.0%** |
| **Multi-Chunk Queries** | 22 | **100.0%** | **95.45%** | **0.9773** | **100.0%** | **100.0%** | **40.0%** |

### 3.3 Performance by Difficulty Tier (Group A)

| Difficulty Level | Question Count | Hit Rate@5 | Rank 1 Accuracy | MRR | Macro Recall@5 | Precision@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Easy** | 7 | **100.0%** | **100.0%** | **1.0** | **100.0%** | **20.0%** |
| **Medium** | 144 | **100.0%** | **75.69%** | **0.8663** | **100.0%** | **20.0%** |
| **Hard** | 27 | **100.0%** | **88.89%** | **0.9444** | **100.0%** | **36.3%** |

---

## 4. Analysis of Group B: Partially Supported Questions (N=5)

The 5 partially supported questions from the original benchmark contain topics where source documents discuss related principles, but specific sub-clauses or full factual coverage is partial:

| Question ID | First Hit Rank | Score | Macro Recall@5 | Expected Source Document | Notes on Partial Evidence |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **Q44** | 1 | 0.5828 | 100.0% | `Configuring K8s HPA for Custom Metrics.pdf` | Mentions custom metrics and HPA controller loop, but omits Prometheus Adapter config details. |
| **Q78** | 1 | 0.6942 | 100.0% | `byzantine_empire_and_justinian_code.txt` | Covers Justinian's Code and legal centralization, partial details on provincial administration. |
| **Q79** | 2 | 0.6934 | 100.0% | `byzantine_empire_and_justinian_code.txt` | Sibling chunk on Nika Riots ranked ahead; target legal reform chunk retrieved at position 2. |
| **Q90** | 1 | 0.6641 | 100.0% | `aztec_empire_spanish_conquest.txt` | Documents Cortés' alliances with Tlaxcala, partial on Aztec tributary economic structures. |
| **Q98** | 1 | 0.6135 | 100.0% | `soviet_command_economy_and_five_year_plans.txt` | Covers Gosplan and heavy industry output targets, partial on agricultural collectivization metrics. |

---

## 5. Analysis of Group C: Negative Controls (N=17)

Group C consists of 17 unanswerable questions whose subjects (e.g. HATEOAS architectural constraints, Python Global Interpreter Lock, CAP Theorem tradeoffs, Paxos leader elections) do not exist anywhere in the corpus.

> [!CAUTION]
> **Risk of Hallucination in Pure Vector Search:**  
> Standard FAISS vector search is strictly nearest-neighbor: it **always** returns the 5 closest vectors regardless of semantic validity. Because no abstention threshold or cross-encoder verifier is enabled in this baseline, **100% of negative control queries return plausible-sounding distractors**.

| ID | Unanswerable Topic | Retrieved Nearest Document (Rank 1) | Distance Score | Top Retrieved Distractor Summary |
| :---: | :--- | :--- | :---: | :--- |
| **Q27** | REST HATEOAS constraints | `Idempotency in RESTful API Design.pdf` | 0.6849 | Returns HTTP idempotency chunk (same API domain, wrong concept). |
| **Q28** | Python Global Interpreter Lock (GIL) | `Implementing Asynchronous Generators in Python.pdf` | 0.7587 | Returns Python async coroutines chunk. |
| **Q29** | Database Sharding & Consistent Hashing | `DOC-NET-802.docx` | 0.6918 | Returns VPC peering chunk due to network partitioning keywords. |
| **Q30** | GraphQL vs REST caching | `Idempotency in RESTful API Design.pdf` | 0.6938 | Returns REST idempotency chunk. |
| **Q31** | MongoDB Aggregation Pipelines | `Optimizing PostgreSQL JSONB Query Performance.pdf` | 0.7769 | Returns PostgreSQL JSONB indexing chunk. |
| **Q32** | Redis Sentinel Failover Elections | `Redis Cluster Data Sharding and High Availability.pdf` | 0.6146 | Returns Redis cluster master-slave replication chunk. |
| **Q33** | CAP Theorem PACELC extension | `Paxos and Raft Consensus Algorithms.pdf` | 0.6628 | Returns consensus algorithm chunk. |
| **Q34** | 2-Phase Commit (2PC) coordinator crash | `Paxos and Raft Consensus Algorithms.pdf` | 0.7693 | Returns consensus algorithm chunk. |
| **Q38** | Distributed tracing with OpenTelemetry | `Circuit Breaker Pattern in Microservices.pdf` | 0.7643 | Returns microservice resilience chunk. |
| **Q39** | Service Mesh sidecar proxy mTLS | `Circuit Breaker Pattern in Microservices.pdf` | 0.7211 | Returns microservice fault tolerance chunk. |
| **Q40** | Kafka log compaction clean-up | `Paxos and Raft Consensus Algorithms.pdf` | 0.6919 | Returns consensus algorithm chunk. |
| **Q41** | Kubernetes CNI overlay networking | `Configuring K8s HPA for Custom Metrics.pdf` | 0.7596 | Returns K8s autoscaling chunk. |
| **Q43** | Kubernetes Pod Disruption Budgets | `Configuring K8s HPA for Custom Metrics.pdf` | 0.7718 | Returns K8s metrics server chunk. |
| **Q45** | Kubernetes StatefulSet volume binding | `Configuring K8s HPA for Custom Metrics.pdf` | 0.7372 | Returns K8s autoscaling chunk. |
| **Q49** | OAuth 2.0 Mutual TLS (mTLS) client certs | `Zero Trust Network Access (ZTNA) Implementation.pdf` | 0.7159 | Returns ZTNA device certificate chunk. |
| **Q50** | FIDO2 WebAuthn public-key credentials | `Zero Trust Network Access (ZTNA) Implementation.pdf` | 0.6244 | Returns Zero Trust micro-segmentation chunk. |
| **Q80** | Justinian Plague economic demographic collapse | `byzantine_empire_and_justinian_code.txt` | 0.7029 | Returns Justinian legal code chunk. |

*Observation:* The distance scores for negative controls fall in the range **0.61 – 0.78**. Setting an abstention score threshold (e.g. rejecting queries with minimum distance > 0.65) or implementing a Cross-Encoder reranker in Step 10 will be essential for filtering out these false positives.

---

## 5. Latency Profile

Execution latencies were recorded independently for the Gemini embedding API and FAISS vector retrieval:

| Latency Component | Minimum | Maximum | Median | Average |
| :--- | :---: | :---: | :---: | :---: |
| **Query Embedding (Gemini API)** | 0.49135s | 0.63676s | 0.51221s | **0.53762s** |
| **FAISS Vector Search (L2 index)** | 0.00025s | 0.00069s | 0.00028s | **0.0003s (0.30 ms)** |
| **Total Query Latency** | 0.49164s | 0.63711s | 0.5125s | **0.53793s** |

---

## 6. Vector Competition Analysis: Non-Rank-1 Retrievals (Group A)

Across the 178 verified questions, **38 questions (21.3%)** retrieved their primary ground-truth chunk at Rank 2 or 3. This showcases genuine competition between semantically related chunks across the 68 documents:

| ID | First Hit Rank | Score | Question Snippet | Nearest Competing Document (Rank 1) |
| :---: | :---: | :---: | :--- | :--- |
| **Q1** | **2** | 0.4984 | What is the fundamental difference between authentication an… | `DOC-SEC-601.docx` |
| **Q8** | **2** | 0.4339 | Why are positional encodings mandatory at the bottom of the … | `DOC-ML-705.docx` |
| **Q12** | **2** | 0.4296 | What is the primary architectural limitation regarding trans… | `DOC-NET-802.docx` |
| **Q15** | **2** | 0.6263 | What distinguishes eBGP (External BGP) from iBGP (Internal B… | `DOC-NET-802.docx` |
| **Q17** | **2** | 0.6299 | How do consumer groups in Kafka facilitate massive scalabili… | `DOC-ARC-901.docx` |
| **Q21** | **2** | 0.5039 | Why are CI/CD pipelines considered prime targets for supply … | `DOC-DEV-1003.docx` |
| **Q22** | **3** | 0.6940 | How does utilizing OpenID Connect (OIDC) to federate identit… | `DOC-SEC-601.docx` |
| **Q51** | **3** | 0.6764 | How did James Watt's improvements to the steam engine act as… | `british_industrial_revolution.txt` |
| **Q52** | **2** | 0.6499 | What were the immediate public health consequences of the ra… | `british_industrial_revolution.txt` |
| **Q53** | **2** | 0.6102 | How did the shift from the domestic cottage industry to the … | `british_industrial_revolution.txt` |
| **Q54** | **4** | 0.6566 | In what ways did the accumulation of industrial capital driv… | `The Scramble for Africa and the Berlin Conference.pdf` |
| **Q55** | **3** | 0.7276 | How did the mass production of high-quality wrought iron via… | `DOC-HIS-2116.docx` |
| **Q56** | **3** | 0.6644 | How did the cessation of territorial expansion in the 2nd ce… | `fall_of_the_western_roman_empire.txt` |
| **Q57** | **3** | 0.6561 | What were the long-term socioeconomic effects of the Roman s… | `fall_of_the_western_roman_empire.txt` |
| **Q58** | **3** | 0.5878 | How did the 'barbarization' of the Roman legions with German… | `fall_of_the_western_roman_empire.txt` |

*(Showing first 15 of 38 non-Rank-1 queries. Full comparison available in `reports/step9_dense_200_comparison_summary.csv`.)*

---

## 7. Retrieval Misses (0 in Top 5)

| Metric | Measured Value |
| :--- | :---: |
| **Total Misses in Group A (Verified)** | **0** |
| **Total Misses in Group B (Partially Supported)** | **0** |
| **Overall Top-5 Retrieval Coverage** | **100.0%** |

*All 183 answerable queries in Groups A and B successfully retrieved at least one relevant ground-truth chunk within the top-5 results.*

---

## 8. Baseline Conclusions and Next Steps for Step 10

1. **Dense Retrieval Strengths:**  
   The Google Gemini `models/gemini-embedding-001` 3072-dimensional space exhibits exceptional semantic recall. With an overall Hit Rate@5 of **100.0%** and Rank 1 Accuracy of **78.65%** on verified questions, dense vectors cleanly separate broad technical domains.
2. **Dense Retrieval Weaknesses Identified:**  
   - **Keyword Exactness vs. Conceptual Generalization:** When queries ask about specific acronyms or terminology shared between documents (e.g. *zonal architectures in ECUs vs SDVs* in Q174), dense embeddings can favor a related document that mentions the concept more prominently.
   - **Zero Abstention on Negative Controls:** Pure vector search has no concept of "I don't know." Negative controls received distance scores around 0.65–0.78 and returned misleading chunks.
3. **Foundation for Step 10 (Hybrid Retrieval + Reranking):**  
   - **BM25 Hybridization:** Will inject lexical matching to prioritize exact keyword matches (e.g. standard identifiers, chemical/physics formulas).
   - **Cross-Encoder Reranking:** Will evaluate query-chunk pairs jointly, pushing Rank 2–3 chunks to Rank 1 and providing calibrated score thresholds to safely reject the 17 negative control queries.