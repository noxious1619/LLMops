# Step 10: Recall@5 Metric and Ground-Truth Audit Report

## Executive Summary

This audit independently reviewed the **100% Macro Recall@5 and 100% Micro Recall@5** reported in Step 9 across the 178 primary verified questions in the 200-question RAG benchmark.

### Primary Audit Verdict

> **FINAL CLASSIFICATION: CATEGORY B — GROUND-TRUTH QUALITY & BENCHMARK CEILING ISSUE**
>
> 1. **No Calculation Bug Detected**: The mathematical formulas for Per-Query Recall@5, Macro Recall@5, and Micro Recall@5 implemented in `tests/test_step9_dense_200_evaluation.py` are strictly correct and verified independently down to 4 decimal places.
> 2. **Single-Chunk Degeneracy ($|GT| = 1$)**: For 156 of the 178 verified questions (87.64%), ground truth consists of exactly 1 chunk. For single-chunk questions, Recall@5 is mathematically degenerate with Hit Rate@5 ($	ext{Recall}@5 = 	ext{Hit}@5 = 100\%$).
> 3. **Corpus Scale & Intra-Document Clustering**: The benchmark corpus consists of only 181 chunks across 68 documents (average 2.66 chunks/doc). In 17 of the 22 multi-chunk questions (77.3%), the document contains only 2 chunks total. Top-5 dense retrieval trivially retrieves both chunks of the matching document.
> 4. **Ground-Truth Mapping Shift (Q12–Q15)**: A severe ground-truth defect exists in the original 150 benchmark: questions Q12, Q13, Q14, and Q15 have their expected answers and chunk mappings shifted by an index offset in `tests/build_ground_truth_dataset.py`. Because `DOC-NET-802.docx` has only 2 chunks in total, FAISS retrieved both chunks regardless, masking this critical error and yielding a spurious 100% recall.
> 5. **Resume & Production Impact**: While 100% Recall@5 is mathematically genuine under the existing dataset definitions, advertising '100% Recall@5' on a resume without corpus context is misleading. It reflects a **corpus ceiling effect** on a small 181-chunk index rather than real-world RAG perfection.

---

## 1. Metric Recalculation Comparison Table

We independently recalculated all retrieval metrics using a newly developed standalone script with zero shared evaluation code:

| Evaluation Cohort | Metric | Step 9 Reported | Step 10 Recalculated | Discrepancy | Audit Status |
|---|---|---|---|---|---|
| **Group A (Verified N=178)** | **Hit Rate@5** | 100.0% | **100.0%** | 0.0% | Verified Valid |
| | **Macro Recall@5** | 100.0% | **100.0%** | 0.0% | Verified Valid (Degenerate) |
| | **Micro Recall@5** | 100.0% | **100.0%** | 0.0% | Verified Valid (200/200 Chunks) |
| | **Rank 1 Accuracy** | 78.65% | **78.65%** | 0.0% | Verified Valid (140/178) |
| | **MRR** | 0.8834 | **0.8834** | 0.0000 | Verified Valid |
| | **Precision@5** | 22.47% | **22.47%** | 0.0% | Verified Valid (200/(178*5)) |
| *• Single-Chunk (N=156)* | **Macro Recall@5** | 100.0% | **100.0%** | 0.0% | Degenerate with Hit@5 |
| *• Multi-Chunk (N=22)* | **Macro Recall@5** | 100.0% | **100.0%** | 0.0% | Intra-doc saturation |
| **Group B (Partial N=5)** | **Hit Rate@5** | 100.0% | **100.0%** | 0.0% | Verified Qualified |
| | **Macro Recall@5** | 100.0% | **100.0%** | 0.0% | Partial evidence matched |
| **Group C (Negative N=17)** | **Recall@5** | N/A | **N/A** | 0.0 | Correctly Excluded |

---

## 2. Why Did Step 9 Achieve 100% Recall@5?

A critical examination of the mathematics and data distribution reveals four structural drivers:

### Driver 1: Single-Chunk Mathematical Degeneracy
$$\text{Recall}@5 = \frac{|\text{Retrieved}@5 \cap \text{GroundTruth}|}{|\text{GroundTruth}|}$$
When $|\text{GroundTruth}| = 1$:
- If the chunk is retrieved in top 5: $\text{Recall}@5 = \frac{1}{1} = 100\%$.
- If the chunk is not in top 5: $\text{Recall}@5 = \frac{0}{1} = 0\%$.

Because 156 of the 178 questions (87.64%) have exactly 1 ground-truth chunk, Recall@5 does not measure recall depth or completeness; **it is identical to Hit Rate@5**.

### Driver 2: Tiny Corpus and High Sampling Ratio
- Total corpus chunks: **181** across 68 documents.
- Retrieving $K=5$ chunks extracts $\frac{5}{181} = \mathbf{2.76\%}$ of the entire vector database in every single query!
- In standard enterprise benchmarks (100K–1M chunks), Top-5 is $0.005\%$, where recall is heavily penalized. In an 181-chunk database, dense semantic search with Gemini 3072D embeddings easily places the target document into the top 2.7% of the vector space.

### Driver 3: Intra-Document Multi-Chunk Clustering
- Total multi-chunk questions: 22.
- **Intra-document questions: 22 (100.0%)**.
- **Cross-document questions: 0 (0.0%)**.
- In **17 of the 22 multi-chunk questions (77.3%)**, the source document has **only 2 chunks in total** in the entire FAISS index.
- When Gemini Embedding 001 maps a query about that document's topic, both Chunk 1 and Chunk 2 have high semantic similarity and rank at positions [1, 2] or [1, 3] (20 of 22 questions had both chunks in ranks 1 and 2!).

### Driver 4: 100-Token Chunk Overlap Duplication
Chunking was performed with `chunk_size=500` tokens and `chunk_overlap=100` tokens. In questions like Q3 (JWT validation claims), the entire section of text containing the required answer was split across the boundary and appears in both Chunk 1 and Chunk 2, rendering both chunks matches even though either one alone was sufficient.

---

## 3. Ground-Truth Quality Defect Audit (Q12–Q15)

During our deep audit of `tests/build_ground_truth_dataset.py`, we identified a major ground-truth misalignment error:

```python
# In tests/data/retrieval_questions_input.json:
# Index 11 (ID 12): 'What is the primary architectural limitation regarding transitivity when managing multiple point-to-point Virtual Private Cloud (VPC) peering connections?'
# Index 12 (ID 13): 'How do BGP communities allow network administrators to tag routes and enforce dynamic routing policies across hybrid cloud boundaries like AWS Direct Connect?'
# Index 13 (ID 14): 'Why does a hub-and-spoke networking architecture utilizing Transit Gateways solve the N*(N-1)/2 administrative burden associated with large-scale VPC peering?'
# Index 14 (ID 15): 'What distinguishes eBGP (External BGP) from iBGP (Internal BGP) when exchanging routes between different Autonomous System Numbers (ASNs)?'
```

However, in `tests/build_ground_truth_dataset.py` lines 210–255, the author mistakenly shifted the expected answers and keyword annotations:
- **Q12** (VPC transitivity) was paired with the answer for **BGP Communities**.
- **Q13** (BGP communities) was paired with the answer for **BGP path selection tie-breaking**.
- **Q14** (Transit Gateway) was paired with the answer for **eBGP RPKI BGP hijacking**.
- **Q15** (eBGP vs iBGP) was paired with the answer for **VPC Peering N*(N-1)/2 Transit Gateway**.

**Why wasn't this caught in Step 9?**
Both BGP routing and VPC peering belong to the exact same source document: `DOC-NET-802.docx`. That document contains **only 2 chunks** (Chunk 87 and Chunk 88). Because the document is so small and contains both topics, dense retrieval retrieved both chunks in the top 5 for all four questions! This completely masked the ground-truth misalignment.

---

## 4. Final Classification and Resume Recommendations

### Classification Verdict

| Category | Description | Applicable? | Justification |
|---|---|---|---|
| **Category A** | Calculation Bug | **NO** | The metric calculation formulas are mathematically sound and independently verified. |
| **Category B** | Ground-Truth Quality & Ceiling Issue | **YES** | Proven ground-truth index shifts (Q12–Q15), intra-document convenience mapping, and corpus size ceiling effect. |
| **Category C** | Both Issues | **NO** | Calculation is sound. |
| **Category D** | No Issue Detected | **NO** | Flaws exist in ground-truth construction and corpus scale saturation. |

### How to Frame This on Your Resume & Technical Interviews

> **❌ Misleading Resume Claim**:
> *'Engineered a RAG retrieval system achieving 100% Recall@5 across a 200-question multi-document benchmark.'*
> *(An experienced interviewer will immediately flag 100% Recall as a sign of data leakage, trivial corpus size, or evaluation bias.)*

> **✅ Professional, High-Impact Engineering Framing**:
> *'Established a 200-query dense retrieval baseline using Gemini 3072D embeddings and FAISS over a 68-document technical corpus, achieving 100.0% Hit Rate@5, 78.65% Rank 1 Accuracy, and 0.8834 MRR.'*
> *'Audited retrieval metrics to uncover a corpus ceiling effect: identified that 87.6% of single-chunk queries rendered Recall@5 mathematically degenerate with Hit Rate@5, and detected a 4-query ground-truth indexing shift in legacy benchmarks masked by small-corpus clustering.'*
> *'Leveraged audit findings to architect a Phase 11 Hybrid Retrieval (BM25 + Dense) and Cross-Encoder Reranking pipeline designed specifically to improve Rank 1 precision from 78.65% toward 95%+.'*

---

## 5. Artifact Links

- Detailed Question-Level Audit CSV: [`reports/step10_question_level_recall_audit.csv`](file:///c:/Users/Dishant%20Pandey/Desktop/Resume/LLM_OPS/reports/step10_question_level_recall_audit.csv)
- Multi-Chunk Evidence Review Dossier: [`reports/step10_multichunk_manual_review.md`](file:///c:/Users/Dishant%20Pandey/Desktop/Resume/LLM_OPS/reports/step10_multichunk_manual_review.md)
- Machine-Readable Audit JSON: [`tests/step10_recall_audit_results.json`](file:///c:/Users/Dishant%20Pandey/Desktop/Resume/LLM_OPS/tests/step10_recall_audit_results.json)
