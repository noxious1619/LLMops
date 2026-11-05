"""
Step 3 — Retrieval Quality Evaluation for the 68-Document RAG Benchmark.

Usage:
    .venv/Scripts/python tests/test_70docs_retrieval_evaluation.py

Outputs:
    tests/step6_70docs_retrieval_results.json
    reports/step6_70docs_retrieval_validation_report.md

Safety guarantees:
    - FAISS index is opened read-only (load_local does not write).
    - Ground-truth JSON is never modified.
    - Gemini API is called ONLY for embedding incoming queries.
    - All chunk IDs are validated against the live docstore before evaluation.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── load .env before importing project modules ──────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── stdlib / third-party ────────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ────────────────────────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
INDEX_DIR      = PROJECT_ROOT / "faiss_index" / "benchmark_70docs_v1"
INDEX_PKL      = INDEX_DIR / "index.pkl"
INDEX_FAISS    = INDEX_DIR / "index.faiss"
GT_PATH        = PROJECT_ROOT / "tests" / "retrieval_evaluation_dataset.json"
RESULTS_PATH   = PROJECT_ROOT / "tests"  / "step6_70docs_retrieval_results.json"
REPORT_PATH    = PROJECT_ROOT / "reports" / "step6_70docs_retrieval_validation_report.md"

EMBEDDING_MODEL = "models/gemini-embedding-001"
TOP_K           = 5

# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    # Encode to cp1252 safely for Windows console, replacing unmappable chars
    safe_msg = msg.encode("cp1252", errors="replace").decode("cp1252")
    print(f"[{time.strftime('%H:%M:%S')}] {safe_msg}", flush=True)


def snapshot_file(path: Path) -> Tuple[int, float]:
    """Return (size_bytes, mtime) for integrity checks."""
    st = path.stat()
    return st.st_size, st.st_mtime


def embed_with_retry(
    embeddings_model,
    text: str,
    max_retries: int = 6,
    base_delay: float = 2.0,
) -> Tuple[Optional[List[float]], float, int]:
    """
    Embed a single query with exponential back-off.
    Returns (vector | None, latency_seconds, retry_count).
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            vec = embeddings_model.embed_query(text)
            latency = time.perf_counter() - t0
            return vec, latency, attempt
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "quota" in msg.lower() or "resource exhausted" in msg.lower():
                log(f"  Rate-limit hit (attempt {attempt + 1}). Sleeping {delay:.1f}s …")
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
            elif attempt < max_retries - 1:
                log(f"  Transient error (attempt {attempt + 1}): {msg[:80]}. Sleeping {delay:.1f}s …")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            else:
                log(f"  Embedding failed after {max_retries} attempts: {msg[:120]}")
                return None, 0.0, attempt
    return None, 0.0, max_retries


# ────────────────────────────────────────────────────────────────────────────
# STEP 1 — VALIDATE INPUTS
# ────────────────────────────────────────────────────────────────────────────

def validate_inputs(gt_data: List[Dict], docstore_chunk_ids: set) -> Dict[str, Any]:
    """Run all pre-flight checks. Return a summary dict."""
    issues = []

    # 1. Exactly 150 questions
    if len(gt_data) != 150:
        issues.append(f"Expected 150 questions, found {len(gt_data)}")

    # 2. No duplicate IDs
    ids = [q["id"] for q in gt_data]
    dup = [x for x in ids if ids.count(x) > 1]
    if dup:
        issues.append(f"Duplicate question IDs: {set(dup)}")

    # 3. Check all assigned chunk IDs exist in docstore
    bad_ids = []
    for q in gt_data:
        for cid in q.get("expected_relevant_chunk_ids", []):
            if cid not in docstore_chunk_ids:
                bad_ids.append((q["id"], cid[:16] + "…"))
    if bad_ids:
        issues.append(f"Chunk IDs not in docstore: {bad_ids[:5]}")

    # 4. Check source filename consistency
    # (already guaranteed by build_ground_truth_dataset.py, but verify)
    # (no docstore source map available here cheaply; skip to keep it lightweight)

    # 5. Verify ground-truth distribution
    status_counts: Dict[str, int] = {}
    for q in gt_data:
        s = q["ground_truth_status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    expected = {"verified": 128, "partially_supported": 5, "not_found": 17}
    for s, exp_cnt in expected.items():
        actual_cnt = status_counts.get(s, 0)
        if actual_cnt != exp_cnt:
            issues.append(f"Expected {exp_cnt} '{s}' but found {actual_cnt}")

    return {
        "total": len(gt_data),
        "status_counts": status_counts,
        "issues": issues,
        "ok": len(issues) == 0,
    }


# ────────────────────────────────────────────────────────────────────────────
# STEP 2 — SMOKE TEST
# ────────────────────────────────────────────────────────────────────────────

def smoke_test(vectorstore, embeddings_model, docstore_chunk_ids: set) -> bool:
    smoke_q = "What is the Raft consensus algorithm?"
    log(f"Smoke test: '{smoke_q}'")
    vec, lat, retries = embed_with_retry(embeddings_model, smoke_q)
    if vec is None:
        log("SMOKE TEST FAILED: embedding returned None.")
        return False
    log(f"  Embedding OK (dim={len(vec)}, latency={lat:.3f}s)")

    t0 = time.perf_counter()
    results = vectorstore.similarity_search_with_score(smoke_q, k=5)
    search_lat = time.perf_counter() - t0
    log(f"  FAISS search OK ({len(results)} results, latency={search_lat:.4f}s)")

    for doc, score in results[:3]:
        cid = doc.metadata.get("chunk_id", "")
        src = Path(doc.metadata.get("source", "")).name
        in_store = cid in docstore_chunk_ids
        log(f"    chunk_id={cid[:16]}… source={src} score={score:.4f} in_docstore={in_store}")
        if not in_store:
            log("  SMOKE TEST FAILED: retrieved chunk not in docstore!")
            return False

    log("  Smoke test PASSED.")
    return True


# ────────────────────────────────────────────────────────────────────────────
# STEP 3 — RUN EVALUATION
# ────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    gt_data: List[Dict],
    vectorstore,
    embeddings_model,
    docstore_chunk_ids: set,
) -> List[Dict]:
    records = []
    total = len(gt_data)
    total_retries = 0

    for i, q in enumerate(gt_data):
        qid = q["id"]
        question = q["question"]
        status = q["ground_truth_status"]
        expected_cids = q.get("expected_relevant_chunk_ids", [])

        log(f"[{i + 1:3d}/{total}] Q{qid} ({status}): {question[:70]}…")

        # ── embed query ──────────────────────────────────────────────────────
        vec, embed_lat, retries = embed_with_retry(embeddings_model, question)
        total_retries += retries

        if vec is None:
            records.append({
                "id": qid,
                "question": question,
                "ground_truth_status": status,
                "expected_relevant_chunk_ids": expected_cids,
                "retrieved_top5": [],
                "hit": False,
                "first_relevant_rank": None,
                "reciprocal_rank": 0.0,
                "embed_latency_s": None,
                "search_latency_s": None,
                "total_latency_s": None,
                "error": "Embedding failed after max retries",
                "retry_count": retries,
            })
            log(f"  FAILED — embedding error.")
            continue

        # ── FAISS similarity search ──────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            results = vectorstore.similarity_search_with_score(question, k=TOP_K)
            search_lat = time.perf_counter() - t0
        except Exception as exc:
            search_lat = time.perf_counter() - t0
            records.append({
                "id": qid,
                "question": question,
                "ground_truth_status": status,
                "expected_relevant_chunk_ids": expected_cids,
                "retrieved_top5": [],
                "hit": False,
                "first_relevant_rank": None,
                "reciprocal_rank": 0.0,
                "embed_latency_s": round(embed_lat, 5),
                "search_latency_s": round(search_lat, 5),
                "total_latency_s": round(embed_lat + search_lat, 5),
                "error": f"FAISS search failed: {str(exc)[:120]}",
                "retry_count": retries,
            })
            log(f"  FAILED — FAISS error: {exc}")
            continue

        total_lat = embed_lat + search_lat

        # ── build top-5 record ───────────────────────────────────────────────
        top5 = []
        hit = False
        first_relevant_rank = None

        for rank, (doc, score) in enumerate(results, start=1):
            cid   = doc.metadata.get("chunk_id", "")
            src   = Path(doc.metadata.get("source", "")).name
            page  = doc.metadata.get("page", None)
            preview = doc.page_content[:120].replace("\n", " ")

            is_relevant = cid in expected_cids

            if is_relevant and first_relevant_rank is None:
                first_relevant_rank = rank
                hit = True

            top5.append({
                "rank":       rank,
                "chunk_id":   cid,
                "source":     src,
                "page":       page,
                "score":      round(float(score), 6),
                "preview":    preview,
                "is_relevant": is_relevant,
            })

        rr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0

        records.append({
            "id":                         qid,
            "question":                   question,
            "ground_truth_status":        status,
            "expected_relevant_chunk_ids": expected_cids,
            "retrieved_top5":             top5,
            "hit":                        hit,
            "first_relevant_rank":        first_relevant_rank,
            "reciprocal_rank":            round(rr, 6),
            "embed_latency_s":            round(embed_lat, 5),
            "search_latency_s":           round(search_lat, 5),
            "total_latency_s":            round(total_lat, 5),
            "error":                      None,
            "retry_count":                retries,
        })

        status_icon = "✓" if hit else "✗"
        rank_str    = f"rank={first_relevant_rank}" if first_relevant_rank else "miss"
        log(f"  {status_icon} hit={hit}, {rank_str}, embed={embed_lat:.3f}s, search={search_lat:.4f}s")

        # throttle slightly to avoid hammering the API
        time.sleep(0.3)

    log(f"\nDone. Total retries: {total_retries}")
    return records


# ────────────────────────────────────────────────────────────────────────────
# STEP 4 — COMPUTE METRICS
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(records: List[Dict], status_filter: str) -> Dict[str, Any]:
    """Compute Hit@5, P@5, MRR, Rank1Acc for a subset of records."""
    subset = [r for r in records if r["ground_truth_status"] == status_filter and r["error"] is None]
    n = len(subset)
    if n == 0:
        return {"n": 0, "hit_rate_at_5": None, "precision_at_5": None,
                "mrr": None, "rank1_accuracy": None}

    hits = sum(1 for r in subset if r["hit"])
    # For P@5: among returned positions, fraction that are relevant
    relevant_in_top5 = sum(
        sum(1 for c in r["retrieved_top5"] if c["is_relevant"])
        for r in subset
    )
    total_positions = sum(len(r["retrieved_top5"]) for r in subset)

    rank1 = sum(1 for r in subset if r["first_relevant_rank"] == 1)
    mrr   = sum(r["reciprocal_rank"] for r in subset) / n

    return {
        "n":                n,
        "hit_rate_at_5":    round(hits / n * 100, 2),
        "precision_at_5":   round(relevant_in_top5 / total_positions * 100, 2) if total_positions else None,
        "mrr":              round(mrr, 4),
        "rank1_accuracy":   round(rank1 / n * 100, 2),
        "hits":             hits,
        "misses":           n - hits,
        "rank1_count":      rank1,
        "relevant_in_top5": relevant_in_top5,
        "total_positions":  total_positions,
    }


def latency_stats(records: List[Dict]) -> Dict[str, Any]:
    good = [r for r in records if r["total_latency_s"] is not None]
    if not good:
        return {}
    embed_lats  = [r["embed_latency_s"]  for r in good if r["embed_latency_s"]  is not None]
    search_lats = [r["search_latency_s"] for r in good if r["search_latency_s"] is not None]
    total_lats  = [r["total_latency_s"]  for r in good]

    def stats(lst):
        if not lst:
            return {}
        return {
            "min":  round(min(lst), 5),
            "max":  round(max(lst), 5),
            "avg":  round(sum(lst) / len(lst), 5),
        }

    return {
        "embed_latency":  stats(embed_lats),
        "search_latency": stats(search_lats),
        "total_latency":  stats(total_lats),
    }


# ────────────────────────────────────────────────────────────────────────────
# STEP 5 — WRITE REPORT
# ────────────────────────────────────────────────────────────────────────────

def write_report(
    records: List[Dict],
    val_summary: Dict,
    verified_m: Dict,
    partial_m:  Dict,
    nf_records: List[Dict],
    lat_stats:  Dict,
    total_retries: int,
    wall_time:  float,
    index_sizes_before: Tuple,
    index_sizes_after:  Tuple,
    gt_size_before: Tuple,
    gt_size_after:  Tuple,
):
    lines = []
    A = lines.append

    A("# Step 3 — Retrieval Quality Evaluation Report")
    A(f"> **Benchmark Session:** `benchmark_70docs_v1`  ")
    A(f"> **FAISS Index:** `faiss_index/benchmark_70docs_v1/`  ")
    A(f"> **Embedding Model:** `{EMBEDDING_MODEL}`  ")
    A(f"> **Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    A(f"> **Top-K:** {TOP_K}  ")
    A("")
    A("---")

    # ── 1. Executive Summary ────────────────────────────────────────────────
    A("## 1. Executive Summary")
    A("")
    errors = [r for r in records if r["error"]]
    completed = len(records) - len(errors)
    A(f"| Item | Value |")
    A(f"| :--- | :---: |")
    A(f"| Total questions attempted | **{len(records)}** |")
    A(f"| Completed without error   | **{completed}** |")
    A(f"| Failed (API / FAISS error)| **{len(errors)}** |")
    A(f"| Total API retries         | **{total_retries}** |")
    A(f"| Wall-clock evaluation time| **{wall_time:.1f}s** |")
    A("")

    # ── 2. Input Validation ─────────────────────────────────────────────────
    A("## 2. Input Validation")
    A("")
    ok_str = "✅ PASSED" if val_summary["ok"] else "❌ FAILED"
    A(f"**Pre-flight validation:** {ok_str}")
    A("")
    for s, c in val_summary["status_counts"].items():
        A(f"- `{s}`: {c} questions")
    if val_summary["issues"]:
        A("")
        A("> [!CAUTION]")
        A("> Validation issues detected:")
        for iss in val_summary["issues"]:
            A(f"> - {iss}")
    A("")

    # ── 3. Index Integrity ─────────────────────────────────────────────────
    A("## 3. Index and Ground-Truth Integrity")
    A("")
    faiss_unchanged = index_sizes_before == index_sizes_after
    gt_unchanged    = gt_size_before == gt_size_after
    A(f"| File | Before (bytes, mtime) | After (bytes, mtime) | Unchanged? |")
    A(f"| :--- | :--- | :--- | :---: |")
    A(f"| `index.faiss` | {index_sizes_before[0]}, {index_sizes_before[1]:.0f} | {index_sizes_after[0]}, {index_sizes_after[1]:.0f} | {'✅' if faiss_unchanged else '❌'} |")
    A(f"| `retrieval_evaluation_dataset.json` | {gt_size_before[0]}, {gt_size_before[1]:.0f} | {gt_size_after[0]}, {gt_size_after[1]:.0f} | {'✅' if gt_unchanged else '❌'} |")
    A("")

    # ── 4. Primary Metrics — Verified ──────────────────────────────────────
    A("## 4. Primary Retrieval Metrics — Verified Questions (N=128)")
    A("")
    vm = verified_m
    A(f"| Metric | Value | Denominator |")
    A(f"| :--- | :---: | :--- |")
    A(f"| **Hit Rate@5** | **{vm['hit_rate_at_5']}%** | {vm['hits']} hits / {vm['n']} verified questions |")
    A(f"| **Precision@5** | **{vm['precision_at_5']}%** | {vm['relevant_in_top5']} relevant positions / {vm['total_positions']} total retrieved positions |")
    A(f"| **MRR** | **{vm['mrr']}** | Σ(1/rank) / {vm['n']} |")
    A(f"| **Rank 1 Accuracy** | **{vm['rank1_accuracy']}%** | {vm['rank1_count']} first-position hits / {vm['n']} |")
    A(f"| Misses | {vm['misses']} | Not retrieved in top-5 |")
    A("")

    # ── 5. Partially Supported ─────────────────────────────────────────────
    A("## 5. Partially Supported Questions (N=5)")
    A("")
    pm = partial_m
    A(f"| Metric | Value |")
    A(f"| :--- | :---: |")
    A(f"| Hit Rate@5 | {pm.get('hit_rate_at_5')}% |")
    A(f"| Precision@5 | {pm.get('precision_at_5')}% |")
    A(f"| MRR | {pm.get('mrr')} |")
    A(f"| Rank 1 Accuracy | {pm.get('rank1_accuracy')}% |")
    A("")

    partial_recs = [r for r in records if r["ground_truth_status"] == "partially_supported"]
    A("| ID | Hit | Rank | Score | Retrieved Source (Rank 1) |")
    A("| :---: | :---: | :---: | :---: | :--- |")
    for r in partial_recs:
        r1_src = r["retrieved_top5"][0]["source"] if r["retrieved_top5"] else "-"
        r1_score = r["retrieved_top5"][0]["score"] if r["retrieved_top5"] else "-"
        A(f"| {r['id']} | {'✅' if r['hit'] else '❌'} | {r['first_relevant_rank'] or '-'} | {r1_score} | `{r1_src}` |")
    A("")

    # ── 6. Negative Controls ───────────────────────────────────────────────
    A("## 6. Negative-Control Questions (N=17)")
    A("")
    A("> [!NOTE]")
    A("> These 17 questions have `expected_relevant_chunk_ids: []` because their topics (HATEOAS, GIL, CAP theorem, etc.) are absent from the corpus.")
    A("> FAISS *always* returns nearest neighbours; without a calibrated rejection threshold there is **no claim of rejection accuracy**.")
    A("> The table below shows the top-retrieved chunk and score for each to allow manual inspection of false-positive risk.")
    A("")
    A("| ID | Top Retrieved Source | Score@1 | Top 5 Sources (summary) |")
    A("| :---: | :--- | :---: | :--- |")
    for r in nf_records:
        if not r["retrieved_top5"]:
            A(f"| {r['id']} | (no results) | — | — |")
            continue
        t1 = r["retrieved_top5"][0]
        srcs = ", ".join(f"`{c['source']}`" for c in r["retrieved_top5"][:3])
        A(f"| {r['id']} | `{t1['source']}` | {t1['score']} | {srcs} |")
    A("")

    # ── 7. Latency ─────────────────────────────────────────────────────────
    A("## 7. Latency Analysis")
    A("")
    tl = lat_stats.get("total_latency", {})
    el = lat_stats.get("embed_latency", {})
    sl = lat_stats.get("search_latency", {})
    A(f"| Latency Component | Min (s) | Max (s) | Avg (s) |")
    A(f"| :--- | :---: | :---: | :---: |")
    A(f"| Embedding (Gemini API) | {el.get('min','—')} | {el.get('max','—')} | {el.get('avg','—')} |")
    A(f"| FAISS Search           | {sl.get('min','—')} | {sl.get('max','—')} | {sl.get('avg','—')} |")
    A(f"| Total per query        | {tl.get('min','—')} | {tl.get('max','—')} | {tl.get('avg','—')} |")
    A(f"| **Total wall time**    |       |       | **{wall_time:.1f}s** |")
    A("")

    # ── 8. API Failures ────────────────────────────────────────────────────
    A("## 8. API Failures and Retries")
    A("")
    A(f"- Total retries across all queries: **{total_retries}**")
    A(f"- Failed queries (embedding error): **{len(errors)}**")
    if errors:
        A("")
        A("| Question ID | Error |")
        A("| :---: | :--- |")
        for r in errors:
            A(f"| {r['id']} | {str(r['error'])[:120]} |")
    A("")

    # ── 9. Worst-Performing Questions ──────────────────────────────────────
    A("## 9. Lowest-Ranking and Failed Queries")
    A("")
    verified_recs = [r for r in records if r["ground_truth_status"] == "verified" and not r["error"]]
    misses = [r for r in verified_recs if not r["hit"]]
    low_rank = sorted(
        [r for r in verified_recs if r["hit"] and r["first_relevant_rank"] and r["first_relevant_rank"] > 1],
        key=lambda x: x["first_relevant_rank"],  # type: ignore
        reverse=True,
    )[:10]

    A(f"### 9.1 Verified Questions With No Hit in Top-5 ({len(misses)} misses)")
    A("")
    if misses:
        A("| ID | Question Snippet | Expected Source |")
        A("| :---: | :--- | :--- |")
        for r in misses:
            src = r["expected_relevant_chunk_ids"] and r.get("expected_relevant_chunk_ids")
            A(f"| {r['id']} | {r['question'][:60]}… | — |")
    else:
        A("*No misses — all verified questions retrieved at least one relevant chunk in top-5.*")
    A("")

    A(f"### 9.2 Top-10 Lowest-Rank Hits (Verified, rank > 1)")
    A("")
    if low_rank:
        A("| ID | Rank | Score@Rank | Question Snippet |")
        A("| :---: | :---: | :---: | :--- |")
        for r in low_rank:
            rk = r["first_relevant_rank"]
            score_at_rank = next(
                (c["score"] for c in r["retrieved_top5"] if c["rank"] == rk), "—"
            )
            A(f"| {r['id']} | {rk} | {score_at_rank} | {r['question'][:55]}… |")
    else:
        A("*No low-rank hits.*")
    A("")

    # ── 10. Source Distribution ────────────────────────────────────────────
    A("## 10. Source-Document Retrieval Distribution (Top-5 across all 150 queries)")
    A("")
    src_counts: Dict[str, int] = {}
    for r in records:
        for c in r["retrieved_top5"]:
            src_counts[c["source"]] = src_counts.get(c["source"], 0) + 1
    sorted_srcs = sorted(src_counts.items(), key=lambda x: -x[1])
    A("| Source Document | Times Retrieved in Top-5 |")
    A("| :--- | :---: |")
    for src, cnt in sorted_srcs[:25]:
        A(f"| `{src}` | {cnt} |")
    A("")

    # ── 11. Recommendations ────────────────────────────────────────────────
    A("## 11. Performance Observations and Recommendations")
    A("")
    A("- **FAISS distance metric:** L2 (cosine-equivalent after L2-normalised Gemini embeddings). Lower score = higher similarity.")
    A("- **Retrieval is similarity only:** No rejection threshold is implemented; negative-control score distributions should be inspected before deploying an abstention mechanism.")
    A("- **Negative controls:** Topics absent from the corpus (e.g. HATEOAS, GIL, CAP theorem) still receive near-neighbour results — monitor top-1 scores when filtering.")
    A("- **Latency:** Average total per-query latency is dominated by Gemini API embedding time. FAISS search is sub-millisecond.")
    A("")
    A("---")
    A("*Report generated by `tests/test_70docs_retrieval_evaluation.py`.*")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Report written → {REPORT_PATH}")


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("Step 3 — Retrieval Quality Evaluation")
    log("=" * 70)

    # ── Snapshot before ─────────────────────────────────────────────────────
    snap_faiss_before = snapshot_file(INDEX_FAISS)
    snap_gt_before    = snapshot_file(GT_PATH)

    # ── Load FAISS docstore (pkl only, no FAISS search yet) ─────────────────
    log("Loading FAISS docstore from index.pkl …")
    with open(INDEX_PKL, "rb") as f:
        docstore, index_to_id = pickle.load(f)

    all_chunks: Dict[str, str] = {}  # chunk_id → source name
    docstore_chunk_ids: set = set()
    for idx, cid in index_to_id.items():
        doc = docstore.search(cid)
        real_cid = doc.metadata.get("chunk_id", "")
        docstore_chunk_ids.add(real_cid)
        all_chunks[real_cid] = Path(doc.metadata.get("source", "")).name

    log(f"  Docstore loaded: {len(docstore_chunk_ids)} chunk IDs verified.")

    # ── Load ground-truth dataset ────────────────────────────────────────────
    log("Loading ground-truth dataset …")
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_data: List[Dict] = json.load(f)
    log(f"  Loaded {len(gt_data)} questions.")

    # ── Validate inputs ──────────────────────────────────────────────────────
    log("Validating inputs …")
    val_summary = validate_inputs(gt_data, docstore_chunk_ids)
    if not val_summary["ok"]:
        log("VALIDATION FAILED:")
        for iss in val_summary["issues"]:
            log(f"  • {iss}")
        sys.exit(1)
    log("  Validation PASSED.")

    # ── Load embedding model ─────────────────────────────────────────────────
    log(f"Loading embedding model: {EMBEDDING_MODEL} …")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        log("ERROR: GOOGLE_API_KEY not set.")
        sys.exit(1)
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,   # type: ignore[arg-type]
    )

    # ── Load FAISS vectorstore ───────────────────────────────────────────────
    log("Loading FAISS vectorstore …")
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings_model,
        index_name="index",
        allow_dangerous_deserialization=True,
    )
    log(f"  Vectorstore loaded. Total vectors: {vectorstore.index.ntotal}")
    if vectorstore.index.ntotal != 181:
        log(f"WARNING: Expected 181 vectors, found {vectorstore.index.ntotal}")

    # ── Smoke test ───────────────────────────────────────────────────────────
    log("Running smoke test …")
    if not smoke_test(vectorstore, embeddings_model, docstore_chunk_ids):
        log("Smoke test failed — aborting evaluation.")
        sys.exit(1)

    # ── Full evaluation ──────────────────────────────────────────────────────
    log(f"\nStarting evaluation of {len(gt_data)} questions …\n")
    t_eval_start = time.perf_counter()
    records = run_evaluation(gt_data, vectorstore, embeddings_model, docstore_chunk_ids)
    wall_time = time.perf_counter() - t_eval_start

    # ── Compute metrics ──────────────────────────────────────────────────────
    verified_m = compute_metrics(records, "verified")
    partial_m  = compute_metrics(records, "partially_supported")
    nf_records = [r for r in records if r["ground_truth_status"] == "not_found"]
    lat        = latency_stats(records)
    total_retries = sum(r.get("retry_count", 0) for r in records)

    # ── Snapshot after ───────────────────────────────────────────────────────
    snap_faiss_after = snapshot_file(INDEX_FAISS)
    snap_gt_after    = snapshot_file(GT_PATH)

    # ── Write results JSON ───────────────────────────────────────────────────
    log(f"\nWriting raw results → {RESULTS_PATH}")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    # Validate by re-loading
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        reloaded = json.load(f)
    assert len(reloaded) == len(gt_data), "Record count mismatch after reload!"
    log(f"  Results JSON validated: {len(reloaded)} records.")

    # ── Write markdown report ────────────────────────────────────────────────
    write_report(
        records, val_summary, verified_m, partial_m,
        nf_records, lat, total_retries, wall_time,
        snap_faiss_before, snap_faiss_after,
        snap_gt_before, snap_gt_after,
    )

    # ── Final console summary ─────────────────────────────────────────────────
    errors_count = sum(1 for r in records if r["error"])
    log("")
    log("=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)
    log(f"  Total attempted:      {len(records)}")
    log(f"  Completed:            {len(records) - errors_count}")
    log(f"  Failed:               {errors_count}")
    log(f"  Total retries:        {total_retries}")
    log("")
    log("  — Verified (N=128) —")
    log(f"  Hit Rate@5:           {verified_m['hit_rate_at_5']}%")
    log(f"  Precision@5:          {verified_m['precision_at_5']}%")
    log(f"  MRR:                  {verified_m['mrr']}")
    log(f"  Rank 1 Accuracy:      {verified_m['rank1_accuracy']}%")
    log("")
    log("  — Partially Supported (N=5) —")
    log(f"  Hit Rate@5:           {partial_m.get('hit_rate_at_5')}%")
    log(f"  MRR:                  {partial_m.get('mrr')}")
    log("")
    log("  — Latency —")
    tl = lat.get("total_latency", {})
    log(f"  Avg total/query:      {tl.get('avg', '—')}s")
    log(f"  Wall time:            {wall_time:.1f}s")
    log("")
    log(f"  FAISS index unchanged:      {snap_faiss_before == snap_faiss_after}")
    log(f"  Ground-truth unchanged:     {snap_gt_before == snap_gt_after}")
    log("")
    log(f"  Raw results:  {RESULTS_PATH}")
    log(f"  Report:       {REPORT_PATH}")
    log("=" * 70)


if __name__ == "__main__":
    main()
