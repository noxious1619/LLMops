"""
Phase 1: 70-Document Extraction and Chunking Validation Test Suite
Dataset: docs_70/
Benchmark Session: benchmark_70docs_v1
Safety: Zero embedding generation, zero API calls, zero FAISS modifications.
"""

import sys
import os
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

# Reconfigure stdout for UTF-8 compatibility on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tiktoken
from multi_doc_chat.utils.document_ops import load_documents
from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor

DOCS_70_DIR = PROJECT_ROOT / "docs_70"
DOCS_20_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_JSON = PROJECT_ROOT / "tests" / "step4_70docs_validation_results.json"
OUTPUT_CSV = PROJECT_ROOT / "reports" / "step4_70docs_extraction_results.csv"


def run_phase1_validation():
    print("=" * 80)
    print("PHASE 1: 70-DOCUMENT EXTRACTION AND CHUNKING VALIDATION")
    print(f"Target Directory: {DOCS_70_DIR}")
    print(f"Benchmark Session: benchmark_70docs_v1")
    print("=" * 80)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    # ---------------------------------------------------------
    # TASK 1: DOCUMENT INVENTORY
    # ---------------------------------------------------------
    print("\n>>> TASK 1: DOCUMENT INVENTORY")
    if not DOCS_70_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {DOCS_70_DIR}")

    raw_items = list(DOCS_70_DIR.iterdir())
    files = sorted([f for f in raw_items if f.is_file()])
    non_files = [f for f in raw_items if not f.is_file()]

    total_files = len(files)
    print(f"Total files discovered: {total_files}")
    if non_files:
        print(f"Warning: Non-file items: {[nf.name for nf in non_files]}")

    ext_counts = defaultdict(int)
    ext_sizes = defaultdict(int)
    file_inventory = []
    sha256_to_files = defaultdict(list)
    empty_files = []
    unreadable_files = []

    for f in files:
        ext = f.suffix.lower()
        ext_counts[ext] += 1
        try:
            size = f.stat().st_size
            ext_sizes[ext] += size
            if size == 0:
                empty_files.append(f.name)
            
            with open(f, "rb") as fp:
                file_bytes = fp.read()
                sha = hashlib.sha256(file_bytes).hexdigest()
                sha256_to_files[sha].append(f.name)

            file_inventory.append({
                "name": f.name,
                "path": str(f),
                "extension": ext,
                "size_bytes": size,
                "sha256": sha
            })
        except Exception as e:
            unreadable_files.append((f.name, str(e)))

    total_bytes = sum(item["size_bytes"] for item in file_inventory)
    min_size_item = min(file_inventory, key=lambda x: x["size_bytes"])
    max_size_item = max(file_inventory, key=lambda x: x["size_bytes"])
    avg_size = total_bytes / total_files if total_files else 0

    duplicate_content_groups = {k: v for k, v in sha256_to_files.items() if len(v) > 1}

    # Compare with docs/
    docs_20_files = set(f.name for f in DOCS_20_DIR.iterdir() if f.is_file()) if DOCS_20_DIR.exists() else set()
    docs_70_files = set(f["name"] for f in file_inventory)
    overlap_with_docs20 = sorted(docs_70_files.intersection(docs_20_files))
    new_in_docs70 = sorted(docs_70_files - docs_20_files)

    print(f"File Type Breakdown:")
    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext.upper()}: {count} files ({ext_sizes[ext]:,} bytes)")
    print(f"Total Dataset Size: {total_bytes:,} bytes ({total_bytes / (1024*1024):.2f} MB)")
    print(f"Min Size: {min_size_item['size_bytes']:,} bytes ({min_size_item['name']})")
    print(f"Max Size: {max_size_item['size_bytes']:,} bytes ({max_size_item['name']})")
    print(f"Average Size: {avg_size:,.1f} bytes")
    print(f"Duplicate content groups (by SHA-256): {len(duplicate_content_groups)}")
    print(f"Empty files (0 bytes): {len(empty_files)}")
    print(f"Unreadable files: {len(unreadable_files)}")
    print(f"Overlap with docs/ (original 20): {len(overlap_with_docs20)} files")
    print(f"New files in docs_70/: {len(new_in_docs70)} files")

    # ---------------------------------------------------------
    # TASK 2: TEXT EXTRACTION
    # ---------------------------------------------------------
    print("\n>>> TASK 2: TEXT EXTRACTION")
    extraction_results = []
    all_extracted_docs = []
    extraction_failures = []
    short_content_warnings = []
    unicode_replacement_warnings = []

    t_start_extraction = time.perf_counter()

    for item in file_inventory:
        file_path = Path(item["path"])
        file_name = item["name"]
        ext = item["extension"]

        try:
            # Call existing load_documents on this single file
            loaded_docs = load_documents([file_path])
            doc_count = len(loaded_docs)
            file_chars = sum(len(d.page_content) for d in loaded_docs)
            file_words = sum(len(d.page_content.split()) for d in loaded_docs)
            file_tokens = sum(len(tokenizer.encode(d.page_content)) for d in loaded_docs)

            # Check metadata preservation
            pages = [d.metadata.get("page") for d in loaded_docs]
            sources = [d.metadata.get("source") for d in loaded_docs]

            # Check for empty or short content (< 100 chars)
            if file_chars < 100:
                short_content_warnings.append((file_name, file_chars))

            # Check for unicode issues (\ufffd replacement char)
            has_unicode_replacement = any("\ufffd" in d.page_content for d in loaded_docs)
            if has_unicode_replacement:
                unicode_replacement_warnings.append(file_name)

            all_extracted_docs.extend(loaded_docs)

            extraction_results.append({
                "file_name": file_name,
                "extension": ext,
                "file_size": item["size_bytes"],
                "status": "SUCCESS",
                "doc_objects": doc_count,
                "chars": file_chars,
                "words": file_words,
                "tokens": file_tokens,
                "pages": pages,
                "source_preserved": all(str(file_path) in str(s) or file_name in str(s) for s in sources),
                "has_page_metadata": any(p is not None for p in pages) if ext == ".pdf" else True,
                "has_replacement_char": has_unicode_replacement
            })
        except Exception as e:
            extraction_failures.append((file_name, str(e)))
            extraction_results.append({
                "file_name": file_name,
                "extension": ext,
                "file_size": item["size_bytes"],
                "status": "FAILED",
                "doc_objects": 0,
                "chars": 0,
                "words": 0,
                "tokens": 0,
                "pages": [],
                "source_preserved": False,
                "has_page_metadata": False,
                "error": str(e)
            })

    extraction_time_sec = time.perf_counter() - t_start_extraction

    total_extracted_docs = len(all_extracted_docs)
    total_extracted_chars = sum(r["chars"] for r in extraction_results)
    total_extracted_words = sum(r["words"] for r in extraction_results)
    total_extracted_tokens = sum(r["tokens"] for r in extraction_results)
    success_files = [r for r in extraction_results if r["status"] == "SUCCESS"]

    print(f"Extraction Time: {extraction_time_sec:.2f} seconds")
    print(f"Files Processed: {total_files}")
    print(f"Files Succeeded: {len(success_files)} / {total_files} ({(len(success_files)/total_files)*100:.1f}%)")
    print(f"Files Failed: {len(extraction_failures)}")
    print(f"Total LangChain Document Objects: {total_extracted_docs}")
    print(f"Total Extracted Characters: {total_extracted_chars:,}")
    print(f"Total Extracted Words: {total_extracted_words:,}")
    print(f"Total Extracted Tokens (cl100k_base): {total_extracted_tokens:,}")
    print(f"Short Content Warnings (< 100 chars): {len(short_content_warnings)}")
    print(f"Unicode Replacement Char (\\ufffd) files: {len(unicode_replacement_warnings)}")

    # ---------------------------------------------------------
    # TASK 3: CHUNK GENERATION
    # ---------------------------------------------------------
    print("\n>>> TASK 3: CHUNK GENERATION")
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    ENCODING_NAME = "cl100k_base"

    print(f"Configuring ChatIngestor._split():")
    print(f"  chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}, encoding_name='{ENCODING_NAME}'")

    ingestor = ChatIngestor.__new__(ChatIngestor)
    t_start_chunking = time.perf_counter()
    chunks = ingestor._split(all_extracted_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunking_time_sec = time.perf_counter() - t_start_chunking

    total_chunks = len(chunks)
    print(f"Total Chunks Generated: {total_chunks}")
    print(f"Chunking Time: {chunking_time_sec:.3f} seconds")

    # Analyze token counts per chunk
    chunk_metrics = []
    chunks_exceeding_limit = []
    missing_chunk_ids = []
    missing_content_hashes = []
    missing_sources = []
    invalid_content_hashes = []
    chunk_id_set = set()
    duplicate_chunk_ids = []

    per_source_chunks = defaultdict(list)

    for idx, c in enumerate(chunks):
        c_text = c.page_content
        c_tokens = len(tokenizer.encode(c_text))
        c_chars = len(c_text)
        c_words = len(c_text.split())

        cid = c.metadata.get("chunk_id")
        chash = c.metadata.get("content_hash")
        src = c.metadata.get("source", "unknown")
        src_name = Path(src).name
        page = c.metadata.get("page")

        per_source_chunks[src_name].append(idx)

        # Validations
        if not cid:
            missing_chunk_ids.append(idx)
        elif cid in chunk_id_set:
            duplicate_chunk_ids.append((idx, cid))
        else:
            chunk_id_set.add(cid)

        if not chash:
            missing_content_hashes.append(idx)
        else:
            actual_sha = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
            if actual_sha != chash:
                invalid_content_hashes.append((idx, chash, actual_sha))

        if not src:
            missing_sources.append(idx)

        if c_tokens > CHUNK_SIZE:
            chunks_exceeding_limit.append((idx, src_name, c_tokens))

        chunk_metrics.append({
            "chunk_index": idx,
            "chunk_id": cid,
            "content_hash": chash,
            "source_name": src_name,
            "page": page,
            "token_count": c_tokens,
            "char_count": c_chars,
            "word_count": c_words
        })

    token_counts = [cm["token_count"] for cm in chunk_metrics]
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    avg_tokens = sum(token_counts) / total_chunks if total_chunks else 0

    # Overlap Analysis between consecutive chunks of the same file
    overlap_samples = []
    for src_name, indices in per_source_chunks.items():
        if len(indices) > 1:
            for i in range(len(indices) - 1):
                idx1, idx2 = indices[i], indices[i+1]
                t1 = chunks[idx1].page_content
                t2 = chunks[idx2].page_content

                # Calculate suffix of t1 matching prefix of t2
                max_check = min(len(t1), len(t2), 2000)
                measured_overlap_chars = 0
                for k in range(max_check, 0, -1):
                    if t1.endswith(t2[:k]):
                        measured_overlap_chars = k
                        break
                
                overlap_text = t2[:measured_overlap_chars]
                overlap_tokens = len(tokenizer.encode(overlap_text)) if measured_overlap_chars > 0 else 0
                overlap_samples.append({
                    "source": src_name,
                    "chunk_a": idx1,
                    "chunk_b": idx2,
                    "overlap_chars": measured_overlap_chars,
                    "overlap_tokens": overlap_tokens
                })

    avg_overlap_tokens = (sum(s["overlap_tokens"] for s in overlap_samples) / len(overlap_samples)) if overlap_samples else 0

    print(f"Chunk Token Statistics:")
    print(f"  Average Tokens: {avg_tokens:.1f}")
    print(f"  Min Tokens: {min_tokens}")
    print(f"  Max Tokens: {max_tokens}")
    print(f"  Chunks > 500 Tokens: {len(chunks_exceeding_limit)}")
    print(f"  Consecutive Chunk Pairs Evaluated: {len(overlap_samples)}")
    print(f"  Average Measured Overlap: {avg_overlap_tokens:.1f} tokens")
    print(f"\nMetadata Integrity:")
    print(f"  Missing Chunk IDs: {len(missing_chunk_ids)}")
    print(f"  Duplicate Chunk IDs: {len(duplicate_chunk_ids)}")
    print(f"  Missing Content Hashes: {len(missing_content_hashes)}")
    print(f"  Invalid Content Hashes: {len(invalid_content_hashes)}")
    print(f"  Missing Sources: {len(missing_sources)}")

    # ---------------------------------------------------------
    # GENERATE OUTPUTS
    # ---------------------------------------------------------
    # Write CSV
    with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
        f.write("file_name,extension,file_size_bytes,status,doc_objects,chars,words,tokens,chunks,page_count\n")
        for r in extraction_results:
            chunks_for_file = len(per_source_chunks.get(r["file_name"], []))
            pg_count = len(r["pages"])
            f.write(f'"{r["file_name"]}","{r["extension"]}",{r["file_size"]},"{r["status"]}",{r["doc_objects"]},{r["chars"]},{r["words"]},{r["tokens"]},{chunks_for_file},{pg_count}\n')
    print(f"\nCSV results written to: {OUTPUT_CSV}")

    # Write summary JSON
    results_summary = {
        "dataset": "docs_70",
        "benchmark_session": "benchmark_70docs_v1",
        "inventory": {
            "total_files": total_files,
            "extension_counts": dict(ext_counts),
            "extension_bytes": dict(ext_sizes),
            "total_bytes": total_bytes,
            "min_file_size": min_size_item,
            "max_file_size": max_size_item,
            "avg_file_size": avg_size,
            "duplicate_content_groups": len(duplicate_content_groups),
            "empty_files": len(empty_files),
            "unreadable_files": len(unreadable_files),
            "overlap_with_docs20": len(overlap_with_docs20),
            "new_files": len(new_in_docs70)
        },
        "extraction": {
            "time_sec": round(extraction_time_sec, 2),
            "files_processed": total_files,
            "files_succeeded": len(success_files),
            "files_failed": len(extraction_failures),
            "total_documents": total_extracted_docs,
            "total_chars": total_extracted_chars,
            "total_words": total_extracted_words,
            "total_tokens": total_extracted_tokens,
            "short_content_warnings": len(short_content_warnings),
            "unicode_warnings": len(unicode_replacement_warnings)
        },
        "chunking": {
            "time_sec": round(chunking_time_sec, 3),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "tokenizer": ENCODING_NAME,
            "total_chunks": total_chunks,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "avg_tokens": round(avg_tokens, 2),
            "chunks_exceeding_limit": len(chunks_exceeding_limit),
            "evaluated_overlap_pairs": len(overlap_samples),
            "avg_overlap_tokens": round(avg_overlap_tokens, 2),
            "missing_chunk_ids": len(missing_chunk_ids),
            "duplicate_chunk_ids": len(duplicate_chunk_ids),
            "missing_content_hashes": len(missing_content_hashes),
            "invalid_content_hashes": len(invalid_content_hashes),
            "missing_sources": len(missing_sources)
        },
        "per_file_summary": extraction_results
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"JSON results written to: {OUTPUT_JSON}")

    return results_summary


if __name__ == "__main__":
    run_phase1_validation()

