"""
Benchmark script for Cross-Encoder model cross-encoder/ms-marco-MiniLM-L-6-v2
on NVIDIA GeForce RTX 3050 GPU (cuda:0) and CPU fallback.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import psutil
import torch
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_ram_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_vram_usage_mb() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2),
    }


def run_benchmark():
    print("=" * 80)
    print("PHASE 1: CROSS-ENCODER MODEL VALIDATION & GPU BENCHMARK")
    print("=" * 80)

    # 1. Environment Detection
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024) if cuda_available else 0.0

    print(f"CUDA Available:     {cuda_available}")
    print(f"CUDA Version:       {torch.version.cuda}")
    print(f"PyTorch Version:    {torch.__version__}")
    print(f"GPU Device:         {device_name}")
    print(f"Total VRAM:         {total_vram:.2f} MB")
    print(f"System RAM Total:   {psutil.virtual_memory().total / (1024 * 1024):.2f} MB")

    # Sample 20 representative query-document pairs from corpus
    pairs = [
        ("What is the CRC16 checksum function used for in Redis Cluster?",
         "Redis Cluster uses a deterministic concept called Hash Slots. There are exactly 16,384 hash slots. To calculate which hash slot a key belongs to, Redis uses the CRC16 checksum of the key modulo 16384."),
        ("What is the CRC16 checksum function used for in Redis Cluster?",
         "Border Gateway Protocol (BGP) is a path-vector routing protocol responsible for exchanging routing and reachability information among Autonomous Systems (AS)."),
        ("What is the difference between authentication and authorization?",
         "Authentication verifies the identity of a subject ('who you are'), whereas authorization determines access rights to specific resources ('what you can do')."),
        ("What is the difference between authentication and authorization?",
         "In modern high-performance microservices, circuit breaker patterns prevent cascading failures by tripping open when error rates exceed thresholds."),
        ("What does the AS-PATH attribute indicate in BGP routing?",
         "BGP selects the best path based on complex path attributes rather than simple metrics, evaluating factors like Weight, Local Preference, AS-PATH length, and Multi-Exit Discriminator."),
        ("What does the AS-PATH attribute indicate in BGP routing?",
         "FlashAttention optimizes standard Transformer self-attention by tiling inputs to minimize GPU SRAM to HBM memory IO overhead."),
        ("How do Redis clusters utilize hash slots to automatically partition data?",
         "Every master node in the cluster is responsible for a contiguous subset of these 16,384 slots. When a client requests a key, if the node holds the slot it serves it; otherwise it replies with -MOVED."),
        ("How do Redis clusters utilize hash slots to automatically partition data?",
         "In the Raft consensus algorithm, the leader acts as the sole ingress for client requests. Followers replicate log entries."),
        ("What is the primary role of the Leader node in Raft consensus?",
         "In the Raft consensus algorithm, the leader acts as the sole ingress for client requests. It appends requests to its log and sends AppendEntries RPCs to follower nodes."),
        ("What is the primary role of the Leader node in Raft consensus?",
         "OAuth 2.0 uses granular custom scopes to enforce the Principle of Least Privilege across microservices."),
        ("How does the Circuit Breaker pattern manage state transitions?",
         "The Circuit Breaker transitions across Closed, Open, and Half-Open states. In Open state, requests fail fast. After a sleep duration, Half-Open permits test requests."),
        ("How does the Circuit Breaker pattern manage state transitions?",
         "Automotive CAN bus architectures use differential signaling across CAN-H and CAN-L to reject common-mode electromagnetic noise."),
        ("How do recent advancements like FlashAttention address memory complexity?",
         "FlashAttention addresses quadratic complexity bottlenecks through hardware-aware IO tiling, avoiding materializing the full N x N attention matrix in GPU HBM."),
        ("How do recent advancements like FlashAttention address memory complexity?",
         "PostgreSQL JSONB columns store semi-structured data in decomposed binary format, supporting GIN index lookups via jsonb_path_ops."),
        ("Why is relying purely on OAuth 2.0 considered a security vulnerability?",
         "Relying purely on OAuth 2.0 for identity verification is a vulnerability because OAuth 2.0 is an authorization framework, not an authentication protocol. OIDC bridges this gap with ID tokens."),
        ("Why is relying purely on OAuth 2.0 considered a security vulnerability?",
         "Hydrogen fuel cell electric vehicles (FCEVs) store hydrogen at 700 bar in carbon-fiber reinforced polymer tanks."),
        ("What occurs during Redis cluster resharding or node addition?",
         "During resharding, slots and their associated keys are migrated from existing nodes to newly added master nodes using CLUSTER SETSLOT commands without downtime."),
        ("What occurs during Redis cluster resharding or node addition?",
         "The Scramble for Africa culminated at the Berlin Conference of 1884-1885 where European powers drew artificial borders."),
        ("What is the function of Raft consensus within etcd in Kubernetes?",
         "etcd uses Raft consensus to guarantee consistent replication of Kubernetes cluster state across a quorum of majority nodes."),
        ("What is the function of Raft consensus within etcd in Kubernetes?",
         "Tire dynamics and slip angle modeling describe lateral friction forces generated during cornering maneuvers.")
    ]
    print(f"\nPrepared {len(pairs)} representative technical query-document pairs.")

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "gpu_name": device_name,
        "total_vram_mb": total_vram,
        "model_id": MODEL_NAME,
        "gpu_benchmark": {},
        "cpu_benchmark": {}
    }

    # -------------------------------------------------------------
    # 2. GPU Benchmark (cuda:0)
    # -------------------------------------------------------------
    if cuda_available:
        print("\n" + "-" * 50)
        print("RUNNING GPU BENCHMARK (cuda:0)")
        print("-" * 50)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        ram_before = get_ram_usage_mb()
        t0 = time.perf_counter()
        model_gpu = CrossEncoder(MODEL_NAME, device="cuda:0", max_length=512)
        load_time_gpu = time.perf_counter() - t0
        ram_after_load = get_ram_usage_mb()
        vram_after_load = get_vram_usage_mb()

        print(f"Model initialization time (GPU): {load_time_gpu:.4f}s")
        print(f"RAM usage after load:           {ram_after_load:.2f} MB (Delta: +{ram_after_load - ram_before:.2f} MB)")
        print(f"VRAM allocated after load:      {vram_after_load['allocated_mb']} MB (Reserved: {vram_after_load['reserved_mb']} MB)")

        # Warmup
        _ = model_gpu.predict([("warmup query", "warmup passage")], show_progress_bar=False)
        torch.cuda.synchronize(0)

        # Benchmark Batch Size = 4 (default)
        torch.cuda.reset_peak_memory_stats(0)
        t1 = time.perf_counter()
        scores_b4 = model_gpu.predict(pairs, batch_size=4, show_progress_bar=False)
        torch.cuda.synchronize(0)
        gpu_time_b4 = time.perf_counter() - t1
        vram_b4 = get_vram_usage_mb()

        # Benchmark Batch Size = 5
        t2 = time.perf_counter()
        scores_b5 = model_gpu.predict(pairs, batch_size=5, show_progress_bar=False)
        torch.cuda.synchronize(0)
        gpu_time_b5 = time.perf_counter() - t2
        vram_b5 = get_vram_usage_mb()

        print(f"Inference latency (20 pairs, batch_size=4): {gpu_time_b4 * 1000:.2f} ms ({gpu_time_b4 / len(pairs) * 1000:.2f} ms/pair)")
        print(f"Throughput (batch_size=4):                   {len(pairs) / gpu_time_b4:.2f} pairs/sec")
        print(f"Peak VRAM during inference:                 {vram_b4['max_allocated_mb']} MB")

        print(f"Inference latency (20 pairs, batch_size=5): {gpu_time_b5 * 1000:.2f} ms ({gpu_time_b5 / len(pairs) * 1000:.2f} ms/pair)")
        print(f"Throughput (batch_size=5):                   {len(pairs) / gpu_time_b5:.2f} pairs/sec")
        print(f"Peak VRAM during inference:                 {vram_b5['max_allocated_mb']} MB")

        results["gpu_benchmark"] = {
            "load_time_s": round(load_time_gpu, 4),
            "ram_load_delta_mb": round(ram_after_load - ram_before, 2),
            "vram_load_allocated_mb": vram_after_load["allocated_mb"],
            "vram_load_reserved_mb": vram_after_load["reserved_mb"],
            "batch_4": {
                "total_time_ms": round(gpu_time_b4 * 1000, 2),
                "per_pair_ms": round(gpu_time_b4 / len(pairs) * 1000, 2),
                "throughput_pairs_per_sec": round(len(pairs) / gpu_time_b4, 2),
                "peak_vram_mb": vram_b4["max_allocated_mb"],
            },
            "batch_5": {
                "total_time_ms": round(gpu_time_b5 * 1000, 2),
                "per_pair_ms": round(gpu_time_b5 / len(pairs) * 1000, 2),
                "throughput_pairs_per_sec": round(len(pairs) / gpu_time_b5, 2),
                "peak_vram_mb": vram_b5["max_allocated_mb"],
            },
            "sample_scores": [round(float(s), 4) for s in scores_b4[:5]]
        }

        del model_gpu
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # 3. CPU Benchmark (Fallback verification)
    # -------------------------------------------------------------
    print("\n" + "-" * 50)
    print("RUNNING CPU FALLBACK BENCHMARK")
    print("-" * 50)
    ram_before_cpu = get_ram_usage_mb()
    t0 = time.perf_counter()
    model_cpu = CrossEncoder(MODEL_NAME, device="cpu", max_length=512)
    load_time_cpu = time.perf_counter() - t0
    ram_after_cpu_load = get_ram_usage_mb()

    # Inference on CPU (batch_size=4)
    t1 = time.perf_counter()
    scores_cpu = model_cpu.predict(pairs, batch_size=4, show_progress_bar=False)
    cpu_time = time.perf_counter() - t1

    print(f"Model initialization time (CPU): {load_time_cpu:.4f}s")
    print(f"RAM usage after CPU load:        {ram_after_cpu_load:.2f} MB")
    print(f"Inference latency (20 pairs, CPU): {cpu_time * 1000:.2f} ms ({cpu_time / len(pairs) * 1000:.2f} ms/pair)")
    print(f"Throughput (CPU):                 {len(pairs) / cpu_time:.2f} pairs/sec")

    results["cpu_benchmark"] = {
        "load_time_s": round(load_time_cpu, 4),
        "ram_load_delta_mb": round(ram_after_cpu_load - ram_before_cpu, 2),
        "total_time_ms": round(cpu_time * 1000, 2),
        "per_pair_ms": round(cpu_time / len(pairs) * 1000, 2),
        "throughput_pairs_per_sec": round(len(pairs) / cpu_time, 2),
        "sample_scores": [round(float(s), 4) for s in scores_cpu[:5]]
    }

    del model_cpu
    gc.collect()

    out_file = PROJECT_ROOT / "reports" / "phase11_model_benchmark_data.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark completed successfully! Raw data written to: {out_file}")
    return results


if __name__ == "__main__":
    run_benchmark()

