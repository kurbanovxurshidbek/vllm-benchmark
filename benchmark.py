#!/usr/bin/env python3
"""
vLLM Stress Test / Benchmark Script
====================================

Sends a configurable number of concurrent chat-completion requests to a
running vLLM OpenAI-compatible server and reports throughput and latency
statistics (mean, median, P95, P99, min, max).

This script was used to produce every row of benchmark_results.csv: each
row corresponds to one run of this script against a server started with
one of the scripts in server/ (qwen.sh, llama.sh, gemma.sh, mistral.sh),
with --max-num-seqs varied across 16, 32, 64, 128, 256, and run once on
each of the two GPU tiers described in the paper (NVIDIA L4 24GB and
NVIDIA RTX PRO 96GB).

Quick start (matches the README)
---------------------------------
1. Start a server, e.g.:  bash server/qwen.sh 64
2. In another terminal:   python benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct

All arguments have defaults, so `python benchmark.py` alone will work
against a server started with the default settings in server/qwen.sh.

Full example with all options:

    python benchmark.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --base-url http://localhost:1105/v1 \\
        --api-key token-abc123 \\
        --total-requests 10000 \\
        --max-workers 128 \\
        --output results_qwen_seqs64_l4.csv
"""

import argparse
import csv
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM stress-test / benchmark client")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help=f"Model name as registered with the vLLM server "
                              f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default="http://localhost:1105/v1",
                         help="Base URL of the vLLM OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default="token-abc123",
                         help="API key configured on the vLLM server (--api-key at launch)")
    parser.add_argument("--total-requests", type=int, default=10000,
                         help="Total number of chat-completion requests to send")
    parser.add_argument("--max-workers", type=int, default=128,
                         help="Number of concurrent worker threads (client-side concurrency)")
    parser.add_argument("--prompt", default="Hello!",
                         help="User prompt sent with every request")
    parser.add_argument("--output", default=None,
                         help="Optional path to write a single-row CSV summary of this run "
                              "(same schema as benchmark_results.csv)")
    return parser.parse_args()


def send_request(client, model, prompt, request_id):
    start = time.perf_counter()
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"success": True, "latency": time.perf_counter() - start}
    except Exception:
        return {"success": False, "latency": time.perf_counter() - start}


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print("=" * 60)
    print("vLLM Benchmark")
    print("=" * 60)
    print(f"Model          : {args.model}")
    print(f"Base URL       : {args.base_url}")
    print(f"Requests       : {args.total_requests}")
    print(f"Max Workers    : {args.max_workers}")
    print()

    results = []
    benchmark_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(send_request, client, args.model, args.prompt, i)
            for i in range(args.total_requests)
        ]
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 100 == 0:
                print(f"Completed {completed}/{args.total_requests}")

    benchmark_time = time.perf_counter() - benchmark_start

    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = sorted(r["latency"] for r in success)
    throughput = len(success) / benchmark_time if benchmark_time > 0 else 0.0

    print()
    print("=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print(f"Total Requests : {args.total_requests}")
    print(f"Success        : {len(success)}")
    print(f"Failed         : {len(failed)}")
    print()
    print(f"Benchmark Time : {benchmark_time:.2f} sec")
    print(f"Throughput     : {throughput:.2f} req/sec")
    print()

    if latencies:
        mean_lat = statistics.mean(latencies)
        median_lat = statistics.median(latencies)
        p95_lat = latencies[int(len(latencies) * 0.95)]
        p99_lat = latencies[int(len(latencies) * 0.99)]
        min_lat = min(latencies)
        max_lat = max(latencies)

        print(f"Average Latency : {mean_lat:.3f} sec")
        print(f"Median Latency  : {median_lat:.3f} sec")
        print(f"P95 Latency     : {p95_lat:.3f} sec")
        print(f"P99 Latency     : {p99_lat:.3f} sec")
        print(f"Min Latency     : {min_lat:.3f} sec")
        print(f"Max Latency     : {max_lat:.3f} sec")

        if args.output:
            with open(args.output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "model", "total_requests", "successful_requests", "failed_requests",
                    "benchmark_time_sec", "throughput_req_per_sec", "mean_latency_sec",
                    "median_latency_sec", "p95_latency_sec", "p99_latency_sec",
                    "min_latency_sec", "max_latency_sec",
                ])
                writer.writerow([
                    args.model, args.total_requests, len(success), len(failed),
                    round(benchmark_time, 2), round(throughput, 2), round(mean_lat, 3),
                    round(median_lat, 3), round(p95_lat, 3), round(p99_lat, 3),
                    round(min_lat, 3), round(max_lat, 3),
                ])
            print(f"\nSummary written to {args.output}")
    else:
        print("No successful requests; latency statistics unavailable.")


if __name__ == "__main__":
    main()
