# vLLM Benchmark: GPU Memory Tier, Batching, and Throughput-Latency Trade-offs in LLM Serving

Companion code and data repository for the paper *"Throughput-Latency Trade-offs in GPU-Tiered LLM Serving"* (Kurbanov & Rakhimov, submitted to PeerJ Computer Science).

## Description

This repository contains everything needed to reproduce the stress-test benchmarks reported in the paper: the benchmarking client used to load-test a running [vLLM](https://github.com/vllm-project/vllm) server, launch scripts for each model, the full curated dataset of results, plotting code, and the analysis script that computes the paper's proposed empirical calibration rule (N*) for choosing `max_num_seqs`.

The study benchmarks four open-weight, instruction-tuned language models — Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct, Gemma-2-2B-Instruct, and Mistral-7B-Instruct-v0.3 — served through vLLM on two GPU tiers (a 24 GB NVIDIA L4 and a 96 GB NVIDIA RTX PRO), sweeping the scheduler's `max_num_seqs` concurrency parameter across five values (16, 32, 64, 128, 256). For each of the 40 resulting configurations, 10,000 concurrent chat-completion requests were sent through a 128-worker thread pool, and throughput and latency percentiles were recorded.

## Repository Structure

```text
.
├── README.md
├── benchmark.py            # benchmarking client
├── make_plots.py           # regenerates plots/*.png from the CSV
├── requirements.txt
├── benchmark_results.csv   # curated dataset (40 rows)
├── benchmark_results.xlsx  # same dataset, Excel format
├── server/
│   ├── qwen.sh
│   ├── llama.sh
│   ├── gemma.sh
│   └── mistral.sh
├── plots/
│   ├── throughput.png
│   ├── latency.png
│   ├── p95_latency.png
│   └── p99_latency.png
├── analysis/
│   └── calibration_rule.py # reproduces the paper's N* diagnostic rule
└── LICENSE
```

## Hardware Configuration

Two GPU configurations were benchmarked.

| Configuration | GPU            | VRAM  |
| -------------- | --------------- | ----- |
| Low Pod        | NVIDIA L4       | 24 GB |
| High Pod       | NVIDIA RTX PRO  | 96 GB |

## Software Environment

The experiments were conducted using:

- Python
- vLLM
- OpenAI Python SDK
- ThreadPoolExecutor
- RunPod
- CUDA
- Hugging Face Transformers

## Tested Models

| Model                    |
| ------------------------- |
| Qwen2.5-1.5B-Instruct     |
| Llama-3.2-3B-Instruct     |
| Gemma-2-2B-Instruct       |
| Mistral-7B-Instruct-v0.3  |

## Dataset Information

The curated dataset is provided in both `benchmark_results.csv` and `benchmark_results.xlsx`. It contains one row per (model, GPU, `max_num_seqs`) configuration — 40 rows in total.

**Columns:**

| Column | Description |
|---|---|
| `model` | Model name (Hugging Face identifier style) |
| `gpu` | GPU used to serve the model (`NVIDIA-L4-24GB` or `NVIDIA-RTX-PRO-96GB`) |
| `max_num_seqs` | vLLM scheduler concurrency limit tested (16, 32, 64, 128, or 256) |
| `kv_cache_dtype` | KV-cache precision used (`auto` or `fp8`; FP8 was used only for Llama-3.2-3B-Instruct) |
| `total_requests` | Total number of chat-completion requests sent in this run (10,000 for every row) |
| `successful_requests` | Number of requests that completed without error |
| `failed_requests` | Number of requests that failed (0 for every configuration in this study) |
| `benchmark_time_sec` | Total wall-clock time for the run, in seconds |
| `throughput_req_per_sec` | `successful_requests / benchmark_time_sec` |
| `mean_latency_sec` | Mean per-request latency, in seconds |
| `median_latency_sec` | Median per-request latency, in seconds |
| `p95_latency_sec` | 95th-percentile per-request latency, in seconds |
| `p99_latency_sec` | 99th-percentile per-request latency, in seconds |
| `min_latency_sec` | Minimum observed per-request latency, in seconds |
| `max_latency_sec` | Maximum observed per-request latency, in seconds |

No personally identifiable or sensitive information is contained in this dataset; all values are system performance measurements. The complete study contains 2 GPU platforms x 4 LLMs x 5 scheduling configurations = 40 benchmark experiments, each with a 100% request success rate.

## Code Information

| File | Purpose |
|---|---|
| `benchmark.py` | Benchmarking client. Sends a configurable number of concurrent chat-completion requests to a running vLLM server and reports throughput and latency statistics. Used to generate every row of `benchmark_results.csv`. |
| `server/qwen.sh`, `server/llama.sh`, `server/gemma.sh`, `server/mistral.sh` | vLLM launch scripts, one per model, with `max_num_seqs` as an optional argument. |
| `make_plots.py` | Regenerates the four PNG charts in `plots/` directly from `benchmark_results.csv`. |
| `analysis/calibration_rule.py` | Reads `benchmark_results.csv` and reproduces the paper's empirical calibration rule: the marginal throughput gain `g(N)` from doubling `max_num_seqs`, and the resulting diagnostic recommendation `N*` under a chosen gain threshold epsilon. |

For each request, the benchmark client measures success/failure and response latency. After all requests complete, it reports: throughput (requests/second), and mean, median, P95, P99, minimum, and maximum latency.

## Usage Instructions

### 1. Install dependencies

```bash
git clone https://github.com/kurbanovxurshidbek/vllm-benchmark.git
cd vllm-benchmark
pip install -r requirements.txt
```

### 2. Start a vLLM server

Pick one of the four launch scripts and optionally pass a `max_num_seqs` value (default: 64):

```bash
bash server/qwen.sh 64
# or: bash server/llama.sh 64
# or: bash server/gemma.sh 64
# or: bash server/mistral.sh 64
```

### 3. Run the benchmark

In a separate terminal:

```bash
python benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct
```

All arguments have defaults, so `python benchmark.py` alone works against a server started with `server/qwen.sh`. To reproduce a specific row of `benchmark_results.csv`, match the `--model` to the launch script and the server's `--max-num-seqs` to your chosen value, then save the summary:

```bash
python benchmark.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --total-requests 10000 \
    --max-workers 128 \
    --output results_llama_seqs64_l4.csv
```

Expected output:

```text
============================================================
Benchmark Summary
============================================================
Total Requests : 10000
Success        : 10000
Failed         : 0

Benchmark Time : xx.xx sec
Throughput     : xxx.xx req/sec

Average Latency : x.xxx sec
Median Latency  : x.xxx sec
P95 Latency     : x.xxx sec
P99 Latency     : x.xxx sec
Min Latency     : x.xxx sec
Max Latency     : x.xxx sec
```

Repeat for each combination of model, GPU, and `max_num_seqs` to reproduce the full 40-configuration sweep reported in the paper.

### 4. Regenerate the plots (optional)

```bash
python make_plots.py
```

### 5. Reproduce the calibration-rule analysis

```bash
python analysis/calibration_rule.py --csv benchmark_results.csv --epsilon 0.15
```

This prints the marginal throughput gain at each doubling step and the resulting N* value for all eight model-GPU pairs, matching Table 5 in the paper.

## Requirements

- Python 3.9+
- A running [vLLM](https://github.com/vllm-project/vllm) server for `benchmark.py` (not required for `analysis/calibration_rule.py` or `make_plots.py`, which only need the CSV)

Recommended Python packages (`requirements.txt`):

```text
openai
vllm
transformers
torch
accelerate
sentencepiece
```

Install with:

```bash
pip install -r requirements.txt
```

## Methodology

1. Deploy a vLLM server for one of the four models, with `dtype=auto`, `gpu-memory-utilization=0.90`, and a fixed `max-num-seqs` (FP8 KV cache additionally enabled for Llama-3.2-3B-Instruct).
2. Launch the benchmark client, which sends 10,000 chat-completion requests (prompt: `"Hello!"`) through a 128-worker thread pool.
3. Record total wall-clock time, per-request latency, and success/failure for every request.
4. Compute throughput and latency percentiles (mean, median, P95, P99, min, max) over all completed requests.
5. Repeat for every combination of model, GPU tier, and `max_num_seqs`, yielding the 40 rows in `benchmark_results.csv`.
6. Apply the marginal-gain diagnostic (`analysis/calibration_rule.py`) to the resulting throughput values to obtain the recommended `max_num_seqs` for each model-GPU pair.

Full methodological detail, including rationale and limitations, is given in the paper's Materials & Methods and Discussion sections.

## Calibration Rule Analysis

In addition to the raw benchmark data, this repository includes the script used to compute the paper's proposed empirical calibration rule for `max_num_seqs`.

For any tested value `N`, the marginal throughput gain from doubling concurrency is:

```
g(N) = [Throughput(2N) - Throughput(N)] / Throughput(N)
```

The recommended value, `N*`, is the smallest tested `N` at which this gain drops below a threshold epsilon (0.15 in the paper):

```
N*(epsilon) = min{ N in {16, 32, 64, 128, 256} : g(N) < epsilon }
```

This requires only measured throughput values; no model-architecture information is needed. Note that `N*` is a throughput-only recommendation and provides no memory-safety guarantee on its own — see the Discussion section of the paper for the full derivation and scope of applicability.

## Citations

If you use this code or dataset, please cite:

> Kurbanov, K., & Rakhimov, M. A. K. (2026). Throughput-Latency Trade-offs in GPU-Tiered LLM Serving. *PeerJ Computer Science* (in press).

```bibtex
@article{Kurbanov2026,
  title   = {Throughput-Latency Trade-offs in GPU-Tiered LLM Serving},
  author  = {Kurbanov, Khurshidbek and Rakhimov, Mukhammad Abdu Kayumbek},
  journal = {PeerJ Computer Science},
  year    = {2026}
}
```

This work builds on and cites the vLLM / PagedAttention paper:

> Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). Efficient memory management for large language model serving with PagedAttention. *Proceedings of the 29th ACM Symposium on Operating Systems Principles (SOSP '23)*.

## License & Contribution Guidelines

This repository is released under the [MIT License](LICENSE).

Contributions are welcome:

1. Open an issue describing the proposed change or bug.
2. Fork the repository and create a feature branch.
3. Submit a pull request referencing the issue.

## Contact

For questions about the paper or this repository, contact the corresponding author at shamsun.com@gmail.com.
