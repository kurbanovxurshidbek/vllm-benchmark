# README

# Throughput and Latency Trade-offs in vLLM-Based LLM Serving: A GPU Memory-Tier Study and an Empirical Calibration Rule for max_num_seqs

## Overview

This repository contains the benchmark implementation, experimental configurations, and benchmark datasets used in the research paper **"Throughput and Latency Trade-offs in vLLM-Based LLM Serving: A GPU Memory-Tier Study and an Empirical Calibration Rule for max_num_seqs."**

The objective of this project is to evaluate how different **max_num_seqs** values influence throughput and latency in vLLM inference serving under different GPU memory capacities.

The experiments compare two GPU memory tiers:

* NVIDIA L4 (24 GB VRAM)
* NVIDIA RTX PRO (96 GB VRAM)

Four open-source Large Language Models (LLMs) were evaluated under identical benchmarking conditions.

---

# Repository Structure

```text
.
├── README.md
├── benchmark.py
├── requirements.txt
├── benchmark_results.csv
├── benchmark_results.xlsx
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
└── LICENSE
```

---

# Hardware Configuration

Two GPU configurations were benchmarked.

| Configuration | GPU            | VRAM  |
| ------------- | -------------- | ----- |
| Low Pod       | NVIDIA L4      | 24 GB |
| High Pod      | NVIDIA RTX PRO | 96 GB |

---

# Software Environment

The experiments were conducted using:

* Python
* vLLM
* OpenAI Python SDK
* ThreadPoolExecutor
* RunPod
* CUDA
* HuggingFace Transformers

---

# Tested Models

The following instruction-tuned models were evaluated.

| Model                    |
| ------------------------ |
| Qwen2.5-1.5B-Instruct    |
| Llama-3.2-3B-Instruct    |
| Gemma-2-2B-Instruct      |
| Mistral-7B-Instruct-v0.3 |

---

# Experimental Design

The benchmark compares inference performance across different batch scheduling configurations.

Each experiment used:

* 10,000 inference requests
* 128 concurrent workers
* OpenAI-compatible API
* Same benchmark client
* Same prompt
* Same request format

Each model was evaluated using:

* max_num_seqs = 16
* max_num_seqs = 32
* max_num_seqs = 64
* max_num_seqs = 128
* max_num_seqs = 256

The complete study therefore contains:

* 2 GPU platforms
* 4 LLMs
* 5 scheduling configurations
* 40 benchmark experiments

---

# Benchmark Client

The benchmark client communicates with the vLLM OpenAI-compatible API.

For each request it measures:

* Success rate
* Response latency
* Throughput
* Benchmark duration

After all requests complete, the following statistics are calculated:

* Average latency
* Median latency
* P95 latency
* P99 latency
* Minimum latency
* Maximum latency
* Throughput (requests/second)

---

# Running the Server

Example:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --dtype auto \
    --api-key token-abc123 \
    --port 1105 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64
```

For Llama experiments FP8 KV Cache was enabled.

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --dtype auto \
    --api-key token-abc123 \
    --port 1105 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --kv-cache-dtype fp8
```

---

# Running the Benchmark

Run the benchmark after the server starts.

```bash
python benchmark.py
```

The benchmark automatically sends 10,000 requests and computes the performance statistics.

---

# Output Metrics

For every experiment the following metrics are reported.

| Metric          | Description           |
| --------------- | --------------------- |
| Benchmark Time  | Total execution time  |
| Throughput      | Requests per second   |
| Average Latency | Mean response latency |
| Median Latency  | 50th percentile       |
| P95 Latency     | 95th percentile       |
| P99 Latency     | 99th percentile       |
| Minimum Latency | Fastest response      |
| Maximum Latency | Slowest response      |

---

# Dataset Description

The repository includes the complete benchmark dataset used in the paper.

Each record contains:

* GPU
* GPU Memory
* Model
* max_num_seqs
* Number of Requests
* Benchmark Time
* Throughput
* Average Latency
* Median Latency
* P95 Latency
* P99 Latency
* Minimum Latency
* Maximum Latency

The benchmark dataset contains all 40 experimental runs reported in the manuscript.

---

# Methodology

The benchmark methodology consists of the following steps.

1. Deploy a vLLM server using one of the selected LLMs.
2. Configure the desired value of `max_num_seqs`.
3. Launch the benchmark client.
4. Send 10,000 concurrent inference requests.
5. Record request completion time.
6. Calculate throughput and latency statistics.
7. Repeat the procedure for every GPU platform and every model.

---

# Reproducing the Results

To reproduce the experiments:

### Step 1

Clone the repository.

```bash
git clone https://github.com/kurbanovxurshidbek/vllm-benchmark.git
```

### Step 2

Install the required packages.

```bash
pip install -r requirements.txt
```

### Step 3

Start the vLLM server.

```bash
bash server/qwen.sh
```

or

```bash
bash server/llama.sh
```

or

```bash
bash server/gemma.sh
```

or

```bash
bash server/mistral.sh
```

### Step 4

Run the benchmark.

```bash
python benchmark.py
```

### Step 5

Collect benchmark statistics.

---

# Expected Output

The benchmark prints results similar to:

```text
====================================================
Benchmark Summary
====================================================

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

---

# Requirements

Recommended Python packages:

```text
openai
vllm
transformers
torch
accelerate
sentencepiece
```

Install using:

```bash
pip install -r requirements.txt
```

---

# Citation

If you use this benchmark or dataset in your research, please cite the associated publication.

```bibtex
@article{Kurbanov2026,
  title={Throughput and Latency Trade-offs in vLLM-Based LLM Serving: A GPU Memory-Tier Study and an Empirical Calibration Rule for max_num_seqs},
  author={Kurbanov, Xurshidbek and co-authors},
  journal={PeerJ Computer Science},
  year={2026}
}
```

---

# License

This repository is released for research and academic purposes.

Please cite the associated publication when using the benchmark implementation or experimental dataset.

---

# Contact

For questions regarding the benchmark implementation, experiments, or dataset, please contact the corresponding author through the contact information provided in the associated publication.
