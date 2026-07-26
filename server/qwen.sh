#!/bin/bash
# Launches Qwen2.5-1.5B-Instruct on vLLM.
# Usage: bash server/qwen.sh [max_num_seqs]
# Example: bash server/qwen.sh 64
MAX_NUM_SEQS=${1:-64}

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --dtype auto \
    --api-key token-abc123 \
    --port 1105 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs "$MAX_NUM_SEQS"
