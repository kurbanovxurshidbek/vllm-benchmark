#!/bin/bash
# Launches Llama-3.2-3B-Instruct on vLLM with an FP8 KV cache.
# Usage: bash server/llama.sh [max_num_seqs]
# Example: bash server/llama.sh 64
MAX_NUM_SEQS=${1:-64}

vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --dtype auto \
    --api-key token-abc123 \
    --port 1105 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --kv-cache-dtype fp8
