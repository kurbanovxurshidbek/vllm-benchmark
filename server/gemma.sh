#!/bin/bash
# Launches Gemma-2-2B-Instruct on vLLM.
# Usage: bash server/gemma.sh [max_num_seqs]
# Example: bash server/gemma.sh 64
MAX_NUM_SEQS=${1:-64}

vllm serve google/gemma-2-2b-it \
    --dtype auto \
    --api-key token-abc123 \
    --port 1105 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs "$MAX_NUM_SEQS"
