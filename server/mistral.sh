#!/bin/bash
# Launches Mistral-7B-Instruct-v0.3 on vLLM.
# Usage: bash server/mistral.sh [max_num_seqs]
# Example: bash server/mistral.sh 64
#
# Note: on some driver/vLLM combinations, disabling the FlashInfer sampler
# avoids a sampler-related crash on this model. If you hit that issue,
# uncomment the line below before launching.
# export VLLM_USE_FLASHINFER_SAMPLER=0
MAX_NUM_SEQS=${1:-64}

vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
    --dtype auto \
    --api-key token-abc123 \
    --port 1105 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs "$MAX_NUM_SEQS"
