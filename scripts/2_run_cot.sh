#!/bin/bash

# ============================================================================
# Chain-of-Thought
# ============================================================================
echo "=== Baseline evaluation ==="
python main.py \
    --sample_size 100 \
    --data_path ./data/paper_dev.jsonl \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --methods cot \
    --chunk_size 64 \
    --gpu_memory 0.95 \
    --cache_dir ./model_cache \
    --verifiable_only \
    --output_dir ./results/baseline
