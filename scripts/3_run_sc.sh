#!/bin/bash

echo "=== Self-Consistency evaluation ==="
python main.py \
    --sample_size 100 \
    --data_path ./data/paper_dev.jsonl \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --methods sc \
    --sc_chunk_size 20 \
    --sc_samples 3 \
    --gpu_memory 0.95 \
    --cache_dir ./model_cache \
    --verifiable_only \
    --output_dir ./results/self_consistency
