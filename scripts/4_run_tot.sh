#!/bin/bash

echo "=== Full evaluation with ToT ==="
python main.py \
    --sample_size 100 \
    --data_path ./data/paper_dev.jsonl \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --methods tot \
    --tot_depth 3 \
    --tot_branches 3 \
    --tot_min_confidence 0.7 \
    --gpu_memory 0.95 \
    --cache_dir ./model_cache \
    --verifiable_only \
    --output_dir ./results/tot_optimized