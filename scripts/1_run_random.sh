#!/bin/bash

echo "=== Random Baseline ==="
python main.py \
    --sample_size 100 \
    --data_path ./data/paper_dev.jsonl \
    --methods random \
    --output_dir ./results/test