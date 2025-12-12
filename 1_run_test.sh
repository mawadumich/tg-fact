#!/bin/bash

# ============================================================================
# PHASE 1: Quick Test 
# ============================================================================
echo "=== PHASE 1: Testing with 10 examples ==="
python main.py \
    --sample_size 10 \
    --data_path ./data/paper_dev.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --methods random \
    --batch_size 10 \
    --gpu_memory 0.95 \
    --output_dir ./results/test

# ============================================================================
# MONITORING COMMANDS
# ============================================================================

# Monitor GPU usage in another terminal:
# watch -n 1 nvidia-smi

# Monitor process with resource usage:
# nvidia-smi dmon -s pucvmet

# Check memory usage during run:
# watch -n 5 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv'