#!/bin/bash

# ============================================================================
# Chain-of-Thought on 1000 examples
# ============================================================================
echo "=== PHASE 2: Baseline evaluation (1000 examples) ==="
python main.py \
    --sample_size 10 \
    --methods cot \
    --batch_size 128 \
    --gpu_memory 0.95 \
    --output_dir ./results/baseline

# Expected results:
# - Random: ~33% accuracy
# - CoT: 55-65% accuracy
# - Time: ~20-30 minutes