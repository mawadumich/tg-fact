#!/bin/bash

# ============================================================================
# PHASE 3: Self-Consistency (1.5-2 hours)
# Add self-consistency with 5 samples
# ============================================================================
echo "=== PHASE 3: Self-Consistency evaluation ==="
python main.py \
    --sample_size 10 \
    --methods random cot sc \
    --batch_size 128 \
    --sc_samples 5 \
    --gpu_memory 0.95 \
    --output_dir ./results/self_consistency

# Expected results:
# - SC: 60-70% accuracy (5-10% improvement over CoT)
# - Time: ~1.5-2 hours