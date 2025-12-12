#!/bin/bash

# ============================================================================
# PHASE 4: Tree-of-Thoughts (4-6 hours)
# Full evaluation with all methods
# ============================================================================
echo "=== PHASE 4: Full evaluation with ToT ==="
python main.py \
    --sample_size 1000 \
    --methods tot \
    --batch_size 128 \
    --sc_samples 5 \
    --tot_depth 3 \
    --tot_branches 3 \
    --gpu_memory 0.95 \
    --output_dir ./results/full

# Expected results:
# - ToT: 65-75% accuracy (best performance)
# - Time: ~4-6 hours