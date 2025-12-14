import argparse
import os
import sys
from data_loader import load_fever_dataset
from fact_checker_vllm import FEVERFactCheckerVLLM
from fever_evaluator import FEVEREvaluator

def main():
    parser = argparse.ArgumentParser(
        description='FEVER Fact-Checking with vLLM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_path', type=str, default='data/paper_dev.jsonl',
                       help='Path to FEVER dataset')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='Number of examples to evaluate')
    parser.add_argument('--methods', nargs='+', default=['random', 'cot'],
                       choices=['random', 'cot', 'sc', 'tot'],
                       help='Methods to evaluate')
    parser.add_argument('--model', type=str,
                       default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                       help='HuggingFace model name')
    parser.add_argument('--gpu_memory', type=float, default=0.95,
                       help='GPU memory utilization (0-1)')
    parser.add_argument('--max_model_len', type=int, default=8192,
                       help='Maximum model sequence length')
    parser.add_argument('--chunk_size', type=int, default=32,
                       help='Chunk size for CoT batched inference')
    parser.add_argument('--sc_chunk_size', type=int, default=10,
                       help='Chunk size for SC (smaller due to n_samples multiplication)')
    parser.add_argument('--sc_samples', type=int, default=5,
                       help='Number of samples for self-consistency')
    parser.add_argument('--tot_depth', type=int, default=3,
                       help='Tree depth for ToT')
    parser.add_argument('--tot_branches', type=int, default=3,
                       help='Branches per node for ToT')
    parser.add_argument('--tot_min_confidence', type=float, default=0.3,
                       help='Minimum confidence threshold for ToT pruning')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--verifiable_only', action='store_true',
                       help='Only use verifiable claims (exclude NOT ENOUGH INFO)')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Directory to cache models (default: HuggingFace cache)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FEVER FACT-CHECKING WITH VLLM")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_path}")
    print(f"Sample size: {args.sample_size}")
    print(f"Methods: {args.methods}")
    print(f"Chunk sizes: CoT={args.chunk_size}, SC={args.sc_chunk_size}")
    print("="*70)
    
    if not os.path.exists(args.data_path):
        print(f"\nError: Data file not found at {args.data_path}")
        sys.exit(1)
    
    print("\n[1/4] Loading model...")
    checker = FEVERFactCheckerVLLM(
        model_name=args.model,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        cache_dir=args.cache_dir
    )
    
    print("\n[2/4] Loading FEVER dataset...")
    df = load_fever_dataset(args.data_path, sample_size=args.sample_size, 
                           verifiable_only=args.verifiable_only)
    print(f"Loaded {len(df)} examples")
    if args.verifiable_only:
        print("(Filtered to verifiable claims only - SUPPORTS/REFUTES)")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    print("\n[3/4] Running evaluation...")
    evaluator = FEVEREvaluator(checker)
    evaluator.evaluate_all_methods(
        df,
        methods=args.methods,
        chunk_size=args.chunk_size,
        sc_chunk_size=args.sc_chunk_size,
        sc_samples=args.sc_samples, 
        tot_depth=args.tot_depth,
        tot_branches=args.tot_branches,
        tot_min_confidence=args.tot_min_confidence
    )
    
    print("\n[4/4] Saving results...")
    evaluator.save_results(args.output_dir)
    
    print("\n" + "="*70)
    print(" EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()