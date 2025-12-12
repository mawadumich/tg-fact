"""
FEVER Fact-Checking with vLLM - FIXED VERSION
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import json
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
import random
from tqdm import tqdm
import re
import argparse
import os
import sys
import time

class FEVERFactCheckerVLLM:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", gpu_memory_utilization=0.95, max_model_len=8192):
        
        print(f"Loading {model_name}...")
        print(f"GPU Memory Utilization: {gpu_memory_utilization*100}%")
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        available_gpus = torch.cuda.device_count()
        # tensor_parallel_size = args.tensor_parallel_size or available_gpus


        # if tensor_parallel_size > available_gpus:
        #     print(f"Warning: Requested {tensor_parallel_size} GPUs but only {available_gpus} available")
        #     tensor_parallel_size = available_gpus

        print(f"Available GPUs: {available_gpus}")
        # print(f"Using tensor_parallel_size: {available_gpus}")
        
        print("Loading vLLM model...")
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=available_gpus, 
            trust_remote_code=True,
            dtype="half",
            enforce_eager=False,  
            max_num_seqs=256
        )
        
        self.model_name = model_name
        print("✓ Model and tokenizer loaded successfully!")   

        # # Configure vLLM LLM instance
        # model_kwargs = {
        #     # 'model': args.model_name,
        #     'tensor_parallel_size': tensor_parallel_size,
        # }
        # if args.cache_dir:
        #     model_kwargs['download_dir'] = args.cache_dir

    def _format_prompt(self, user_message: str, system_message: str = None) -> str:
        if system_message is None:
            system_message = "You are a precise fact-checking assistant. You must analyze claims carefully."
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_special_tokens=False,
            add_generation_prompt=True,
        )
        
        return formatted
    
    def _format_evidence(self, evidence_raw):
        if not evidence_raw or evidence_raw == "No evidence provided.":
            return "No evidence provided."
        
        urls = []
        seen_articles = set()
        
        for evidence_set in evidence_raw:
            if evidence_set:
                for evidence in evidence_set:
                    if len(evidence) >= 3 and evidence[2] is not None:
                        article_title = evidence[2]
                        
                        if article_title in seen_articles:
                            continue
                        seen_articles.add(article_title)
                        
                        url = f"https://en.wikipedia.org/wiki/{article_title}"
                        urls.append(url)
        
        if urls:
            return "\n".join(urls)
        return "No evidence provided."
            
    def random_baseline(self, claims, evidences_raw) -> List[Dict]:
        results = []
        for _, _ in zip(claims, evidences_raw):
            label = random.choice(['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'])
            results.append({
                'method': 'random',
                'label': label,
                'rationale': 'Random selection'
            })
        return results
    
    def chain_of_thought(self, claims, evidences_raw) -> List[Dict]:
        few_shot_examples = """
        Example 1:
        Claim: Albert Einstein failed mathematics as a student.
        Reasoning: This is a common myth, so I would think about biographical information on Einstein or imagine searching for details about his school performance. Reputable sources consistently note that Einstein excelled in mathematics at a young age, which contradicts the claim. The answer is REFUTES.

        Now evaluate this claim:"""

        prompts = []
        for claim, _ in zip(claims, evidences_raw):
            
            prompt_text = f"""{few_shot_examples}

                        Claim: {claim}

                        Think step by step about whether this claim is SUPPORTS, REFUTES, or NOT ENOUGH INFO.

                        Reasoning:"""
                
            prompts.append(self._format_prompt(prompt_text))

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=512,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for i, output in enumerate(outputs):
            response_text = output.outputs[0].text
            label = self._extract_label(response_text)

            results.append({
                'method': 'chain_of_thought',
                'label': label,
                'rationale': response_text,
                'evidence_formatted': self._format_evidence(evidences_raw[i])
            })

        return results
    
    def self_consistency_cot(self, claims, evidences_raw, n_samples=5) -> List[Dict]:   
        few_shot_examples = """
        Example 1:
        Claim: Albert Einstein failed mathematics as a student.
        Reasoning: This is a common myth, so I would think about biographical information on Einstein or imagine searching for details about his school performance. Reputable sources consistently note that Einstein excelled in mathematics at a young age, which contradicts the claim. The answer is REFUTES.

        Now evaluate this claim:"""

        all_results = []

        for claim, _ in tqdm(zip(claims, evidences_raw), total=len(claims), desc="Self-Consistency"):
            prompt_text = f"""
                        You are a fact-checking assistant. Evaluate the claim below using step-by-step reasoning. Use your general knowledge and reasoning to decide if the claim is exactly one of: SUPPORTS, REFUTES, or NOT ENOUGH INFO.

                        {few_shot_examples}

                        Claim: {claim}

                        Think step by step and explain your reasoning in a concise paragraph, then provide your final verdict.

                        Your response:"""
            
            prompts = [self._format_prompt(prompt_text)] * n_samples

            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                stop_token_ids=[self.tokenizer.eos_token_id]
            )

            outputs = self.llm.generate(prompts, sampling_params, show_progress=False)

            labels = []
            reasoning_paths = []

            for output in outputs:
                response_text = output.outputs[0].text
                label = self._extract_label(response_text)
                labels.append(label)
                reasoning_paths.append(response_text)

            label_counts = Counter(labels)
            final_label = label_counts.most_common(1)[0][0]
            confidence = label_counts[final_label] / n_samples

            all_results.append({
                'method': 'self_consistency',
                'label': final_label,
                'confidence': confidence,
                'label_distribution': dict(label_counts),
                'rationale': f"Majority vote: {final_label} ({label_counts[final_label]}/{n_samples})",
                'reasoning_paths': reasoning_paths,
                'evidence_formatted': None 
            })

        return all_results

    
    def tree_of_thoughts(self, claims, evidences_raw, depth=3, branches=3) -> List[Dict]:
        all_results = []
        
        for claim in tqdm(claims, desc="Tree-of-Thoughts"):
            result = self._tot_single_example(claim, depth, branches)
            all_results.append(result)
        return all_results
    
    def _tot_single_example(self, claim, depth, branches) -> Dict:
        """Generate initial reasoning approaches"""
        root_prompt = f"""You are fact-checking this claim using your knowledge:

                        Claim: {claim}

                        Generate {branches} different reasons why this claim may be SUPPORTED, REFUTED, or NOT ENOUGH INFO. Each should be ONE clear reasoning step.

                        Format EXACTLY as:
                        REASON 1: [one sentence]
                        REASON 2: [one sentence]
                        REASON 3: [one sentence]"""

        formatted_prompt = self._format_prompt(root_prompt)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=300,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        output = self.llm.generate([formatted_prompt], sampling_params)[0]
        initial_approaches = self._parse_approaches(output.outputs[0].text, branches)
        
        best_score = -1
        best_path = []
        best_label = None
        
        # Explore each initial approach
        for approach in initial_approaches[:branches]:
            path, label, score = self._expand_branch(claim, approach, depth - 1, branches)
            
            if score > best_score:
                best_score = score
                best_path = [approach] + path
                best_label = label
        
        return {
            'method': 'tree_of_thoughts',
            'label': best_label,
            'score': best_score,
            'reasoning_paths': best_path
        }
    
    def _expand_branch(self, claim, current_reasoning, remaining_depth, branches) -> Tuple[List[str], str, float]:
        """Recursively expand reasoning tree"""
        
        if remaining_depth == 0:
            final_prompt = f"""Based on this reasoning chain:

                                Claim: {claim}
                                Reasoning so far: {current_reasoning}

                                Provide your final fact-checking verdict.

                                Respond EXACTLY in this format:
                                VERDICT: [SUPPORTS or REFUTES or NOT ENOUGH INFO]
                                CONFIDENCE: [0.0 to 1.0]
                                """

            formatted_prompt = self._format_prompt(final_prompt)
            
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=150,
                stop_token_ids=[self.tokenizer.eos_token_id]
            )
            
            output = self.llm.generate([formatted_prompt], sampling_params)[0]
            response = output.outputs[0].text
            
            label = self._extract_label(response)
            confidence = self._extract_confidence(response)
            
            return [], label, confidence
        
        expand_prompt = f"""Continue this reasoning chain:

                            Claim: {claim}
                            Current reasoning: {current_reasoning}

                            Generate {branches} different next reasoning steps that build on the current reasoning.

                            Format EXACTLY as:
                            STEP 1: [one sentence]
                            STEP 2: [one sentence]  
                            STEP 3: [one sentence]"""

        formatted_prompt = self._format_prompt(expand_prompt)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=300,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )
        
        output = self.llm.generate([formatted_prompt], sampling_params)[0]
        next_steps = self._parse_approaches(output.outputs[0].text, branches)
        
        best_score = -1
        best_continuation = []
        best_label = None
        
        # Explore top branches
        for step in next_steps[:2]:
            new_reasoning = f"{current_reasoning}\n{step}"
            path, label, score = self._expand_branch(
                claim,
                new_reasoning,
                remaining_depth - 1,
                branches
            )
            
            if score > best_score:
                best_score = score
                best_continuation = [step] + path
                best_label = label
        
        return best_continuation, best_label, best_score

    def _extract_label(self, text) -> str:
        """Extract label with improved logic"""
        text_upper = text.upper()
        
        verdict_match = re.search(r'VERDICT:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)', text_upper)
        if verdict_match:
            return verdict_match.group(1)
        
        answer_match = re.search(r'ANSWER:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO)', text_upper)
        if answer_match:
            return answer_match.group(1)
        
        if 'NOT ENOUGH INFO' in text_upper:
            return 'NOT ENOUGH INFO'
        elif 'REFUTES' in text_upper or 'REFUTED' in text_upper:
            return 'REFUTES'
        elif 'SUPPORTS' in text_upper or 'SUPPORTED' in text_upper:
            return 'SUPPORTS'
        
        return 'NOT ENOUGH INFO'  
    
    def _extract_confidence(self, text) -> float:
        """Extract confidence score"""
        match = re.search(r'CONFIDENCE:\s*([\d.]+)', text, re.IGNORECASE)
        if match:
            try:
                conf = float(match.group(1))
                return min(max(conf, 0.0), 1.0) 
            except:
                pass
        return 0.5 
    
    def _parse_approaches(self, text, n) -> List[str]:
        """Parse numbered approaches/steps from text"""
        approaches = []
        
        patterns = [
            r'(?:APPROACH|STEP|REASON)\s+\d+:\s*(.+?)(?=(?:APPROACH|STEP|REASON)\s+\d+:|$)',
            r'\d+[\.)]\s*(.+?)(?=\d+[\.)]|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                approaches = [m.strip() for m in matches if m.strip() and len(m.strip()) > 10]
                break
        
        if not approaches:
            lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 10]
            approaches = lines[:n]
        
        return approaches[:n] if approaches else ["Analyze the factual accuracy of this claim"]


def load_fever_dataset(filepath, sample_size=None) -> pd.DataFrame:
    """
    Load FEVER dataset - keep raw evidence structure
    """
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append(item)
            except:
                continue
    
    df = pd.DataFrame(data)
    
    if 'label' in df.columns:
        df['label'] = df['label'].str.upper()
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    return df


class FEVEREvaluator:
    def __init__(self, checker):
        self.checker = checker
        self.results = {}
        
    def evaluate_all_methods(self, df, methods, batch_size=32, sc_samples=5, tot_depth=3, tot_branches=3):
        claims = df['claim'].tolist()
        evidences_raw = df['evidence'].tolist() 
        true_labels = df['label'].tolist()
        
        print(f"\n{'='*70}")
        print(f"Evaluating {len(df)} examples")
        print(f"Methods: {', '.join(methods)}")
        print(f"{'='*70}\n")
        
        for method in methods:
            print(f"\n--- Evaluating {method.upper()} ---")
            start_time = time.time()
            
            try:
                if method == 'random':
                    results = self.checker.random_baseline(claims, evidences_raw)
                    
                elif method == 'cot':
                    results = []
                    for i in tqdm(range(0, len(claims), batch_size), desc="CoT batches"):
                        batch_claims = claims[i:i+batch_size]
                        batch_evidences = evidences_raw[i:i+batch_size]
                        batch_results = self.checker.chain_of_thought(batch_claims, batch_evidences)
                        results.extend(batch_results)
                
                elif method == 'sc':
                    results = self.checker.self_consistency_cot(
                        claims, evidences_raw, n_samples=sc_samples
                    )
                
                elif method == 'tot':
                    results = self.checker.tree_of_thoughts(
                        claims, evidences_raw, depth=tot_depth, branches=tot_branches
                    )
                
                for i, result in enumerate(results):
                    result['true_label'] = true_labels[i]
                    result['claim'] = claims[i]
                    result['correct'] = (result['label'] == true_labels[i])
                
                self.results[method] = results
                
                elapsed = time.time() - start_time
                accuracy = sum(r['correct'] for r in results) / len(results) * 100
                
                print(f"  Completed in {elapsed:.1f}s")
                print(f"  Accuracy: {accuracy:.2f}%")
                print(f"  Time per example: {elapsed/len(results):.2f}s")
                
            except Exception as e:
                print(f"  Error in {method}: {e}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def compute_metrics(self, output_dir) -> pd.DataFrame:
        metrics = []
        
        for method, results in self.results.items():
            if not results:
                continue
            
            df = pd.DataFrame(results)
            accuracy = df['correct'].mean() * 100
            
            class_metrics = {}
            for label in ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']:
                class_df = df[df['true_label'] == label]
                if len(class_df) > 0:
                    class_acc = class_df['correct'].mean() * 100
                    class_metrics[f'{label}_acc'] = class_acc
            
            metrics.append({
                'Method': method,
                'Accuracy': accuracy,
                **class_metrics,
                'Num_Samples': len(df)
            })
            
            data_sum = [{
                'Method': method,
                'Accuracy': accuracy,
                **class_metrics,
                'Num_Samples': len(df)
            }]
            metrics_df = pd.DataFrame(data_sum).round(2)
            metrics_df.to_csv(f'{output_dir}/{method}_metrics_summary.csv', index=False)
        
        return pd.DataFrame(metrics).round(2)
    
    def save_results(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        for method, results in self.results.items():
            if results:
                df = pd.DataFrame(results)
                df.to_csv(f'{output_dir}/{method}_results.csv', index=False)
        
        metrics_df = self.compute_metrics(output_dir)
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(metrics_df.to_string(index=False))
        print("="*70)
        
        print(f"\n✓ Results saved to {output_dir}/")


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
                       default='meta-llama/Meta-Llama-3-8B-Instruct',
                       help='HuggingFace model name')
    parser.add_argument('--gpu_memory', type=float, default=0.95,
                       help='GPU memory utilization (0-1)')
    parser.add_argument('--max_model_len', type=int, default=8192,
                       help='Maximum model sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for CoT inference')
    parser.add_argument('--sc_samples', type=int, default=5,
                       help='Number of samples for self-consistency')
    parser.add_argument('--tot_depth', type=int, default=3,
                       help='Tree depth for ToT')
    parser.add_argument('--tot_branches', type=int, default=3,
                       help='Branches per node for ToT')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FEVER FACT-CHECKING WITH WIKIPEDIA URLS")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_path}")
    print(f"Sample size: {args.sample_size}")
    print(f"Methods: {args.methods}")
    # print(f"Evidence format: Wikipedia URLs")
    print("="*70)
    
    if not os.path.exists(args.data_path):
        print(f"\n✗ Error: Data file not found at {args.data_path}")
        sys.exit(1)
    
    print("\n[1/4] Loading model...")
    checker = FEVERFactCheckerVLLM(
        model_name=args.model,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len
    )
    
    print("\n[2/4] Loading FEVER dataset...")
    df = load_fever_dataset(args.data_path, sample_size=args.sample_size)
    print(f"Loaded {len(df)} examples")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    if len(df) > 0:
        sample_evidence = checker._format_evidence(df['evidence'].iloc[0])
        print(f"\nExample evidence format:")
        print(f"{sample_evidence[:200]}...")
    
    print("\n[3/4] Running evaluation...")
    evaluator = FEVEREvaluator(checker)
    evaluator.evaluate_all_methods(
        df,
        methods=args.methods,
        batch_size=args.batch_size,
        sc_samples=args.sc_samples, 
        tot_depth=args.tot_depth,
        tot_branches=args.tot_branches
    )
    
    print("\n[4/4] Saving results...")
    evaluator.save_results(args.output_dir)
    
    print("\n" + "="*70)
    print("✓ EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()