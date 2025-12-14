from typing import List, Dict
import pandas as pd
import os
import time


class FEVEREvaluator:
    def __init__(self, checker):
        self.checker = checker
        self.results = {}
        
    def evaluate_all_methods(self, df, methods, chunk_size=32, sc_samples=5, 
                            tot_depth=3, tot_branches=3, sc_chunk_size=10, tot_min_confidence=0.3):
        claims = df['claim'].tolist()
        evidences_raw = df['evidence'].tolist() 
        true_labels = df['label'].tolist()
        
        print(f"\n{'='*70}")
        print(f"Evaluating {len(df)} examples")
        print(f"Methods: {', '.join(methods)}")
        print(f"Chunk size: {chunk_size} (CoT), {sc_chunk_size} (SC)")
        print(f"{'='*70}\n")
        
        for method in methods:
            print(f"\n--- Evaluating {method.upper()} ---")
            start_time = time.time()
            
            try:
                if method == 'random':
                    results = self.checker.random_baseline(claims, evidences_raw)
                    
                elif method == 'cot':
                    results = self.checker.chain_of_thought(
                        claims, evidences_raw, chunk_size=chunk_size
                    )
                
                elif method == 'sc':
                    results = self.checker.self_consistency_cot(
                        claims, evidences_raw, 
                        n_samples=sc_samples, 
                        chunk_size=sc_chunk_size
                    )
                
                elif method == 'tot':
                    
                    results = self.checker.tree_of_thoughts(
                        claims, evidences_raw, 
                        depth=tot_depth, 
                        branches=tot_branches,
                        min_confidence_threshold=tot_min_confidence
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
            
            
            pred_dist = df['label'].value_counts(normalize=True) * 100
            pred_metrics = {f'pred_{k}_pct': v for k, v in pred_dist.items()}
            
            
            if 'NOT ENOUGH INFO' not in df['true_label'].unique():
                not_enough_count = (df['label'] == 'NOT ENOUGH INFO').sum()
                pred_metrics['uncertain_rate'] = (not_enough_count / len(df)) * 100
            
            metrics.append({
                'Method': method,
                'Accuracy': accuracy,
                **class_metrics,
                **pred_metrics,
                'Num_Samples': len(df)
            })
            
            
            metrics_df = pd.DataFrame([{
                'Method': method,
                'Accuracy': accuracy,
                **class_metrics,
                **pred_metrics,
                'Num_Samples': len(df)
            }]).round(2)
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
        
        print(f"\n Results saved to {output_dir}/")
