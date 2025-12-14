
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from typing import List, Dict, Tuple
from collections import Counter
import random
from tqdm import tqdm
import re

class FEVERFactCheckerVLLM:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                 gpu_memory_utilization=0.95, max_model_len=8192, cache_dir=None):
        
        print(f"Loading {model_name}...")
        print(f"GPU Memory Utilization: {gpu_memory_utilization*100}%")
        if cache_dir:
            print(f"Cache directory: {cache_dir}")
        
        print("Loading tokenizer...")
        tokenizer_kwargs = {'trust_remote_code': True}
        if cache_dir:
            tokenizer_kwargs['cache_dir'] = cache_dir
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **tokenizer_kwargs
        )

        available_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {available_gpus}")
        
        tensor_parallel = max(1, available_gpus)
        
        print("Loading vLLM model...")
        llm_kwargs = {
            'model': model_name,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_model_len': max_model_len,
            'tensor_parallel_size': tensor_parallel,
            'trust_remote_code': True,
            'dtype': "half",
            'enforce_eager': False,
            'max_num_seqs': 512  
        }
        if cache_dir:
            llm_kwargs['download_dir'] = cache_dir
        
        self.llm = LLM(**llm_kwargs)
        
        self.model_name = model_name
        print("Model and tokenizer loaded successfully!")

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
    
    def chain_of_thought(self, claims, evidences_raw, chunk_size=32) -> List[Dict]:
        few_shot_examples = """Example 1:
                                Claim: Albert Einstein failed mathematics as a student.
                                Reasoning: This is a common myth, so I would think about biographical information on Einstein or imagine searching for details about his school performance. Reputable sources consistently note that Einstein excelled in mathematics at a young age, which contradicts the claim. The answer is REFUTES.

                                Now evaluate this claim:"""

        all_results = []
        
        
        for i in tqdm(range(0, len(claims), chunk_size), desc="CoT chunks", total=(len(claims)+chunk_size-1)//chunk_size):
            chunk_claims = claims[i:i+chunk_size]
            chunk_evidences = evidences_raw[i:i+chunk_size]
            
            prompts = []
            for claim in chunk_claims:
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

            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

            for j, output in enumerate(outputs):
                response_text = output.outputs[0].text
                label = self._extract_label(response_text)

                all_results.append({
                    'method': 'chain_of_thought',
                    'label': label,
                    'rationale': response_text,
                    'evidence_formatted': self._format_evidence(chunk_evidences[j])
                })

        return all_results
    
    def self_consistency_cot(self, claims, evidences_raw, n_samples=5, chunk_size=10) -> List[Dict]:
        few_shot_examples = """Example 1:
                                Claim: Albert Einstein failed mathematics as a student.
                                Reasoning: This is a common myth, so I would think about biographical information on Einstein or imagine searching for details about his school performance. Reputable sources consistently note that Einstein excelled in mathematics at a young age, which contradicts the claim. The answer is REFUTES.

                                Now evaluate this claim:"""

        all_results = []

        
        for i in tqdm(range(0, len(claims), chunk_size), desc="SC chunks", total=(len(claims)+chunk_size-1)//chunk_size):
            chunk_claims = claims[i:i+chunk_size]
            
            
            all_prompts = []
            claim_indices = []  
            
            for claim_idx, claim in enumerate(chunk_claims):
                prompt_text = f"""{few_shot_examples}

                                    Claim: {claim}

                                    Think step by step and explain your reasoning in a concise paragraph, then provide your final verdict as SUPPORTS, REFUTES, or NOT ENOUGH INFO.

                                    Your response:"""
                                                    
                formatted = self._format_prompt(prompt_text)
                
                
                for _ in range(n_samples):
                    all_prompts.append(formatted)
                    claim_indices.append(claim_idx)

            sampling_params = SamplingParams(
                temperature=0.7,  
                top_p=0.9,
                max_tokens=512,
                stop_token_ids=[self.tokenizer.eos_token_id]
            )

            
            outputs = self.llm.generate(all_prompts, sampling_params, use_tqdm=False)

            
            claim_outputs = {idx: [] for idx in range(len(chunk_claims))}
            for output_idx, output in enumerate(outputs):
                claim_idx = claim_indices[output_idx]
                response_text = output.outputs[0].text
                label = self._extract_label(response_text)
                claim_outputs[claim_idx].append({
                    'label': label,
                    'reasoning': response_text
                })

            
            for claim_idx in range(len(chunk_claims)):
                labels = [o['label'] for o in claim_outputs[claim_idx]]
                reasoning_paths = [o['reasoning'] for o in claim_outputs[claim_idx]]
                
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

    def tree_of_thoughts(self, claims, evidences_raw, depth=3, branches=3, min_confidence_threshold=0.3) -> List[Dict]:
        all_results = []
        
        for claim in tqdm(claims, desc="Tree-of-Thoughts"):
            result = self._tot_single_example(claim, depth, branches, min_confidence_threshold)
            all_results.append(result)
        return all_results
    
    def _tot_single_example(self, claim, depth, branches, min_confidence_threshold=0.3) -> Dict:
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
        
        output = self.llm.generate([formatted_prompt], sampling_params, use_tqdm=False)[0]
        initial_approaches = self._parse_approaches(output.outputs[0].text, branches)
        
        best_score = -1
        best_path = []
        best_label = None
        
        for approach in initial_approaches[:branches]:
            path, label, score = self._expand_branch(claim, approach, depth - 1, branches, min_confidence_threshold)
            
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
    
    def _expand_branch(self, claim, current_reasoning, remaining_depth, branches, min_confidence_threshold=0.3) -> Tuple[List[str], str, float]:
        
        if remaining_depth == 0:
            
            final_prompt = f"""Based on this reasoning chain:

                                Claim: {claim}
                                Reasoning so far: {current_reasoning}

                                Provide your final fact-checking verdict.

                                Respond EXACTLY in this format:
                                VERDICT: [SUPPORTS or REFUTES or NOT ENOUGH INFO]
                                CONFIDENCE: [0.0 to 1.0]"""

            formatted_prompt = self._format_prompt(final_prompt)
            
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=150,
                stop_token_ids=[self.tokenizer.eos_token_id]
            )
            
            output = self.llm.generate([formatted_prompt], sampling_params, use_tqdm=False)[0]
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
        
        output = self.llm.generate([formatted_prompt], sampling_params, use_tqdm=False)[0]
        next_steps = self._parse_approaches(output.outputs[0].text, branches)
        
        best_score = -1
        best_continuation = []
        best_label = None
        
        
        for step in next_steps[:2]:  
            
            if best_score >= 0.9:
                break
            
            new_reasoning = f"{current_reasoning}\n{step}"
            path, label, score = self._expand_branch(
                claim,
                new_reasoning,
                remaining_depth - 1,
                branches,
                min_confidence_threshold  
            )
            
            
            if score >= min_confidence_threshold and score > best_score:
                best_score = score
                best_continuation = [step] + path
                best_label = label
        
        
        if best_score == -1:
            return [], 'NOT ENOUGH INFO', 0.3
        
        return best_continuation, best_label, best_score

    def _extract_label(self, text) -> str:
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
        match = re.search(r'CONFIDENCE:\s*([\d.]+)', text, re.IGNORECASE)
        if match:
            try:
                conf = float(match.group(1))
                return min(max(conf, 0.0), 1.0) 
            except:
                pass
        return 0.5 
    
    def _parse_approaches(self, text, n) -> List[str]:
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