#!/usr/bin/env python3
"""
Essay Scoring Assignment

This script scores essays using vLLM. Your task is to:
1. Write a prompt that instructs the model to score essays
2. Extract the scores from the model's responses
3. Evaluate your results

INSTRUCTIONS:
============
1. Fill in the TODO sections marked with "YOUR CODE HERE"
2. Write an effective prompt in the create_prompt() function
3. Implement parsing logic to extract scores from model responses
4. Run the script and analyze your results

The input CSV (train_dev.csv) has the following columns:
- essay_id: Unique identifier for each essay
- assignment: The essay prompt/assignment description
- holistic_essay_score: The ground truth score (for evaluation)
- full_text: The essay text to be scored
"""

import argparse
import csv
import json
import re
import torch
from typing import Dict, List, Optional
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ============================================================================
# TODO SECTION 1: PROMPT ENGINEERING
# ============================================================================
# Write a prompt that instructs the LLM to score essays.
#
# Your prompt should:
# 1. Explain the task (scoring essays on a scale, typically 1-6 or similar)
# 2. Include the assignment/prompt that students were given
# 3. Include the essay text to be scored
# 4. Ask the model to provide a score (and optionally justification)
# 5. Specify the output format you want (e.g., JSON, plain text with score)
# ============================================================================

def create_prompt(assignment: str, essay_text: str, prompt_id: int = 0) -> str:
    """
    Create a prompt for scoring an essay.

    Args:
        assignment: The essay prompt/assignment description
        essay_text: The full text of the essay to be scored
        prompt_id: Which prompt variant to use (default: 0)

    Returns:
        A formatted prompt string

    TODO: Write your prompt here. Make it clear, specific, and effective.
    You can use prompt_id to implement different prompt variants.
    """

    match prompt_id:
        case 0:
            prompt = f"""
            You are an expert essay grader. Score this essay on a holistic scale from 1 to 6.

            Assignment:
            {assignment}

            Scoring Rubric:
            - Score 6: Exceptional essay with clear thesis, well developed arguments, strong evidence, advanced vocabulary, and little to no errors 
            - Score 5: Strong essay with clear organization, good supporting details, appropriate vocabulary, and few errors
            - Score 4: Competent essay with adequate organization, some supporting details, acceptable vocabulary, and some errors 
            - Score 3: Basic essay with simple organization, limited supporting details, basic vocabulary, and noticeable errors 
            - Score 2: Weak essay with poor organization, minimal details, limited vocabulary, and many errors 
            - Score 1: Very weak essay with no clear organization, insufficient content, poor vocabulary, and extensive amounts of errors

            Essay:
            {essay_text}

            Provide your score using just a plain number from 1 to 6.

            Score: 
            """
        case 1:
            prompt = f"""
            Score this student essay on a 1-6 scale where:
            - 1-2: Below basic - significant problems with content, organization, or language
            - 3-4: Basic to proficient - adequate response with some development
            - 5-6: Advanced - sophisticated ideas, strong organization, effective language

            Assignment:
            {assignment}

            Essay to score:
            {essay_text}

            Think about it step by step. 

            Provide only the numerical score (1, 2, 3, 4, 5, or 6) followed by a one sentence justification to why you chose that score.

            Score: 

            Justification: 
            """
        case 2:
            prompt = f"""
                You are an experienced English teacher grading student essays. Evaluate the following essay and assign a holistic score from 1 to 6.

                Assignment Given to Students:
                {assignment}

                Student's Essay:
                {essay_text}

                Evaluate based on these criteria:
                1. Content and Ideas: Does the essay address the prompt with relevant ideas?
                2. Organization: Is the essay logically structured with clear transitions?
                3. Language Use: Does the student demonstrate appropriate vocabulary and sentence variety?
                4. Conventions: Is grammar, spelling, and punctuation correct?

                Score the essay from 1 (lowest) to 6 (highest) based on overall quality.

                Provide your score and reason in the following JSON format:
                {{"score": <number from 1-6>, "reason": "<brief explanation>"}}
                """
        case _:
            prompt = f"""err""" 

    return prompt.strip()


# ============================================================================
# TODO SECTION 2: RESPONSE PARSING
# ============================================================================
# Extract the score from the model's response.
# The model might return:
# - Just a number: "4"
# - JSON: {"score": 4, "justification": "..."}
# - Text with score: "The score is 4 because..."
#
# You need to parse the response and extract the numeric score.
# ============================================================================

def parse_score(response_text: str) -> Optional[float]:
    """
    Extract the essay score from the model's response.

    Args:
        response_text: The raw text response from the LLM

    Returns:
        The score as a float (or None if parsing fails)

    TODO: Implement parsing logic to extract the score from various response formats.
    """

    try:
        json_match = re.search(r'\{[^}]*["\']score["\'][^}]*\}', response_text, re.DOTALL| re.IGNORECASE)
        if json_match:
            data = json.loads(json_match.group())
            if 'score' in data:
                score = float(data['score'])
                if 1 <= score <= 6:
                    return score
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
        
    try:
        score_match = re.search(r'score\s*(?:\:|is|of)?\s*(\d+)', response_text, re.MULTILINE | re.IGNORECASE)
        if score_match:
                score = float(score_match.group(1))
                if 1 <= score <= 6:
                    return score
    except ValueError:
        pass
    
    try:
        numbers = re.findall(r'\b([1-6])\b', response_text)
        for num in numbers:
            value = float(num)
            if 1 <= value <= 6:
                return value
    except Exception:
        pass

    return None


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_chunk(chunk_data: List[Dict[str, str]], model: LLM, tokenizer: AutoTokenizer,
                  sampling_params: SamplingParams, use_chat_template: bool = True,
                  system_message: Optional[str] = None, prompt_id: int = 0) -> List[Dict]:
    """
    Process a chunk of essays through vLLM in batch.

    Args:
        chunk_data: List of essay dictionaries
        model: vLLM LLM instance
        tokenizer: HuggingFace tokenizer
        sampling_params: vLLM SamplingParams object
        use_chat_template: Whether to use chat template formatting (for chat models)
        system_message: Optional system message for chat templates
        prompt_id: Which prompt variant to use (default: 0)

    Returns:
        List of result dictionaries with original data + model outputs
    """
    input_list = []
    metadata_list = []  # Store original essay data for each item

    for essay in chunk_data:
        essay_id = essay['essay_id']
        assignment = essay['assignment']
        ground_truth_score = essay.get('holistic_essay_score', '')
        essay_text = essay['full_text']

        # Create prompt using the create_prompt function
        try:
            prompt = create_prompt(assignment, essay_text, prompt_id=prompt_id)
        except Exception as e:
            print(f"Warning: Error creating prompt for essay {essay_id}: {e}")
            input_list.append(None)
            metadata_list.append({
                'essay_id': essay_id,
                'assignment': assignment,
                'ground_truth_score': ground_truth_score,
                'error': f'Prompt creation error: {e}'
            })
            continue

        # Skip empty prompts
        if not prompt or not prompt.strip():
            input_list.append(None)
            metadata_list.append({
                'essay_id': essay_id,
                'assignment': assignment,
                'ground_truth_score': ground_truth_score,
                'error': 'Empty prompt'
            })
            continue

        # Format as chat messages if using chat template
        if use_chat_template:
            system_msg = system_message or "You are a helpful assistant."
            input_list.append(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
            )
        else:
            # For non-chat models, use the prompt directly
            input_list.append(prompt)

        metadata_list.append({
            'essay_id': essay_id,
            'assignment': assignment,
            'ground_truth_score': ground_truth_score,
            'error': None
        })

    # Filter out None entries and track indices
    valid_indices = []
    valid_inputs = []
    for i, inp in enumerate(input_list):
        if inp is not None:
            valid_indices.append(i)
            valid_inputs.append(inp)

    if not valid_inputs:
        return [m for m in metadata_list if m.get('error')]

    # Apply chat template if needed
    if use_chat_template:
        formatted_inputs = [
            tokenizer.apply_chat_template(
                user_input,
                tokenize=False,
                add_special_tokens=False,
                add_generation_prompt=True,
                enable_thinking=False
            ) for user_input in valid_inputs
        ]
    else:
        formatted_inputs = valid_inputs

    # Generate outputs using vLLM (batch processing)
    outputs = model.generate(formatted_inputs, sampling_params)
    output_texts = [output.outputs[0].text.strip() for output in outputs]

    # Process results: combine original data with model outputs
    results = []
    output_idx = 0
    for i, metadata in enumerate(metadata_list):
        if i in valid_indices:
            # Get the model's response
            response_text = output_texts[output_idx]
            output_idx += 1

            # Parse the score from the response
            parsed_score = parse_score(response_text)

            result = {
                **metadata,
                'model_response': response_text,
                'predicted_score': parsed_score
            }
            results.append(result)
        else:
            # Include essays that had errors
            results.append(metadata)

    return results


def read_essays_csv(filepath: str) -> List[Dict[str, str]]:
    """
    Read essays from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        List of dictionaries, one per essay
    """
    essays = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            essays.append(row)
    return essays


def write_results_jsonl(results: List[Dict], output_file: str):
    """
    Write results to JSONL file (one JSON object per line).

    Args:
        results: List of result dictionaries
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def evaluate_predictions(results: List[Dict]) -> Dict:
    """
    Evaluate predictions against ground truth scores.

    Args:
        results: List of result dictionaries with 'predicted_score' and 'ground_truth_score'

    Returns:
        Dictionary with evaluation metrics
    """
    predictions = []
    ground_truths = []

    for result in results:
        if result.get('predicted_score') is not None and result.get('ground_truth_score'):
            try:
                pred = float(result['predicted_score'])
                truth = float(result['ground_truth_score'])
                predictions.append(pred)
                ground_truths.append(truth)
            except (ValueError, TypeError):
                continue

    if not predictions:
        return {
            'num_valid_predictions': 0,
            'error': 'No valid predictions to evaluate'
        }

    # Calculate metrics
    import numpy as np
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    mae = np.mean(np.abs(predictions - ground_truths))
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    correlation = np.corrcoef(predictions, ground_truths)[0, 1]

    # Exact match accuracy
    exact_matches = np.sum(predictions == ground_truths)
    accuracy = exact_matches / len(predictions)

    # Within 1 point accuracy
    within_one = np.sum(np.abs(predictions - ground_truths) <= 1)
    within_one_accuracy = within_one / len(predictions)

    return {
        'num_valid_predictions': len(predictions),
        'num_total_essays': len(results),
        'mean_absolute_error': float(mae),
        'root_mean_squared_error': float(rmse),
        'correlation': float(correlation),
        'exact_match_accuracy': float(accuracy),
        'within_one_accuracy': float(within_one_accuracy)
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Score essays using vLLM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output files
    parser.add_argument(
        '--input-file',
        type=str,
        default='train_dev.csv',
        help='Path to input CSV file with essays'
    )

    parser.add_argument(
        '--output-file',
        type=str,
        default='essay_scores.jsonl',
        help='Path to output JSONL file with results'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of essays to process (for testing)'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate predictions against ground truth scores'
    )

    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model name or path (e.g., "meta-llama/Llama-2-7b-chat-hf", "Qwen/Qwen2.5-7B-Instruct")'
    )

    parser.add_argument(
        '--cache-dir',
        type=str,
        default=None,
        help='Directory to cache models (default: HuggingFace cache)'
    )

    parser.add_argument(
        '--tensor-parallel-size',
        type=int,
        default=None,
        help='Number of GPUs for tensor parallelism (default: auto-detect)'
    )

    # Processing settings
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Number of essays to process in each batch (default: 100)'
    )

    parser.add_argument(
        '--use-chat-template',
        action='store_true',
        default=True,
        help='Use chat template formatting (for chat models)'
    )

    parser.add_argument(
        '--no-chat-template',
        dest='use_chat_template',
        action='store_false',
        help='Disable chat template (for base/completion models)'
    )

    parser.add_argument(
        '--system-message',
        type=str,
        default=None,
        help='System message for chat templates (default: "You are a helpful assistant.")'
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature (0.0 = deterministic, higher = more random)'
    )

    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Top-p (nucleus) sampling parameter'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=200,
        help='Maximum number of tokens to generate'
    )

    parser.add_argument(
        '--prompt-id',
        type=int,
        default=0,
        help='Which prompt variant to use (default: 0)'
    )

    parser.add_argument(
        '--metrics-file',
        type=str,
        default=None,
        help='File to save each runs metrics to'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Essay Scoring Assignment (vLLM)")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Model: {args.model_name}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Prompt ID: {args.prompt_id}")
    if args.limit:
        print(f"Processing limit: {args.limit} essays")
    print("=" * 60)

    # ===================================================================
    # STEP 1: Load Tokenizer
    # ===================================================================
    print("\n[1/4] Loading tokenizer...")
    tokenizer_kwargs = {'trust_remote_code': True}
    if args.cache_dir:
        tokenizer_kwargs['cache_dir'] = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        **tokenizer_kwargs
    )
    print("✓ Tokenizer loaded")

    # ===================================================================
    # STEP 2: Configure vLLM Model
    # ===================================================================
    print("\n[2/4] Configuring vLLM model...")

    # Auto-detect number of GPUs if not specified
    available_gpus = torch.cuda.device_count()
    tensor_parallel_size = args.tensor_parallel_size or available_gpus

    if tensor_parallel_size > available_gpus:
        print(f"Warning: Requested {tensor_parallel_size} GPUs but only {available_gpus} available")
        tensor_parallel_size = available_gpus

    print(f"Available GPUs: {available_gpus}")
    print(f"Using tensor_parallel_size: {tensor_parallel_size}")

    # Configure vLLM LLM instance
    model_kwargs = {
        'model': args.model_name,
        'tensor_parallel_size': tensor_parallel_size,
    }
    if args.cache_dir:
        model_kwargs['download_dir'] = args.cache_dir

    print("Loading model (this may take a few minutes for large models)...")
    model = LLM(**model_kwargs)
    print("✓ Model loaded")

    # ===================================================================
    # STEP 3: Configure Sampling Parameters
    # ===================================================================
    print("\n[3/4] Setting up sampling parameters...")
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Max tokens: {args.max_tokens}")
    print("✓ Sampling parameters configured")

    # ===================================================================
    # STEP 4: Read and Process Essays
    # ===================================================================
    print(f"\n[4/4] Reading essays from {args.input_file}...")
    essays = read_essays_csv(args.input_file)

    if args.limit:
        essays = essays[:args.limit]

    print(f"Loaded {len(essays)} essays")
    print(f"Processing in chunks of {args.chunk_size}...")

    # Open output file for writing
    output_fp = open(args.output_file, 'w', encoding='utf-8')

    # Process essays in chunks
    results = []
    chunk_num = 0
    total_results = 0

    with tqdm(total=len(essays), desc="Processing essays", unit="essays") as pbar:
        for i in range(0, len(essays), args.chunk_size):
            chunk_data = essays[i:i + args.chunk_size]

            # Process chunk through vLLM
            chunk_results = process_chunk(
                chunk_data,
                model,
                tokenizer,
                sampling_params,
                use_chat_template=args.use_chat_template,
                system_message=args.system_message,
                prompt_id=args.prompt_id
            )

            # Write results to output file immediately
            for result in chunk_results:
                output_fp.write(json.dumps(result, ensure_ascii=False) + '\n')
                total_results += 1

            output_fp.flush()  # Ensure data is written to disk

            results.extend(chunk_results)
            chunk_num += 1
            pbar.update(len(chunk_data))
            pbar.set_postfix({
                'chunks': chunk_num,
                'results': total_results
            })

    output_fp.close()

    # Evaluate if requested
    if args.evaluate:
        print("\nEvaluating predictions...")
        metrics = evaluate_predictions(results)

        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("=" * 60)

        if args.metrics_file:
            with open(args.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Metrics saved to: {args.metrics_file}")

    # Summary
    num_successful = sum(1 for r in results if r.get('predicted_score') is not None)
    num_errors = sum(1 for r in results if r.get('error') is not None)

    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Total essays: {len(results)}")
    print(f"Successful predictions: {num_successful}")
    print(f"Errors: {num_errors}")
    print(f"Output file: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

