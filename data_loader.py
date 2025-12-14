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

def load_fever_dataset(filepath, sample_size=None, verifiable_only=False) -> pd.DataFrame:
    data = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                
                
                if verifiable_only and item.get('verifiable') != 'VERIFIABLE':
                    continue
                    
                data.append(item)
            except:
                continue
    
    df = pd.DataFrame(data)
    
    if 'label' in df.columns:
        df['label'] = df['label'].str.upper()
    
    if sample_size:
        if verifiable_only:
            
            available = len(df)
            sample_size = min(sample_size, available)
            print(f"Sampling {sample_size} from {available} verifiable claims")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    return df