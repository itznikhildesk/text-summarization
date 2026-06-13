import os
import psutil
import torch
from rouge_score import rouge_scorer
from typing import Dict, Any

def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    """Calculates ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def get_memory_usage() -> float:
    """Returns the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def calculate_model_size(model: torch.nn.Module, is_quantized: bool = False) -> float:
    """
    Estimates model size in MB.
    For FP32 use 4 bytes per param.
    For INT8 quantized model use approximately 1 byte per param.
    """
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 1 if is_quantized else 4
    model_size_mb = (total_params * bytes_per_param) / (1024 * 1024)
    return model_size_mb

def calculate_efficiency_score(time_val: float, memory_val: float, energy_val: float) -> float:
    """
    Calculates a custom Green Efficiency Score from 0 to 100.
    Lower values of time, memory, and energy lead to a higher score.
    Using a simple reciprocal formula with normalization.
    """
    # Normalized weights (can be adjusted)
    w_time = 0.4
    w_mem = 0.3
    w_energy = 0.3
    
    # Avoid division by zero
    time_val = max(time_val, 0.001)
    memory_val = max(memory_val, 0.001)
    energy_val = max(energy_val, 0.001)
    
    # We use a base-line to normalize (roughly)
    # these are just heuristic normalizers for the prototype
    norm_time = time_val / 10.0 
    norm_mem = memory_val / 500.0
    norm_energy = energy_val / 100.0
    
    weighted_cost = (w_time * norm_time) + (w_mem * norm_mem) + (w_energy * norm_energy)
    
    efficiency_score = 100 / (1 + weighted_cost)
    return round(efficiency_score, 2)
