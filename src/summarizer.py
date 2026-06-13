import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import time
from typing import Tuple, Optional
from src.config import BASELINE_MODEL, OPTIMIZED_MODEL, FALLBACK_MODEL

class Summarizer:
    def __init__(self, model_name: str, use_quantization: bool = False):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads the model and tokenizer with optional quantization."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            if self.use_quantization and self.device == -1:
                # Dynamic quantization is typically for CPU
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            elif self.device == 0:
                self.model = self.model.to("cuda")
                
        except Exception as e:
            st.error(f"Error loading model {self.model_name}: {e}")
            if self.model_name != FALLBACK_MODEL:
                st.warning(f"Falling back to {FALLBACK_MODEL}...")
                self.model_name = FALLBACK_MODEL
                self.load_model()
            else:
                raise e

    def summarize(self, text: str, min_length: int = 30, max_length: int = 150) -> str:
        """Generates a summary for the given text."""
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        
        if self.device == 0:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        summary_ids = self.model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            min_length=min_length, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

@st.cache_resource
def get_summarizer(model_name: str, use_quantization: bool = False):
    return Summarizer(model_name, use_quantization)
