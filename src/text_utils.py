import re
from typing import List

def clean_text(text: str) -> str:
    """Cleans the input text."""
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def count_words(text: str) -> int:
    """Counts words in a text."""
    return len(text.split())

def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Chunks long text into smaller pieces for summarization.
    Very simple sentence-based chunking.
    """
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def calculate_compression_ratio(input_text: str, summary_text: str) -> float:
    """Calculates the compression ratio."""
    input_words = count_words(input_text)
    summary_words = count_words(summary_text)
    if input_words == 0:
        return 0.0
    return summary_words / input_words
