#!/usr/bin/env python3
"""perplexity_audit.py

Quick perplexity audit using OpenAI's best public models.

Measures token-level perplexity by calling Chat Completions with logprobs=5,
calculating exp(-mean log-prob) over the entire text.

Usage:
    python perplexity_audit.py <text_file> <model_name>
    
    Examples:
        python perplexity_audit.py grants43.txt gpt-4o-2024-05-13
        python perplexity_audit.py wiki_control.txt gpt-4o-2024-05-13

Model: gpt-4o-2024-05-13
"""

import os
import sys
import math
from pathlib import Path
from typing import List, Tuple

import openai
import tiktoken
from dotenv import load_dotenv


def get_encoding(model: str) -> tiktoken.Encoding:
    """Get the appropriate tiktoken encoding for the model."""
    if model.startswith('gpt-4'):
        return tiktoken.encoding_for_model(model)
    else:
        raise ValueError(f"Unsupported model: {model}")


def chunk_text(text: str, encoding: tiktoken.Encoding, max_tokens: int = 8000) -> List[str]:
    """Split text into chunks of max_tokens or fewer."""
    tokens = encoding.encode(text)
    
    if len(tokens) > 25000:  # Cap at 25k tokens for cost control
        tokens = tokens[:25000]
        print(f"Warning: Truncated to 25,000 tokens for cost control")
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens:
        if current_length + 1 > max_tokens:
            # Start new chunk
            chunks.append(encoding.decode(current_chunk))
            current_chunk = [token]
            current_length = 1
        else:
            current_chunk.append(token)
            current_length += 1
    
    # Add final chunk
    if current_chunk:
        chunks.append(encoding.decode(current_chunk))
    
    return chunks


def get_logprobs(text: str, model: str) -> List[float]:
    """Get token-level log probabilities for text using OpenAI API."""
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ""},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": text}
            ],
            logprobs=True,
            top_logprobs=5,
            max_tokens=0  # Don't generate new tokens
        )
        
        # Extract logprobs from the response
        logprobs = []
        for choice in response.choices:
            if hasattr(choice, 'logprobs') and choice.logprobs:
                for token_info in choice.logprobs.content:
                    # Get the logprob for the actual token
                    token_logprob = token_info.logprob
                    logprobs.append(token_logprob)
        
        return logprobs
        
    except Exception as e:
        print(f"API Error: {e}")
        sys.exit(1)


def calculate_perplexity(logprobs: List[float]) -> float:
    """Calculate perplexity as exp(-mean log-prob)."""
    if not logprobs:
        return float('inf')
    
    mean_logprob = sum(logprobs) / len(logprobs)
    perplexity = math.exp(-mean_logprob)
    return perplexity


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python perplexity_audit.py <text_file> <model_name> [api_key]")
        print("Model: gpt-4o-2024-05-13")
        print("API key can be provided as 3rd argument or set OPENAI_API_KEY env var")
        sys.exit(1)
    
    file_path = sys.argv[1]
    model = sys.argv[2]
    api_key_arg = sys.argv[3] if len(sys.argv) == 4 else None
    
    # Validate model first
    if model != 'gpt-4o-2024-05-13':
        print(f"Error: Only gpt-4o-2024-05-13 is supported")
        sys.exit(1)
    
    # Load .env file if it exists
    load_dotenv()
    
    # Check API key - try command line arg first, then environment
    api_key = api_key_arg or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key not found")
        print("Set OPENAI_API_KEY in .env file, environment variable, or pass as 3rd argument")
        sys.exit(1)
    
    # Set the API key
    os.environ['OPENAI_API_KEY'] = api_key
    
    # Check file exists
    if not Path(file_path).exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    # Read and process file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Get encoding and chunk text
    encoding = get_encoding(model)
    chunks = chunk_text(text, encoding)
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Get logprobs for all chunks
    all_logprobs = []
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}/{len(chunks)}...")
        try:
            chunk_logprobs = get_logprobs(chunk, model)
            all_logprobs.extend(chunk_logprobs)
            print(f"  Got {len(chunk_logprobs)} logprobs")
        except Exception as e:
            print(f"  Error processing chunk {i+1}: {e}")
            sys.exit(1)
    
    # Calculate perplexity
    perplexity = calculate_perplexity(all_logprobs)
    
    # Print result
    print(f"{file_path} | {model} | avg_ppl {perplexity:.2f}")


if __name__ == '__main__':
    main() 