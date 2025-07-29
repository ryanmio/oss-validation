"""overlap_audit.py

A lightweight CPU-only audit that checks a local text corpus
against several public language-model training corpora for near-
verbatim overlap.

Workflow
1.  Normalise the input text (lower-case, ASCII punctuation).
2.  Generate all overlapping 13-token shingles.
3.  Build a 128-permutation MinHash signature for each shingle.
4.  Count exact MinHash matches against published MinHash indices:
      •  mosaicml/c4-dedup-minhash
      •  mosaicml/the-pile-dedup-minhash
      •  mosaicml/books3-minhash
      •  mosaicml/commoncrawl-2023-minhash (if accessible)
   The indices are streamed from Hugging Face (datasets API) and
   cached on disk.  Each unique signature is stored in a Python set
   so membership checks run in O(1) time.
5.  Emit a concise report showing collisions per million shingles
   plus a boolean is_clean flag (collisions < 0.1 ppm).

Usage
  python overlap_audit.py [input_file]

  Default input: data/raw/CavaliersandPioneersvol3/cavpion.txt

Requirements
  pip install datasketch datasets tqdm

Note: The MosaicML MinHash datasets are currently unavailable on
Hugging Face. This script generates MinHash signatures for the input
text and provides a framework for future dataset comparisons.
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Iterator, Set, Tuple

from datasketch import MinHash
from datasets import load_dataset  # streaming access to C4
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Normalize text to lowercase with ASCII punctuation."""
    # Convert to lowercase
    text = text.lower()
    # Replace non-ASCII characters with ASCII equivalents
    text = text.replace('—', '-').replace('–', '-').replace('…', '...')
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    # Remove other non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Simple word-based tokenization."""
    return text.split()


def generate_shingles(tokens: list[str], window_size: int = 13) -> Iterator[str]:
    """Generate overlapping shingles of specified window size."""
    if len(tokens) < window_size:
        return
    
    for i in range(len(tokens) - window_size + 1):
        shingle = ' '.join(tokens[i:i + window_size])
        yield shingle


def create_minhash(shingle: str, num_perm: int = 128) -> MinHash:
    """Create a MinHash signature for a shingle."""
    mh = MinHash(num_perm=num_perm)
    # Use word-level hashing for better performance
    words = shingle.split()
    for word in words:
        mh.update(word.encode('utf-8'))
    return mh


def process_text_file(filepath: str) -> Tuple[int, Set[bytes]]:
    """Process text file and return shingle count and MinHash signatures."""
    print(f"Scanning {filepath} …")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Normalize text
    normalized_text = normalize_text(text)
    tokens = tokenize(normalized_text)
    
    # Generate shingles and MinHash signatures
    shingles = list(generate_shingles(tokens))
    signatures = set()
    
    print(f"Processing {len(shingles)} shingles...")
    for shingle in tqdm(shingles, desc="Generating MinHash signatures"):
        mh = create_minhash(shingle)
        # Convert MinHash to hashable bytes for storage
        signature_bytes = bytes(mh.digest())
        signatures.add(signature_bytes)
    
    return len(shingles), signatures


def load_reference_dataset(dataset_name: str) -> Set[bytes]:
    """Load reference dataset MinHash signatures.
    
    Note: This is a placeholder function since the MosaicML datasets
    are currently unavailable on Hugging Face.
    """
    print(f"Note: {dataset_name} is not available on Hugging Face")
    print("This is a placeholder for future dataset integration")
    return set()


def count_collisions(input_signatures: Set[bytes], reference_signatures: Set[bytes]) -> int:
    """Count exact MinHash matches between input and reference."""
    return len(input_signatures.intersection(reference_signatures))


def calculate_ppm(collisions: int, total_shingles: int) -> float:
    """Calculate collisions per million shingles."""
    if total_shingles == 0:
        return 0.0
    return (collisions / total_shingles) * 1_000_000


def compare_with_c4_sample(input_signatures: Set[bytes], total_input_shingles: int, sample_docs: int = 10_000) -> Tuple[int, float, bool]:
    """Stream a sample of *allenai/c4* (English) and count MinHash collisions.

    Returns (collisions_unique, ppm, is_clean)
    """
    print(f"\nStreaming {sample_docs:,} documents from allenai/c4 (English)…")
    ds_iter = load_dataset('allenai/c4', 'en', split='train', streaming=True)

    collisions_seen: Set[bytes] = set()
    processed_docs = 0

    # Iterate over the dataset
    for sample in tqdm(ds_iter, desc="Processing C4 docs"):
        text = sample.get('text', '')
        if not text:
            continue
        tokens = tokenize(normalize_text(text))
        for shingle in generate_shingles(tokens):
            sig = create_minhash(shingle)
            sig_bytes = bytes(sig.digest())
            # Check collision with input signatures
            if sig_bytes in input_signatures and sig_bytes not in collisions_seen:
                collisions_seen.add(sig_bytes)
        processed_docs += 1
        if processed_docs >= sample_docs:
            break

    collisions = len(collisions_seen)
    ppm = calculate_ppm(collisions, total_input_shingles)
    is_clean = ppm < 0.1
    return collisions, ppm, is_clean


def main():
    parser = argparse.ArgumentParser(description="Audit text corpus for overlap with training datasets")
    parser.add_argument('input_file', nargs='?', 
                       default='data/raw/CavaliersandPioneersvol3/cavpion.txt',
                       help='Input text file to audit')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Process input text
    total_shingles, input_signatures = process_text_file(args.input_file)
    
    print(f"\nResults for {args.input_file}:")
    print(f"Total shingles processed: {total_shingles:,}")
    print(f"Unique MinHash signatures: {len(input_signatures):,}")
    
    # Perform real-time C4 comparison (10k-doc sample)
    collisions, ppm, is_clean = compare_with_c4_sample(input_signatures, total_shingles, sample_docs=10_000)

    print("\n" + "=" * 50)
    print("C4 Overlap Report (10k-document sample)")
    print(f"Collisions: {collisions:,}")
    print(f"Collisions per million shingles: {ppm:.3f}")
    print(f"is_clean: {is_clean}")
    print("=" * 50)
    if is_clean:
        print("Result: The text is considered clean against the sampled C4 corpus (ppm < 0.1).")
    else:
        print("Result: Overlap exceeds threshold – manual review recommended.")


if __name__ == '__main__':
    main() 