#!/usr/bin/env python3
"""positive_control.py

Positive control test for overlap_audit.py pipeline.

Downloads a known web-visible text (Project Gutenberg's Frankenstein)
and runs the same overlap detection against C4 to verify the pipeline
correctly flags collisions.

Usage:
    python positive_control.py [input_file]

    Default: gutenberg_test.txt (Frankenstein from Project Gutenberg)
"""

import sys
from pathlib import Path

# Import functions from overlap_audit.py
from overlap_audit import (
    process_text_file,
    compare_with_c4_sample,
    calculate_ppm
)


def main():
    # Default to Frankenstein if no file specified
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'gutenberg_test.txt'
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        print("Available test files:")
        for test_file in ['gutenberg_test.txt', 'wiki_test.txt']:
            if Path(test_file).exists():
                size = Path(test_file).stat().st_size
                print(f"  {test_file} ({size:,} bytes)")
        sys.exit(1)
    
    print(f"Positive Control Test: {input_file}")
    print("=" * 50)
    
    # Process the test file using existing functions
    total_shingles, input_signatures = process_text_file(input_file)
    
    print(f"\nTest file statistics:")
    print(f"Total shingles processed: {total_shingles:,}")
    print(f"Unique MinHash signatures: {len(input_signatures):,}")
    
    # Run C4 comparison (10k docs)
    collisions, ppm, is_clean = compare_with_c4_sample(
        input_signatures, total_shingles, sample_docs=10_000
    )
    
    print("\n" + "=" * 50)
    print("POSITIVE CONTROL RESULTS")
    print(f"Collisions: {collisions:,}")
    print(f"Collisions per million shingles: {ppm:.3f}")
    print(f"is_clean: {is_clean}")
    print("=" * 50)
    
    # Determine if positive control passed
    if ppm > 0.1:
        print("✅ POSITIVE CONTROL PASSED")
        print("Pipeline correctly detects overlap as expected.")
        print(f"Found {collisions:,} collisions ({ppm:.3f} ppm) - above threshold.")
    else:
        print("❌ POSITIVE CONTROL FAILED")
        print("Pipeline did not detect expected overlap.")
        print(f"Only {collisions:,} collisions ({ppm:.3f} ppm) - below threshold.")
        print("\nPossible reasons:")
        print("- Test file too small/short")
        print("- C4 sample doesn't contain similar content")
        print("- Need larger C4 sample or different test text")


if __name__ == '__main__':
    main() 