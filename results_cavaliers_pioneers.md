# Cavaliers & Pioneers Overlap Audit Results

**File:** `data/raw/CavaliersandPioneersvol3/cavpion.txt`  
**Date:** $(date)  
**Test:** MinHash overlap detection against C4 corpus sample

## Test Configuration
- **Shingle size:** 13 words (overlapping)
- **MinHash permutations:** 128
- **C4 sample size:** 10,000 documents
- **Processing time:** ~57 minutes (CPU-only)

## Results

### Input Text Statistics
- **Total shingles processed:** 273,599
- **Unique MinHash signatures:** 265,657
- **Uniqueness rate:** 97.1%

### C4 Overlap Analysis
- **Collisions detected:** 0
- **Collisions per million shingles:** 0.000
- **is_clean:** ✅ **True**

## Conclusion

The Cavaliers & Pioneers text shows **zero overlap** with the sampled C4 corpus. This indicates:

1. **No verbatim or near-verbatim matches** with the 10k-document C4 sample
2. **Text is considered clean** against the threshold of 0.1 ppm
3. **Low likelihood** of contamination in major language model training corpora

## Technical Notes

- Test used real-time streaming of C4 documents from Hugging Face
- MinHash signatures were generated on-the-fly for both input text and C4 sample
- Processing speed: ~2.92 C4 documents/second on Intel i7 Mac
- All collisions are exact MinHash matches (128-permutation signatures)

## Recommendations

- **For production use:** Consider testing against larger C4 samples or full corpus
- **For comprehensive audit:** Test against additional corpora (The Pile, Books3, etc.)
- **Current result:** Text appears safe for use in training data 