"""generate_stratified_worksheet.py
===================================
Create a **stratified random sample of 100 C&P grants** across (historical)
county × decade strata and write a worksheet for manual anchor geolocation.

• Input:  `data/processed/matched_grants.csv` (+ county info from
           `data/processed/spatial_validation.csv`)
• Output: `data/processed/stratified_100_anchor_worksheet.csv`

Key parameters
--------------
SAMPLE_SIZE = 100   # number of grants to draw
SEED        = 42    # RNG seed so the same 100 grants are selected every time

The worksheet adds helper columns (`anchor_score`, `suggested_anchor_phrase`)
and blank fields (`manual_lat_lon`, `notes`) for the reviewer to fill in.
Those two columns are later picked up by `compute_distances.py`.

Run from repo root:
    python tier2_positional_accuracy/scripts/generate_stratified_worksheet.py
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
MATCHED_CSV = Path('data/processed/matched_grants.csv')
OUT_CSV = Path('data/processed/stratified_100_anchor_worksheet.csv')

SAMPLE_SIZE = 100
SEED = 42  # For reproducibility

# Anchor potential keywords/patterns (inspired by anchor_extraction.py)
ANCHOR_PATTERNS = [
    r'\bmouth\b', r'\bfork\b', r'\bconfluence\b', r'\bhead\b',
    r'\bcreek\b', r'\bswamp\b', r'\bbranch\b', r'\brun\b', r'\briver\b'
]

def load_data() -> pd.DataFrame:
    df = pd.read_csv(MATCHED_CSV, dtype={'oss_id': str})
    spatial = pd.read_csv(Path('data/processed/spatial_validation.csv'), dtype={'oss_id': str})
    df = df.merge(spatial[['oss_id', 'cp_county']], on='oss_id', how='left')
    df['year'] = df['year_cp'].fillna(df['year_oss']).astype(int)
    df['decade'] = (df['year'] // 10) * 10
    df['stratum'] = df['cp_county'].fillna('Unknown').astype(str) + '_' + df['decade'].astype(str)
    return df

def stratified_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    group_sizes = df.groupby('stratum').size()
    total = len(df)
    fractional = group_sizes * sample_size / total
    quotas = np.floor(fractional).astype(int)
    remainder = sample_size - quotas.sum()
    if remainder > 0:
        frac_part = fractional - quotas
        top_up = frac_part.sort_values(ascending=False).index[:remainder]
        quotas.loc[top_up] += 1
    sampled = []
    rng = np.random.default_rng(SEED)
    for stratum, n in quotas.items():
        subgroup = df[df['stratum'] == stratum]
        if n > 0:
            sampled.append(subgroup.sample(min(n, len(subgroup)), random_state=rng))
    return pd.concat(sampled, ignore_index=True)

def score_anchor_potential(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    score = 0
    for pat in ANCHOR_PATTERNS:
        score += len(re.findall(pat, text.lower()))
    return score / len(text) * 100  # Normalize by length for fairness

def extract_anchor_phrase(text: str) -> str:
    if not isinstance(text, str):
        return ''
    # Look for anchor patterns and extract the phrase (up to comma/semicolon)
    patterns = [
        r'(mouth of [^,;\n]+)',
        r'(fork of [^,;\n]+)',
        r'(head of [^,;\n]+)',
        r'(confluence of [^,;\n]+)',
        r'(at [^,;\n]+ creek)',
        r'(at [^,;\n]+ branch)',
        r'(at [^,;\n]+ river)',
        r'(on [^,;\n]+ creek)',
        r'(on [^,;\n]+ branch)',
        r'(on [^,;\n]+ river)',
        r'(at [^,;\n]+ swamp)',
        r'(on [^,;\n]+ swamp)'
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            # Return phrase as it appears in original text (preserve case)
            start = m.start(1)
            end = m.end(1)
            return text[start:end].strip()
    return ''

def main():
    df = load_data()
    sample = stratified_sample(df, SAMPLE_SIZE)
    sample['anchor_score'] = sample['cp_raw'].apply(score_anchor_potential)
    sample = sample.sort_values('anchor_score', ascending=False).reset_index(drop=True)
    sample['initial_20'] = 0
    sample.loc[:19, 'initial_20'] = 1
    sample['suggested_anchor_phrase'] = sample['cp_raw'].apply(extract_anchor_phrase)
    sample['manual_lat_lon'] = ''
    sample['notes'] = ''
    columns = ['cp_id', 'cp_county', 'decade', 'cp_raw', 'suggested_anchor_phrase', 'manual_lat_lon', 'notes', 'initial_20', 'anchor_score']
    sample[columns].to_csv(OUT_CSV, index=False)
    print(f'Worksheet created at {OUT_CSV} with 100 stratified grants. Top 20 marked with initial_20=1.')

if __name__ == '__main__':
    main() 