#!/usr/bin/env python3
"""
Get exact count of reviews after applying ALL filters.
"""

from datasets import load_from_disk
import pandas as pd

print('='*80)
print('EXACT FILTERED COUNT WITH ALL FILTERS')
print('='*80)
print()

# Load data
print('Loading data...')
metadata = load_from_disk('/export/fs06/mzhong8/doctor_review_metadata')
corpus = load_from_disk('/export/fs06/mzhong8/doctor_review_corpus')

metadata_df = metadata.to_pandas()
corpus_df = corpus.to_pandas()

print(f'Original corpus: {len(corpus_df):,} reviews')
print()

# ============================================================================
# STEP 1: Doctor-Level Filters
# ============================================================================
print('STEP 1: Doctor-Level Filters')
print('-' * 80)

# Filter 1: Specialty = Family Medicine
print('Filter 1: Specialty = Family Medicine Physician')
filtered_metadata = metadata_df[metadata_df['Specialty'] == 'Family Medicine Physician']
print(f'  → {len(filtered_metadata):,} doctors')

# Filter 2: Review count >= 20
print('Filter 2: Review count >= 20')
filtered_metadata = filtered_metadata[filtered_metadata['num_reviews'] >= 20]
print(f'  → {len(filtered_metadata):,} doctors')

# Filter 3: Review count <= 100 (NEW)
print('Filter 3: Review count <= 100')
filtered_metadata = filtered_metadata[filtered_metadata['num_reviews'] <= 100]
print(f'  → {len(filtered_metadata):,} doctors')

total_expected_reviews = filtered_metadata['num_reviews'].sum()
print(f'\nExpected reviews from metadata: {total_expected_reviews:,}')
print()

# Get filtered PhyIDs
filtered_phyids = set(filtered_metadata['PhyID'].tolist())

# Filter corpus by PhyID
print('Filtering corpus by doctor PhyIDs...')
filtered_corpus = corpus_df[corpus_df['PhyID'].isin(filtered_phyids)].copy()
print(f'  → {len(filtered_corpus):,} reviews')
print()

# ============================================================================
# STEP 2: Review-Level Quality Filters
# ============================================================================
print('STEP 2: Review-Level Quality Filters')
print('-' * 80)

# Calculate text metrics
print('Calculating text metrics...')
filtered_corpus['text_length'] = filtered_corpus['text'].astype(str).str.len()
filtered_corpus['word_count'] = filtered_corpus['text'].astype(str).str.split().str.len()

# Filter 4: Word count >= 15
print('Filter 4: Word count >= 15')
before = len(filtered_corpus)
filtered_corpus = filtered_corpus[filtered_corpus['word_count'] >= 15]
after = len(filtered_corpus)
print(f'  → Removed {before - after:,} reviews')
print(f'  → Remaining: {after:,} reviews')

# Filter 5: Word count <= 200
print('Filter 5: Word count <= 200')
before = len(filtered_corpus)
filtered_corpus = filtered_corpus[filtered_corpus['word_count'] <= 200]
after = len(filtered_corpus)
print(f'  → Removed {before - after:,} reviews')
print(f'  → Remaining: {after:,} reviews')

# Filter 6: NOT all caps
print('Filter 6: NOT all caps')
before = len(filtered_corpus)
filtered_corpus = filtered_corpus[~filtered_corpus['text'].astype(str).str.isupper()]
after = len(filtered_corpus)
print(f'  → Removed {before - after:,} reviews')
print(f'  → Remaining: {after:,} reviews')

# Filter 7: Digit ratio < 30%
print('Filter 7: Digit ratio < 30%')
def high_digit_ratio(text):
    text_str = str(text)
    if len(text_str) == 0:
        return False
    digit_ratio = sum(c.isdigit() for c in text_str) / len(text_str)
    return digit_ratio >= 0.3

before = len(filtered_corpus)
filtered_corpus = filtered_corpus[~filtered_corpus['text'].apply(high_digit_ratio)]
after = len(filtered_corpus)
print(f'  → Removed {before - after:,} reviews')
print(f'  → Remaining: {after:,} reviews')

# Filter 8: Character diversity >= 10
print('Filter 8: Character diversity >= 10 unique chars')
def low_char_diversity(text):
    text_str = str(text).lower()
    if len(text_str) < 20:
        return False
    unique_chars = len(set(text_str))
    return unique_chars < 10

before = len(filtered_corpus)
filtered_corpus = filtered_corpus[~filtered_corpus['text'].apply(low_char_diversity)]
after = len(filtered_corpus)
print(f'  → Removed {before - after:,} reviews')
print(f'  → Remaining: {after:,} reviews')

# Filter 9: Non-ASCII ratio < 50% (removes Spanish)
print('Filter 9: Non-ASCII ratio < 50% (English-focused)')
def high_non_ascii(text):
    text_str = str(text)
    if len(text_str) == 0:
        return False
    non_ascii = sum(1 for c in text_str if ord(c) > 127)
    return non_ascii / len(text_str) >= 0.5

before = len(filtered_corpus)
filtered_corpus = filtered_corpus[~filtered_corpus['text'].apply(high_non_ascii)]
after = len(filtered_corpus)
print(f'  → Removed {before - after:,} reviews')
print(f'  → Remaining: {after:,} reviews')

print()

# ============================================================================
# FINAL RESULT
# ============================================================================
print('='*80)
print('FINAL RESULT')
print('='*80)
print()

final_count = len(filtered_corpus)
final_doctors = filtered_corpus['PhyID'].nunique()

print(f'✅ EXACT FINAL COUNT: {final_count:,} reviews')
print(f'✅ From {final_doctors:,} doctors')
print(f'✅ Average {final_count/final_doctors:.1f} reviews per doctor')
print()

# Comparison to targets
print('Comparison:')
print(f'  - Original corpus: 257,541 reviews')
print(f'  - Final corpus: {final_count:,} reviews')
print(f'  - Reduction: {(1 - final_count/257541)*100:.1f}%')
print(f'  - vs 200K target: {(final_count/200000)*100:.0f}% ({final_count - 200000:+,})')
print()

# Save exact count to file
with open('/home/srangre1/Iterative-Query-Refinement/exact_filtered_count.txt', 'w') as f:
    f.write(f'EXACT FILTERED COUNT: {final_count:,} reviews\n')
    f.write(f'Doctors: {final_doctors:,}\n')
    f.write(f'Date: 2026-01-13\n')

print('Count saved to: /home/srangre1/Iterative-Query-Refinement/exact_filtered_count.txt')
print()
print('='*80)
