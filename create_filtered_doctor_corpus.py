#!/usr/bin/env python3
"""
Create filtered Family Medicine doctor reviews corpus.

Applies all 9 filters and saves to disk in a format compatible with
the end-to-end evaluation pipeline.

Filters applied:
  Doctor-level:
    1. Specialty = "Family Medicine Physician"
    2. Review count >= 20
    3. Review count <= 100

  Review-level:
    4. Word count >= 15
    5. Word count <= 200
    6. NOT all caps
    7. Digit ratio < 30%
    8. Character diversity >= 10 unique characters
    9. Non-ASCII ratio < 50% (removes Spanish)

Output format matches TREC-COVID:
  - _id: unique document ID (from text_id)
  - title: empty string (for compatibility)
  - text: review text
"""

from datasets import load_from_disk, Dataset
import pandas as pd
import os

print('='*80)
print('CREATING FILTERED DOCTOR REVIEWS CORPUS')
print('='*80)
print()

# ============================================================================
# Configuration
# ============================================================================
METADATA_PATH = '/export/fs06/mzhong8/doctor_review_metadata'
CORPUS_PATH = '/export/fs06/mzhong8/doctor_review_corpus'
OUTPUT_PATH = '/home/srangre1/datasets/doctor_reviews_family_med_filtered'

# ============================================================================
# Step 1: Load data
# ============================================================================
print('Step 1: Loading data...')
metadata = load_from_disk(METADATA_PATH)
corpus = load_from_disk(CORPUS_PATH)

metadata_df = metadata.to_pandas()
corpus_df = corpus.to_pandas()

print(f'  Original metadata: {len(metadata_df):,} doctors')
print(f'  Original corpus: {len(corpus_df):,} reviews')
print()

# ============================================================================
# Step 2: Apply doctor-level filters
# ============================================================================
print('Step 2: Applying doctor-level filters...')

# Filter 1: Specialty = Family Medicine
print('  Filter 1: Specialty = Family Medicine Physician')
filtered_metadata = metadata_df[metadata_df['Specialty'] == 'Family Medicine Physician']
print(f'    → {len(filtered_metadata):,} doctors')

# Filter 2: Review count >= 20
print('  Filter 2: Review count >= 20')
filtered_metadata = filtered_metadata[filtered_metadata['num_reviews'] >= 20]
print(f'    → {len(filtered_metadata):,} doctors')

# Filter 3: Review count <= 100
print('  Filter 3: Review count <= 100')
filtered_metadata = filtered_metadata[filtered_metadata['num_reviews'] <= 100]
print(f'    → {len(filtered_metadata):,} doctors')

# Get filtered PhyIDs
filtered_phyids = set(filtered_metadata['PhyID'].tolist())
print(f'  Filtered PhyIDs: {len(filtered_phyids):,}')
print()

# ============================================================================
# Step 3: Filter corpus by PhyID
# ============================================================================
print('Step 3: Filtering corpus by doctor PhyIDs...')
filtered_corpus = corpus_df[corpus_df['PhyID'].isin(filtered_phyids)].copy()
print(f'  → {len(filtered_corpus):,} reviews')
print()

# ============================================================================
# Step 4: Apply review-level quality filters
# ============================================================================
print('Step 4: Applying review-level quality filters...')

# Calculate text metrics
print('  Calculating text metrics...')
filtered_corpus['word_count'] = filtered_corpus['text'].astype(str).str.split().str.len()

# Filter 4: Word count >= 15
print('  Filter 4: Word count >= 15')
before = len(filtered_corpus)
filtered_corpus = filtered_corpus[filtered_corpus['word_count'] >= 15]
print(f'    → Removed {before - len(filtered_corpus):,}, remaining: {len(filtered_corpus):,}')

# Filter 5: Word count <= 200
print('  Filter 5: Word count <= 200')
before = len(filtered_corpus)
filtered_corpus = filtered_corpus[filtered_corpus['word_count'] <= 200]
print(f'    → Removed {before - len(filtered_corpus):,}, remaining: {len(filtered_corpus):,}')

# Filter 6: NOT all caps
print('  Filter 6: NOT all caps')
before = len(filtered_corpus)
filtered_corpus = filtered_corpus[~filtered_corpus['text'].astype(str).str.isupper()]
print(f'    → Removed {before - len(filtered_corpus):,}, remaining: {len(filtered_corpus):,}')

# Filter 7: Digit ratio < 30%
print('  Filter 7: Digit ratio < 30%')
def high_digit_ratio(text):
    text_str = str(text)
    if len(text_str) == 0:
        return False
    digit_ratio = sum(c.isdigit() for c in text_str) / len(text_str)
    return digit_ratio >= 0.3

before = len(filtered_corpus)
filtered_corpus = filtered_corpus[~filtered_corpus['text'].apply(high_digit_ratio)]
print(f'    → Removed {before - len(filtered_corpus):,}, remaining: {len(filtered_corpus):,}')

# Filter 8: Character diversity >= 10
print('  Filter 8: Character diversity >= 10 unique chars')
def low_char_diversity(text):
    text_str = str(text).lower()
    if len(text_str) < 20:
        return False
    unique_chars = len(set(text_str))
    return unique_chars < 10

before = len(filtered_corpus)
filtered_corpus = filtered_corpus[~filtered_corpus['text'].apply(low_char_diversity)]
print(f'    → Removed {before - len(filtered_corpus):,}, remaining: {len(filtered_corpus):,}')

# Filter 9: Non-ASCII ratio < 50% (removes Spanish)
print('  Filter 9: Non-ASCII ratio < 50% (English-focused)')
def high_non_ascii(text):
    text_str = str(text)
    if len(text_str) == 0:
        return False
    non_ascii = sum(1 for c in text_str if ord(c) > 127)
    return non_ascii / len(text_str) >= 0.5

before = len(filtered_corpus)
filtered_corpus = filtered_corpus[~filtered_corpus['text'].apply(high_non_ascii)]
print(f'    → Removed {before - len(filtered_corpus):,}, remaining: {len(filtered_corpus):,}')
print()

# ============================================================================
# Step 5: Prepare output format (match TREC-COVID schema)
# ============================================================================
print('Step 5: Preparing output format...')

# Create output dataframe with TREC-COVID compatible schema
output_df = pd.DataFrame({
    '_id': filtered_corpus['text_id'].astype(str),  # Rename text_id -> _id
    'title': '',  # Empty title for compatibility
    'text': filtered_corpus['text'].astype(str)
})

# Reset index
output_df = output_df.reset_index(drop=True)

print(f'  Output columns: {list(output_df.columns)}')
print(f'  Output size: {len(output_df):,} reviews')
print()

# ============================================================================
# Step 6: Save to disk
# ============================================================================
print('Step 6: Saving to disk...')

# Create output directory if needed
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Convert to HuggingFace Dataset and save
output_dataset = Dataset.from_pandas(output_df)
output_dataset.save_to_disk(OUTPUT_PATH)

print(f'  ✅ Saved to: {OUTPUT_PATH}')
print()

# ============================================================================
# Step 7: Verify
# ============================================================================
print('Step 7: Verifying saved dataset...')
verification = load_from_disk(OUTPUT_PATH)
print(f'  Loaded {len(verification):,} documents')
print(f'  Columns: {verification.column_names}')
print(f'  Sample record:')
print(f'    _id: {verification[0]["_id"]}')
print(f'    title: "{verification[0]["title"]}"')
print(f'    text: {verification[0]["text"][:100]}...')
print()

# ============================================================================
# Summary
# ============================================================================
print('='*80)
print('SUMMARY')
print('='*80)
print(f'Original corpus: 4,991,218 reviews')
print(f'After doctor filters: 210,136 reviews')
print(f'After quality filters: {len(output_df):,} reviews')
print(f'Output path: {OUTPUT_PATH}')
print()
print('✅ Filtered corpus created successfully!')
print('='*80)
