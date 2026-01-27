#!/usr/bin/env python3
"""
Data Quality Analysis for Family Medicine ≥20 Reviews

Analyzes potential data quality issues:
1. Language detection (Spanish vs English)
2. Review length distribution (find junk/too-short reviews)
3. Character quality (special chars, encoding issues)
4. Duplicate detection
5. Content quality indicators

Goal: Identify what preprocessing/filtering is needed before topic modeling.
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
from collections import Counter
import re
import json

print('='*80)
print('DATA QUALITY ANALYSIS: Family Medicine ≥20 Reviews')
print('='*80)
print()

# ============================================================================
# STEP 1: Load Filtered Data
# ============================================================================
print('STEP 1: Loading filtered data...')
metadata = load_from_disk('/export/fs06/mzhong8/doctor_review_metadata')
corpus = load_from_disk('/export/fs06/mzhong8/doctor_review_corpus')

# Apply filter
print('  Applying Family Medicine ≥20 filter...')
metadata_df = metadata.to_pandas()
filtered_metadata = metadata_df[
    (metadata_df['Specialty'] == 'Family Medicine Physician') &
    (metadata_df['num_reviews'] >= 20)
]

filtered_phyids = set(filtered_metadata['PhyID'].tolist())
print(f'  Filtered PhyIDs: {len(filtered_phyids):,}')

corpus_df = corpus.to_pandas()
filtered_corpus = corpus_df[corpus_df['PhyID'].isin(filtered_phyids)].copy()
print(f'  Filtered reviews: {len(filtered_corpus):,}')
print()

# ============================================================================
# STEP 2: Check for Language Field
# ============================================================================
print('='*80)
print('STEP 2: LANGUAGE DETECTION')
print('='*80)
print()

print('Corpus columns:', list(filtered_corpus.columns))
print()

# Check if language field exists
if 'language' in filtered_corpus.columns or 'lang' in filtered_corpus.columns:
    lang_col = 'language' if 'language' in filtered_corpus.columns else 'lang'
    print(f'✅ Language field found: "{lang_col}"')
    print()

    lang_counts = filtered_corpus[lang_col].value_counts()
    print('Language distribution:')
    for lang, count in lang_counts.head(10).items():
        pct = (count / len(filtered_corpus) * 100)
        print(f'  {lang}: {count:,} ({pct:.1f}%)')
    print()
else:
    print('❌ No language field in corpus')
    print('   Will perform heuristic language detection on sample...')
    print()

    # Heuristic Spanish detection on sample
    sample_size = min(10000, len(filtered_corpus))
    sample = filtered_corpus.sample(n=sample_size, random_state=42)

    def detect_spanish_heuristic(text):
        """Simple heuristic for Spanish text"""
        text_lower = str(text).lower()
        spanish_indicators = [
            'el ', 'la ', 'los ', 'las ', 'un ', 'una ', 'es ', 'está', 'muy ',
            'que ', 'con ', 'para ', 'por ', 'del ', 'de la', 'al ', 'en el',
            'señor', 'señora', 'doctor', 'doctora', 'excelente', 'muy bien'
        ]
        count = sum(1 for indicator in spanish_indicators if indicator in text_lower)
        return count >= 3  # If 3+ Spanish indicators, likely Spanish

    spanish_count = sample['text'].apply(detect_spanish_heuristic).sum()
    spanish_pct = (spanish_count / len(sample) * 100)

    print(f'Heuristic Spanish detection (sample of {sample_size:,}):')
    print(f'  Likely Spanish: {spanish_count:,} ({spanish_pct:.1f}%)')
    print(f'  Likely English: {len(sample) - spanish_count:,} ({100-spanish_pct:.1f}%)')
    print()
    print(f'Estimated Spanish reviews in full corpus: ~{int(len(filtered_corpus) * spanish_pct / 100):,}')
    print()

# ============================================================================
# STEP 3: Review Length Distribution (Character-Level)
# ============================================================================
print('='*80)
print('STEP 3: REVIEW LENGTH DISTRIBUTION (Character-Level)')
print('='*80)
print()

filtered_corpus['text_length'] = filtered_corpus['text'].astype(str).str.len()
filtered_corpus['word_count'] = filtered_corpus['text'].astype(str).str.split().str.len()

print('Character Length Statistics:')
print(filtered_corpus['text_length'].describe())
print()

# Define length buckets
char_buckets = [
    (0, 50, '0-50 (very short)'),
    (51, 100, '51-100 (short)'),
    (101, 200, '101-200 (medium-short)'),
    (201, 400, '201-400 (medium)'),
    (401, 800, '401-800 (medium-long)'),
    (801, 1500, '801-1500 (long)'),
    (1501, float('inf'), '1500+ (very long)')
]

print('Review Distribution by Character Length:')
print()
for min_len, max_len, label in char_buckets:
    bucket_reviews = filtered_corpus[
        (filtered_corpus['text_length'] >= min_len) &
        (filtered_corpus['text_length'] <= max_len)
    ]
    count = len(bucket_reviews)
    pct = (count / len(filtered_corpus) * 100)
    print(f'  {label:>25}: {count:>7,} reviews ({pct:>5.1f}%)')
print()

# ============================================================================
# STEP 4: Word Count Distribution
# ============================================================================
print('='*80)
print('STEP 4: WORD COUNT DISTRIBUTION')
print('='*80)
print()

print('Word Count Statistics:')
print(filtered_corpus['word_count'].describe())
print()

word_buckets = [
    (0, 5, '0-5 (useless)'),
    (6, 10, '6-10 (very short)'),
    (11, 20, '11-20 (short)'),
    (21, 50, '21-50 (medium)'),
    (51, 100, '51-100 (substantial)'),
    (101, 200, '101-200 (long)'),
    (201, float('inf'), '200+ (very long)')
]

print('Review Distribution by Word Count:')
print()
for min_words, max_words, label in word_buckets:
    bucket_reviews = filtered_corpus[
        (filtered_corpus['word_count'] >= min_words) &
        (filtered_corpus['word_count'] <= max_words)
    ]
    count = len(bucket_reviews)
    pct = (count / len(filtered_corpus) * 100)
    print(f'  {label:>22}: {count:>7,} reviews ({pct:>5.1f}%)')
print()

# ============================================================================
# STEP 5: Identify "Junk" Reviews (Quality Issues)
# ============================================================================
print('='*80)
print('STEP 5: POTENTIAL "JUNK" REVIEW DETECTION')
print('='*80)
print()

# Define quality issues
print('Analyzing potential quality issues...')
print()

# 1. Too short (likely uninformative)
too_short = filtered_corpus['word_count'] <= 10
print(f'1. Too Short (≤10 words): {too_short.sum():,} ({too_short.sum()/len(filtered_corpus)*100:.1f}%)')

# 2. Too long (possibly spam/copy-paste)
too_long = filtered_corpus['word_count'] >= 300
print(f'2. Too Long (≥300 words): {too_long.sum():,} ({too_long.sum()/len(filtered_corpus)*100:.1f}%)')

# 3. Single sentence (low information)
single_period = filtered_corpus['text'].astype(str).str.count(r'[.!?]') <= 1
print(f'3. Single Sentence: {single_period.sum():,} ({single_period.sum()/len(filtered_corpus)*100:.1f}%)')

# 4. All caps (shouting/spam)
all_caps = filtered_corpus['text'].astype(str).str.isupper()
print(f'4. All Caps: {all_caps.sum():,} ({all_caps.sum()/len(filtered_corpus)*100:.1f}%)')

# 5. High digit ratio (phone numbers, spam)
def high_digit_ratio(text):
    text_str = str(text)
    if len(text_str) == 0:
        return False
    digit_ratio = sum(c.isdigit() for c in text_str) / len(text_str)
    return digit_ratio > 0.3

high_digits = filtered_corpus['text'].apply(high_digit_ratio)
print(f'5. High Digit Ratio (>30%): {high_digits.sum():,} ({high_digits.sum()/len(filtered_corpus)*100:.1f}%)')

# 6. Extremely low character diversity (repeated chars)
def low_char_diversity(text):
    text_str = str(text).lower()
    if len(text_str) < 20:
        return False
    unique_chars = len(set(text_str))
    return unique_chars < 10  # Very few unique characters

low_diversity = filtered_corpus['text'].apply(low_char_diversity)
print(f'6. Low Character Diversity: {low_diversity.sum():,} ({low_diversity.sum()/len(filtered_corpus)*100:.1f}%)')

# 7. Non-ASCII heavy (encoding issues or non-English)
def high_non_ascii(text):
    text_str = str(text)
    if len(text_str) == 0:
        return False
    non_ascii = sum(1 for c in text_str if ord(c) > 127)
    return non_ascii / len(text_str) > 0.5

non_ascii_heavy = filtered_corpus['text'].apply(high_non_ascii)
print(f'7. High Non-ASCII (>50%): {non_ascii_heavy.sum():,} ({non_ascii_heavy.sum()/len(filtered_corpus)*100:.1f}%)')

print()

# Combined "junk" criteria
junk_mask = (
    too_short |
    too_long |
    all_caps |
    high_digits |
    low_diversity |
    non_ascii_heavy
)

junk_count = junk_mask.sum()
junk_pct = (junk_count / len(filtered_corpus) * 100)

print(f'📊 POTENTIAL JUNK REVIEWS (any of above): {junk_count:,} ({junk_pct:.1f}%)')
print(f'   Clean reviews: {len(filtered_corpus) - junk_count:,} ({100-junk_pct:.1f}%)')
print()

# ============================================================================
# STEP 6: Show Examples of Problematic Reviews
# ============================================================================
print('='*80)
print('STEP 6: EXAMPLES OF PROBLEMATIC REVIEWS')
print('='*80)
print()

def show_examples(df, mask, category_name, n=3):
    examples = df[mask].head(n)
    if len(examples) == 0:
        print(f'  No {category_name} found')
        return

    print(f'{category_name} Examples:')
    print('-' * 80)
    for idx, row in examples.iterrows():
        text = str(row['text'])[:200]
        print(f'  Review {idx}:')
        print(f'    Length: {row["text_length"]} chars, {row["word_count"]} words')
        print(f'    Text: {text}...')
        print()
    print()

show_examples(filtered_corpus, too_short, 'Too Short (≤10 words)')
show_examples(filtered_corpus, all_caps, 'All Caps')
show_examples(filtered_corpus, high_digits, 'High Digit Ratio')

# ============================================================================
# STEP 7: Duplicate Detection
# ============================================================================
print('='*80)
print('STEP 7: DUPLICATE DETECTION')
print('='*80)
print()

print('Checking for exact duplicates...')
text_duplicates = filtered_corpus['text'].duplicated()
dup_count = text_duplicates.sum()
dup_pct = (dup_count / len(filtered_corpus) * 100)

print(f'  Exact duplicate texts: {dup_count:,} ({dup_pct:.1f}%)')
print()

# Near-duplicate check (first 100 chars)
filtered_corpus['text_prefix'] = filtered_corpus['text'].astype(str).str[:100]
prefix_duplicates = filtered_corpus['text_prefix'].duplicated()
near_dup_count = prefix_duplicates.sum()
near_dup_pct = (near_dup_count / len(filtered_corpus) * 100)

print(f'  Near-duplicates (same first 100 chars): {near_dup_count:,} ({near_dup_pct:.1f}%)')
print()

# ============================================================================
# STEP 8: Recommended Filtering Thresholds
# ============================================================================
print('='*80)
print('STEP 8: RECOMMENDED FILTERING STRATEGY')
print('='*80)
print()

print('Based on analysis, recommended filters:')
print()

# Conservative filter (minimal removal)
conservative_filter = (
    (filtered_corpus['word_count'] >= 5) &  # Remove very short
    (filtered_corpus['word_count'] <= 500) &  # Remove extremely long
    ~all_caps &  # Remove all caps
    ~low_diversity  # Remove low diversity
)

conservative_removed = len(filtered_corpus) - conservative_filter.sum()
conservative_pct = (conservative_removed / len(filtered_corpus) * 100)

print('1. CONSERVATIVE FILTER (Recommended):')
print(f'   - Minimum 5 words')
print(f'   - Maximum 500 words')
print(f'   - No all-caps reviews')
print(f'   - No low character diversity')
print(f'   → Removes: {conservative_removed:,} reviews ({conservative_pct:.1f}%)')
print(f'   → Keeps: {conservative_filter.sum():,} reviews ({100-conservative_pct:.1f}%)')
print()

# Moderate filter (quality-focused)
moderate_filter = (
    (filtered_corpus['word_count'] >= 10) &  # More substantial minimum
    (filtered_corpus['word_count'] <= 300) &  # Reasonable maximum
    ~all_caps &
    ~high_digits &
    ~low_diversity
)

moderate_removed = len(filtered_corpus) - moderate_filter.sum()
moderate_pct = (moderate_removed / len(filtered_corpus) * 100)

print('2. MODERATE FILTER (Balanced):')
print(f'   - Minimum 10 words')
print(f'   - Maximum 300 words')
print(f'   - No all-caps, high digits, or low diversity')
print(f'   → Removes: {moderate_removed:,} reviews ({moderate_pct:.1f}%)')
print(f'   → Keeps: {moderate_filter.sum():,} reviews ({100-moderate_pct:.1f}%)')
print()

# Aggressive filter (high quality only)
aggressive_filter = (
    (filtered_corpus['word_count'] >= 15) &  # Substantial content
    (filtered_corpus['word_count'] <= 200) &  # Not too long
    ~all_caps &
    ~high_digits &
    ~low_diversity &
    ~non_ascii_heavy  # English-focused
)

aggressive_removed = len(filtered_corpus) - aggressive_filter.sum()
aggressive_pct = (aggressive_removed / len(filtered_corpus) * 100)

print('3. AGGRESSIVE FILTER (High Quality):')
print(f'   - Minimum 15 words')
print(f'   - Maximum 200 words')
print(f'   - No quality issues')
print(f'   - Mostly ASCII (English)')
print(f'   → Removes: {aggressive_removed:,} reviews ({aggressive_pct:.1f}%)')
print(f'   → Keeps: {aggressive_filter.sum():,} reviews ({100-aggressive_pct:.1f}%)')
print()

# ============================================================================
# STEP 9: Final Corpus Size After Filtering
# ============================================================================
print('='*80)
print('STEP 9: FINAL CORPUS SIZES WITH FILTERS')
print('='*80)
print()

print('Starting corpus: 257,541 reviews')
print()

filters = {
    'Conservative': conservative_filter,
    'Moderate': moderate_filter,
    'Aggressive': aggressive_filter
}

for name, filt in filters.items():
    final_size = filt.sum()
    removed = len(filtered_corpus) - final_size
    removed_pct = (removed / len(filtered_corpus) * 100)

    # Comparison to 200K target
    target_diff = final_size - 200000
    target_pct = (final_size / 200000 * 100)

    print(f'{name} Filter:')
    print(f'  Final size: {final_size:,} reviews')
    print(f'  Removed: {removed:,} ({removed_pct:.1f}%)')
    print(f'  vs 200K target: {target_pct:.0f}% ({target_diff:+,})')
    print()

# ============================================================================
# STEP 10: Save Summary
# ============================================================================
print('='*80)
print('STEP 10: SAVING QUALITY ANALYSIS SUMMARY')
print('='*80)
print()

summary = {
    'corpus_size': {
        'original': int(len(filtered_corpus)),
        'conservative_filter': int(conservative_filter.sum()),
        'moderate_filter': int(moderate_filter.sum()),
        'aggressive_filter': int(aggressive_filter.sum())
    },
    'quality_issues': {
        'too_short_10words': int(too_short.sum()),
        'too_long_300words': int(too_long.sum()),
        'all_caps': int(all_caps.sum()),
        'high_digits': int(high_digits.sum()),
        'low_diversity': int(low_diversity.sum()),
        'high_non_ascii': int(non_ascii_heavy.sum()),
        'any_issue': int(junk_count)
    },
    'duplicates': {
        'exact_duplicates': int(dup_count),
        'near_duplicates': int(near_dup_count)
    },
    'length_stats': {
        'mean_chars': float(filtered_corpus['text_length'].mean()),
        'median_chars': float(filtered_corpus['text_length'].median()),
        'mean_words': float(filtered_corpus['word_count'].mean()),
        'median_words': float(filtered_corpus['word_count'].median())
    },
    'recommendations': {
        'suggested_filter': 'conservative',
        'min_words': 5,
        'max_words': 500,
        'expected_final_size': int(conservative_filter.sum())
    }
}

output_file = '/home/srangre1/Iterative-Query-Refinement/family_medicine_quality_analysis.json'
with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Quality analysis summary saved to: {output_file}')
print()

print('='*80)
print('QUALITY ANALYSIS COMPLETE')
print('='*80)
print()
print('🎯 RECOMMENDATION: Use CONSERVATIVE filter (removes <5% of data)')
print('   This balances data quality with corpus size.')
