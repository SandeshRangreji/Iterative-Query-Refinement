#!/usr/bin/env python3
"""
Detailed analysis of Family Medicine ≥20 reviews filter.

This creates a comprehensive report on the filtered subset to verify it's suitable
for topic modeling experiments (~200K reviews target).
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
from collections import Counter
import json

print('='*80)
print('FAMILY MEDICINE ≥20 REVIEWS FILTER ANALYSIS')
print('='*80)
print()

# ============================================================================
# STEP 1: Load and Filter Metadata
# ============================================================================
print('STEP 1: Loading and filtering metadata...')
metadata = load_from_disk('/export/fs06/mzhong8/doctor_review_metadata')
df = metadata.to_pandas()

print(f'  Original dataset: {len(df):,} doctors')

# Apply filter: Family Medicine with ≥20 reviews
filtered_df = df[
    (df['Specialty'] == 'Family Medicine Physician') &
    (df['num_reviews'] >= 20)
].copy()

print(f'  Filtered dataset: {len(filtered_df):,} doctors')
print(f'  Reduction: {(1 - len(filtered_df)/len(df))*100:.1f}%')
print()

# ============================================================================
# STEP 2: Basic Statistics
# ============================================================================
print('='*80)
print('STEP 2: FILTERED DATASET STATISTICS')
print('='*80)
print()

total_reviews = filtered_df['num_reviews'].sum()
print(f'Total Reviews (Expected): {total_reviews:,}')
print(f'Total Doctors: {len(filtered_df):,}')
print(f'Average Reviews per Doctor: {filtered_df["num_reviews"].mean():.1f}')
print(f'Median Reviews per Doctor: {filtered_df["num_reviews"].median():.0f}')
print()

print('Review Count Distribution:')
print(filtered_df['num_reviews'].describe())
print()

print('Review Count Quantiles:')
quantiles = filtered_df['num_reviews'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
for q, val in quantiles.items():
    print(f'  {int(q*100)}th percentile: {val:.0f} reviews')
print()

# ============================================================================
# STEP 3: Review Count Buckets
# ============================================================================
print('='*80)
print('STEP 3: DOCTOR DISTRIBUTION BY REVIEW COUNT BUCKETS')
print('='*80)
print()

buckets = [
    (20, 29, '20-29'),
    (30, 49, '30-49'),
    (50, 99, '50-99'),
    (100, 199, '100-199'),
    (200, 499, '200-499'),
    (500, float('inf'), '500+')
]

bucket_stats = []
for min_rev, max_rev, label in buckets:
    bucket_df = filtered_df[(filtered_df['num_reviews'] >= min_rev) &
                             (filtered_df['num_reviews'] <= max_rev)]
    count = len(bucket_df)
    total_revs = bucket_df['num_reviews'].sum()
    pct_doctors = (count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    pct_reviews = (total_revs / total_reviews * 100) if total_reviews > 0 else 0

    bucket_stats.append({
        'bucket': label,
        'doctors': count,
        'pct_doctors': pct_doctors,
        'reviews': int(total_revs),
        'pct_reviews': pct_reviews
    })

    print(f'{label:>10} reviews: {count:>5,} doctors ({pct_doctors:>5.1f}%) → {int(total_revs):>7,} reviews ({pct_reviews:>5.1f}%)')

print()

# ============================================================================
# STEP 4: Gender Distribution
# ============================================================================
print('='*80)
print('STEP 4: GENDER DISTRIBUTION')
print('='*80)
print()

gender_counts = filtered_df['Gender'].value_counts(dropna=False)
for gender, count in gender_counts.items():
    pct = (count / len(filtered_df) * 100)
    total_revs = filtered_df[filtered_df['Gender'] == gender]['num_reviews'].sum()
    print(f'  {gender}: {count:,} doctors ({pct:.1f}%) → {int(total_revs):,} reviews')
print()

# ============================================================================
# STEP 5: Credential Distribution
# ============================================================================
print('='*80)
print('STEP 5: CREDENTIAL DISTRIBUTION')
print('='*80)
print()

cred_counts = filtered_df['Credential'].value_counts(dropna=False).head(10)
for cred, count in cred_counts.items():
    pct = (count / len(filtered_df) * 100)
    total_revs = filtered_df[filtered_df['Credential'] == cred]['num_reviews'].sum()
    print(f'  {cred}: {count:,} doctors ({pct:.1f}%) → {int(total_revs):,} reviews')
print()

# ============================================================================
# STEP 6: Geographic Distribution (Top States from ZIP codes)
# ============================================================================
print('='*80)
print('STEP 6: GEOGRAPHIC DISTRIBUTION (Top 20 Zip Codes by Doctor Count)')
print('='*80)
print()

zip_counts = filtered_df['PracticeZip5'].value_counts(dropna=False).head(20)
for zip_code, count in zip_counts.items():
    pct = (count / len(filtered_df) * 100)
    if pd.isna(zip_code):
        print(f'  Missing: {count:,} doctors ({pct:.1f}%)')
    else:
        print(f'  {int(zip_code):05d}: {count:,} doctors ({pct:.1f}%)')
print()

# ============================================================================
# STEP 7: Load and Verify Actual Corpus
# ============================================================================
print('='*80)
print('STEP 7: CORPUS VERIFICATION')
print('='*80)
print()

print('Loading full corpus (this may take a moment)...')
corpus = load_from_disk('/export/fs06/mzhong8/doctor_review_corpus')
print(f'  Full corpus size: {len(corpus):,} reviews')
print()

# Get filtered PhyIDs
filtered_phyids = set(filtered_df['PhyID'].tolist())
print(f'Filtered PhyIDs: {len(filtered_phyids):,}')
print()

# Count reviews matching filtered PhyIDs
print('Counting reviews in corpus for filtered doctors...')
corpus_df = corpus.to_pandas()
filtered_corpus_df = corpus_df[corpus_df['PhyID'].isin(filtered_phyids)]

actual_review_count = len(filtered_corpus_df)
actual_phyids = filtered_corpus_df['PhyID'].nunique()

print(f'  Actual reviews in corpus: {actual_review_count:,}')
print(f'  Expected reviews (metadata sum): {total_reviews:,}')
print(f'  Difference: {actual_review_count - total_reviews:,}')
print(f'  Match: {"✅ YES" if abs(actual_review_count - total_reviews) < 100 else "⚠️  MISMATCH"}')
print()
print(f'  Unique doctors with reviews: {actual_phyids:,}')
print(f'  Average reviews per doctor (actual): {actual_review_count / actual_phyids:.1f}')
print()

# ============================================================================
# STEP 8: Review Text Statistics
# ============================================================================
print('='*80)
print('STEP 8: REVIEW TEXT STATISTICS')
print('='*80)
print()

print('Analyzing review text lengths (sample of 10,000 reviews)...')
sample_size = min(10000, len(filtered_corpus_df))
sample_reviews = filtered_corpus_df.sample(n=sample_size, random_state=42)

# Check column names
print(f'Corpus columns: {list(corpus_df.columns)}')
print()

# Determine text column (likely 'text', 'review_text', 'content', or 'Review')
text_col = None
for col in ['text', 'review_text', 'content', 'Review', 'review']:
    if col in corpus_df.columns:
        text_col = col
        break

if text_col:
    print(f'Text column found: "{text_col}"')
    print()

    # Calculate lengths
    sample_reviews['text_length'] = sample_reviews[text_col].astype(str).str.len()
    sample_reviews['word_count'] = sample_reviews[text_col].astype(str).str.split().str.len()

    print('Text Length Statistics (characters):')
    print(sample_reviews['text_length'].describe())
    print()

    print('Word Count Statistics:')
    print(sample_reviews['word_count'].describe())
    print()

    # Show sample reviews
    print('Sample Reviews (3 random examples):')
    print('-' * 80)
    for idx, row in sample_reviews.head(3).iterrows():
        text = str(row[text_col])[:200]  # First 200 chars
        print(f'\nReview {idx}:')
        print(f'  PhyID: {row["PhyID"]}')
        print(f'  Length: {row["text_length"]} chars, {row["word_count"]} words')
        print(f'  Text: {text}...')
    print('-' * 80)
    print()
else:
    print('⚠️  Could not find text column. Available columns:', list(corpus_df.columns))
    print()

# ============================================================================
# STEP 9: Comparison to TREC-COVID
# ============================================================================
print('='*80)
print('STEP 9: COMPARISON TO TREC-COVID DATASET')
print('='*80)
print()

trec_covid_size = 171332
ratio = actual_review_count / trec_covid_size

print(f'TREC-COVID documents: {trec_covid_size:,}')
print(f'Family Medicine ≥20 reviews: {actual_review_count:,}')
print(f'Ratio: {ratio:.2f}x TREC-COVID')
print()

if ratio < 1.0:
    print(f'  → Smaller dataset ({(1-ratio)*100:.0f}% smaller)')
elif ratio < 1.5:
    print(f'  → Similar size (within 1.5x)')
elif ratio < 3:
    print(f'  → Moderately larger (1.5-3x)')
else:
    print(f'  → Much larger (>3x)')
print()

# Runtime estimate (rough)
trec_covid_runtime_min = 20  # ~20 min per query for TREC-COVID
estimated_runtime = trec_covid_runtime_min * ratio
print(f'Estimated Runtime per Query:')
print(f'  TREC-COVID baseline: ~{trec_covid_runtime_min} min/query')
print(f'  Family Medicine ≥20: ~{estimated_runtime:.0f} min/query ({estimated_runtime/60:.1f} hours)')
print()

# ============================================================================
# STEP 10: Save Summary Stats
# ============================================================================
print('='*80)
print('STEP 10: SAVING SUMMARY STATISTICS')
print('='*80)
print()

summary = {
    'filter': {
        'specialty': 'Family Medicine Physician',
        'min_reviews': 20,
        'description': 'Family Medicine doctors with at least 20 reviews'
    },
    'size': {
        'total_doctors': int(len(filtered_df)),
        'total_reviews_expected': int(total_reviews),
        'total_reviews_actual': int(actual_review_count),
        'avg_reviews_per_doctor': float(filtered_df['num_reviews'].mean()),
        'median_reviews_per_doctor': float(filtered_df['num_reviews'].median())
    },
    'comparison': {
        'trec_covid_size': trec_covid_size,
        'ratio_to_trec_covid': float(ratio),
        'estimated_runtime_minutes_per_query': float(estimated_runtime)
    },
    'demographics': {
        'gender': {k: int(v) for k, v in gender_counts.items()},
        'top_credentials': {k: int(v) for k, v in cred_counts.head(5).items()}
    },
    'review_distribution': {
        'buckets': bucket_stats
    }
}

if text_col:
    summary['text_stats'] = {
        'sample_size': int(sample_size),
        'avg_length_chars': float(sample_reviews['text_length'].mean()),
        'median_length_chars': float(sample_reviews['text_length'].median()),
        'avg_word_count': float(sample_reviews['word_count'].mean()),
        'median_word_count': float(sample_reviews['word_count'].median())
    }

output_file = '/home/srangre1/Iterative-Query-Refinement/family_medicine_20plus_analysis.json'
with open(output_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Summary saved to: {output_file}')
print()

# ============================================================================
# STEP 11: Final Recommendation
# ============================================================================
print('='*80)
print('STEP 11: FINAL ASSESSMENT')
print('='*80)
print()

print(f'✅ Filter: Family Medicine Physician with ≥20 reviews')
print(f'✅ Size: {actual_review_count:,} reviews ({len(filtered_df):,} doctors)')
print(f'✅ Target: ~200K reviews → {(actual_review_count/200000)*100:.0f}% of target ({actual_review_count - 200000:+,})')
print()

if 150000 <= actual_review_count <= 300000:
    print('✅ EXCELLENT: Size is within reasonable range of 200K target')
elif 100000 <= actual_review_count <= 400000:
    print('✅ GOOD: Size is acceptable for experiments')
else:
    print('⚠️  WARNING: Size may be too far from 200K target')
print()

print('Suitability for Topic Modeling:')
quality_score = 0

# Quality checks
if len(filtered_df) >= 5000:
    print('  ✅ Sufficient doctors (≥5,000)')
    quality_score += 1
else:
    print('  ⚠️  Limited doctors (<5,000)')

if filtered_df['num_reviews'].median() >= 25:
    print('  ✅ Good review depth (median ≥25)')
    quality_score += 1
else:
    print('  ✅ Adequate review depth')
    quality_score += 1

if 150000 <= actual_review_count <= 300000:
    print('  ✅ Optimal size for topic modeling')
    quality_score += 1
elif 100000 <= actual_review_count <= 400000:
    print('  ✅ Acceptable size for topic modeling')
    quality_score += 1

print(f'\nOverall Quality Score: {quality_score}/3')
print()

if quality_score >= 2:
    print('🎯 RECOMMENDATION: This filter is SUITABLE for your experiments!')
else:
    print('⚠️  RECOMMENDATION: Consider adjusting the filter')

print()
print('='*80)
print('ANALYSIS COMPLETE')
print('='*80)
