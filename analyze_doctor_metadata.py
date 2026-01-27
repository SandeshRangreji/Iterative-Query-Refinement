#!/usr/bin/env python3
"""
Analyze doctor review metadata to identify filtering strategies for subset creation.
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np

print('Loading metadata...')
metadata = load_from_disk('/export/fs06/mzhong8/doctor_review_metadata')
df = metadata.to_pandas()

print(f'\n{"="*80}')
print(f'DATASET OVERVIEW')
print(f'{"="*80}')
print(f'Total records: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print()

# 1. Gender
print(f'\n{"="*80}')
print('1. GENDER Distribution:')
print(f'{"="*80}')
gender_counts = df['Gender'].value_counts(dropna=False)
for val, count in gender_counts.items():
    pct = (count / len(df) * 100)
    print(f'  {val}: {count:,} ({pct:.1f}%)')

# 2. Credential
print(f'\n{"="*80}')
print('2. CREDENTIAL Distribution (Top 15):')
print(f'{"="*80}')
cred_counts = df['Credential'].value_counts(dropna=False).head(15)
for val, count in cred_counts.items():
    pct = (count / len(df) * 100)
    print(f'  {val}: {count:,} ({pct:.1f}%)')
print(f'  Total unique credentials: {df["Credential"].nunique()}')

# 3. Specialty
print(f'\n{"="*80}')
print('3. SPECIALTY Distribution (Top 25):')
print(f'{"="*80}')
spec_counts = df['Specialty'].value_counts(dropna=False).head(25)
for val, count in spec_counts.items():
    pct = (count / len(df) * 100)
    val_display = str(val)[:50]  # Truncate long values
    print(f'  {val_display}: {count:,} ({pct:.1f}%)')
print(f'  Total unique specialties: {df["Specialty"].nunique()}')

# 4. PhysicianType
print(f'\n{"="*80}')
print('4. PHYSICIAN TYPE Distribution:')
print(f'{"="*80}')
phys_counts = df['PhysicianType'].value_counts(dropna=False)
for val, count in phys_counts.items():
    pct = (count / len(df) * 100)
    print(f'  {val}: {count:,} ({pct:.1f}%)')

# 5. Number of reviews
print(f'\n{"="*80}')
print('5. NUMBER OF REVIEWS Statistics:')
print(f'{"="*80}')
print(df['num_reviews'].describe())
print()
print('Reviews distribution (quantiles):')
quantiles = df['num_reviews'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
for q, val in quantiles.items():
    print(f'  {int(q*100)}th percentile: {val:.0f} reviews')

# 6. Geography - Practice Zip
print(f'\n{"="*80}')
print('6. PRACTICE ZIP (Top 15 locations by doctor count):')
print(f'{"="*80}')
zip_counts = df['PracticeZip5'].value_counts(dropna=False).head(15)
for val, count in zip_counts.items():
    pct = (count / len(df) * 100)
    print(f'  {val}: {count:,} doctors ({pct:.1f}%)')

# 7. Check actual review corpus size
print(f'\n{"="*80}')
print('7. REVIEW CORPUS SIZE:')
print(f'{"="*80}')
print('Loading review corpus...')
corpus = load_from_disk('/export/fs06/mzhong8/doctor_review_corpus')
print(f'  Total reviews in corpus: {len(corpus):,}')

# Join to see review coverage
corpus_df = corpus.to_pandas()
print(f'  Unique PhyIDs with reviews: {corpus_df["PhyID"].nunique():,}')
print(f'  Average reviews per doctor (in corpus): {len(corpus_df) / corpus_df["PhyID"].nunique():.1f}')

# 8. Null values
print(f'\n{"="*80}')
print('8. NULL VALUE ANALYSIS:')
print(f'{"="*80}')
for col in df.columns:
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df) * 100)
    if null_count > 0:
        print(f'  {col}: {null_count:,} ({null_pct:.1f}%)')

# 9. Combined filter examples
print(f'\n{"="*80}')
print('9. EXAMPLE FILTER COMBINATIONS:')
print(f'{"="*80}')

# Common specialties with good review counts (use actual specialty names from data)
print('BY SPECIALTY:')
common_specs = [
    'Family Medicine Physician',
    'Internal Medicine Physician',
    'Pediatrics Physician',
    'Obstetrics & Gynecology Physician',
    'Psychiatry Physician'
]
for spec in common_specs:
    for threshold in [5, 10, 20]:
        filtered = df[(df['Specialty'] == spec) & (df['num_reviews'] >= threshold)]
        total_reviews = filtered['num_reviews'].sum()
        print(f'  {spec} with ≥{threshold} reviews: {total_reviews:,} reviews ({len(filtered):,} doctors)')

print()

# High-review doctors across all specialties
print('ALL SPECIALTIES BY REVIEW THRESHOLD:')
for threshold in [5, 10, 20, 50, 100]:
    filtered = df[df['num_reviews'] >= threshold]
    total_reviews = filtered['num_reviews'].sum()
    print(f'  All doctors with ≥{threshold} reviews: {total_reviews:,} reviews ({len(filtered):,} doctors)')

print()

# By credential
print('BY CREDENTIAL:')
for cred in ['MD', 'DO']:
    for threshold in [5, 10, 20]:
        filtered = df[(df['Credential'] == cred) & (df['num_reviews'] >= threshold)]
        total_reviews = filtered['num_reviews'].sum()
        print(f'  {cred} with ≥{threshold} reviews: {total_reviews:,} reviews ({len(filtered):,} doctors)')

print()

# By PhysicianType
print('BY PHYSICIAN TYPE:')
for ptype in ['Primary Care', 'Specialty', 'Super Specialties']:
    for threshold in [5, 10, 20]:
        filtered = df[(df['PhysicianType'] == ptype) & (df['num_reviews'] >= threshold)]
        total_reviews = filtered['num_reviews'].sum()
        print(f'  {ptype} with ≥{threshold} reviews: {total_reviews:,} reviews ({len(filtered):,} doctors)')

print()

# Combined filters - high quality subsets
print('COMBINED FILTER EXAMPLES:')
print()

# Primary care with good reviews
filtered = df[(df['PhysicianType'] == 'Primary Care') & (df['num_reviews'] >= 10)]
total_reviews = filtered['num_reviews'].sum()
print(f'  Primary Care with ≥10 reviews: {total_reviews:,} reviews ({len(filtered):,} doctors)')

# MD/DO physicians with moderate reviews
filtered = df[(df['Credential'].isin(['MD', 'DO'])) & (df['num_reviews'] >= 10)]
total_reviews = filtered['num_reviews'].sum()
print(f'  MD/DO with ≥10 reviews: {total_reviews:,} reviews ({len(filtered):,} doctors)')

# Top 5 specialties with adequate reviews
top_5_specs = [
    'Family Medicine Physician',
    'Internal Medicine Physician',
    'Obstetrics & Gynecology Physician',
    'Pediatrics Physician',
    'Psychiatry Physician'
]
filtered = df[(df['Specialty'].isin(top_5_specs)) & (df['num_reviews'] >= 10)]
total_reviews = filtered['num_reviews'].sum()
print(f'  Top 5 specialties with ≥10 reviews: {total_reviews:,} reviews ({len(filtered):,} doctors)')

# Individual specialty breakdown (no threshold, just total reviews)
print()
print('TOTAL REVIEWS PER SPECIALTY (all doctors):')
for spec in common_specs:
    filtered = df[df['Specialty'] == spec]
    total_reviews = filtered['num_reviews'].sum()
    print(f'  {spec}: {total_reviews:,} reviews ({len(filtered):,} doctors)')

# High-quality single specialty subsets
print()
print('SINGLE SPECIALTY HIGH-QUALITY SUBSETS:')
for spec in common_specs:
    for threshold in [20, 50]:
        filtered = df[(df['Specialty'] == spec) & (df['num_reviews'] >= threshold)]
        total_reviews = filtered['num_reviews'].sum()
        print(f'  {spec} with ≥{threshold} reviews: {total_reviews:,} reviews ({len(filtered):,} doctors)')

print('\nAnalysis complete!')
