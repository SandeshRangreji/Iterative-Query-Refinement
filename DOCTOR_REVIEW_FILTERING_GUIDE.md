# Doctor Review Dataset Filtering Guide

This document provides filtering strategies for creating manageable subsets of the doctor review dataset for topic modeling experiments.

## Dataset Overview

- **Total Doctors**: 492,565
- **Total Reviews**: 4,991,218
- **Average Reviews per Doctor**: 10.1
- **Metadata Path**: `/export/fs06/mzhong8/doctor_review_metadata`
- **Corpus Path**: `/export/fs06/mzhong8/doctor_review_corpus`
- **Join Key**: `PhyID` column

---

## Key Dataset Characteristics

### Review Distribution (Highly Skewed)
- **Median**: 4 reviews per doctor
- **75th percentile**: 10 reviews
- **90th percentile**: 22 reviews
- **95th percentile**: 36 reviews
- **99th percentile**: 98 reviews
- **Maximum**: 5,643 reviews

**Implication**: The distribution is extremely skewed. Most doctors have very few reviews, so filtering by minimum review count is essential for quality topic modeling.

### Physician Categories
- **Primary Care**: 181,323 doctors (36.8%)
- **Specialty**: 271,581 doctors (55.1%)
- **Super Specialties**: 39,661 doctors (8.1%)

### Top Specialties
1. **Family Medicine**: 76,033 doctors (15.4%) - 613,243 total reviews
2. **Internal Medicine**: 74,116 doctors (15.0%) - 567,465 total reviews
3. **OB/GYN**: 26,913 doctors (5.5%) - 355,333 total reviews
4. **Pediatrics**: 26,150 doctors (5.3%) - 148,115 total reviews
5. **Psychiatry**: 21,280 doctors (4.3%) - 149,341 total reviews

### Credentials
- **MD**: 437,930 doctors (88.9%)
- **DO**: 44,883 doctors (9.1%)

---

## Filtering Strategies with Exact Corpus Sizes

### Strategy 1: Filter by Review Count Only (All Specialties)

**Simplest approach** - just set a minimum review threshold.

| Threshold | Reviews | Doctors | Comparison to TREC-COVID |
|-----------|---------|---------|--------------------------|
| ≥5 | **4,459,651** | 231,703 | 26x larger |
| ≥10 | **3,767,749** | 127,577 | 22x larger |
| ≥20 | **2,809,640** | 56,397 | 16x larger |
| ≥50 | **1,608,740** | 15,757 | 9x larger |
| ≥100 | **866,176** | 4,799 | 5x larger |

**Recommendation**:
- **≥10** for comprehensive experiments (3.8M reviews)
- **≥20** for higher quality (2.8M reviews)
- **≥50** for focused, high-quality subset (1.6M reviews)

**Python Code**:
```python
from datasets import load_from_disk

metadata = load_from_disk('/export/fs06/mzhong8/doctor_review_metadata')
corpus = load_from_disk('/export/fs06/mzhong8/doctor_review_corpus')

# Filter metadata (e.g., ≥10 reviews)
filtered_metadata = metadata.filter(lambda x: x['num_reviews'] >= 10)

# Get filtered PhyIDs
filtered_phyids = set(filtered_metadata['PhyID'])

# Filter corpus by PhyID
filtered_corpus = corpus.filter(lambda x: x['PhyID'] in filtered_phyids)

# Save filtered dataset
filtered_corpus.save_to_disk('/path/to/doctor_reviews_10plus')
```

---

### Strategy 2: Primary Care Focus

**Goal**: Focus on primary care physicians for general health topics.

| Threshold | Reviews | Doctors |
|-----------|---------|---------|
| ≥5 | **1,154,942** | 78,499 |
| ≥10 | **888,666** | 38,240 |
| ≥20 | **560,794** | 13,614 |

**Use Case**: General health topics, preventive care, chronic disease management

**Comparison**: Primary Care ≥10 (888K reviews) is **5x larger than TREC-COVID**

**Python Code**:
```python
filtered_metadata = metadata.filter(
    lambda x: x['PhysicianType'] == 'Primary Care' and x['num_reviews'] >= 10
)
```

---

### Strategy 3: Specialty Focus

**Goal**: Focus on specialized medicine for domain-specific topics.

| Threshold | Reviews | Doctors |
|-----------|---------|---------|
| ≥5 | **2,965,012** | 137,340 |
| ≥10 | **2,586,185** | 80,592 |
| ≥20 | **2,018,379** | 38,635 |

**Use Case**: Specialized medical procedures, specific conditions, surgical outcomes

**Comparison**: Specialty ≥10 (2.6M reviews) is **15x larger than TREC-COVID**

**Python Code**:
```python
filtered_metadata = metadata.filter(
    lambda x: x['PhysicianType'] == 'Specialty' and x['num_reviews'] >= 10
)
```

---

### Strategy 4: Single Specialty Subsets

Perfect for **domain-specific topic modeling** on focused medical areas.

#### By Individual Specialty (≥10 reviews):

| Specialty | Reviews (≥10) | Reviews (≥20) | Reviews (≥50) |
|-----------|---------------|---------------|---------------|
| **Family Medicine** | 409,373 | 257,541 | 113,266 |
| **Internal Medicine** | 373,620 | 235,739 | 103,602 |
| **OB/GYN** | 288,203 | 216,357 | 115,712 |
| **Pediatrics** | 80,355 | 49,711 | 23,381 |
| **Psychiatry** | 96,974 | 63,890 | 26,546 |

**Recommendations by Specialty**:

**Family Medicine (≥10)**: 409K reviews
- **Topics**: Preventive care, chronic disease, general health
- **Comparison**: 2.4x TREC-COVID size

**Internal Medicine (≥10)**: 374K reviews
- **Topics**: Adult medicine, chronic conditions, hospital medicine
- **Comparison**: 2.2x TREC-COVID size

**OB/GYN (≥10)**: 288K reviews
- **Topics**: Pregnancy, childbirth, women's health, reproductive care
- **Comparison**: 1.7x TREC-COVID size

**Psychiatry (≥10)**: 97K reviews
- **Topics**: Mental health, therapy, medication management
- **Comparison**: Similar to TREC-COVID size (0.6x)

**Pediatrics (≥10)**: 80K reviews
- **Topics**: Child health, vaccinations, developmental concerns, parent experience
- **Comparison**: 0.5x TREC-COVID size

**Python Code** (example: Psychiatry):
```python
filtered_metadata = metadata.filter(
    lambda x: x['Specialty'] == 'Psychiatry Physician' and x['num_reviews'] >= 10
)
```

---

### Strategy 5: Credential-Based Filtering

**Goal**: Focus on traditional physicians (MD/DO).

| Credential | Threshold | Reviews | Doctors |
|------------|-----------|---------|---------|
| **MD** | ≥5 | 3,966,597 | 206,186 |
| **MD** | ≥10 | 3,349,407 | 113,279 |
| **MD** | ≥20 | 2,497,317 | 49,984 |
| **DO** | ≥5 | 381,217 | 21,061 |
| **DO** | ≥10 | 318,175 | 11,610 |
| **DO** | ≥20 | 229,616 | 5,018 |
| **MD/DO Combined** | ≥10 | **3,667,582** | 124,889 |

**Use Case**: Exclude non-physician providers, focus on credentialed physicians

**Comparison**: MD/DO ≥10 (3.7M reviews) is **21x larger than TREC-COVID**

**Python Code**:
```python
filtered_metadata = metadata.filter(
    lambda x: x['Credential'] in ['MD', 'DO'] and x['num_reviews'] >= 10
)
```

---

### Strategy 6: Combined Multi-Specialty Subsets

**Top 5 Specialties** (Family Med, Internal Med, OB/GYN, Pediatrics, Psychiatry):

| Threshold | Reviews | Doctors |
|-----------|---------|---------|
| ≥10 | **1,248,525** | 51,271 |

**Use Case**: Broad coverage across major specialties, good generalization

**Comparison**: 7x larger than TREC-COVID

**Python Code**:
```python
top_5_specs = [
    'Family Medicine Physician',
    'Internal Medicine Physician',
    'Obstetrics & Gynecology Physician',
    'Pediatrics Physician',
    'Psychiatry Physician'
]
filtered_metadata = metadata.filter(
    lambda x: x['Specialty'] in top_5_specs and x['num_reviews'] >= 10
)
```

---

## Recommended Subsets for Your Experiments

### For Testing/Development (Similar to TREC-COVID size)
**Filter**: Psychiatry ≥10 reviews
- **Reviews**: 96,974
- **Size**: 0.6x TREC-COVID
- **Runtime**: Similar to TREC-COVID (~15-20 min/query)
- **Use**: Test pipeline, debug, quick iterations

### For Medium Experiments (Similar to TREC-COVID)
**Filter**: Single specialty ≥20 reviews
- **Examples**:
  - OB/GYN: 216,357 reviews (women's health)
  - Family Medicine: 257,541 reviews (primary care)
  - Internal Medicine: 235,739 reviews (adult medicine)
- **Size**: 1.3-1.5x TREC-COVID
- **Runtime**: ~20-30 min per query estimate
- **Use**: Domain-specific topic modeling, comparable to TREC-COVID scale

### For Large Experiments (5x TREC-COVID)
**Filter**: Primary Care ≥10 reviews
- **Reviews**: 888,666
- **Size**: 5x TREC-COVID
- **Runtime**: ~1-2 hours per query estimate
- **Use**: Comprehensive primary care topic modeling

### For Very Large Experiments (22x TREC-COVID)
**Filter**: All doctors ≥10 reviews
- **Reviews**: 3,767,749
- **Size**: 22x TREC-COVID
- **Runtime**: ~4-8 hours per query estimate
- **Use**: Full dataset coverage, cross-specialty comparison

---

## Sample Queries for Doctor Reviews

Unlike TREC-COVID (scientific literature), queries should focus on **patient experience themes**:

### General Patient Experience Queries:
1. "doctor communication and bedside manner"
2. "appointment wait times and scheduling"
3. "accurate diagnosis and treatment effectiveness"
4. "office staff professionalism"
5. "insurance and billing issues"
6. "pain management and symptom relief"
7. "medication side effects and management"
8. "office cleanliness and comfort"
9. "follow-up care and accessibility"
10. "medical records and test results communication"

### Specialty-Specific Queries:

**For Psychiatry**:
- "therapy effectiveness and therapeutic relationship"
- "medication management for mental health"
- "counseling and coping strategies"

**For OB/GYN**:
- "pregnancy care and prenatal visits"
- "childbirth and delivery experience"
- "postpartum care and support"

**For Pediatrics**:
- "child-friendly care and environment"
- "vaccination experience and education"
- "parent communication and guidance"

**For Primary Care**:
- "preventive care and health screenings"
- "chronic disease management"
- "referral coordination and specialist communication"

---

## Implementation Workflow

### Step 1: Choose Your Filter Strategy

Based on your research goals:
- **Broad coverage**: All doctors ≥10 or ≥20
- **Primary care focus**: Primary Care ≥10
- **Specialty focus**: Single specialty ≥10 or ≥20
- **Testing**: Psychiatry ≥10 (similar to TREC-COVID size)

### Step 2: Create Filtered Dataset

```python
from datasets import load_from_disk

# Load full datasets
metadata = load_from_disk('/export/fs06/mzhong8/doctor_review_metadata')
corpus = load_from_disk('/export/fs06/mzhong8/doctor_review_corpus')

# Example: Primary Care ≥10 reviews
filtered_metadata = metadata.filter(
    lambda x: x['PhysicianType'] == 'Primary Care' and x['num_reviews'] >= 10
)

# Get PhyIDs
filtered_phyids = set(filtered_metadata['PhyID'])

# Filter corpus
filtered_corpus = corpus.filter(lambda x: x['PhyID'] in filtered_phyids)

# Verify size
print(f"Filtered corpus size: {len(filtered_corpus):,} reviews")

# Save
filtered_corpus.save_to_disk('/export/fs06/mzhong8/doctor_reviews_primarycare_10plus')
```

### Step 3: Inspect the Data

```python
# Check column names
print("Corpus columns:", filtered_corpus.column_names)

# View sample reviews
print("\nSample review:")
print(filtered_corpus[0])

# Check text field name (likely 'text', 'review_text', or 'content')
```

### Step 4: Adapt the Topic Modeling Pipeline

Modify `end_to_end_evaluation.py`:

1. **Replace dataset loading** (around line 3107):
```python
# OLD: TREC-COVID
# corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]

# NEW: Doctor reviews
corpus_dataset = load_from_disk('/export/fs06/mzhong8/doctor_reviews_primarycare_10plus')
```

2. **Define healthcare queries** instead of TREC-COVID queries:
```python
# Create a simple queries dataset
queries = [
    {"_id": "1", "text": "doctor communication and bedside manner"},
    {"_id": "2", "text": "appointment wait times and scheduling"},
    {"_id": "3", "text": "accurate diagnosis and treatment effectiveness"},
    # ... add more
]
```

3. **Update text field references** if needed:
```python
# Check if corpus uses 'text' field or something else
# Update references in search.py if different field name
```

### Step 5: Test on Small Subset First

```bash
# Modify QUERY_IDS in main() to test 1-2 queries first
QUERY_IDS = ["1", "2"]  # Just 2 queries for testing

# Run
sbatch /home/srangre1/run_search.sh
```

---

## Dataset Comparison Table

| Dataset | Documents | Domain | Text Type | Avg Length |
|---------|-----------|--------|-----------|------------|
| **TREC-COVID** | 171,332 | Medical/Scientific | Abstracts | Medium |
| **Doctor Reviews (≥10)** | 3,767,749 | Healthcare/Patient | Reviews | Short-Medium |
| **Primary Care (≥10)** | 888,666 | Primary Care | Reviews | Short-Medium |
| **Psychiatry (≥10)** | 96,974 | Mental Health | Reviews | Short-Medium |

---

## Important Considerations

### Data Quality
- **User-generated content**: Expect more spelling errors, informal language, slang
- **Rating bias**: Reviews may skew negative (complaints) or positive (gratitude)
- **Variable length**: Some reviews are 1 sentence, others are paragraphs

### Text Processing Adjustments
- **Stopwords**: May need custom healthcare stopwords ("doctor", "appointment", "office")
- **Vocabulary**: More informal than scientific literature
- **Ngrams**: Consider trigrams for phrases like "wait time", "front desk"

### Computational Resources
- **Larger datasets** (>1M reviews): Increase memory allocation, longer runtimes
- **Embedding computation**: Scales linearly with corpus size
- **BM25 indexing**: One-time cost, then cached

### Privacy & Ethics
- Reviews are public but may contain sensitive patient experiences
- Ensure compliance with data usage policies
- Consider anonymization if publishing results

---

## Quick Reference: Filter Commands

```bash
# Check corpus size
python -c "from datasets import load_from_disk; c=load_from_disk('/export/fs06/mzhong8/doctor_review_corpus'); print(f'Total: {len(c):,} reviews')"

# Check metadata columns
python -c "from datasets import load_from_disk; m=load_from_disk('/export/fs06/mzhong8/doctor_review_metadata'); print(m.column_names)"

# Check corpus columns
python -c "from datasets import load_from_disk; c=load_from_disk('/export/fs06/mzhong8/doctor_review_corpus'); print(c.column_names)"

# View sample record
python -c "from datasets import load_from_disk; c=load_from_disk('/export/fs06/mzhong8/doctor_review_corpus'); print(c[0])"
```

---

## Summary: Recommended Starting Points

| Goal | Filter | Reviews | TREC-COVID Ratio | Runtime Estimate |
|------|--------|---------|------------------|------------------|
| **Quick test** | Psychiatry ≥10 | 96,974 | 0.6x (smaller) | ~15-20 min/query |
| **TREC-COVID scale** | OB/GYN ≥20 or Family Med ≥20 | 216-258K | 1.3-1.5x (similar) | ~20-30 min/query |
| **Large experiment** | Primary Care ≥10 | 888,666 | 5x (larger) | ~1-2 hours/query |
| **Very large** | All doctors ≥20 | 2,809,640 | 16x (much larger) | ~3-5 hours/query |
| **Comprehensive** | All doctors ≥10 | 3,767,749 | 22x (massive) | ~5-8 hours/query |

**My recommendation for first experiment**:
- **Testing pipeline**: Start with **Psychiatry ≥10** (97K reviews, 0.6x TREC-COVID)
- **Actual experiments**: Use **OB/GYN ≥20** or **Family Medicine ≥20** (216-258K reviews, ~1.3-1.5x TREC-COVID) for similar scale and manageable runtimes
- **Comprehensive study**: Scale up to **Primary Care ≥10** (889K reviews, 5x TREC-COVID) if you have compute resources
