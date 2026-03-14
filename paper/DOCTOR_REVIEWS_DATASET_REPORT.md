# Doctor Reviews Dataset Report

This document provides a comprehensive overview of the Online Doctor Reviews dataset and the filtered Family Medicine subset used for topic modeling experiments.

---

## Executive Summary

| Aspect | Full Dataset | Filtered Subset |
|--------|-------------|-----------------|
| **Total Documents** | 4,991,218 reviews | 171,216 reviews |
| **Total Doctors** | 492,565 | 5,986 |
| **Domain** | All medical specialties | Family Medicine only |
| **Comparison to TREC-COVID** | 29x larger | ~1.0x (same scale) |
| **Data Quality** | Mixed (user-generated) | High (9 quality filters applied) |

---

## Part 1: Full Dataset Overview

### Source Information

- **Dataset Name**: Online Doctor Reviews
- **Metadata Path**: `/export/fs06/mzhong8/doctor_review_metadata`
- **Corpus Path**: `/export/fs06/mzhong8/doctor_review_corpus`
- **Join Key**: `PhyID` column
- **Content Type**: Patient-written reviews of healthcare providers

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Doctors | 492,565 |
| Total Reviews | 4,991,218 |
| Average Reviews per Doctor | 10.1 |
| Median Reviews per Doctor | 4 |
| Maximum Reviews for Single Doctor | 5,643 |

### Review Distribution (Highly Skewed)

| Percentile | Reviews per Doctor |
|------------|-------------------|
| 10th | 1 |
| 25th | 2 |
| 50th (Median) | 4 |
| 75th | 10 |
| 90th | 22 |
| 95th | 36 |
| 99th | 98 |

**Key Insight**: The distribution is extremely skewed. Most doctors have very few reviews, making minimum review count filtering essential for quality topic modeling.

### Physician Categories

| Category | Doctors | Percentage |
|----------|---------|------------|
| Specialty | 271,581 | 55.1% |
| Primary Care | 181,323 | 36.8% |
| Super Specialties | 39,661 | 8.1% |

### Top 10 Specialties by Doctor Count

| Rank | Specialty | Doctors | % of Total | Total Reviews |
|------|-----------|---------|------------|---------------|
| 1 | Family Medicine | 76,033 | 15.4% | 613,243 |
| 2 | Internal Medicine | 74,116 | 15.0% | 567,465 |
| 3 | OB/GYN | 26,913 | 5.5% | 355,333 |
| 4 | Pediatrics | 26,150 | 5.3% | 148,115 |
| 5 | Psychiatry | 21,280 | 4.3% | 149,341 |
| 6 | Orthopaedic Surgery | 18,268 | 3.7% | - |
| 7 | Surgery | 17,602 | 3.6% | - |
| 8 | Emergency Medicine | 17,314 | 3.5% | - |
| 9 | Cardiovascular Disease | 14,659 | 3.0% | - |
| 10 | Ophthalmology | 13,827 | 2.8% | - |

### Demographics

**Gender Distribution:**
| Gender | Doctors | Percentage |
|--------|---------|------------|
| Male | 334,254 | 67.9% |
| Female | 158,311 | 32.1% |

**Credential Distribution:**
| Credential | Doctors | Percentage |
|------------|---------|------------|
| MD | 437,930 | 88.9% |
| DO | 44,883 | 9.1% |
| MD,PHD | 3,105 | 0.6% |
| MD,MPH | 1,624 | 0.3% |
| Other | ~2,000 | 0.4% |

### Metadata Schema

```
Columns: ['PhyID', 'NPI', 'FirstName', 'LastName', 'Gender', 'Credential',
          'Specialty', 'PracticeZip5', 'BusinessZip5', 'biography_doc',
          'education_doc', 'num_reviews', 'DocName', 'PhysicianType']
```

### Null Value Analysis

| Field | Missing Count | Percentage |
|-------|---------------|------------|
| biography_doc | 207,732 | 42.2% |
| education_doc | 149,782 | 30.4% |
| PracticeZip5 | 46,953 | 9.5% |
| BusinessZip5 | 47,459 | 9.6% |
| Specialty | 5,522 | 1.1% |
| Names/DocName | <15 | <0.01% |

---

## Part 2: Filtered Subset - Family Medicine Quality Corpus

### Filter Configuration

The filtered subset applies **9 multi-level filters** to create a high-quality corpus suitable for topic modeling:

#### Doctor-Level Filters (3)

| Filter | Criteria | Purpose |
|--------|----------|---------|
| 1. Specialty | `Specialty == 'Family Medicine Physician'` | Focus on single domain |
| 2. Min Reviews | `num_reviews >= 20` | Ensure established doctors |
| 3. Max Reviews | `num_reviews <= 100` | Remove outlier super-popular doctors |

#### Review-Level Quality Filters (6)

| Filter | Criteria | Purpose |
|--------|----------|---------|
| 4. Min Words | `word_count >= 15` | Remove too-short reviews |
| 5. Max Words | `word_count <= 200` | Remove extremely long reviews |
| 6. No All Caps | `NOT text.isupper()` | Remove spam/shouting |
| 7. Low Digits | `digit_ratio < 30%` | Remove phone numbers, dates |
| 8. Char Diversity | `unique_chars >= 10` | Remove repetitive text |
| 9. ASCII Focus | `non_ascii_ratio < 50%` | Remove encoding issues/non-Latin scripts |

### Filtered Dataset Statistics

| Metric | Value |
|--------|-------|
| **Final Review Count** | 171,216 |
| **Doctors Included** | 5,986 |
| **Pre-Quality Filter Reviews** | 210,136 |
| **Reviews Removed by Quality Filters** | 38,920 (18.5%) |
| **Ratio to TREC-COVID** | ~1.0x (comparable scale) |

### Filter Pipeline Results

```
Step 1: Original corpus                           → 4,991,218 reviews
Step 2: Specialty = Family Medicine               → (filtered doctors)
Step 3: Review count >= 20                        → (filtered doctors)
Step 4: Review count <= 100                       → 5,986 doctors
Step 5: Filter corpus by PhyIDs                   → 210,136 reviews
Step 6: Word count >= 15                          → (quality filter)
Step 7: Word count <= 200                         → (quality filter)
Step 8: NOT all caps                              → (quality filter)
Step 9: Digit ratio < 30%                         → (quality filter)
Step 10: Character diversity >= 10                → (quality filter)
Step 11: Non-ASCII ratio < 50%                    → (quality filter)
═══════════════════════════════════════════════════════════════════
FINAL:                                            → 171,216 reviews
```

### Doctor Distribution by Review Count

*Note: Only doctors with 20-100 reviews are included (≤100 filter applied).*

| Bucket | Doctors | % of Doctors | Reviews | % of Reviews |
|--------|---------|--------------|---------|--------------|
| 20-29 | 3,099 | 51.8% | 73,385 | 34.9% |
| 30-49 | 1,895 | 31.7% | 70,890 | 33.7% |
| 50-100 | 992 | 16.6% | 65,861 | 31.3% |
| **Total** | **5,986** | 100% | **210,136** | 100% |

### Text Statistics

| Metric | Value |
|--------|-------|
| Average Length (chars) | ~363 |
| Median Length (chars) | ~243 |
| Average Word Count | ~66 |
| Median Word Count | ~44 |

---

## Part 3: Output Format & Location

### Filtered Dataset Location

```
datasets/doctor_reviews_family_med_filtered/
├── data-00000-of-00001.arrow    # 58.5 MB
├── dataset_info.json
└── state.json
```

### Schema (TREC-COVID Compatible)

```json
{
  "features": {
    "_id": {"dtype": "string"},
    "title": {"dtype": "string"},   // Empty for compatibility
    "text": {"dtype": "string"}
  }
}
```

### Loading the Dataset

```python
from datasets import load_from_disk

corpus = load_from_disk('datasets/doctor_reviews_family_med_filtered')
print(f"Loaded {len(corpus)} reviews")
# Output: Loaded 171,216 reviews
```

---

## Part 4: Queries for Topic Modeling

Six manually-designed queries focused on patient experience themes:

| ID | Query Text |
|----|------------|
| 1 | How do patients find and choose their doctors? |
| 2 | What are patients' experiences with specialist referrals? |
| 3 | What breathing problems do patients report and how are they treated? |
| 4 | How do doctors manage patients with asthma? |
| 5 | What do patients like about their doctors? |
| 6 | What do patients dislike about their doctors? |

### Query Categories

- **Doctor Selection** (Query 1): Patient decision-making process
- **Care Coordination** (Query 2): Referral experiences
- **Medical Conditions** (Queries 3-4): Specific health topics (respiratory)
- **Satisfaction Analysis** (Queries 5-6): Positive/negative experiences

---

## Part 5: Comparison with TREC-COVID

| Aspect | TREC-COVID | Doctor Reviews (Filtered) |
|--------|------------|---------------------------|
| **Size** | 171,332 documents | 171,216 reviews |
| **Ratio** | 1.0x | ~1.0x |
| **Domain** | Scientific/Medical | Healthcare/Patient Experience |
| **Text Type** | Abstracts | Reviews (user-generated) |
| **Text Length** | Medium-long | Short-medium |
| **Vocabulary** | Technical, formal | Informal, colloquial |
| **Quality** | High (peer-reviewed) | Mixed (filtered) |
| **Relevance Judgments** | Yes (QRELs) | No |
| **Ground Truth** | Available | Not available |


---

*Report generated: January 2026*
*Dataset creation date: January 19, 2026*
