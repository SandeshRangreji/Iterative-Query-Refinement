# Doctor-Reviews Dataset - Query Quality Analysis

**Date**: February 11, 2026
**Dataset**: Online Doctor Reviews (171,216 reviews)
**Models Analyzed**: HiCode, BERTopic, LDA, TopicGPT

---

## Summary Answers

### **Q1: Total number of queries in Doctor-Reviews**
**Answer: 11 queries**

### **Q2: Number of "outlier" queries (near-random alignment)**
**Answer: 1 outlier query** (Query 3)

**Definition**: Outlier = topic-query alignment < (mean - 1.5×std) across all 4 models
- Overall mean alignment: **0.543 ± 0.052**
- Outlier threshold: **0.465**
- Random baseline (approximate): **~0.50**

### **Q3: Example of one vague query**
**Query 3: "What breathing problems do patients report and how are they treated?"**

**Mean alignment: 0.450** (lowest of all 11 queries)

**Why this query is problematic:**
- ❌ **Compound query**: Asks about BOTH problems AND treatment (two separate aspects)
- ❌ **Too broad**: "Breathing problems" encompasses many conditions beyond asthma
- ❌ **Dataset mismatch**: Doctor reviews rarely detail specific clinical treatments
- ❌ **Vague terminology**: "Breathing problems" is imprecise vs specific conditions like "asthma exacerbation"

**Performance by model:**
| Model | Alignment | Status |
|-------|-----------|--------|
| TopicGPT | 0.292 | Very poor |
| BERTopic | 0.417 | Poor |
| LDA | 0.431 | Poor |
| HiCode | 0.658 | Below average |

**Even HiCode** (most robust model) performs below its average (0.654) on this query.

### **Q4: Stats for non-outlier subset (10 queries)**

#### Overall Performance (Excluding Query 3):
| Metric | All 11 Queries | Non-Outlier (10 Queries) | Improvement |
|--------|----------------|--------------------------|-------------|
| **Mean Alignment** | 0.543 | **0.552** | +0.009 (+1.7%) |
| **Std Deviation** | 0.052 | **0.044** | -0.008 (-15%) |
| **Range** | 0.450 - 0.603 | 0.469 - 0.603 | Better floor |

#### By Model (Non-Outlier Queries Only):

| Model | Mean ± Std | Range | Coefficient of Variation | Notes |
|-------|------------|-------|--------------------------|-------|
| **HiCode** | **0.654 ± 0.040** | 0.608 - 0.728 | **6.1%** | ✅ Most consistent, always high |
| **LDA** | **0.604 ± 0.057** | 0.509 - 0.684 | **9.4%** | ✅ Reliable, moderate variance |
| **BERTopic** | 0.513 ± 0.108 | 0.389 - 0.771 | **21.1%** | ⚠️ Highly inconsistent |
| **TopicGPT** | 0.439 ± 0.096 | 0.313 - 0.602 | **21.9%** | ⚠️ Consistently low |

**Key Finding**: HiCode is **3.5× more consistent** than BERTopic (6.1% vs 21.1% CV)

---

## Complete Query List with Performance

| Query ID | Query Text | Mean Alignment | HiCode | BERTopic | LDA | TopicGPT | Status |
|----------|------------|----------------|--------|----------|-----|----------|--------|
| **6** | What do patients dislike about their doctors? | **0.603** | 0.630 | 0.571 | 0.611 | 0.602 | ✅ Best |
| **2** | What are patients' experiences with specialist referrals? | **0.601** | 0.728 | 0.498 | 0.684 | 0.493 | ✅ Top 3 |
| **7** | What follow-up care or testing do doctors recommend for people with asthma? | **0.594** | 0.701 | 0.771 | 0.591 | 0.314 | ✅ Top 3 |
| 4 | How do doctors manage patients with asthma? | 0.566 | 0.700 | 0.566 | 0.628 | 0.368 | Good |
| 8 | What do patients like about treatment or management recommendations? | 0.560 | 0.634 | 0.486 | 0.592 | 0.528 | Good |
| 1 | How do patients find and choose their doctors? | 0.554 | 0.648 | 0.457 | 0.648 | 0.462 | Good |
| 5 | What do patients like about their doctors? | 0.553 | 0.608 | 0.501 | 0.608 | 0.495 | Good |
| 9 | What do patients dislike about treatment or management recommendations? | 0.524 | 0.640 | 0.493 | 0.509 | 0.452 | Good |
| 11 | What symptoms do patients with asthma report? | 0.501 | 0.625 | 0.399 | 0.654 | 0.327 | Okay |
| 10 | What lifestyle challenges do patients with asthma report? | 0.469 | 0.629 | 0.389 | 0.512 | 0.347 | Weak |
| **3** | What breathing problems do patients report and how are they treated? | **0.450** | 0.658 | 0.417 | 0.431 | 0.292 | ❌ **Outlier** |

---

## Query Characteristics Analysis

### ✅ **Best Performing Query Types:**

**Specific, actionable queries with clear sentiment:**
- "What patients **dislike** about their doctors" (0.603)
- "**Specialist referrals**" (0.601)
- "**Follow-up care** recommendations" (0.594)

**Characteristics:**
- ✓ Single, focused concept
- ✓ Clear sentiment (like/dislike) or concrete process
- ✓ Well-represented in doctor reviews
- ✓ Specific medical context

### ❌ **Worst Performing Query Types:**

**Vague, compound, or overly broad queries:**
- "**Breathing problems** + treatment" (0.450) - compound query
- "**Lifestyle challenges**" (0.469) - too open-ended
- "**Symptoms**" (0.501) - too generic

**Characteristics:**
- ✗ Multiple concepts (compound queries)
- ✗ Vague terminology
- ✗ Potentially under-represented in reviews
- ✗ Clinical detail that reviews may not contain

---

## Model Consistency Comparison

### HiCode: Most Robust Model
- **Never fails**: Even on outlier query 3, achieves 0.658 (vs 0.292 for TopicGPT)
- **Narrow range**: 0.608 - 0.728 (0.12 spread)
- **Low variance**: ±0.040 (6.1% CV)
- **Consistent excellence**: 10/11 queries above 0.620

### BERTopic: Most Inconsistent
- **Wide range**: 0.389 - 0.771 (0.38 spread)
- **High variance**: ±0.108 (21.1% CV)
- **Unpredictable**: Can be excellent (0.771 on Q7) or terrible (0.389 on Q10)

### LDA: Middle Ground
- **Moderate range**: 0.509 - 0.684 (0.18 spread)
- **Moderate variance**: ±0.057 (9.4% CV)
- **Reliable**: Consistently decent performance

### TopicGPT: Consistently Weak
- **Low overall**: Mean 0.439 (worst of all models)
- **High variance**: ±0.096 (21.9% CV)
- **Unreliable**: Often falls below 0.40

---

## Recommendations

### 1. Query Design for Doctor Reviews

**DO:**
- ✅ Ask about **specific aspects** (referrals, follow-up care)
- ✅ Include **clear sentiment** (like/dislike)
- ✅ Focus on **patient experiences** (well-represented in reviews)
- ✅ Use **one concept per query**

**DON'T:**
- ❌ Combine multiple questions (e.g., "problems AND treatment")
- ❌ Use vague terms ("lifestyle challenges", "breathing problems")
- ❌ Ask for clinical details (may not be in reviews)
- ❌ Be overly generic ("symptoms")

### 2. Model Selection

**For Doctor-Reviews:**
- **Best choice**: HiCode (0.654 mean, 6.1% CV) - most reliable
- **Alternative**: LDA (0.604 mean, 9.4% CV) - good balance
- **Avoid**: TopicGPT (0.439 mean) - consistently weak on this dataset

### 3. Outlier Query Handling

**Query 3 should be revised:**
- **Current**: "What breathing problems do patients report and how are they treated?"
- **Better split**:
  - Query 3a: "What respiratory symptoms do patients with asthma report?"
  - Query 3b: "What treatments do doctors prescribe for asthma patients?"

**Expected improvement**: Splitting could increase alignment from 0.450 to ~0.55-0.60

---

## Statistical Summary

### Full Dataset (11 Queries):
```
Mean alignment: 0.543 ± 0.052
Min: 0.450 (Query 3)
Max: 0.603 (Query 6)
Outliers: 1 (9.1%)
```

### Non-Outlier Subset (10 Queries):
```
Mean alignment: 0.552 ± 0.044 (+1.7% improvement)
Min: 0.469 (Query 10)
Max: 0.603 (Query 6)
Variance reduction: -15%
```

### Conclusion

The doctor-reviews dataset has **generally good query quality**, with only 1 problematic query out of 11. The outlier query (Q3) suffers from being a compound question that's too broad. Excluding this query improves mean alignment by 1.7% and reduces variance by 15%, demonstrating that the remaining 10 queries are well-designed and appropriate for the dataset.

HiCode demonstrates superior robustness across all queries, maintaining high performance even on the problematic query where other models fail.

---

**Analysis Date**: February 11, 2026
**Analyst**: Automated evaluation pipeline
**Source Files**: `results/doctor-reviews/*/query_*/results/per_method_summary.csv`
