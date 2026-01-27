# Family Medicine ≥20 Reviews Filter Analysis

This directory contains scripts to analyze the **Family Medicine ≥20 reviews** filter for reducing the doctor review dataset to ~200K reviews.

## Quick Start

### Run the Analysis

```bash
cd /home/srangre1
sbatch analyze_family_medicine.sh
```

### Check Progress

```bash
# Check job status
squeue -u $USER

# View output in real-time (replace JOBID with actual job ID)
tail -f family_med_analysis_JOBID.out

# View errors (if any)
tail -f family_med_analysis_JOBID.err
```

### View Results

After the job completes:

```bash
# View the detailed output log
cat family_med_analysis_*.out

# View the JSON summary
cat ~/Iterative-Query-Refinement/family_medicine_20plus_analysis.json
```

---

## What This Analysis Does

The script performs a comprehensive analysis of the **Family Medicine Physician with ≥20 reviews** filter:

### 1. **Filter Statistics**
- Total doctors after filtering
- Total reviews (expected vs actual)
- Review distribution statistics
- Comparison to original dataset size

### 2. **Doctor Distribution**
- Breakdown by review count buckets (20-29, 30-49, 50-99, etc.)
- Percentage of doctors in each bucket
- Percentage of reviews in each bucket

### 3. **Demographics**
- Gender distribution
- Credential distribution (MD, DO, etc.)
- Geographic distribution (top ZIP codes)

### 4. **Corpus Verification**
- Loads actual review corpus
- Verifies expected review count matches actual
- Checks data integrity

### 5. **Review Text Analysis**
- Text length statistics (characters and words)
- Sample reviews for inspection
- Column name detection

### 6. **Comparison to TREC-COVID**
- Size comparison (ratio)
- Estimated runtime per query
- Suitability assessment

### 7. **Quality Assessment**
- Overall quality score
- Recommendations for use

---

## Expected Output

The analysis generates:

1. **Console Output** (in `.out` file):
   - Detailed statistics and breakdowns
   - Sample reviews
   - Quality assessment
   - Final recommendations

2. **JSON Summary** (`family_medicine_20plus_analysis.json`):
   - Structured data for programmatic access
   - Key metrics and statistics
   - Easy to parse for downstream use

---

## Key Metrics to Look For

When reviewing the output, pay attention to:

1. **Total Reviews**: Should be around 257K (close to 200K target)
2. **Doctor Count**: Should be around 6,283 doctors
3. **Review Distribution**: Check if most doctors are in the 20-50 review range
4. **Text Length**: Verify reviews have reasonable length (not too short)
5. **Quality Score**: Should be 2/3 or 3/3 for suitability
6. **TREC-COVID Ratio**: Should be around 1.5x

---

## File Descriptions

### Analysis Script
- **`analyze_family_medicine_filter.py`**
  - Python script that performs the analysis
  - Loads metadata and corpus
  - Generates statistics and summary

### Shell Script
- **`analyze_family_medicine.sh`**
  - SLURM batch script
  - Requests CPU resources (no GPU needed)
  - 32GB memory, 2-hour time limit
  - Runs on CPU partition

---

## Troubleshooting

### Job Fails with Memory Error
Increase memory in the shell script:
```bash
#SBATCH --mem=64G  # Increase from 32G to 64G
```

### Job Takes Too Long
The corpus loading step can take 5-10 minutes. This is normal.

### Can't Find Text Column
The script auto-detects common text column names:
- `text`
- `review_text`
- `content`
- `Review`
- `review`

If it fails, check the actual column names in the corpus.

### Different Review Count Than Expected
Small differences (<1%) are normal due to:
- Rounding in metadata
- Missing reviews in corpus
- Data processing artifacts

---

## Next Steps After Analysis

Once you verify the filter looks good:

1. **Create the filtered corpus** using the script in the guide
2. **Save it to disk** for reuse
3. **Adapt the topic modeling pipeline** to use the new corpus
4. **Design healthcare-relevant queries** for Family Medicine topics

---

## Filter Details

**Filter Criteria:**
```python
Specialty == 'Family Medicine Physician' AND num_reviews >= 20
```

**Expected Results:**
- ~257K reviews
- ~6,283 doctors
- ~1.5x TREC-COVID size
- High quality (doctors with established reputations)

**Domain:**
- Primary care
- Chronic disease management
- Preventive care
- General health
- Patient experience themes

---

## Questions This Analysis Answers

1. ✅ How many reviews will we have after filtering?
2. ✅ How does this compare to our 200K target?
3. ✅ What's the quality of the filtered data?
4. ✅ How are reviews distributed across doctors?
5. ✅ What's the demographic breakdown?
6. ✅ How long are the reviews (text length)?
7. ✅ How does this compare to TREC-COVID?
8. ✅ What's the estimated runtime for experiments?
9. ✅ Is this filter suitable for topic modeling?

---

## Contact & Support

If you encounter issues:
1. Check the `.err` file for error messages
2. Verify the conda environment is activated
3. Ensure the corpus path is correct: `/export/fs06/mzhong8/doctor_review_corpus`
4. Check available memory with `squeue -u $USER`
