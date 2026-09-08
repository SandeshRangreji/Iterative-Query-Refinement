# Iterative Query Refinement — Retrieval-Guided Topic Modeling

**Research question:** how does controlling document relevance via different retrieval methods affect downstream topic modeling (coherence, diversity, query alignment, topic structure)?

This is an exploratory, descriptive ablation study — see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the full research framing. It compares 7 document-sampling methods spanning a spectrum of relevance bias (random uniform → BM25 → SBERT → hybrid retrieval → hybrid+MMR → query expansion) across 4 topic modeling approaches (BERTopic, LDA, TopicGPT, HiCode) and 2 datasets (TREC-COVID, Doctor-Reviews), using 40+ metrics with Wilcoxon-signed-rank significance testing.

## Quick start

```bash
conda create -n coreset_proj python=3.11
conda activate coreset_proj
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

python src/end_to_end_evaluation.py    # runs the default config: trec-covid + bertopic
```

For the CLSP SLURM grid: `sbatch grid/run_search.sh` (see [GRID_WORKFLOW.md](GRID_WORKFLOW.md)). For every configuration knob — dataset/topic-model selection, the two separate embedding-model settings, per-sampling-method parameters, and which hardcoded paths you need to change for your own account — see [USAGE.md](USAGE.md).

## Project transition in progress

This project is transitioning from srangre1 to mzhong8. **[TRANSITION_PLAN.md](TRANSITION_PLAN.md)** is the checklist for that — which results/cache directory is authoritative (validated by timestamp/content, not assumption), the data transfer plan via `/export/fs06/`, and what mzhong8 needs to update on their end. Remove this section once the transition is complete.

## How the pieces fit together

This repo is not one script — it's five pipeline stages (corpus prep, indexing, search evaluation, keyword generation, and the main sampling/topic-modeling/evaluation run, plus a parallel HiCode track) implemented across ~10 scripts. **Start with [PIPELINE.md](PIPELINE.md)** before editing anything — it maps every script to its stage, shows the data flow between them, and has a "which script do I run for X" table.

## Documentation index

**Core — read these first:**

| Doc | What it covers |
|---|---|
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Research question, experimental design, interpretation framework |
| [PIPELINE.md](PIPELINE.md) | Architecture: every script, its stage, and how they connect |
| [USAGE.md](USAGE.md) | Full configuration reference — every flag, env var, and hardcoded path |
| [CACHING.md](CACHING.md) | What's cached, where, and what does/doesn't auto-invalidate |
| [METRICS_GUIDE.md](METRICS_GUIDE.md) | Definitions and interpretation for all 40+ metrics |
| [GRID_WORKFLOW.md](GRID_WORKFLOW.md) | Running on the CLSP SLURM grid, `grid/*.sh` scripts |
| [FUTURE_WORK.md](FUTURE_WORK.md) | Planned metrics/ablations, roadmap |

**Reference — dataset & infrastructure background (still accurate, narrower scope):**

| Doc | What it covers |
|---|---|
| [DOCTOR_REVIEW_FILTERING_GUIDE.md](DOCTOR_REVIEW_FILTERING_GUIDE.md) | The 9 filters applied to build the doctor-reviews corpus |
| [DOCTOR_REVIEWS_DATASET_REPORT.md](DOCTOR_REVIEWS_DATASET_REPORT.md) | Doctor-reviews dataset provenance and characteristics |
| [PRECISION_RECALL_FIX.md](PRECISION_RECALL_FIX.md) | Explains the asymmetric precision/recall heatmap semantics (referenced from METRICS_GUIDE.md) |
| [CLSP_GPU_SOLUTION.md](CLSP_GPU_SOLUTION.md), [GPU_COMPATIBILITY_ISSUE.md](GPU_COMPATIBILITY_ISSUE.md), [GPU_IMPLEMENTATION_GUIDE.md](GPU_IMPLEMENTATION_GUIDE.md) | CLSP grid GPU/CUDA setup troubleshooting |

**Historical analysis snapshots — point-in-time results, useful context but not live state (check `results/` directly for current numbers):**

`RESULTS_REPORT.md`, `RESULTS_ANALYSIS.md`, `FINDINGS_REPORT.md`, `FINDINGS_REPORT_BERTOPIC_ONLY.md`, `OUTLIER_QUERY_ANALYSIS_REPORT.md`, `HICODE_PRECISION_SUMMARY.md`, `HICODE_QUERY43_PRECISION_ANALYSIS.md`, `FAMILY_MEDICINE_ANALYSIS_README.md`, `RUN_BERTOPIC_EVAL_ONLY.md`

These were each accurate as of the date in their own header — treat them as "here's what we found when we looked, and why," not as current configuration or results. If a historical doc and a core doc above disagree, trust the core doc.

## Repo layout

```
src/                          # all pipeline code — see PIPELINE.md
grid/                         # CLSP SLURM submission scripts — see GRID_WORKFLOW.md
generate_doctor_review_keywords.py, create_filtered_doctor_corpus.py   # dataset-prep scripts, run once (root level, not src/)
cache/                        # search indices, keyword cache, IDF cache — see CACHING.md
results/                      # per-query and aggregate outputs (large, not versioned)
```
