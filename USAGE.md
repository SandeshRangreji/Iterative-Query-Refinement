# Usage Guide — Configuration Reference

This is the detailed configuration reference for `src/end_to_end_evaluation.py` (the main pipeline — see [PIPELINE.md](PIPELINE.md) for how it relates to the other scripts in this repo, and [CACHING.md](CACHING.md) for cache behavior referenced throughout).

## Quick Start

### Local execution

```bash
cd /path/to/Iterative-Query-Refinement
conda activate coreset_proj
python src/end_to_end_evaluation.py
```

### SLURM/HPC execution (CLSP grid)

```bash
# Default (trec-covid, bertopic):
sbatch grid/run_search.sh

# Override dataset/topic model:
sbatch --export=ALL,DATASET=doctor-reviews,MODEL=lda grid/run_search.sh

# For TopicGPT, also export an API key:
sbatch --export=ALL,MODEL=topicgpt,OPENAI_API_KEY=sk-... grid/run_search.sh
```

`grid/run_search.sh` requests the **GPU partition** (1 GPU, 8 CPUs, 16GB) — GPU accelerates SBERT embedding generation but nothing here strictly requires it; CPU-only execution works, just slower. See [GRID_WORKFLOW.md](GRID_WORKFLOW.md) for all grid scripts, what each maps to, and account-specific setup.

---

## Configuration Reference

### Top-level parameters (edited in `main()`, or overridden via env var where noted)

All of this is in `main()` in `src/end_to_end_evaluation.py`:

```python
def main():
    # Dataset selection — override via DATASET env var
    DATASET_NAME = os.environ.get("DATASET", "trec-covid")   # "trec-covid" | "doctor-reviews"

    QUERY_IDS = dataset_config["query_ids"]   # all queries for the selected dataset by default
    SAMPLE_SIZE = 1000                        # documents per sample, fixed across all methods

    # Embedding models — TWO separate knobs, see "Why two embedding models?" below
    EMBEDDING_MODEL = "all-mpnet-base-v2"                # retrieval, BERTopic, KeyBERT
    METRICS_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"    # metric computation only

    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Topic model selection — override via MODEL env var
    TOPIC_MODEL_TYPE = os.environ.get("MODEL", "bertopic")   # "bertopic" | "lda" | "topicgpt"
    TOPIC_MODEL_PARAMS = { ... }   # model-specific, see "Topic Model Parameters" below

    DEVICE = "cpu"   # "cpu" | "cuda" | "mps"

    SAVE_TOPIC_MODELS = False            # True saves full ~420MB BERTopic models
    FORCE_REGENERATE_SAMPLES = False
    FORCE_REGENERATE_TOPICS = False
    FORCE_REGENERATE_EVALUATION = False

    OUTPUT_DIR = "results"     # see "Hardcoded paths" below — several places don't respect this
    CACHE_DIR = "cache"
    RANDOM_SEED = 42
```

**HiCode is not selected via `TOPIC_MODEL_TYPE`.** It's a separate entry point (`src/evaluate_hicode_all_queries.py`) that evaluates an externally-produced HiCode run rather than fitting a topic model itself — see PIPELINE.md's Stage 4b.

### Why two embedding models?

`EMBEDDING_MODEL` drives retrieval (BM25+SBERT hybrid search) and BERTopic's own clustering. `METRICS_EMBEDDING_MODEL` is used *only* when computing topic-query similarity, semantic diversity, and every other embedding-based metric — completely independent of what the topic model itself used internally.

This split exists because the two candidate models disagree substantially: BGE (`BAAI/bge-base-en-v1.5`) inflates raw cosine similarity by roughly 0.25-0.30 absolute points versus MPNet (`all-mpnet-base-v2`) on the same text pairs. Any metric with a hard similarity threshold (e.g. "query-relevant" = similarity ≥ 0.5) is sensitive to this — under MPNet, `random_uniform` samples almost never clear 0.5 similarity to the query, which biases paired significance tests toward a tiny, non-representative subset of queries. Running metrics under both embedding models and comparing is how that sensitivity gets caught rather than silently baked into a single number. Results from each model land in separate tagged subdirectories (`results/.../{metrics_tag}/` — see [CACHING.md](CACHING.md)) so neither overwrites the other.

### Sampling methods — what's configurable vs. hardcoded

All 7 methods are always run; there's no flag to select a subset. Most of their internal parameters (MMR λ, RRF fusion weights, candidate pool sizes) are **hardcoded inside each method's function body**, not exposed as `main()` constants — to change them you edit the method directly:

| Method | Function (file:line) | Key parameters (hardcoded unless noted) |
|---|---|---|
| `random_uniform` | `sample_random_uniform` (`src/end_to_end_evaluation.py:1416`) | none |
| `keyword_search` | `sample_keyword_search` (`:1550`) | BM25 only, `top_k=SAMPLE_SIZE` |
| `sbert` | `sample_sbert` (`:1580`) | SBERT only |
| `direct_retrieval` | `sample_direct_retrieval` (`:1447`) | Hybrid, `HybridStrategy.SIMPLE_SUM` |
| `direct_retrieval_mmr` | `sample_direct_retrieval_mmr` (`:1610`) | candidate pool = 5000, MMR `lambda_param=0.3` |
| `query_expansion` | `sample_query_expansion` (`:1478`) | RRF weights fixed 0.7 (baseline) / 0.3 split across keywords; reads keyword cache (path is a **constructor parameter**, `keyword_cache_path`, set per-dataset in `main()`'s `DATASET_CONFIGS`) |
| `retrieval_random` | `sample_retrieval_random` (`:1680`) | pool size = 5000 (constructor default parameter) |

BM25/SBERT/hybrid/MMR/cross-encoder parameters *are* exposed as named constants, but only in `src/search.py`'s own standalone `main()` (used for Stage 2 research, not by the main pipeline) — see PIPELINE.md. If you want to experiment with, say, a different MMR λ across the whole pipeline, you're editing `src/end_to_end_evaluation.py` directly.

### Topic Model Parameters

#### BERTopic (default)

```python
TOPIC_MODEL_TYPE = "bertopic"
TOPIC_MODEL_PARAMS = {
    # "min_cluster_size": 5,
    # "metric": "euclidean",
    # "cluster_selection_method": "eom",
    # "min_df": 2, "ngram_range": (1, 2), "max_features": 10000
}
```
Auto-determines topic count via HDBSCAN; some documents become outliers (topic = -1); variable document coverage; requires `EMBEDDING_MODEL`; ~5-8 min/query; ~420MB/saved model.

#### LDA

```python
TOPIC_MODEL_TYPE = "lda"
TOPIC_MODEL_PARAMS = {
    "n_topics": "auto",       # matches BERTopic's discovered count per method, or pass an int
    "alpha": "symmetric", "eta": 0.01, "passes": 15, "iterations": 100,
    "random_state": 42, "workers": 20,
    "min_df": 2, "ngram_range": (1, 2), "max_features": 10000
}
FORCE_REGENERATE_TOPICS = True     # samples are shared with BERTopic; only topics need regenerating
FORCE_REGENERATE_EVALUATION = True
```
`"auto"` reads `n_topics` per method from `results/{dataset}/bertopic/query_{id}/results/{metrics_tag}/per_method_summary.csv` (falls back to 20 if not found) — this is what makes LDA-vs-BERTopic an apples-to-apples topic-count comparison. Bag-of-words (CPU-only, no embeddings); 100% document coverage (no outliers); ~2-5 min/query, faster than BERTopic.

| `eta` | Effect |
|---|---|
| `0.01` | Focused, domain-specific topics (recommended for biomedical/technical text) |
| `None` | Moderate sparsity (general text) |
| `1.0` | Broad, less focused topics |

#### TopicGPT

```python
TOPIC_MODEL_TYPE = "topicgpt"
TOPIC_MODEL_PARAMS = {
    "generation_model": "gpt-4o-mini", "assignment_model": "gpt-4o-mini",
    "generation_sample_size": 500,
    "min_df": 2, "ngram_range": (1, 2), "max_features": 10000, "verbose": True
}
FORCE_REGENERATE_TOPICS = True
FORCE_REGENERATE_EVALUATION = True
```
Requires `export OPENAI_API_KEY="sk-..."`. Two-stage: generation (500 docs) → assignment (all docs). CPU-only (sentence-transformers on CPU); 100% coverage; ~10-30 min/query depending on rate limits; **incurs real API costs** — `gpt-4o-mini`/`gpt-4o-mini` is ~$1.50/sample (~$135 for a full 15-query × 6-method-ish run). See git history / RESULTS_REPORT.md for cost-vs-quality notes if switching to `gpt-4o` or `gpt-4-turbo`.

#### HiCode

Not configured here — see `src/evaluate_hicode_all_queries.py` and [GRID_WORKFLOW.md](GRID_WORKFLOW.md). It evaluates a pre-existing external HiCode run rather than fitting anything itself, so there's no `TOPIC_MODEL_PARAMS` equivalent — the tunable is whichever HiCode run's output directory you point it at.

### Hardcoded paths — must change if you're not running as the original author

These are baked into the Python source as literal strings, not exposed via env var or `main()` config. If you're running under a different account (or a different grid), you need to edit these directly:

| File:line | Current value | What it is |
|---|---|---|
| `src/end_to_end_evaluation.py:5411`, `:5415` (inside `DATASET_CONFIGS`) | `/home/srangre1/cache/keywords/*.json` | Keyword cache paths per dataset (Stage 3 output) |
| `generate_doctor_review_keywords.py` (`CORPUS_PATH`, `OUTPUT_PATH`) | `/home/srangre1/datasets/...`, `/home/srangre1/cache/keywords/...` | Doctor-reviews corpus input + keyword cache output |
| `create_filtered_doctor_corpus.py` | corpus input/output paths | Check the script's top-level constants before running |
| `src/evaluate_hicode_all_queries.py` (argparse defaults: `--hicode-dir`, `--samples-dir`, `--output-dir`) | `/export/fs06/mzhong8/...`, `/home/srangre1/results/...` | HiCode external input + output location — override with CLI flags, don't rely on defaults |
| `regenerate_diversity_plots.py` | `/home/srangre1/results/trec-covid/...` | Legacy utility, stale paths regardless — see GRID_WORKFLOW.md |
| `grid/*.sh` | `CONFIGURATION` block at top of each script | See [GRID_WORKFLOW.md](GRID_WORKFLOW.md) |

`OUTPUT_DIR = "results"` in `end_to_end_evaluation.py`'s `main()` is a *relative* path — where it resolves to depends on your working directory when you launch the script. The results referenced throughout this project's reports live under `/home/srangre1/results` (i.e. the script was launched from `/home/srangre1/`, not from inside the repo). If you run from inside the repo instead, you'll get a separate, disconnected `results/` tree inside it — don't assume the two are the same or in sync.

---

## Output Structure

```
results/{dataset}/{topic_model}/query_{id}/
├── config.json
├── samples/{method}.pkl                         # shared across topic models (see CACHING.md)
├── topic_models/{method}_results.pkl            # always saved
├── topic_models/{method}_model.pkl              # only if SAVE_TOPIC_MODELS=True
└── results/{metrics_tag}/
    ├── per_method_summary.csv
    ├── pairwise_metrics.csv
    ├── topics_summary/{method}_topics.{txt,json}
    └── plots/                                   # ~20+ plots incl. Wilcoxon + coverage heatmaps

results/{dataset}/{topic_model}/aggregate_results/{metrics_tag}/
├── statistical_tests/                           # Wilcoxon + t-test CSVs, coverage-significance CSVs, heatmaps
├── per_method_aggregates.json
└── plots/
```

`{metrics_tag}` = `METRICS_EMBEDDING_MODEL` with `/` and `-` replaced by `_` (e.g. `BAAI_bge_base_en_v1.5`). See [CACHING.md](CACHING.md) for the full cache-tier breakdown this structure sits on top of.

---

## Statistical Significance Testing

Cross-query aggregate comparisons run two paired tests per method-pair, per metric:

- **Wilcoxon signed-rank (primary)** — `wilcoxon_test()`, BH-FDR corrected. This is the number to lead with in any significance claim.
- **Paired t-test (secondary/backup)** — kept for continuity with earlier analysis, also BH-FDR corrected, plotted with a "secondary" label.

Both use **listwise NaN deletion** per method-pair (queries where either method's metric is NaN are dropped from that specific test) and flag `bias_warning=True` in the output CSV when this reduces the query count below the total — check that flag before trusting a significance result computed on a shrunken query set (this is exactly the failure mode that makes MPNet-based Relevant-Topic-Diversity claims unreliable; see METRICS_GUIDE.md).

A separate test, `run_pairwise_coverage_statistical_tests()`, applies the same Wilcoxon+BH treatment specifically to the "Relevant Coverage A→B" pairwise metrics (METRICS_GUIDE.md #30-32), symmetrized so every method is evaluated as both a coverage "source" and "target," with its own per-target heatmaps (`coverage_wilcoxon_significance_at07_{target}.png`).

Aggregate testing covers 11 metrics by default (see `run_aggregate_statistical_analysis()`'s `metrics_to_test` list) — topic-query similarity, semantic diversity, topic specificity, plus the full relevant-topic-diversity family.

---

## Common Workflows

### Force regenerate everything (clean run)
```python
FORCE_REGENERATE_SAMPLES = True
FORCE_REGENERATE_TOPICS = True
FORCE_REGENERATE_EVALUATION = True
```

### Iterative metric development (don't re-run sampling/topic-modeling)
```python
FORCE_REGENERATE_SAMPLES = False
FORCE_REGENERATE_TOPICS = False
FORCE_REGENERATE_EVALUATION = True
```

### Switching topic models mid-project
Samples are shared automatically (see CACHING.md's cross-model fallback) — set `FORCE_REGENERATE_SAMPLES = False`, `FORCE_REGENERATE_TOPICS = True` when moving from e.g. BERTopic to LDA. Results land in separate, non-conflicting directories per model type.

### Comparing across the two embedding models
Run once with default `METRICS_EMBEDDING_MODEL`, then again after changing it — both land in separate `{metrics_tag}` subdirectories automatically, nothing to clean up between runs.

---

## Dependencies

```bash
conda create -n coreset_proj python=3.11
conda activate coreset_proj
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

See `requirements.txt` for pinned versions (BERTopic, sentence-transformers, gensim, topicgpt_python, rank-bm25, keybert, etc.). GPU acceleration (optional): `cupy-cuda11x` is already listed; add `cuml-cu11` for GPU-accelerated HDBSCAN clustering if desired.

---

## Troubleshooting

**Missing keyword cache** (`FileNotFoundError: Keyword cache not found`) — the path itself is missing entirely (different from the graceful per-query fallback described in CACHING.md, which only triggers when the *file* exists but a specific query_id key doesn't). Check the path is correct for your account (see "Hardcoded paths" above) and that the corresponding Stage 3 generation script has been run.

**BERTopic: "HDBSCAN could not find any clusters"** — sample too small/homogeneous. Handled gracefully (method skipped, warning logged, evaluation continues) — usually not an error, just means that method has no result for that query.

**Low document overlap → ARI/NMI near 0 or NaN** — expected; retrieval-based and random samples are mostly disjoint by construction, so document-overlap clustering-agreement metrics are naturally weak. Not a bug.

**CUDA out of memory** — reduce SBERT `batch_size` (hardcoded in `search.py`'s index-building calls), set `DEVICE = "cpu"`, or reduce `SAMPLE_SIZE`.

**Results look suspicious (all methods identical, alignment always 0)** — check `samples/*.pkl` and `topics_summary/*.txt` directly; force-regenerate everything; confirm `QUERY_IDS` are valid for the selected `DATASET_NAME`. Also check whether you're looking at a doctor-reviews `query_expansion` result for queries 7-11 — see the keyword-cache caveat in CACHING.md before assuming a bug.

---

## Related Documentation

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) — research question, experimental design
- [PIPELINE.md](PIPELINE.md) — how all the scripts fit together, which stage does what
- [CACHING.md](CACHING.md) — full cache-tier reference and invalidation footguns
- [GRID_WORKFLOW.md](GRID_WORKFLOW.md) — running on the CLSP SLURM grid
- [METRICS_GUIDE.md](METRICS_GUIDE.md) — metric definitions and interpretation
- [README.md](README.md) — project front door and full doc index

## Last Updated

2026-09 — Documented the dual-embedding-model split, Wilcoxon/coverage significance testing, HiCode, `sbert`/doctor-reviews, `DATASET`/`MODEL` env vars, and the hardcoded-path inventory. Superseded the 2025-11-13 revision, which predated all of the above.
