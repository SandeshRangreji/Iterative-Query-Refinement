# Usage Guide - End-to-End Evaluation

## Quick Start

### Local Execution
```bash
cd /home/srangre1/Iterative-Query-Refinement
conda activate coreset_proj
python src/end_to_end_evaluation.py
```

### SLURM/HPC Execution
```bash
# Set environment variables (if needed)
export OPENAI_API_KEY="sk-..."  # For TopicGPT

# Submit job to SLURM
sbatch --export=ALL run_search.sh
```

The SLURM script (`run_search.sh`) runs on CPU partition with 24 cores and 32GB memory. No GPU required.

---

## Configuration

### Editing Parameters

All configuration is in the `main()` function of `src/end_to_end_evaluation.py`:

```python
def main():
    # Query configuration
    QUERY_IDS = ["2", "9", "10", ..., "48"]  # List of 15 queries or single query
    SAMPLE_SIZE = 1000                       # Documents per sample

    # Model configuration
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DATASET_NAME = "trec-covid"

    # Topic modeling configuration
    TOPIC_MODEL_TYPE = "bertopic"      # Options: "bertopic", "lda", "topicgpt"
    TOPIC_MODEL_PARAMS = {}            # Model-specific parameters (see below)

    # Device configuration
    DEVICE = "cpu"                     # or "cuda", "mps" for GPU

    # Caching configuration
    SAVE_TOPIC_MODELS = False          # Set True to save full models
    FORCE_REGENERATE_SAMPLES = False   # Force re-run sampling
    FORCE_REGENERATE_TOPICS = False    # Force re-run topic modeling
    FORCE_REGENERATE_EVALUATION = False  # Force re-compute metrics

    # Directory configuration
    OUTPUT_DIR = "results"
    CACHE_DIR = "cache"
    RANDOM_SEED = 42
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `QUERY_IDS` | ["2", "9", ..., "48"] | List of query IDs or single query string |
| `SAMPLE_SIZE` | 1000 | Number of documents to sample per method |
| `TOPIC_MODEL_TYPE` | "bertopic" | Topic model type: "bertopic", "lda", "topicgpt" |
| `TOPIC_MODEL_PARAMS` | {} | Model-specific parameters (see below) |
| `EMBEDDING_MODEL` | "all-mpnet-base-v2" | SentenceTransformer model for embeddings |
| `CROSS_ENCODER_MODEL` | "cross-encoder/ms-marco-MiniLM-L-6-v2" | Cross-encoder for reranking |
| `DATASET_NAME` | "trec-covid" | Dataset name (used for caching) |
| `DEVICE` | "cpu" | Device for embeddings ("cpu", "cuda", "mps") |
| `SAVE_TOPIC_MODELS` | False | Whether to save full topic models |
| `FORCE_REGENERATE_SAMPLES` | False | Force re-run document sampling |
| `FORCE_REGENERATE_TOPICS` | False | Force re-run topic modeling |
| `FORCE_REGENERATE_EVALUATION` | False | Force re-compute evaluation metrics |
| `OUTPUT_DIR` | "results" | Output directory for results |
| `CACHE_DIR` | "cache" | Cache directory for indices and models |
| `RANDOM_SEED` | 42 | Random seed for reproducibility |

---

## Switching Between Topic Models

The evaluation framework supports multiple topic modeling methods. All metrics and visualizations work identically across models.

### Available Topic Models

1. **BERTopic** - Embedding-based topic modeling with HDBSCAN clustering
2. **LDA** - Latent Dirichlet Allocation (bag-of-words, probabilistic)
3. **TopicGPT** - LLM-based topic modeling using OpenAI API

### Using BERTopic (Default)

```python
# In main() of src/end_to_end_evaluation.py

TOPIC_MODEL_TYPE = "bertopic"

# BERTopic parameters
TOPIC_MODEL_PARAMS = {
    # Using defaults that match current behavior
    # Optional overrides:
    # "min_cluster_size": 5,
    # "metric": "euclidean",
    # "cluster_selection_method": "eom",
    # "min_df": 2,
    # "ngram_range": (1, 2),
    # "max_features": 10000
}
```

**BERTopic Characteristics:**
- **Auto-determines topic count** using HDBSCAN clustering
- **Outliers**: Some documents marked as outliers (topic ID = -1)
- **Document Coverage**: Variable (typically 80-95%)
- **Requires**: Embeddings (uses `EMBEDDING_MODEL`)
- **Runtime**: ~5-8 min per query (1000 docs)
- **Model Size**: ~420 MB per saved model

---

### Using LDA (Latent Dirichlet Allocation)

```python
# In main() of src/end_to_end_evaluation.py

TOPIC_MODEL_TYPE = "lda"

# LDA parameters (biomedical-optimized for TREC-COVID)
TOPIC_MODEL_PARAMS = {
    "n_topics": "auto",       # Match BERTopic results, or specify int (e.g., 20)
    "alpha": "symmetric",     # Document-topic prior
    "eta": 0.01,              # Topic-word prior (sparse for focused topics)
    "passes": 15,             # Training passes
    "iterations": 100,        # Iterations per pass
    "random_state": 42,
    "workers": 20,            # Multi-core training (adjust for your system)
    # Vocabulary params (match BERTopic for fair comparison)
    "min_df": 2,
    "ngram_range": (1, 2),
    "max_features": 10000
}

# Force flags for LDA (samples are shared with BERTopic)
FORCE_REGENERATE_SAMPLES = False   # Use existing samples
FORCE_REGENERATE_TOPICS = True     # Generate LDA models
FORCE_REGENERATE_EVALUATION = True # Generate LDA results
```

**LDA Characteristics:**
- **Fixed topic count** (must specify or use "auto" to match BERTopic)
- **No outliers**: All documents assigned to topics (100% coverage)
- **Probabilistic**: Returns full topic distribution per document
- **Bag-of-words**: Does NOT use embeddings (CPU-only)
- **Runtime**: ~2-5 min per query (1000 docs) - **faster than BERTopic**
- **Model Size**: ~10-50 MB per saved model

**LDA-Specific: "auto" n_topics**

The special value `"auto"` makes LDA match BERTopic's discovered topic counts:

1. Reads: `/home/srangre1/results/trec-covid/bertopic/query_{id}/results/per_method_summary.csv`
2. Extracts `n_topics` for each sampling method
3. Uses those exact values for LDA training
4. **Result**: Perfect apples-to-apples comparison!

Falls back to `n_topics=20` if BERTopic results not found.

**LDA Hyperparameter Guide:**

| Parameter | Recommended | Alternative | Effect |
|-----------|-------------|-------------|--------|
| `n_topics` | `"auto"` | `20` | Match BERTopic or fixed count |
| `alpha` | `"symmetric"` | `0.01`, `1.0` | Document-topic sparsity |
| `eta` | `0.01` | `None`, `0.05` | Topic-word sparsity (0.01 = focused topics) |
| `passes` | `15` | `10`, `20` | More passes = better convergence |
| `iterations` | `100` | `50`, `200` | More iterations = better quality |
| `workers` | `20` | `4`, `os.cpu_count()-1` | Multi-core parallelization |

**When to use each value:**
- `eta=0.01`: **Biomedical/technical text** (focused, domain-specific topics)
- `eta=None`: General text (moderate sparsity)
- `eta=1.0`: Broad topics (less focused)
- `alpha=0.01`: Few topics per document
- `alpha=1.0`: Many topics per document

---

### Installation Requirements

**BERTopic** (included by default):
```bash
pip install bertopic==0.17.0
pip install sentence-transformers==3.4.1
```

**LDA** (requires Gensim):
```bash
pip install gensim==4.3.2
```

**TopicGPT** (requires OpenAI API):
```bash
pip install topicgpt_python>=0.1.0
export OPENAI_API_KEY="your-api-key"
```

Already added to `requirements.txt`.

---

### Using TopicGPT (LLM-Based Topic Modeling)

TopicGPT uses large language models to generate and assign topics. It provides more interpretable topic descriptions but requires an OpenAI API key and incurs API costs.

```python
# In main() of src/end_to_end_evaluation.py

TOPIC_MODEL_TYPE = "topicgpt"

# TopicGPT parameters
TOPIC_MODEL_PARAMS = {
    # Model selection (use cheaper models for cost efficiency)
    "generation_model": "gpt-4o-mini",    # Model for topic generation
    "assignment_model": "gpt-4o-mini",    # Model for topic assignment

    # Sampling parameters
    "generation_sample_size": 500,        # Documents to sample for topic generation

    # Vocabulary (match BERTopic/LDA)
    "min_df": 2,
    "ngram_range": (1, 2),
    "max_features": 10000,

    # Output
    "verbose": True
}

# Force flags for TopicGPT
FORCE_REGENERATE_SAMPLES = False   # Use existing samples
FORCE_REGENERATE_TOPICS = True     # Generate TopicGPT models
FORCE_REGENERATE_EVALUATION = True # Generate TopicGPT results
```

**Environment Setup:**
```bash
# Set API key (required)
export OPENAI_API_KEY="sk-..."

# Run evaluation
python src/end_to_end_evaluation.py
```

**TopicGPT Characteristics:**
- **LLM-generated topics**: Natural language topic descriptions via OpenAI API
- **Two-stage process**: Topic generation (500 docs) → Topic assignment (all docs)
- **CPU-only execution**: Uses sentence-transformers on CPU (no GPU required)
- **No outliers**: All documents assigned (100% coverage)
- **Requires**: OpenAI API key (OPENAI_API_KEY environment variable)
- **Runtime**: Varies by model and rate limits (~10-30 min per query)
- **Cost**: ~$1-15 per sample depending on models used

**Model Selection Guide:**

| Model | Input/1M | Output/1M | Best For |
|-------|----------|-----------|----------|
| `gpt-4o-mini` | $0.15 | $0.60 | **Recommended default** |
| `gpt-3.5-turbo` | $0.50 | $1.50 | Legacy cheap option |
| `gpt-4o` | $5.00 | $15.00 | Higher quality |
| `gpt-4-turbo` | $10.00 | $30.00 | Best quality |

**Cost Estimation (1000 docs per sample):**

| Configuration | Cost/Sample | 90 Samples (Full Run) |
|---------------|-------------|------------------------|
| gpt-4o-mini / gpt-4o-mini | ~$1.50 | ~$135 |
| gpt-4o / gpt-4o-mini | ~$8.00 | ~$720 |
| gpt-4-turbo / gpt-4o-mini | ~$12.00 | ~$1,080 |

**TopicGPT-Specific Outputs:**

TopicGPT results include additional fields:
- `topic_descriptions`: Natural language topic descriptions from LLM
- `assignment_quotes`: Supporting quotes from documents

These are saved in `topic_models/{method}_results.pkl` and can be accessed for qualitative analysis.

---

### Output Structure by Model Type

Results are saved separately by model type:

```
results/
└── trec-covid/
    ├── bertopic/              # BERTopic results
    │   └── query_2/
    │       ├── samples/
    │       ├── topic_models/
    │       └── results/
    │
    ├── lda/                   # LDA results
    │   └── query_2/
    │       ├── samples/       # Can share with BERTopic
    │       ├── topic_models/
    │       └── results/
    │
    └── topicgpt/              # TopicGPT results
        └── query_2/
            ├── samples/       # Can share with BERTopic/LDA
            ├── topic_models/
            └── results/
```

**Sample Sharing:**
- Document samples are **identical** across model types (determined by retrieval method)
- Set `FORCE_REGENERATE_SAMPLES = False` to reuse existing samples
- LDA will use BERTopic's samples if they exist

---

### Comparing Topic Models

Run multiple models on the same queries, then compare results:

```bash
# Step 1: Run BERTopic (if not already done)
# Edit main(): TOPIC_MODEL_TYPE = "bertopic"
python src/end_to_end_evaluation.py

# Step 2: Run LDA
# Edit main(): TOPIC_MODEL_TYPE = "lda"
python src/end_to_end_evaluation.py

# Step 3: Run TopicGPT
# Edit main(): TOPIC_MODEL_TYPE = "topicgpt"
# Set OPENAI_API_KEY environment variable
python src/end_to_end_evaluation.py

# Step 4: Compare results
# BERTopic: results/trec-covid/bertopic/query_2/results/per_method_summary.csv
# LDA:      results/trec-covid/lda/query_2/results/per_method_summary.csv
# TopicGPT: results/trec-covid/topicgpt/query_2/results/per_method_summary.csv
```

**Key Metrics to Compare:**

| Metric | BERTopic | LDA | TopicGPT |
|--------|----------|-----|----------|
| **NPMI Coherence** | Variable | Variable | Variable |
| **Embedding Coherence** | Usually highest (uses embeddings) | Lower (BoW) | Medium |
| **Semantic Diversity** | Usually highest (HDBSCAN) | Medium | Depends on LLM |
| **Document Coverage** | Variable (has outliers) | Always 1.0 | Always 1.0 |
| **Interpretability** | Word lists | Word lists | Natural language |
| **Cost** | Free (local) | Free (local) | API costs |
| **Speed** | Medium | Fast | Slow (API calls) |

---

### Switching Models Mid-Project

**Scenario**: Already ran BERTopic, now want to run LDA.

**Steps:**

1. **Keep existing BERTopic results** (don't regenerate)

2. **Configure LDA in `main()`:**
```python
TOPIC_MODEL_TYPE = "lda"
TOPIC_MODEL_PARAMS = {"n_topics": "auto", ...}  # Matches BERTopic
FORCE_REGENERATE_SAMPLES = False    # Reuse BERTopic samples
FORCE_REGENERATE_TOPICS = True      # Generate new LDA models
```

3. **Run evaluation:**
```bash
python src/end_to_end_evaluation.py
```

4. **Results saved separately:**
   - BERTopic: `results/trec-covid/bertopic/`
   - LDA: `results/trec-covid/lda/`

**No conflicts** - models are completely isolated.

---

## Common Workflows

### 1. Run Evaluation for Different Query

Edit `QUERY_ID` in `main()`:

```python
QUERY_ID = "1"  # Change to desired query ID (1-50 available in TREC-COVID)
```

Then run:
```bash
python src/end_to_end_evaluation.py
```

Results saved to: `results/topic_evaluation/query_1/`

---

### 2. Force Regenerate Everything (Clean Run)

Set all force flags to `True`:

```python
FORCE_REGENERATE_SAMPLES = True
FORCE_REGENERATE_TOPICS = True
FORCE_REGENERATE_EVALUATION = True
```

Use case: Modified sampling logic or BERTopic configuration

---

### 3. Iterative Metric Development

When developing new metrics, avoid re-running expensive sampling/modeling:

```python
FORCE_REGENERATE_SAMPLES = False      # Keep cached samples
FORCE_REGENERATE_TOPICS = False       # Keep cached topic models
FORCE_REGENERATE_EVALUATION = True    # Re-compute metrics only
```

Workflow:
1. Modify metric computation code in `end_to_end_evaluation.py`
2. Set `FORCE_REGENERATE_EVALUATION = True`
3. Re-run script
4. Check updated results in `results/topic_evaluation/query_X/`

---

### 4. Save BERTopic Models for Exploration

By default, full BERTopic models are NOT saved (saves ~25 GB for 15 queries × 4 methods).

To save models for interactive exploration:

```python
SAVE_TOPIC_MODELS = True
```

Models saved to: `results/topic_evaluation/query_X/topic_models/{method}_model.pkl`

Load with:
```python
from bertopic import BERTopic
model = BERTopic.load("path/to/model.pkl")
```

---

### 5. GPU Acceleration

If you have CUDA-compatible GPU:

```python
DEVICE = "cuda"
```

For Apple Silicon (M1/M2/M3):
```python
DEVICE = "mps"
```

Speeds up:
- SBERT embedding generation (~2-3x faster)
- HDBSCAN clustering (if using cuML - optional)

---

## Output Structure

After running, results are organized as:

```
results/topic_evaluation/query_{query_id}/
├── config.json                          # Run configuration
│
├── samples/                             # Cached document samples
│   ├── random_uniform.pkl
│   ├── keyword_search.pkl
│   ├── direct_retrieval.pkl
│   └── query_expansion.pkl
│
├── topic_models/                        # Cached BERTopic results
│   ├── random_uniform_results.pkl       # Topic results (always saved)
│   ├── random_uniform_model.pkl         # Full model (optional, 420 MB)
│   ├── keyword_search_results.pkl
│   ├── direct_retrieval_results.pkl
│   └── query_expansion_results.pkl
│
└── results/                             # Evaluation outputs
    ├── per_method_summary.csv           # All per-method metrics
    ├── pairwise_metrics.csv             # All pairwise method comparisons
    │
    ├── topics_summary/                  # Human-readable topic lists
    │   ├── random_uniform_topics.txt
    │   ├── keyword_search_topics.txt
    │   ├── direct_retrieval_topics.txt
    │   └── query_expansion_topics.txt
    │
    └── plots/                           # Visualizations (17 plots)
        ├── intrinsic_quality_metrics.png
        ├── query_alignment_metrics.png
        ├── diversity_scatter.png
        ├── pairwise_topic_similarity.png
        ├── pairwise_topic_overlap.png
        ├── pairwise_f1_05.png
        ├── pairwise_f1_06.png
        ├── pairwise_f1_07.png
        └── ... (14 pairwise heatmaps total)
```

### Key Output Files

- **`per_method_summary.csv`**: Compare intrinsic quality and query alignment across methods
- **`pairwise_metrics.csv`**: All pairwise method comparisons (topic matching, diversity differences, etc.)
- **`topics_summary/*.txt`**: Human-readable topic lists for qualitative validation
- **`plots/`**: All visualizations for pattern identification

---

## Expected Runtime

Runtime depends on caching state:

### Query 43 (sample_size=1000)

| Step | Cold Run (No Cache) | Cached |
|------|---------------------|--------|
| Index building (BM25 + SBERT) | ~10-15 min (one-time) | < 1 sec |
| Sampling (4 methods) | ~2-3 min | < 1 sec |
| Topic modeling (4 models) | ~5-8 min | < 1 sec |
| Evaluation | ~2-3 min | ~2-3 min |
| **Total** | **~20-30 min** | **~2-3 min** |

### 15 Queries (Full Evaluation)

| Configuration | Total Runtime |
|---------------|---------------|
| Cold run (no cache) | ~5-8 hours |
| Cached samples + models | ~30-45 min |
| Cached everything | ~30-45 min (evaluation not cached across queries) |

*Times assume CPU. GPU reduces topic modeling time by ~30-50%.*

---

## Caching Behavior

### Three-Tier Caching

1. **Search Indices** (shared across all queries)
   - `cache/{dataset}/bm25_index.pkl` - BM25 index
   - `cache/{dataset}/sbert_embeddings_{model}.npy` - SBERT embeddings
   - Built once, reused for all queries

2. **Document Samples** (per query, per method)
   - `results/topic_evaluation/query_X/samples/{method}.pkl`
   - Controlled by `FORCE_REGENERATE_SAMPLES`

3. **Topic Models** (per query, per method)
   - `results/topic_evaluation/query_X/topic_models/{method}_results.pkl` (always saved)
   - `results/topic_evaluation/query_X/topic_models/{method}_model.pkl` (optional, 420 MB)
   - Controlled by `FORCE_REGENERATE_TOPICS` and `SAVE_TOPIC_MODELS`

4. **Evaluation Results** (per query)
   - `results/topic_evaluation/query_X/results/`
   - Controlled by `FORCE_REGENERATE_EVALUATION`

### Cache Logic

```python
if cache_exists and not force_flag:
    load_from_cache()
else:
    compute()
    save_to_cache()
```

### Clearing Cache

To force full recomputation:

```bash
# Clear all results for a query
rm -rf results/topic_evaluation/query_43/

# Clear all search indices (requires rebuild)
rm -rf cache/trec-covid/

# Clear all results (all queries)
rm -rf results/topic_evaluation/
```

---

## Dependencies

### Required Packages

All dependencies in `requirements.txt`:

```txt
bertopic==0.17.0
sentence-transformers==3.4.1
datasets==3.4.1
matplotlib==3.10.1
seaborn==0.13.2
scikit-learn>=1.0.0
scipy>=1.7.0
tqdm>=4.62.0
pandas>=1.3.0
numpy>=1.21.0
rank-bm25>=0.2.2
nltk>=3.6.0
```

### Installation

```bash
conda create -n coreset_proj python=3.11
conda activate coreset_proj
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Optional (GPU Acceleration)

For CUDA support:
```bash
pip install cupy-cuda11x  # Replace 11x with your CUDA version
pip install cuml-cu11     # GPU-accelerated clustering
```

---

## Troubleshooting

### 1. Missing Keywords Cache

**Error:**
```
FileNotFoundError: Keyword cache not found: /home/srangre1/cache/keywords/keybert_k10_div0.7_top10docs_mpnet_k1000_ngram1-2.json
```

**Cause:** Query expansion method requires pre-extracted KeyBERT keywords

**Solution 1 (Quick):** Keywords should already exist at the path above. Check if file exists.

**Solution 2 (Regenerate):**
```python
from src.keyword_extraction import KeywordExtractor
# Run keyword extraction (see keyword_extraction.py documentation)
```

**Solution 3 (Skip method):** Comment out query expansion method in `main()` if not needed

---

### 2. BERTopic Clustering Fails

**Error:**
```
HDBSCAN could not find any clusters
```

**Cause:** Sample too small or too homogeneous for HDBSCAN density-based clustering

**Behavior:** Method is skipped gracefully, warning logged, evaluation continues

**Solution:**
- Increase `min_cluster_size` in BERTopic configuration (current: 5)
- Increase `SAMPLE_SIZE` (current: 1000)
- Normal behavior for some queries - method simply excluded from results

---

### 3. Low Document Overlap Warning

**Warning:**
```
Insufficient overlap (12 docs) for clustering metrics (ARI/NMI)
```

**Cause:** Different sampling methods naturally have little document overlap

**Behavior:** ARI and NMI reported as 0.0 or N/A

**Explanation:** Expected behavior - retrieval and random samples are mostly disjoint. Not an error.

---

### 4. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size in SBERT encoding (edit `batch_size` parameter)
- Use `DEVICE = "cpu"` instead
- Reduce `SAMPLE_SIZE` (e.g., 500 instead of 1000)
- Clear GPU cache: `torch.cuda.empty_cache()`

---

### 5. Slow Evaluation on CPU

**Symptom:** Evaluation takes >30 min per query

**Solutions:**
- Enable GPU: `DEVICE = "cuda"` or `"mps"`
- Reduce sample size: `SAMPLE_SIZE = 500`
- Skip saving models: `SAVE_TOPIC_MODELS = False` (default)
- Run on subset of queries initially

---

### 6. Results Look Suspicious

**Symptoms:**
- All metrics are identical across methods
- Query alignment is 0.0 for all methods
- Topics are all generic

**Debugging steps:**
1. Check cached samples: `results/topic_evaluation/query_X/samples/*.pkl`
2. Check topic summaries: `results/topic_evaluation/query_X/results/topics_summary/*.txt`
3. Force regenerate: Set all force flags to `True`
4. Check query text: Ensure `QUERY_ID` corresponds to valid query

---

## Advanced Usage

### Running Batch Evaluation (Multiple Queries)

Create a wrapper script:

```python
# run_batch_evaluation.py
import subprocess

queries = ["1", "5", "10", "15", "20", "25", "30", "35", "40", "43", "45", "48", "50"]

for query_id in queries:
    print(f"\n{'='*50}")
    print(f"Running evaluation for Query {query_id}")
    print(f"{'='*50}\n")

    # Edit main() to use query_id, or pass as argument
    subprocess.run(["python", "src/end_to_end_evaluation.py", "--query", query_id])
```

**Note:** Currently requires editing `QUERY_ID` in `main()` for each query. Could be extended to accept command-line arguments.

---

### Analyzing Results Across Queries

After running multiple queries, aggregate results:

```python
import pandas as pd
import glob

# Load all per_method_summary.csv files
all_summaries = []
for csv_path in glob.glob("results/topic_evaluation/query_*/results/per_method_summary.csv"):
    query_id = csv_path.split("query_")[1].split("/")[0]
    df = pd.read_csv(csv_path)
    df["query_id"] = query_id
    all_summaries.append(df)

# Combine and aggregate
combined = pd.concat(all_summaries)
aggregated = combined.groupby("method").agg(["mean", "std", "median", "min", "max"])
print(aggregated)
```

---

## Tips for Efficient Experimentation

1. **Start with one query** - Debug on Query 43, then scale to 15 queries
2. **Use force flags strategically** - Only regenerate what changed
3. **Save models sparingly** - 420 MB × 4 methods × 15 queries = 25 GB
4. **Check topic summaries** - Qualitative validation catches issues metrics miss
5. **Use GPU when available** - 2-3x speedup for embeddings
6. **Monitor cache size** - SBERT embeddings are ~5-10 GB for full corpus
7. **Version control configs** - Save `config.json` shows exact parameters used

---

## Related Documentation

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)** - Research question, experimental design, conceptual framework
- **[METRICS_GUIDE.md](METRICS_GUIDE.md)** - Detailed metric definitions and interpretations
- **[FUTURE_WORK.md](FUTURE_WORK.md)** - Planned extensions and experiments
- **[README.md](README.md)** - Installation and component documentation

---

## Contact

For questions or issues, see project README or create an issue in the repository.

## Last Updated

2025-11-13 - Added LDA topic modeling support and model switching guide
