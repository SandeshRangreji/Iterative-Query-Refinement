# Project Overview: Retrieval-Guided Topic Modeling

## Research Question

**How does control of relevancy in sampling methods (by using different retrieval techniques) affect downstream topic modeling?**

This project investigates the effects of retrieval-based document sampling on topic modeling outcomes. Rather than claiming retrieval produces "better" topic models, we aim to **characterize and quantify** how different degrees of relevance bias affect various aspects of topic modeling: coherence, diversity, query alignment, topic structure, and document clustering.

---

## Overview

This is an **exploratory ablation study** examining how retrieval techniques control document relevance in samples, and how this relevance control affects topic modeling with BERTopic. The investigation is **descriptive and observational** - we measure effects and document trends without predetermined hypotheses about what constitutes "better" topic modeling.

### Key Observation

From preliminary qualitative analysis: Topics discovered from retrieval-based samples appear "less generic and more nuanced" compared to random sampling. This project aims to operationalize and quantify these observations across multiple metrics and queries.

---

## Experimental Design

### Ablation of Relevance Control

We compare sampling methods across a **spectrum of relevance bias**:

| Method | Relevance Control | Retrieval Mechanism |
|--------|------------------|---------------------|
| **Random Uniform** | None (unbiased control) | Pure random sampling from corpus |
| **Keyword Search** | Low (lexical only) | BM25 lexical matching |
| **Direct Retrieval** | Medium (lexical + semantic) | Hybrid BM25+SBERT (Simple Sum fusion) |
| **Direct Retrieval + MMR** | Medium + High Diversity | Hybrid retrieval (5000 candidates) → MMR reranking (λ=0.3) → Top 1000 |
| **Query Expansion** | High (expanded semantic) | KeyBERT keywords + Weighted RRF fusion |

**Independent Variable:** Retrieval method (controls degree of relevance bias and diversity)
**Dependent Variables:** 50+ topic modeling metrics (includes IDF-based specificity, relevance concentration, document overlap)
**Sample Size:** 1,000 documents per method (fixed)
**Dataset:** TREC-COVID corpus (171K documents)
**Queries:** 15 queries for cross-query generalization

### Future Ablations

Planned extensions include:
- **Cross-encoder reranking**: Enhanced relevance precision
- **Additional retrieval variants**: Different fusion strategies, embedding models, MMR lambda values
- **Additional datasets**: Generalization beyond TREC-COVID

---

## Topic Modeling Pipeline

### Sampling Methods

1. **Random Uniform Sampling**
   - Pure random selection from corpus
   - No relevance bias (control condition)
   - Represents unbiased topic discovery

2. **Keyword Search (BM25 only)**
   - Lexical retrieval based on term matching
   - Low relevance control (exact term overlap)

3. **Direct Retrieval (Hybrid BM25+SBERT)**
   - Combines lexical and semantic retrieval
   - Medium relevance control (term + concept matching)
   - Simple Sum fusion strategy

4. **Query Expansion + Retrieval**
   - KeyBERT extracts semantically related keywords
   - Weighted RRF fusion (70% original query, 30% keywords)
   - High relevance control (expanded semantic coverage)

### BERTopic Configuration

All methods use **identical BERTopic parameters** for fair comparison:

- **Embeddings:** all-mpnet-base-v2 (SentenceTransformer)
- **Dimensionality Reduction:** UMAP (default params)
- **Clustering:** HDBSCAN (min_cluster_size=5)
- **Topic Representation:** c-TF-IDF with CountVectorizer
- **Vocabulary:** English stopwords removed, unigrams + bigrams

This ensures differences in topic modeling outcomes are attributable to **sampling method**, not modeling configuration.

---

## Evaluation Metrics (40+)

Metrics are organized into three categories:

### A. Intrinsic Quality Metrics (Per-Method)

Measure topic quality independent of other methods:

1. **NPMI Coherence** (-1 to 1) - Semantic coherence based on word co-occurrence
2. **Embedding Coherence** (0-1) - Semantic tightness in embedding space
3. **Semantic Diversity** (0-1) - Conceptual distinctness between topics
4. **Lexical Diversity** (0-1) - Vocabulary redundancy across topics
5. **Document Coverage** (0-1) - Fraction of documents assigned to topics
6. **Number of Topics** - Total topics discovered by HDBSCAN
7. **Average Topic Size** - Mean documents per topic

### B. Query Alignment Metrics (Per-Method)

Measure how well topics align with the query:

8. **Topic-Query Similarity (Avg)** (0-1) - Average cosine similarity between all topics and query embedding
9. **Max Query Similarity** (0-1) - Highest topic-query similarity score
10. **Query-Relevant Ratio** (0-1) - Fraction of topics above 0.5 similarity threshold
11. **Top-3 Avg Similarity** (0-1) - Average similarity of top-3 most query-relevant topics

### C. Pairwise Comparison Metrics (Between Methods)

Compare two methods against each other:

12. **Topic Word Overlap (Jaccard)** (0-1) - Lexical overlap of Hungarian-matched topic pairs
13. **Topic Semantic Similarity** (0-1) - Embedding similarity of matched topics
14-16. **F1 @ Thresholds (0.5, 0.6, 0.7)** - Topic matching quality at different strictness levels
17-19. **Precision @ Thresholds** - Fraction of method B's topics matching method A
20-22. **Recall @ Thresholds** - Fraction of method A's topics matching method B
23. **NPMI Coherence Difference** - Coherence delta between methods
24. **Embedding Coherence Difference** - Embedding coherence delta
25. **Semantic Diversity Difference** - Semantic diversity delta
26. **Lexical Diversity Difference** - Lexical diversity delta
27. **ARI (Adjusted Rand Index)** - Document clustering agreement (on overlap only)
28. **NMI (Normalized Mutual Information)** - Shared clustering information (on overlap only)
29. **Document Overlap (Jaccard)** - Proportion of shared documents between samples

See [METRICS_GUIDE.md](METRICS_GUIDE.md) for detailed metric definitions and interpretations.

---

## Interpretation Framework

### No "Better" or "Worse"

Metrics are **descriptive dimensions**, not quality judgments. We do not have ground truth for topic modeling quality. Instead, we:

- **Characterize effects:** "Retrieval increases query alignment by X%"
- **Identify trade-offs:** "Higher relevance correlates with lower lexical diversity"
- **Document patterns:** "BM25 produces more topics than SBERT on average"
- **Observe consistency:** "Effect X appears across 12/15 queries"

### Trade-off Analysis

We investigate potential trade-offs:

- **Query Alignment vs. Semantic Diversity** - Do query-focused samples produce narrow topics?
- **Coherence vs. Coverage** - Do tighter clusters leave more outliers?
- **Lexical vs. Semantic Diversity** - Query-biased samples may repeat query terms but cover different concepts
- **Topic Count vs. Topic Size** - Clustering granularity effects

### Cross-Query Generalization

We demonstrate generalizability by:
- Evaluating on **15 independent queries**
- Aggregating statistics (mean, std, median, range)
- Identifying **consistent effects** vs. query-specific patterns
- Future work: Multiple datasets beyond TREC-COVID

---

## Output Structure

```
results/topic_evaluation/query_{query_id}/
├── config.json                          # Experiment configuration
├── samples/                             # Cached document samples
│   ├── random_uniform.pkl
│   ├── keyword_search.pkl
│   ├── direct_retrieval.pkl
│   └── query_expansion.pkl
├── topic_models/                        # Cached BERTopic results
│   ├── random_uniform_results.pkl
│   ├── keyword_search_results.pkl
│   ├── direct_retrieval_results.pkl
│   └── query_expansion_results.pkl
└── results/                             # Evaluation outputs
    ├── per_method_summary.csv           # All per-method metrics
    ├── pairwise_metrics.csv             # All pairwise comparisons
    ├── topics_summary/                  # Human-readable topic summaries
    │   ├── random_uniform_topics.txt
    │   ├── keyword_search_topics.txt
    │   ├── direct_retrieval_topics.txt
    │   └── query_expansion_topics.txt
    └── plots/                           # Visualizations (17 plots)
        ├── intrinsic_quality_metrics.png
        ├── query_alignment_metrics.png
        ├── diversity_scatter.png
        ├── relevancy_vs_diversity.png   # Methods plotted by relevance and diversity
        └── 13 pairwise heatmaps (similarity, overlap, F1, precision, recall, etc.)
```

---

## Caching Strategy

Three-tier caching for efficient experimentation:

1. **Search Indices** (reused across all queries)
   - BM25 index
   - SBERT embeddings (all-mpnet-base-v2)

2. **Document Samples** (per query, per method)
   - Cached in `samples/` directory
   - Force regeneration flag: `force_regenerate_samples`

3. **Topic Models** (per query, per method)
   - Cached in `topic_models/` directory
   - Force regeneration flag: `force_regenerate_topics`
   - Optional: Save full BERTopic models (420 MB each, only needed for interactive exploration)

4. **Evaluation Results** (per query)
   - Cached in `results/` directory
   - Force regeneration flag: `force_regenerate_evaluation`
   - Enables iterative metric development without re-running sampling/modeling

---

## Usage

### Running the Evaluation

```bash
cd /home/srangre1/Iterative-Query-Refinement
conda activate coreset_proj
python src/end_to_end_evaluation.py
```

### Configuration

Edit parameters in `main()` function of `end_to_end_evaluation.py`:

```python
QUERY_ID = "43"                    # Query to evaluate
SAMPLE_SIZE = 1000                 # Documents per sample
EMBEDDING_MODEL = "all-mpnet-base-v2"
DEVICE = "cpu"                     # or "cuda", "mps"
SAVE_TOPIC_MODELS = False          # Set True to save 420MB models
FORCE_REGENERATE_SAMPLES = False
FORCE_REGENERATE_TOPICS = False
FORCE_REGENERATE_EVALUATION = False
```

### Expected Runtime

- **Cold run** (no cache): ~15-20 min per query
- **Cached samples/models**: ~2-3 min per query
- **15 queries total**: ~30-45 min with cache

---

## Roadmap and Future Work

See [FUTURE_WORK.md](FUTURE_WORK.md) for:

- Planned metrics (Topic Specificity, Relevant Document Concentration, Vendiscore, etc.)
- Additional retrieval ablations (MMR, cross-encoder reranking)
- Guided topic modeling experiments
- Cross-dataset evaluation
- Inductive labeling downstream task

---

## Technology Stack

- **Topic Modeling:** BERTopic 0.17.0, HDBSCAN, UMAP
- **Retrieval:** rank-bm25, sentence-transformers 3.4.1
- **Embeddings:** all-mpnet-base-v2 (default)
- **Evaluation:** scikit-learn, scipy, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Dataset:** TREC-COVID via HuggingFace datasets
- **GPU Support:** Optional CUDA/MPS acceleration

---

## Key Files

- [`src/end_to_end_evaluation.py`](src/end_to_end_evaluation.py) - Main evaluation pipeline
- [`METRICS_GUIDE.md`](METRICS_GUIDE.md) - Comprehensive metrics documentation
- [`FUTURE_WORK.md`](FUTURE_WORK.md) - Planned experiments and metrics
- [`README.md`](README.md) - Installation and component documentation
- [`requirements.txt`](requirements.txt) - Python dependencies
