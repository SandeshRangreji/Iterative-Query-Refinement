# Metrics Guide - Topic Modeling Evaluation

## Overview

This guide documents all metrics used to evaluate the **effects of retrieval-based sampling on topic modeling**. Metrics are **descriptive dimensions** rather than quality judgments - we characterize how different sampling methods affect topic modeling outcomes across multiple axes.

---

## Metric Interpretation Philosophy

### No "Better" or "Worse"

We do not have ground truth for topic modeling quality. Instead:

- **Metrics describe effects:** "Method X produces Y% higher coherence than method Z"
- **Metrics reveal trade-offs:** "Higher query alignment correlates with lower lexical diversity"
- **Metrics characterize patterns:** "Retrieval-based methods consistently produce fewer but larger topics"

### Effect Magnitude Matters

- **Large effects:** Substantial differences between methods (>20% relative change)
- **Small effects:** Minimal practical difference (<5% relative change)
- **Directional consistency:** Effect direction consistent across queries
- **Query-specific effects:** Effect appears for some queries but not others

### Cross-Query Generalization

We evaluate 15 independent queries and aggregate results to identify:
- **Consistent effects** - patterns appearing across most/all queries
- **Query-dependent effects** - patterns specific to query characteristics
- **Method stability** - variance in metric values across queries

---

## Intrinsic Quality Metrics (Per-Method)

These measure topic characteristics independent of other methods.

### 1. NPMI Coherence

**Range:** -1 to 1
**Computation:** Normalized Pointwise Mutual Information of word pairs within each topic, averaged across all topics
**Measures:** Semantic coherence of topic words based on document co-occurrence

**Interpretation:**
- Higher values indicate topic words frequently co-occur in documents (coherent)
- Lower values indicate topic words rarely appear together (incoherent)
- Negative values indicate words anti-correlate (appear in different contexts)

**Relevance effects:**
- Retrieval-based samples may produce higher coherence (query-focused documents share vocabulary)
- Or lower coherence (diverse query aspects don't co-occur)
- Effect direction depends on query specificity and sample diversity

**Related metrics:** Embedding Coherence (semantic vs. statistical coherence)

---

### 2. Embedding Coherence

**Range:** 0 to 1
**Computation:** Average pairwise cosine similarity of word embeddings within each topic
**Measures:** Semantic tightness of topic words in embedding space

**Interpretation:**
- Higher values indicate topic words are semantically related in embedding space
- Lower values indicate topic words are semantically scattered
- Complements NPMI (embedding-based vs. co-occurrence-based)

**Relevance effects:**
- Query-focused samples may have tighter semantic clusters
- Generic samples may have broader semantic coverage per topic

**Related metrics:** NPMI Coherence (statistical coherence)

---

### 3. Semantic Diversity

**Range:** 0 to 1
**Computation:** Average pairwise distance between topic embeddings (1 - cosine similarity)
**Measures:** Conceptual distinctness between topics

**Interpretation:**
- Higher values indicate topics cover different concepts
- Lower values indicate topics are conceptually redundant/overlapping
- Balance needed: too high = incoherent, too low = redundant

**Relevance effects:**
- Retrieval may reduce diversity (topics focus on query aspects)
- Or maintain diversity (different facets of the query)
- Trade-off with query alignment expected

**Related metrics:** Lexical Diversity (vocabulary vs. semantic diversity)

---

### 4. Lexical Diversity

**Range:** 0 to 1
**Computation:** Unique words / (num_topics × top_k words per topic)
**Measures:** Vocabulary redundancy across topics

**Interpretation:**
- Higher values indicate less word repetition across topics
- Lower values indicate shared vocabulary (query terms may repeat)
- **Context-dependent:** Query-focused methods may have lower lexical diversity but high semantic diversity (same query words, different concepts)

**Relevance effects:**
- Retrieval-based samples likely have lower lexical diversity (query terms repeat)
- But may maintain semantic diversity (different query aspects)
- This is NOT necessarily negative - indicates focused topic modeling

**Related metrics:** Semantic Diversity (vocabulary vs. semantic diversity)

---

### 5. Document Coverage

**Range:** 0 to 1
**Computation:** 1 - outlier_ratio (fraction of documents assigned to topics vs. outlier cluster -1)
**Measures:** Fraction of documents successfully clustered into topics

**Interpretation:**
- Higher values indicate most documents fit into discovered topics
- Lower values indicate many documents are outliers (don't fit any topic)
- Depends on sample coherence and HDBSCAN clustering behavior

**Relevance effects:**
- Retrieval-based samples may have higher coverage (more coherent, easier to cluster)
- Or lower coverage (diverse query aspects create scattered clusters)
- Interacts with Number of Topics and Average Topic Size

**Related metrics:** Average Topic Size, Number of Topics

---

### 6. Number of Topics

**Range:** 1 to N
**Computation:** Count of distinct topics discovered by HDBSCAN (excluding outlier cluster -1)
**Measures:** Topic granularity produced by clustering

**Interpretation:**
- Higher values indicate finer-grained topic structure
- Lower values indicate coarser clustering
- Depends on sample diversity and HDBSCAN density threshold
- **No inherent "better" value** - depends on use case

**Relevance effects:**
- Retrieval may produce fewer topics (focused samples cluster into fewer dense groups)
- Or more topics (diverse query aspects create more clusters)
- Observe trend: does relevance increase or decrease granularity?

**Related metrics:** Average Topic Size (inverse relationship)

---

### 7. Average Topic Size

**Range:** 1 to N documents
**Computation:** Mean number of documents per topic (excluding outliers)
**Measures:** Clustering balance

**Interpretation:**
- Higher values indicate larger, coarser topics
- Lower values indicate smaller, finer-grained topics
- Balance vs. imbalance: highly variable sizes indicate imbalanced clustering
- **No inherent "better" value** - depends on use case

**Relevance effects:**
- Fewer topics → larger average size
- More topics → smaller average size
- Observe whether retrieval creates balanced or imbalanced distributions

**Related metrics:** Number of Topics (inverse relationship)

---

## Query Alignment Metrics (Per-Method)

These measure how well discovered topics align with the query.

### 8. Topic-Query Similarity (Average)

**Range:** 0 to 1
**Computation:** Average cosine similarity between query embedding and each topic embedding (topic = concatenated top-10 words)
**Measures:** Overall query alignment across all topics

**Interpretation:**
- Higher values indicate topics are more semantically related to the query
- Lower values indicate topics are less query-relevant
- **Expected effect:** Retrieval-based methods should have higher values

**Relevance effects:**
- **Primary metric for measuring relevance effect**
- Retrieval-based samples expected to have substantially higher query alignment
- Magnitude of increase quantifies relevance bias

**Related metrics:** Max Query Similarity, Query-Relevant Ratio, Top-3 Avg Similarity

---

### 9. Max Query Similarity

**Range:** 0 to 1
**Computation:** Highest cosine similarity between query and any single topic
**Measures:** Peak query alignment

**Interpretation:**
- Higher values indicate at least one highly query-relevant topic exists
- Complements average: high max + low avg = one relevant topic + many irrelevant
- High max + high avg = consistently query-aligned topics

**Relevance effects:**
- Retrieval methods should have higher max similarity
- Difference between max and average reveals topic alignment variance

**Related metrics:** Topic-Query Similarity (Average), Top-3 Avg Similarity

---

### 10. Query-Relevant Ratio

**Range:** 0 to 1
**Computation:** Fraction of topics with similarity ≥ 0.5 threshold
**Measures:** Proportion of query-relevant topics

**Interpretation:**
- Higher values indicate most topics are query-relevant
- Lower values indicate few topics are query-relevant (many are noise)
- Threshold-dependent (currently 0.5)

**Relevance effects:**
- Retrieval methods should have higher ratios
- Random sampling likely has low ratio (most topics unrelated to query)
- Quantifies "signal vs. noise" in topic discovery

**Related metrics:** Topic-Query Similarity (Average), Max Query Similarity

---

### 11. Top-3 Average Similarity

**Range:** 0 to 1
**Computation:** Average cosine similarity of the 3 most query-relevant topics
**Measures:** Peak query alignment quality

**Interpretation:**
- Higher values indicate best topics are highly query-aligned
- Robust to outliers (ignores low-relevance topics)
- Useful when only top topics matter (e.g., summarization tasks)

**Relevance effects:**
- Retrieval methods should have higher top-3 alignment
- Even random sampling may produce 1-2 relevant topics by chance
- Top-3 filters this noise

**Related metrics:** Max Query Similarity, Topic-Query Similarity (Average)

---

## Pairwise Comparison Metrics (Between Methods)

These compare two methods to quantify differences and similarities.

### 12. Topic Word Overlap (Jaccard)

**Range:** 0 to 1
**Computation:** Jaccard similarity of top-10 words for Hungarian-matched topic pairs
**Measures:** Lexical overlap between matched topics

**Interpretation:**
- Higher values indicate methods use similar vocabulary for topics
- Lower values indicate methods use different words (may still be semantically similar)
- Sensitive to synonym differences (use Topic Semantic Similarity for robustness)

**Method comparisons:**
- High overlap = methods discover similar topics lexically
- Low overlap + high semantic similarity = synonymous topics
- Low overlap + low semantic similarity = different topics entirely

**Related metrics:** Topic Semantic Similarity (robust to synonyms)

---

### 13. Topic Semantic Similarity

**Range:** 0 to 1
**Computation:** Cosine similarity of topic embeddings for Hungarian-matched pairs
**Measures:** Semantic similarity between matched topics

**Interpretation:**
- Higher values indicate methods discover conceptually similar topics
- More robust than word overlap (captures synonyms and paraphrases)
- Used for F1/Precision/Recall thresholds

**Method comparisons:**
- High similarity = methods produce similar topic structures
- Low similarity = methods produce different topic structures
- Compare retrieval methods to each other (stability) vs. random (difference)

**Related metrics:** Topic Word Overlap (lexical vs. semantic), F1 @ thresholds

---

### 14-16. F1 @ Thresholds (0.5, 0.6, 0.7)

**Range:** 0 to 1
**Computation:** Harmonic mean of precision and recall at different semantic similarity thresholds
**Measures:** Topic matching quality at varying strictness levels

**Interpretation:**
- Higher values indicate more topics match across methods
- Threshold sensitivity:
  - **0.5**: Lenient matching (topics somewhat similar)
  - **0.6**: Moderate matching (topics reasonably similar)
  - **0.7**: Strict matching (topics highly similar)
- Decreasing F1 across thresholds = topics partially similar but not identical

**Method comparisons:**
- **High F1 (retrieval vs. retrieval)**: Topic discovery is stable across retrieval methods
- **Low F1 (retrieval vs. random)**: Retrieval produces different topics than random
- **Threshold degradation**: Topics match loosely but not strictly

**Related metrics:** Precision @ thresholds, Recall @ thresholds

---

### 17-19. Precision @ Thresholds (0.5, 0.6, 0.7)

**Range:** 0 to 1
**Computation:** Fraction of method B's topics that match method A's topics
**Measures:** Coverage of B by A (directional)

**Interpretation:**
- Higher values indicate most of B's topics appear in A
- Asymmetric: Precision_B ≠ Precision_A
- Low precision = B discovers topics not in A

**Method comparisons:**
- If Precision_B is high, A "covers" B's topics
- If Precision_B is low, B discovers novel topics

**Related metrics:** Recall @ thresholds (inverse direction), F1 @ thresholds

---

### 20-22. Recall @ Thresholds (0.5, 0.6, 0.7)

**Range:** 0 to 1
**Computation:** Fraction of method A's topics that match method B's topics
**Measures:** Coverage of A by B (directional)

**Interpretation:**
- Higher values indicate most of A's topics appear in B
- Asymmetric: Recall_A ≠ Recall_B
- Low recall = A discovers topics not in B

**Method comparisons:**
- If Recall_A is high, B "covers" A's topics
- If Recall_A is low, A discovers novel topics

**Related metrics:** Precision @ thresholds (inverse direction), F1 @ thresholds

---

### 23. NPMI Coherence Difference

**Range:** -1 to 1
**Computation:** NPMI_A - NPMI_B
**Measures:** Coherence delta between methods

**Interpretation:**
- Positive values indicate A has higher coherence than B
- Negative values indicate B has higher coherence than A
- **No "better"** - just magnitude and direction of effect

**Method comparisons:**
- Characterize which methods produce more coherent topics
- Observe trade-offs (e.g., higher alignment → lower coherence?)

**Related metrics:** Embedding Coherence Difference

---

### 24. Embedding Coherence Difference

**Range:** -1 to 1
**Computation:** EmbeddingCoherence_A - EmbeddingCoherence_B
**Measures:** Embedding coherence delta between methods

**Interpretation:**
- Positive values indicate A has tighter semantic clusters than B
- Negative values indicate B has tighter semantic clusters than A
- Complements NPMI difference (embedding-based vs. co-occurrence)

**Method comparisons:**
- Characterize which methods produce semantically tight topics
- May differ from NPMI difference (different coherence definitions)

**Related metrics:** NPMI Coherence Difference

---

### 25. Semantic Diversity Difference

**Range:** -1 to 1
**Computation:** SemanticDiversity_A - SemanticDiversity_B
**Measures:** Semantic diversity delta between methods

**Interpretation:**
- Positive values indicate A has more distinct topics than B
- Negative values indicate B has more distinct topics than A
- Observe trade-off with query alignment

**Method comparisons:**
- Characterize diversity effects of retrieval
- Expected: Retrieval may reduce diversity (query-focused)

**Related metrics:** Lexical Diversity Difference, Topic-Query Similarity

---

### 26. Lexical Diversity Difference

**Range:** -1 to 1
**Computation:** LexicalDiversity_A - LexicalDiversity_B
**Measures:** Lexical diversity delta between methods

**Interpretation:**
- Positive values indicate A has less word redundancy than B
- Negative values indicate B has less word redundancy than A
- **Context-dependent:** Lower lexical diversity in retrieval may reflect query-term focus (not necessarily negative)

**Method comparisons:**
- Expected: Retrieval has lower lexical diversity (query words repeat)
- Check if semantic diversity remains high (different concepts, shared vocabulary)

**Related metrics:** Semantic Diversity Difference

---

### 27. ARI (Adjusted Rand Index)

**Range:** -1 to 1 (typically 0 to 1)
**Computation:** Adjusted Rand Index on document clustering assignments (only on overlapping documents)
**Measures:** Document clustering agreement between methods

**Interpretation:**
- 1.0 = perfect agreement (identical cluster assignments)
- 0.0 = random agreement (no clustering relationship)
- Negative = worse than random (rare)
- **Only computed on document overlap** (typically small for different sampling methods)

**Method comparisons:**
- High ARI = methods cluster shared documents similarly
- Low ARI = methods cluster shared documents differently
- Limited applicability (small overlap between samples)

**Related metrics:** NMI (information-theoretic clustering agreement)

---

### 28. NMI (Normalized Mutual Information)

**Range:** 0 to 1
**Computation:** Normalized Mutual Information on document clustering assignments (only on overlapping documents)
**Measures:** Shared clustering information between methods

**Interpretation:**
- 1.0 = perfect information sharing (identical clustering)
- 0.0 = no shared information (independent clustering)
- Information-theoretic alternative to ARI
- **Only computed on document overlap** (typically small)

**Method comparisons:**
- High NMI = methods share clustering structure
- Low NMI = methods have independent clustering structure

**Related metrics:** ARI (clustering agreement)

---

### 29. Document Overlap (Jaccard)

**Range:** 0 to 1
**Computation:** Jaccard similarity of sampled document IDs between methods
**Measures:** Proportion of shared documents in samples

**Interpretation:**
- 1.0 = identical samples
- 0.0 = completely disjoint samples
- Higher overlap = more shared documents
- Expected: Low overlap between retrieval and random, moderate overlap among retrieval methods

**Method comparisons:**
- Helps interpret topic differences (different topics due to different documents)
- Low overlap + high topic similarity = robust topic discovery
- High overlap + low topic similarity = method-dependent topic structure

**Related metrics:** ARI, NMI (only applicable when overlap > 0)

---

## Visualizations

### 1. Intrinsic Quality Metrics Plot

**Type:** 3×2 bar chart grid
**Shows:** Per-method values for NPMI, Embedding Coherence, Semantic Diversity, Lexical Diversity, Document Coverage, Number of Topics

**Usage:**
- Compare intrinsic quality across methods
- Identify which method has highest/lowest for each metric
- Observe patterns (e.g., retrieval consistently higher coherence)

---

### 2. Query Alignment Metrics Plot

**Type:** 1×3 bar chart grid
**Shows:** Per-method values for Avg Similarity, Max Similarity, Query-Relevant Ratio

**Usage:**
- Primary visualization for relevance effect
- Expected: Retrieval methods substantially higher than random
- Quantifies query alignment increase

---

### 3. Diversity Scatter Plot

**Type:** 2D scatter (Semantic Diversity × Lexical Diversity)
**Points:** One per method
**Axes:** X = Lexical Diversity, Y = Semantic Diversity

**Quadrant interpretation:**
- **Top-right:** High semantic + lexical diversity (distinct topics, varied vocabulary)
- **Top-left:** High semantic, low lexical (distinct concepts, shared vocabulary - expected for retrieval)
- **Bottom-right:** Low semantic, high lexical (redundant concepts, varied words)
- **Bottom-left:** Low semantic + lexical diversity (redundant topics, shared vocabulary)

**Usage:**
- Visualize diversity trade-offs
- Expected: Retrieval in top-left (focused vocabulary, distinct aspects)

---

### 4. Relevancy vs. Diversity Scatter Plot

**Type:** 2D scatter (Avg Query-Topic Similarity × Semantic Diversity)
**Points:** One per method
**Axes:** X = Relevance metric (e.g., Topic-Query Similarity), Y = Diversity metric (e.g., Semantic Diversity)

**Usage:**
- Visualize **primary trade-off** hypothesis
- Expected pattern: Retrieval methods have higher relevance, potentially lower diversity
- Pareto frontier analysis: which methods achieve best balance?

---

### 5-17. Pairwise Heatmaps

**Type:** Method × Method heatmap (4×4 grid)
**Metrics:** 13 pairwise metrics (similarity, overlap, F1, precision, recall, coherence diff, etc.)

**Usage:**
- Compare all method pairs
- Identify which methods are most similar/different
- Diagonal = self-comparison (not meaningful)
- Symmetric metrics (e.g., F1) produce symmetric heatmaps
- Asymmetric metrics (e.g., precision/recall) produce asymmetric heatmaps

---

## Trade-off Analysis Framework

### Expected Trade-offs

1. **Query Alignment ↔ Semantic Diversity**
   - Higher relevance control → higher query alignment
   - Higher relevance control → potentially lower topic diversity
   - Visualize: Relevancy vs. Diversity scatter plot

2. **Coherence ↔ Coverage**
   - Tighter clusters (higher coherence) → more outliers (lower coverage)
   - Looser clusters (lower coherence) → fewer outliers (higher coverage)

3. **Lexical Diversity ↔ Query Alignment**
   - Query-focused samples → lower lexical diversity (query terms repeat)
   - But semantic diversity may remain high (different query aspects)
   - This is **descriptive**, not negative

4. **Topic Count ↔ Topic Size**
   - More topics → smaller average size
   - Fewer topics → larger average size
   - Neither is inherently "better"

### Observing Trade-offs

For each metric pair:
1. **Scatter plot** - visualize relationship across methods
2. **Correlation** - compute Pearson/Spearman correlation
3. **Consistency** - check if pattern holds across multiple queries
4. **Magnitude** - quantify strength of relationship

---

## Cross-Query Analysis

### Aggregation Statistics

For each metric, compute across 15 queries:
- **Mean** - central tendency
- **Std** - variance across queries
- **Median** - robust central tendency
- **Min/Max** - range of effects
- **Coefficient of Variation** - normalized variance (std/mean)

### Effect Consistency

Classify effects as:
- **Highly consistent** - same direction in >90% of queries
- **Moderately consistent** - same direction in 70-90% of queries
- **Inconsistent** - direction varies across queries

### Query-Specific Analysis

Investigate:
- Do broad queries show different patterns than narrow queries?
- Do queries with more/fewer relevant documents behave differently?
- Are there outlier queries that deviate from trends?

---

## Future Metrics Under Consideration

See [FUTURE_WORK.md](FUTURE_WORK.md) for detailed plans on:

- **Topic Specificity (IDF-based)** - Operationalize "less generic" observation
- **Query Term Overlap** - Measure query-term bias in topics
- **Corpus Coverage** - How many corpus documents match discovered topics
- **Relevant Document Concentration** - % of sample that's relevant (from QRELs)
- **Topic Purity** - Cluster purity using pseudo-clusters (half-baked idea)
- **Vendiscore** - External topic quality metric with Python package

---

## Usage

### Running Evaluation

```bash
python src/end_to_end_evaluation.py
```

Results saved to:
```
results/topic_evaluation/query_X/
├── per_method_summary.csv       # All per-method metrics
├── pairwise_metrics.csv         # All pairwise comparisons
└── plots/                       # All visualizations
```

### Analyzing Results

1. **Per-method summary**: Compare metrics across methods in `per_method_summary.csv`
2. **Pairwise comparisons**: Examine method differences in `pairwise_metrics.csv`
3. **Visualizations**: Review plots for patterns and trade-offs
4. **Topic summaries**: Qualitatively validate topics in `topics_summary/`

### Interpreting Metrics

- **Look for large effects** (>20% relative change)
- **Check consistency** across queries
- **Identify trade-offs** (e.g., alignment vs. diversity)
- **Avoid normative language** ("better/worse") - describe effects neutrally
- **Combine quantitative + qualitative** - metrics + human topic evaluation

---

## References

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for research question and experimental design.
See [FUTURE_WORK.md](FUTURE_WORK.md) for planned extensions.
