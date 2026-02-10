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
- Word-based models: Uses top words from topic distributions
- LLM models: Uses TF-IDF words from document clusters (measures **document** vocabulary coherence)
**Measures:** Semantic coherence of topic words based on document co-occurrence

**Note for LLM models:** Measures whether documents in the cluster use coherent vocabulary, not whether the topic concept itself is coherent. See "Word-Level Metrics for LLM Models" section.

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
- Word-based models: Uses top words from topic distributions
- LLM models: Uses TF-IDF words from document clusters (measures **document** vocabulary tightness)
**Measures:** Semantic tightness of topic words in embedding space

**Note for LLM models:** Measures semantic tightness of document vocabulary, not topic concept coherence. See "Word-Level Metrics for LLM Models" section.

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
- Uses **model-aware representations** (see "Model-Specific Topic Representations" section)
- Word-based models: Top 5 words concatenated
- LLM models: Semantic descriptions/labels directly
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
- Word-based models: Uses top words from topic distributions
- LLM models: Uses TF-IDF words from document clusters (measures document vocabulary diversity)
**Measures:** Vocabulary redundancy across topics

**Note for LLM models:** Measures document vocabulary redundancy, not topic label/concept redundancy.

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

### 8. Relevant Document Concentration

**Range:** 0 to 1
**Computation:** Fraction of sampled documents that are relevant according to TREC-COVID QRELs
**Measures:** Sample relevance validated against ground-truth relevance judgments

**Interpretation:**
- Higher values indicate more relevant documents in sample
- 0.0 means no relevant documents (pure random or failed retrieval)
- 1.0 means all documents are relevant (perfect retrieval)
- Validates that retrieval actually increases sample relevance

**Relevance effects:**
- Expected: Retrieval > Random
- Expected: Query Expansion ≥ Direct Retrieval ≥ Keyword Search > Random
- Directly validates the relevance control mechanism

**Related metrics:** Topic-Query Similarity (measures topic alignment, not sample relevance)

---

### 9. Topic Specificity (IDF-based)

**Range:** 0 to max(IDF) (typically 2-10 for biomedical corpus)
**Computation:** Average IDF score of topic words (top-10 per topic), averaged across all topics
- Word-based models: Uses top words from topic distributions
- LLM models: Uses TF-IDF words from document clusters (measures **document** vocabulary specificity)
**Measures:** How specific/technical vs. generic/common the topic words are

**Note for LLM models:** Measures specificity of document vocabulary, not specificity of topic concepts. See "Word-Level Metrics for LLM Models" section.

**Interpretation:**
- Higher values indicate more specific, technical, domain-specific terms (rare in corpus)
- Lower values indicate more generic, common terms (frequent in corpus)
- Operationalizes "less generic, more nuanced" qualitative observation
- IDF computed over full corpus (171K documents)

**Relevance effects:**
- Expected: Retrieval-based topics use more specific terminology
- Expected: Random topics use more generic terms
- Hypothesis: Query-focused samples bias toward query-specific terminology

**Related metrics:** NPMI Coherence (both measure topic quality, but different aspects)

---

## Query Alignment Metrics (Per-Method)

These measure how well discovered topics align with the query.

### 10. Topic-Query Similarity (Average)

**Range:** 0 to 1
**Computation:** Average cosine similarity between query embedding and each topic embedding
- Uses **model-aware representations** (see "Model-Specific Topic Representations" section)
- Word-based models: Top 10 words concatenated
- LLM models: Semantic descriptions/labels directly
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

### 11. Max Query Similarity

**Range:** 0 to 1
**Computation:** Highest cosine similarity between query and any single topic
- Uses **model-aware representations** (same as Topic-Query Similarity)
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

### 12. Query-Relevant Ratio

**Range:** 0 to 1
**Computation:** Fraction of topics with similarity ≥ 0.5 threshold
- Uses **model-aware representations** (same as Topic-Query Similarity)
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

### 13. Top-3 Average Similarity

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

## Relevant Topic Diversity Metrics (Per-Method)

These metrics address a key insight: **overall diversity can be inflated by "noise" topics unrelated to the query**. By focusing on query-relevant topics, we measure whether the method discovers distinct facets of the query rather than generic corpus topics.

### 14. Relevant Topic Diversity

**Range:** 0 to 1
**Computation:** Average pairwise distance (1 - cosine similarity) between topic embeddings, computed only on topics with query similarity ≥ 0.5
- Uses **model-aware representations** (see "Model-Specific Topic Representations" section)
**Measures:** Semantic diversity among query-relevant topics only

**Interpretation:**
- Higher values indicate query-relevant topics cover different facets of the query
- Lower values indicate query-relevant topics are redundant/overlapping
- Unlike overall diversity, this excludes "noise" topics that artificially inflate diversity
- Requires at least 2 relevant topics to compute (otherwise returns None)

**Relevance effects:**
- Expected: Retrieval methods may have **higher** relevant diversity than keyword search
- The key insight: Keyword Search's high overall diversity is inflated by off-topic noise
- QE/DR may discover distinct, nuanced facets (gun violence, domestic violence, cybercrime) while KS drowns them in noise

**Related metrics:** Semantic Diversity (overall), Relevance-Weighted Diversity, Relevant Diversity Ratio

---

### 15. Relevance-Weighted Diversity

**Range:** 0 to 1
**Computation:** Weighted average pairwise distance between topics, where weights are the product of query similarities for each topic pair: `weight_ij = sim_i × sim_j`
**Measures:** Diversity with more weight given to highly query-relevant topic pairs

**Interpretation:**
- Higher values indicate diverse, query-relevant topics
- Lower values indicate either low diversity or low relevance
- No hard threshold - uses continuous weighting by relevance
- More robust than hard threshold when relevance values are near 0.5

**Relevance effects:**
- Combines relevance and diversity into a single score
- High relevance + high diversity = high weighted diversity
- High relevance + low diversity = moderate weighted diversity
- Low relevance (regardless of diversity) = low weighted diversity

**Related metrics:** Relevant Topic Diversity (hard threshold version), Topic-Query Similarity

---

### 16. Top-K Relevant Diversity

**Range:** 0 to 1
**Computation:** Average pairwise distance among the top-K most query-relevant topics (default K=5)
**Measures:** Diversity among the best query-aligned topics

**Interpretation:**
- Higher values indicate top topics cover different query facets
- Lower values indicate top topics are redundant
- Fixed K makes it comparable across methods with different topic counts
- Robust to methods that produce many low-relevance topics

**Relevance effects:**
- Focuses on the "best" topics regardless of total topic count
- Useful when only top topics matter (e.g., summarization)
- May reveal that retrieval methods produce more diverse top topics

**Related metrics:** Relevant Topic Diversity, Top-3 Average Similarity

---

### 17. Number of Relevant Topics

**Range:** 0 to N (total topics)
**Computation:** Count of topics with query similarity ≥ 0.5 threshold
**Measures:** How many topics are meaningfully related to the query

**Interpretation:**
- Higher values indicate more query-relevant topics discovered
- Lower values indicate fewer relevant topics (more noise)
- Combined with Query-Relevant Ratio, shows absolute vs. relative relevance
- Zero relevant topics indicates complete failure to find query-related structure

**Relevance effects:**
- Expected: Retrieval methods discover more relevant topics
- KS may have fewer relevant topics despite having more total topics
- This is the absolute count; Query-Relevant Ratio is the percentage

**Related metrics:** Query-Relevant Ratio, Number of Topics

---

### 18. Relevant Diversity Ratio

**Range:** 0 to 1+ (can exceed 1.0)
**Computation:** `relevant_topic_diversity / overall_semantic_diversity`
**Measures:** What proportion of a method's diversity comes from meaningful (query-relevant) topics

**Interpretation:**
- Values near 1.0 indicate diversity is mostly from relevant topics
- Values < 1.0 indicate diversity is inflated by noise topics
- Values > 1.0 indicate relevant topics are MORE diverse than overall (rare)
- **Key metric for detecting "fake diversity"** from noise

**Relevance effects:**
- **Primary metric for the noise-diversity hypothesis**
- Expected: QE/DR have higher ratios (75-80%) than KS (50-60%)
- KS's high overall diversity is "fake" - inflated by off-topic noise
- Random sampling has lowest ratio (diversity is almost entirely noise)

**Example (Query 43 - Violence):**
| Method | Overall Diversity | Relevant Diversity | Ratio |
|--------|------------------|-------------------|-------|
| Query Expansion | 0.64 | 0.51 | **0.80** |
| Direct Retrieval | 0.51 | 0.38 | **0.75** |
| Keyword Search | 0.82 | 0.43 | **0.53** |
| Random Uniform | 0.86 | 0.36 | **0.42** |

**Related metrics:** Semantic Diversity, Relevant Topic Diversity, Query-Relevant Ratio

---

## Model-Specific Topic Representations

Different topic models represent topics in fundamentally different ways. To ensure valid semantic comparisons, we use **model-appropriate representations** when computing pairwise metrics.

### Word-Based Models (BERTopic, LDA)

**Topic Definition:** Topics are defined by word probability distributions
- **BERTopic:** c-TF-IDF scores determine top words for each topic
- **LDA:** Word probability distributions from Dirichlet allocation

**Representation for Matching:** Top 10 words concatenated as a string
- Example: `"domestic violence abuse partner intimate household family assault women protection"`

**Rationale:** For these models, words ARE the topic definition. Comparing word distributions directly compares what the model learned.

### LLM-Based Models (TopicGPT, HiCODE)

**Topic Definition:** Topics are semantic concepts with natural language labels/descriptions
- **TopicGPT:** LLM generates full descriptions (e.g., "Mental Health and Pandemic Impact: The effects of pandemics on psychological well-being and communication")
- **HiCODE:** LLM generates hierarchical semantic labels (e.g., "domestic and gender-based violence")

**Representation for Matching:** Direct use of semantic description/label
- TopicGPT Example: `"Mental Health and Pandemic Impact: The effects of pandemics on psychological well-being and communication"`
- HiCODE Example: `"domestic and gender-based violence"`

**Rationale:** For LLM models, the semantic concept IS the topic definition. The model's understanding is captured in the natural language description, not in derived word lists. Using TF-IDF words extracted from assigned documents would measure document similarity, not topic similarity.

### Methodological Implications

**Within-Model Consistency:** All comparisons within a single topic model use the same representation method
- BERTopic experiment: All methods (Random, BM25, Query Expansion, etc.) use top 10 words
- HiCODE experiment: All methods use semantic labels
- TopicGPT experiment: All methods use topic descriptions

**Cross-Model Comparisons:** Not recommended
- Different representations produce different numerical ranges
- E.g., "domestic violence" (4 chars) vs "domestic violence abuse partner..." (50+ chars)
- Embedding characteristics differ (short phrase vs long text)

**Metrics Affected:**
- **Topic Semantic Similarity:** Uses appropriate representation (primary metric)
- **Topic Word Overlap:** Only applicable to word-based models (reported as 0.0 for LLM models)
- **Precision/Recall/F1 @ thresholds:** Uses semantic similarity from appropriate representation

**Validity:** Using model-native representations increases construct validity - we measure what each model actually produces rather than forcing all models into a word-distribution framework designed for BERTopic/LDA.

### Word-Level Metrics for LLM Models

Some metrics are inherently **word-level corpus statistics** (NPMI coherence, embedding coherence, lexical diversity, topic specificity). For LLM-based models, these metrics measure different properties than for word-based models:

**For Word-Based Models (BERTopic, LDA):**
- Use top words from topic distributions (c-TF-IDF scores, probability distributions)
- These words ARE the topic definition
- Metrics measure properties of the **topic itself**

**For LLM-Based Models (TopicGPT, HiCODE):**
- Topics are semantic concepts, not word distributions
- Use TF-IDF words extracted from documents assigned to each topic
- Metrics measure properties of the **document collection**, not the topic concept
- This is a fundamentally different measurement

**Affected Metrics:**
1. **NPMI Coherence:** For LLM models, measures if documents in the cluster use coherent vocabulary (not if the topic concept is coherent)
2. **Embedding Coherence:** For LLM models, measures semantic tightness of document vocabulary (not concept coherence)
3. **Lexical Diversity:** For LLM models, measures vocabulary redundancy across document clusters (not across topic labels)
4. **Topic Specificity (IDF):** For LLM models, measures specificity of document vocabulary (not concept specificity)

**HiCODE-Specific:** For hierarchical word lists, we use the **first hierarchical level** (most specific/fine-grained) as it best represents the document cluster's core vocabulary.

**Interpretation:**
- These metrics remain useful for LLM models but answer different questions
- Document-level coherence and specificity provide insights into cluster quality
- Should not be directly compared across model types (different constructs)
- Within a single LLM-based topic model, comparisons across methods remain valid

**Example:**
- **Topic concept (HiCODE):** "domestic and gender-based violence"
- **Document words (TF-IDF):** ["domestic", "violence", "assault", "abuse", "intimate", "partner", "household", "family", "women", "protection"]
- **NPMI coherence** measures: Do these document words co-occur frequently? (High = documents are coherent)
- **Does NOT measure:** Is the concept "domestic and gender-based violence" coherent? (That's inherently clear from the semantic label)

---

## Pairwise Comparison Metrics (Between Methods)

These compare two methods to quantify differences and similarities.

**Note:** Pairwise metrics use model-appropriate topic representations (see above). Within each experiment using a single topic model, all comparisons are consistent and directly comparable.

### 19. Topic Word Overlap (Jaccard)

**Range:** 0 to 1
**Computation:** Jaccard similarity of top-10 words for Hungarian-matched topic pairs
**Measures:** Lexical overlap between matched topics

**Applicability:** **Word-based models only** (BERTopic, LDA)
- For LLM-based models (TopicGPT, HiCODE), reported as 0.0
- Reason: LLM models use semantic descriptions/labels, not word lists
- Word overlap is not meaningful for comparing phrases like "domestic and gender-based violence" vs "intimate partner violence"

**Interpretation (for word-based models):**
- Higher values indicate methods use similar vocabulary for topics
- Lower values indicate methods use different words (may still be semantically similar)
- Sensitive to synonym differences (use Topic Semantic Similarity for robustness)

**Method comparisons (for word-based models):**
- High overlap = methods discover similar topics lexically
- Low overlap + high semantic similarity = synonymous topics
- Low overlap + low semantic similarity = different topics entirely

**Related metrics:** Topic Semantic Similarity (robust to synonyms and applicable to all model types)

---

### 20. Topic Semantic Similarity

**Range:** 0 to 1
**Computation:**
1. Get model-appropriate topic representations:
   - **Word-based models:** Concatenate top-10 words
   - **LLM models:** Use semantic descriptions/labels directly
2. Encode representations using sentence transformer (all-mpnet-base-v2)
3. Compute cosine similarity matrix between all topic pairs
4. **Hungarian algorithm** finds optimal 1-to-1 matching that maximizes total similarity
5. Report mean similarity of matched pairs
6. Creates `min(n_topics_A, n_topics_B)` matched pairs

**Measures:** Semantic similarity between optimally-matched topics

**Interpretation:**
- Higher values indicate methods discover conceptually similar topics
- More robust than word overlap (captures synonyms and paraphrases)
- Used for thresholding in F1/Precision/Recall metrics
- **Hungarian matching ensures fairness**: Each topic matched to its best counterpart (1-to-1 constraint)
- **Model-aware representation ensures validity**: Compares what each model actually outputs

**Method comparisons:**
- High similarity = methods produce similar topic structures
- Low similarity = methods produce different topic structures
- Compare retrieval methods to each other (stability) vs. random (difference)
- **Within-model only**: Only compare methods using the same topic model (e.g., all HiCODE or all BERTopic)

**Important constraint:**
- **1-to-1 matching only**: If method A has 50 topics and method B has 30, only 30 pairs are created
- Unmatched topics (20 in this case) are ignored in similarity calculation
- This affects precision/recall: they measure coverage of the matchable topic space

**Related metrics:** Topic Word Overlap (lexical vs. semantic), Precision/Recall @ thresholds (use this similarity for matching)

---

### 21-23. F1 @ Thresholds (0.5, 0.6, 0.7)

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

### 24-26. Precision_B @ Thresholds (0.5, 0.6, 0.7)

**Range:** 0 to 1
**Computation:**
1. Hungarian algorithm creates optimal 1-to-1 topic matchings between methods A and B
2. Count matched pairs with semantic similarity ≥ threshold (0.5, 0.6, or 0.7)
3. `precision_b = matched_count / n_topics_B`

**Measures:** What fraction of method B's topics are covered by method A at the given similarity threshold

**Interpretation:**
- Higher values indicate most of B's topics have high-similarity matches in A
- Lower values indicate many of B's topics are NOT covered by A (unique to B)
- **Asymmetric**: `precision_b(A,B) ≠ precision_b(B,A)` when topic counts differ
- Same numerator as recall_a, but different denominator (n_topics_B vs n_topics_A)

**Method comparisons:**
- **High precision_b**: A "covers" most of B's topic space
- **Low precision_b**: B discovers many topics not present in A
- **Example**: If Direct Retrieval (37 topics) vs Query Expansion (50 topics) has 31 matched pairs at threshold 0.7:
  - `precision_b = 31/50 = 0.62` (62% of QE's topics covered by DR)
  - `recall_a = 31/37 = 0.84` (84% of DR's topics covered by QE)

**Mathematical relationship:**
- `precision_b(A, B) = recall_a(B, A)` (they measure the same coverage, from opposite perspectives)

**Related metrics:** Recall_A @ thresholds (complementary metric), F1 @ thresholds

---

### 27-29. Recall_A @ Thresholds (0.5, 0.6, 0.7)

**Range:** 0 to 1
**Computation:**
1. Hungarian algorithm creates optimal 1-to-1 topic matchings between methods A and B
2. Count matched pairs with semantic similarity ≥ threshold (0.5, 0.6, or 0.7)
3. `recall_a = matched_count / n_topics_A`

**Measures:** What fraction of method A's topics are covered by method B at the given similarity threshold

**Interpretation:**
- Higher values indicate most of A's topics have high-similarity matches in B
- Lower values indicate many of A's topics are NOT covered by B (unique to A)
- **Asymmetric**: `recall_a(A,B) ≠ recall_a(B,A)` when topic counts differ
- Same numerator as precision_b, but different denominator (n_topics_A vs n_topics_B)

**Method comparisons:**
- **High recall_a**: B "covers" most of A's topic space
- **Low recall_a**: A discovers many topics not present in B
- **Example**: If Keyword Search (51 topics) vs Direct Retrieval (37 topics) has 19 matched pairs at threshold 0.7:
  - `recall_a = 19/51 = 0.37` (37% of KS's topics covered by DR)
  - `precision_b = 19/37 = 0.51` (51% of DR's topics covered by KS)

**Mathematical relationship:**
- `recall_a(A, B) = precision_b(B, A)` (they measure the same coverage, from opposite perspectives)

**Related metrics:** Precision_B @ thresholds (complementary metric), F1 @ thresholds

---

### 30-32. Relevant Coverage A→B @ Thresholds (0.5, 0.6, 0.7)

**Range:** 0 to 1
**Computation:**
1. Filter topics by query-relevance: topics with query-topic similarity ≥ 0.5
2. Match only query-relevant topics using Hungarian algorithm
3. Count matched pairs with topic-topic similarity ≥ threshold (0.5, 0.6, or 0.7)
4. `relevant_coverage_a_to_b = matched_count / n_relevant_topics_B`

**Measures:** What fraction of method B's **query-relevant** topics are covered by method A's **query-relevant** topics

**Key difference from precision_b:**
- **Precision_B**: Matches ALL topics (including off-topic/generic topics)
- **Relevant Coverage**: Matches ONLY query-relevant topics (similarity to query ≥ 0.5)

**Interpretation:**
- Higher values indicate A's relevant topics cover most of B's relevant topic space
- Lower values indicate B discovers many relevant topics not covered by A (complementarity)
- **Asymmetric**: `relevant_coverage_a_to_b(A,B) ≠ relevant_coverage_b_to_a(A,B)`

**Use cases:**
- **Coverage analysis**: "What % of Keyword Search's relevant topics are covered by Direct Retrieval's relevant topics?"
- **Complementarity**: Low coverage = methods discover different query-relevant aspects
- **Redundancy**: High coverage = methods discover similar query-relevant topics

**Example:**
- Direct Retrieval: 5 relevant topics (out of 8 total)
- Keyword Search: 5 relevant topics (out of 59 total)
- Matched relevant topics @0.7: 1
- `relevant_coverage_a_to_b = 1/5 = 0.20` (20% of KS's relevant topics covered by DR)
- `relevant_coverage_b_to_a = 1/5 = 0.20` (20% of DR's relevant topics covered by KS)
- → 80% complementarity: each method discovers unique relevant topics!

**Mathematical relationship:**
- `relevant_coverage_a_to_b(A, B) = relevant_coverage_b_to_a(B, A)` (complementary perspectives)

**Related metrics:**
- Unique Relevant Topics A/B @ thresholds (count of non-overlapping relevant topics)
- Shared Relevant Topics @ thresholds (count of overlapping relevant topics)
- Relevant Coverage B→A @ thresholds (reverse direction)

---

### 33-35. Unique Relevant Topics A @ Thresholds (0.5, 0.6, 0.7)

**Range:** 0 to n_relevant_topics_A
**Computation:**
1. Filter topics by query-relevance (query-topic similarity ≥ 0.5)
2. Match only relevant topics using Hungarian algorithm
3. Count relevant topics in A with NO match in B at the threshold
4. `unique_relevant_a = count(A_relevant topics with max_similarity_to_B < threshold)`

**Measures:** How many of method A's query-relevant topics are NOT covered by method B's query-relevant topics

**Interpretation:**
- Higher values indicate A discovers many unique relevant aspects
- Lower values indicate most of A's relevant topics are also in B
- **Complementarity indicator**: High unique counts = methods discover different things

**Use cases:**
- **Unique contribution**: "Direct Retrieval discovers 3 unique relevant topics not in Keyword Search"
- **Method diversity**: Comparing unique counts across method pairs
- **Ensemble potential**: High unique counts suggest combining methods would increase coverage

**Example:**
- Direct Retrieval: 5 relevant topics @0.7
  - 3 topics have NO match in Keyword Search (similarity < 0.7)
  - `unique_relevant_a = 3`
  - Unique topics: "Victimization", "Social Compliance", "Public Health Interventions (youth violence)"

**Related metrics:**
- Unique Relevant Topics B (symmetric metric for method B)
- Shared Relevant Topics (topics covered by both methods)
- Unique Relevant Ratio A (fraction of A's relevant topics that are unique)

---

### 36-38. Shared Relevant Topics @ Thresholds (0.5, 0.6, 0.7)

**Range:** 0 to min(n_relevant_topics_A, n_relevant_topics_B)
**Computation:**
1. Filter topics by query-relevance (query-topic similarity ≥ 0.5)
2. Match only relevant topics using Hungarian algorithm
3. Count matched pairs with similarity ≥ threshold
4. `shared_relevant = matched_count`

**Measures:** How many query-relevant topics appear in BOTH methods (matched at threshold)

**Interpretation:**
- Higher values indicate significant overlap in what methods discover
- Lower values indicate methods discover mostly different things (complementary)
- **Redundancy indicator**: High shared counts = methods are redundant

**Use cases:**
- **Core topics**: "2 topics appear in both Direct Retrieval and Keyword Search"
- **Stability**: Topics consistently discovered across methods
- **Ensemble efficiency**: Low shared counts suggest combining methods adds unique value

**Example:**
- Direct Retrieval vs Keyword Search @0.7:
  - Shared: 2 topics ("Crime and Public Safety", "Criminal Justice Policy")
  - These are core topics discovered by both methods
  - Unique to DR: 3 topics (violence-specific)
  - Unique to KS: 3 topics (mental health, race, environment)

**Visualization:**
```
DR (5 relevant)  KS (5 relevant)
[3 unique] + [2 shared] + [3 unique]
   60%         40%          60%
```

**Related metrics:**
- Unique Relevant Topics A/B (non-overlapping topics)
- Relevant Coverage metrics (same numerator, different denominators)

---

### 39-41. Unique Relevant Ratio A @ Thresholds (0.5, 0.6, 0.7)

**Range:** 0 to 1
**Computation:** `unique_relevant_ratio_a = unique_relevant_a / n_relevant_topics_A`

**Measures:** Fraction of method A's query-relevant topics that are unique (not covered by B)

**Interpretation:**
- 1.0 = All of A's relevant topics are unique (complete complementarity)
- 0.0 = None of A's relevant topics are unique (complete coverage by B)
- 0.6 = 60% of A's relevant topics are unique (high complementarity)

**Use cases:**
- **Complementarity score**: Quantify how much unique value each method provides
- **Ensemble selection**: High ratios indicate method should be included in ensemble
- **Method comparison**: Compare uniqueness across different method pairs

**Example:**
- Direct Retrieval: 5 relevant topics, 3 unique @0.7
  - `unique_relevant_ratio_a = 3/5 = 0.60`
  - 60% of DR's relevant topics are unique contributions
- Keyword Search: 5 relevant topics, 3 unique @0.7
  - `unique_relevant_ratio_b = 3/5 = 0.60`
  - 60% of KS's relevant topics are unique contributions
- **High complementarity**: Both methods contribute substantial unique value!

**Related metrics:**
- Unique Relevant Ratio B (symmetric metric for method B)
- Complementarity Score: average of unique_ratio_a and unique_ratio_b

---

### 42. NPMI Coherence Difference

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

### 31. Embedding Coherence Difference

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

### 32. Semantic Diversity Difference

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

### 33. Lexical Diversity Difference

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

### 34. ARI (Adjusted Rand Index)

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

### 35. NMI (Normalized Mutual Information)

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

### 36. Document Overlap (Jaccard)

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

**Type:** 4×2 bar chart grid
**Shows:** Per-method values for NPMI Coherence, Embedding Coherence, Semantic Diversity, Lexical Diversity, Document Coverage, Topic Specificity, Number of Topics, Relevant Document Concentration

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

**Type:** Method × Method heatmap (6×6 grid)
**Metrics:** 13 pairwise metrics (similarity, overlap, F1, precision, recall, coherence diff, etc.)

**Usage:**
- Compare all method pairs
- Identify which methods are most similar/different
- Diagonal = self-comparison (always 1.0 for precision/recall)
- **Symmetric metrics** (e.g., F1, topic_semantic_similarity) produce symmetric heatmaps
- **Asymmetric metrics** (e.g., precision_b, recall_a) produce **asymmetric heatmaps**

**Important:**
- **Precision_B and Recall_A heatmaps are asymmetric** (fixed in Dec 2025)
  - Row i, Col j for precision_b: "How much of method j's topics are covered by method i?"
  - Row i, Col j for recall_a: "How much of method i's topics are covered by method j?"
  - `heatmap[i,j] ≠ heatmap[j,i]` when topic counts differ
- See [PRECISION_RECALL_FIX.md](PRECISION_RECALL_FIX.md) for details on the visualization bug fix

---

### 18-20. Relevant Coverage Heatmaps (NEW - 2026-02)

**Type:** Method × Method heatmap at thresholds 0.5, 0.6, 0.7
**Files:** `pairwise_relevant-coverage-a-to-b--at-05_heatmap.png`, etc.

**How to Read:**
- **Row = Method A, Column = Method B**
- **Value = fraction of B's query-relevant topics covered by A's query-relevant topics**
- Read as: "A covers X% of B's relevant topics"

**Example:**
- Row: `direct_retrieval`, Col: `sbert`, Value: `0.470`
- Interpretation: direct_retrieval covers 47% of sbert's relevant topics

**Key Difference from Precision Heatmap:**
- **Precision_B**: Matches ALL topics (including off-topic/generic topics)
- **Relevant Coverage**: Matches ONLY query-relevant topics (similarity to query ≥ 0.5)

**Usage:**
- Identify complementary methods (low coverage = discover different relevant topics)
- Identify redundant methods (high coverage = discover same relevant topics)
- Compare across thresholds to see matching strictness sensitivity

---

### 21. Relevant vs All Coverage Comparison (NEW - 2026-02)

**Type:** Scatter plot
**File:** `aggregate_relevant_vs_all_coverage_comparison.png`

**Axes:**
- **X-axis:** All Topics Coverage (precision_b @0.7)
- **Y-axis:** Relevant Topics Only Coverage (relevant_coverage_a_to_b @0.7)

**Interpretation:**
- **Diagonal line:** Equal coverage for all vs relevant topics
- **Points ABOVE diagonal:** Relevant topics are more focused (A covers B's relevant topics better than overall)
- **Points BELOW diagonal:** Relevant topics are more distinctive (coverage drops when filtering to relevant only)

**Key Insight:**
- Most points fall BELOW the diagonal
- This means query-relevant topics are MORE distinctive between methods than general topics
- Methods share generic topics but discover DIFFERENT relevant topics

---

### 22. Unique/Shared Relevant Topics Distribution (NEW - 2026-02)

**Type:** Stacked bar chart
**File:** `aggregate_unique_shared_distribution.png`

**Shows:** For each method pair, the breakdown of:
- **Unique to A:** Relevant topics only in method A
- **Shared:** Relevant topics in both methods
- **Unique to B:** Relevant topics only in method B

**Usage:**
- Visualize complementarity at a glance
- Tall "unique" bars = high complementarity (methods discover different things)
- Tall "shared" bars = high redundancy (methods discover same things)

---

### 23. Cross-Query Coverage Analysis (NEW - 2026-02)

**Type:** Multi-panel plot showing coverage metrics across all queries
**File:** `cross_query_coverage_analysis.png`

**Shows:**
- Per-query breakdown of relevant coverage metrics
- Variance in coverage across different queries
- Query-specific complementarity patterns

**Usage:**
- Identify queries where methods are more/less complementary
- Understand query-dependent effects on topic discovery

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

- **Query Term Overlap** - Measure query-term bias in topics
- **Corpus Coverage** - How many corpus documents match discovered topics
- **Topic Purity** - Cluster purity using pseudo-clusters (half-baked idea)
- **Vendiscore** - External topic quality metric with Python package

**Recently Implemented (2025-11):**
- ✅ **Topic Specificity (IDF-based)** - Now metric #9
- ✅ **Relevant Document Concentration** - Now metric #8
- ✅ **Document Overlap (Jaccard)** - Now metric #36

**Recently Implemented (2026-01):**
- ✅ **Relevant Topic Diversity** - Now metric #14 (diversity among query-relevant topics only)
- ✅ **Relevance-Weighted Diversity** - Now metric #15 (diversity weighted by query relevance)
- ✅ **Top-K Relevant Diversity** - Now metric #16 (diversity among top-K most relevant topics)
- ✅ **Number of Relevant Topics** - Now metric #17 (count of topics above relevance threshold)
- ✅ **Relevant Diversity Ratio** - Now metric #18 (key metric for detecting "fake diversity" from noise)

**Recently Implemented (2026-02):**
- ✅ **Relevant Coverage A→B @ Thresholds** - Now metrics #30-32 (pairwise: fraction of B's relevant topics covered by A)
- ✅ **Relevant Coverage B→A @ Thresholds** - Symmetric metric (fraction of A's relevant topics covered by B)
- ✅ **Unique Relevant Topics A/B @ Thresholds** - Now metrics #33-35 (count of unique relevant topics per method)
- ✅ **Shared Relevant Topics @ Thresholds** - Now metrics #36-38 (count of relevant topics in both methods)
- ✅ **Unique Relevant Ratio A/B @ Thresholds** - Now metrics #39-41 (fraction of relevant topics that are unique)
- ✅ **New Visualizations** - Relevant coverage heatmaps, unique/shared distribution plots, coverage comparison scatter

---

## Usage

### Running Evaluation

```bash
python src/end_to_end_evaluation.py
```

Results saved to:
```
results/topic_evaluation/query_X/
├── per_method_summary.csv       # All per-method metrics (36 metrics)
├── pairwise_metrics.csv         # All pairwise comparisons
└── plots/                       # All visualizations

# Per-method summary columns include:
# - method, n_topics, n_docs
# - diversity_semantic, diversity_lexical
# - npmi_coherence, embedding_coherence
# - relevant_concentration, topic_specificity
# - outlier_ratio, document_coverage, avg_topic_size
# - topic_query_similarity, max_query_similarity, query_relevant_ratio, top3_avg_similarity
# - relevant_topic_diversity, relevance_weighted_diversity, topk_relevant_diversity
# - n_relevant_topics, relevant_diversity_ratio
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
