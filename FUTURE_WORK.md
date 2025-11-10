# Future Work - Planned Extensions

## Overview

This document outlines planned experiments, metrics, and extensions to the retrieval-guided topic modeling investigation.

---

## Planned Metrics

### High Priority

#### 1. Topic Specificity (IDF-based)

**Motivation:** Operationalize the observed "less generic, more nuanced" topics from retrieval-based samples

**Computation:**
```python
# For each topic:
topic_specificity = mean(IDF(word) for word in top_k_words)

# IDF computed over full corpus:
IDF(word) = log(total_docs / docs_containing_word)
```

**Interpretation:**
- Higher values = more specific/technical terms (rare in corpus)
- Lower values = more generic terms (common in corpus)
- Expected: Retrieval-based topics have higher specificity

**Implementation:** Use sklearn's TfidfVectorizer with fit on full corpus, then compute IDF for topic words

---

#### 2. Relevant Document Concentration

**Motivation:** Validate that retrieval actually increases sample relevance using ground-truth QRELs

**Computation:**
```python
# For each sample:
relevant_docs = set(doc_ids) & set(qrels[query_id].keys())
concentration = len(relevant_docs) / len(doc_ids)
```

**Interpretation:**
- Range: 0 to 1
- Higher values = more relevant documents in sample
- Expected: Retrieval > Random
- Validates relevance control mechanism

**Implementation:** Load QRELs, intersect with sample doc_ids, compute ratio

---

#### 3. Document Overlap (Jaccard) Between Samples

**Motivation:** Quantify how much samples differ across methods, helps interpret topic differences

**Computation:**
```python
# For each method pair:
overlap_jaccard = len(set(doc_ids_A) & set(doc_ids_B)) / len(set(doc_ids_A) | set(doc_ids_B))
```

**Interpretation:**
- Range: 0 (disjoint) to 1 (identical)
- Expected: Low overlap (random vs. retrieval), moderate overlap (retrieval methods)
- Low overlap + high topic similarity = robust topic discovery

**Implementation:** Already computed in code (metrics["overlap_count"]), need to compute Jaccard explicitly

**Status:** ⚠️ PARTIALLY IMPLEMENTED - overlap count exists, need Jaccard similarity

---

#### 4. Relevancy vs. Diversity Visualization

**Motivation:** Visualize primary trade-off between query alignment and topic diversity

**Type:** 2D scatter plot
- **X-axis:** Topic-Query Similarity (Avg) or Relevant Document Concentration
- **Y-axis:** Semantic Diversity
- **Points:** One per method
- **Annotations:** Method labels

**Interpretation:**
- Top-right quadrant: High relevance + high diversity (ideal?)
- Top-left quadrant: Low relevance + high diversity (random sampling expected)
- Bottom-right quadrant: High relevance + low diversity (retrieval trade-off?)
- Bottom-left quadrant: Low relevance + low diversity (worst case)

**Implementation:** Add to visualization pipeline in `end_to_end_evaluation.py`

**Status:** 📝 NOT IMPLEMENTED

---

### Medium Priority

#### 5. Query Term Overlap

**Motivation:** Measure query-term bias in discovered topics (complements lexical diversity)

**Computation:**
```python
# For each topic:
query_terms = set(query_text.lower().split())
topic_terms = set(topic_words[:10])
overlap = len(query_terms & topic_terms) / len(topic_terms)

# Aggregate:
query_term_overlap = mean(overlap for all topics)
```

**Interpretation:**
- Range: 0 to 1
- Higher values = topics contain more query words
- Expected: Retrieval > Random
- Explains lower lexical diversity in retrieval methods

**Implementation:** Tokenize query, compute overlap with each topic, average

---

#### 6. Corpus Coverage

**Motivation:** Measure topic generalizability - do discovered topics apply to corpus beyond the sample?

**Computation:**
```python
# For each topic:
topic_signature = " ".join(topic_words[:10])
corpus_matches = search_corpus_for_topic(topic_signature, top_k=1000)
corpus_coverage = len(corpus_matches) / len(corpus)

# Alternative: Use topic classifier on full corpus
```

**Interpretation:**
- Higher values = topic applies to many corpus documents (generalizable)
- Lower values = topic specific to sample (query-focused)
- Trade-off: High coverage = generic topics, low coverage = specific topics

**Implementation:** Complex - requires either search or classifier application to full corpus

---

#### 7. Topic Purity (Cluster Purity from QRELs)

**Motivation:** Use QRELs as ground-truth labels to measure clustering quality

**Status:** ⚠️ HALF-BAKED IDEA - needs refinement

**Computation (draft):**
```python
# For each topic (pseudo-cluster):
relevant_docs_in_topic = set(topic_doc_ids) & set(qrels[query_id].keys())
non_relevant_docs_in_topic = set(topic_doc_ids) - relevant_docs_in_topic

# Purity: majority class fraction
purity = max(len(relevant_docs), len(non_relevant_docs)) / len(topic_doc_ids)

# Aggregate across topics
avg_purity = mean(purity for all topics)
```

**Issues to resolve:**
- QRELs only cover small subset of corpus (most docs unlabeled)
- Purity is asymmetric (favors small clusters)
- May need to use only topics containing labeled documents

**Alternative:** Use entropy or Gini impurity instead of purity

---

### Low Priority / Exploratory

#### 8. Vendiscore

**Motivation:** External topic quality metric with established Python package

**Package:** `vendiscore` (available on PyPI)

**Computation:** See package documentation

**Interpretation:** TBD - investigate metric properties

**Status:** 📝 EXPLORATORY - need to research metric definition and applicability

---

#### 9. Topic Stability Across Runs

**Motivation:** Measure randomness in topic modeling (HDBSCAN/UMAP stochasticity)

**Experiment:**
- Run BERTopic 5 times on same sample with different random seeds
- Compute topic similarity across runs (Hungarian matching)
- Report mean ± std similarity

**Interpretation:**
- High similarity = stable topics (deterministic)
- Low similarity = unstable topics (stochastic)

**Status:** 📝 FUTURE EXPERIMENT

---

#### 10. Topic Evolution Across Sample Sizes

**Motivation:** Understand how sample size affects topic quality

**Experiment:**
- Sample sizes: [100, 250, 500, 1000, 2500, 5000]
- Run all methods at each size
- Plot metrics vs. sample size

**Questions:**
- At what sample size do topics stabilize?
- Does retrieval require smaller samples than random?

**Status:** 📝 FUTURE EXPERIMENT (conflicts with fixed sample size design)

---

## Planned Retrieval Ablations

### High Priority

#### 1. Direct Retrieval + MMR

**Motivation:** Test diversity enforcement alongside relevance control

**Configuration:**
- Base: Direct Retrieval (Hybrid BM25+SBERT, Simple Sum)
- Enhancement: MMR with λ=0.7 (70% relevance, 30% diversity)

**Expected effect:**
- Higher semantic diversity than Direct Retrieval
- Similar query alignment to Direct Retrieval
- Tests: "Can diversity enforcement counteract narrowing from retrieval?"

**Implementation:** Add `use_mmr=True, mmr_lambda=0.7` to search call

**Status:** 📝 PLANNED - easy to implement

---

### Medium Priority

#### 2. Cross-Encoder Reranking

**Motivation:** Test highest-quality relevance estimation (BERT-based)

**Configuration:**
- Retrieve top-2000 with Hybrid search
- Rerank with cross-encoder (ms-marco-MiniLM-L-6-v2)
- Take top-1000

**Expected effect:**
- Highest query alignment (most accurate relevance)
- May have lowest diversity (most focused)

**Implementation:** Add `use_cross_encoder=True` to search call

**Status:** 📝 PLANNED - moderate runtime increase (~2-3x)

---

#### 3. SBERT-Only Retrieval

**Motivation:** Pure semantic retrieval (no lexical component)

**Configuration:**
- SBERT retrieval only (no BM25)
- Compare to BM25-only (Keyword Search) and Hybrid (Direct Retrieval)

**Expected effect:**
- Higher semantic coherence (embedding-based)
- May miss exact-match important terms (lower precision?)

**Implementation:** Use `method=RetrievalMethod.SBERT`

**Status:** 📝 PLANNED - easy to implement

---

#### 4. Alternative Fusion Strategies

**Motivation:** Test different ways to combine BM25 and SBERT

**Configurations:**
- Weighted Sum (0.3 BM25 + 0.7 SBERT)
- Reciprocal Rank Fusion (RRF)
- Compare to Simple Sum (current)

**Expected effect:**
- Weight toward SBERT → more semantic alignment
- Weight toward BM25 → more lexical overlap

**Implementation:** Add `hybrid_strategy=HybridStrategy.WEIGHTED` or `RRF`

**Status:** 📝 PLANNED - implemented but not tested in pipeline

---

### Low Priority

#### 5. Alternative Embedding Models

**Motivation:** Test embedding model effect on retrieval and topic modeling

**Models:**
- Current: all-mpnet-base-v2 (768 dim, general-purpose)
- Alternative 1: all-MiniLM-L6-v2 (384 dim, faster)
- Alternative 2: msmarco-distilbert-base-v4 (768 dim, passage retrieval optimized)

**Expected effect:**
- Domain-specific models may improve relevance
- Smaller models may reduce coherence (less semantic nuance)

**Status:** 📝 FUTURE WORK - requires cache rebuild

---

## Guided Topic Modeling

### Motivation

Move from **post-hoc query alignment** (current) to **query-informed topic modeling** (future)

### Approaches to Explore

#### 1. Seed Topics from Query

**Method:** Provide query terms as seed topic to BERTopic

**Implementation:**
```python
seed_topic_list = [query_text.split()]  # Query as seed topic
topic_model = BERTopic(seed_topic_list=seed_topic_list)
```

**Expected effect:** At least one topic will match query directly

---

#### 2. Guided UMAP/HDBSCAN

**Method:** Weight embeddings toward query before dimensionality reduction

**Implementation:**
```python
# Weighted embeddings
query_emb = model.encode(query_text)
weighted_embs = doc_embs + 0.1 * query_emb  # Add query signal

# Then UMAP + HDBSCAN
```

**Expected effect:** Topics biased toward query similarity

---

#### 3. Constrained Clustering

**Method:** Force QRELs-relevant documents into specific clusters

**Implementation:** Use semi-supervised clustering with must-link constraints

**Expected effect:** Relevant documents form coherent topics

---

### Evaluation for Guided Topic Modeling

**New metrics needed:**
- Guidance effectiveness: How much do guided topics differ from unguided?
- Overfitting: Do guided topics only work for that query?
- Transfer: Do guided topics generalize to other queries?

**Status:** 📝 FUTURE WORK - major research direction

---

## Cross-Dataset Evaluation

### Motivation

Demonstrate generalizability beyond TREC-COVID

### Candidate Datasets

1. **TREC-DL (Deep Learning Track)** - MS MARCO passages
   - Similar to TREC-COVID (IR evaluation)
   - ~8.8M passages, 200 queries

2. **Natural Questions** - Google Search queries
   - Different domain (web search vs. scientific literature)
   - Question-answering format

3. **HotpotQA** - Multi-hop reasoning
   - Complex information needs
   - Tests diverse query aspects

4. **SciFact** - Scientific claim verification
   - Similar domain to TREC-COVID (biomedical)
   - Different task (verification vs. retrieval)

### Implementation Considerations

- Adapt QRELs loading for different formats
- May need different sample sizes (corpus size varies)
- Some datasets lack relevance judgments (unsupervised evaluation only)

**Status:** 📝 FUTURE WORK - depends on TREC-COVID results

---

## Inductive Labeling Downstream Task

### Motivation

Current work is **prototyping with topic modeling**. End goal is **inductive labeling** for machine learning.

### Workflow (Draft)

1. **Sample documents** (retrieval-based or random)
2. **Discover concepts** (topic modeling or clustering)
3. **Induce labels** (assign topics as pseudo-labels)
4. **Train classifier** (on pseudo-labeled data)
5. **Evaluate** (on held-out ground-truth labels)

### Evaluation Metrics for Inductive Labeling

- **Classification accuracy** - how well do pseudo-labels predict ground truth?
- **Label quality** - coherence/interpretability of induced labels
- **Sample efficiency** - performance vs. sample size
- **Transfer** - do labels work on different queries/datasets?

### Research Questions

- Does retrieval-based sampling improve downstream classification?
- Do "less generic" topics produce better pseudo-labels?
- Trade-off: Query alignment vs. generalizability?

**Status:** 📝 MAJOR FUTURE DIRECTION - needs separate experimental design

---

## Implementation Priorities

### Immediate (Can Implement Now)

1. ✅ **Document Overlap (Jaccard)** - Add to pairwise metrics
2. ✅ **Relevant Document Concentration** - Simple QRELs intersection
3. ✅ **Topic Specificity (IDF)** - Compute from TfidfVectorizer
4. ✅ **Relevancy vs. Diversity Plot** - Add to visualization pipeline
5. ✅ **Direct Retrieval + MMR** - Add as 5th sampling method

### Short-term (1-2 weeks)

1. **Cross-Encoder Reranking** - Add as retrieval variant
2. **SBERT-Only Retrieval** - Add as sampling method
3. **Query Term Overlap** - Simple metric addition
4. **Topic Purity (refined)** - Resolve computation issues

### Medium-term (1-2 months)

1. **Corpus Coverage** - Requires search infrastructure
2. **Vendiscore** - Research and integrate package
3. **Alternative Fusion Strategies** - Test RRF, Weighted Sum
4. **Cross-Query Stability Analysis** - Aggregate results across 15 queries
5. **Guided Topic Modeling (simple)** - Seed topics from query

### Long-term (3+ months)

1. **Cross-Dataset Evaluation** - TREC-DL, Natural Questions
2. **Guided Topic Modeling (advanced)** - Constrained clustering, weighted embeddings
3. **Inductive Labeling Pipeline** - Full downstream task implementation
4. **Topic Stability Experiments** - Multiple runs, random seed analysis
5. **Sample Size Ablation** - Requires substantial compute

---

## Questions to Resolve

### Metric Design

1. **Topic Purity:** How to handle unlabeled documents in QRELs?
2. **Corpus Coverage:** Search-based or classifier-based?
3. **Query Term Overlap:** Lemmatization/stemming or exact match?

### Experimental Design

1. **MMR lambda:** Test multiple values (0.5, 0.7, 0.9) or fix at 0.7?
2. **Cross-encoder:** Rerank full 1000 or retrieve 2000 → rerank → top 1000?
3. **Visualizations:** Which scatter plots are most informative?

### Generalization

1. **Cross-dataset:** Which datasets are highest priority?
2. **Guided topic modeling:** Which guidance method to start with?
3. **Inductive labeling:** What classifier to use (logistic regression, BERT)?

---

## Notes for Future Researchers

- **Caching is critical:** ~420 MB per BERTopic model × 4 methods × 15 queries = 25 GB
- **GPU acceleration:** SBERT embeddings and HDBSCAN benefit from CUDA
- **Reproducibility:** Fix random seeds for HDBSCAN, UMAP, sampling
- **Human evaluation:** Qualitative topic assessment needed alongside metrics
- **Effect sizes matter:** Focus on large, consistent effects (>20% change)
- **Trade-offs over optimization:** No single "best" method - characterize dimensions

---

## Contact / Collaboration

For questions about planned work or collaboration opportunities, see project README.

## Last Updated

2025-11-10
