# Retrieval Guided Coreset Construction for Inductive Labelling

This project implements a comprehensive pipeline for evaluating different document sampling strategies in the context of topic modeling and concept induction. The system compares how well various sampling methods preserve the topical structure of a full corpus when working with limited document subsets.

## Overview

The project addresses the challenge of maintaining topic coherence when sampling documents from large corpora. It implements multiple sampling strategies and evaluates their effectiveness using various metrics, with a focus on how well sampled subsets preserve the topic structure of the full dataset.

## Key Components

### 1. Hybrid Retrieval System (`src/search.py`)
- **BM25 + Dense Retrieval**: Combines lexical (BM25) and semantic (SBERT) search
- **Multiple Fusion Strategies**: RRF, weighted combination, simple sum
- **Diversity Enhancement**: MMR (Maximal Marginal Relevance) for result diversification
- **Cross-encoder Reranking**: Optional reranking for improved relevance

### 2. Query Expansion (`src/query_expansion.py`)
- **KeyBERT-based Expansion**: Extracts semantically relevant keywords
- **Multiple Combination Strategies**: 
  - Weighted RRF for combining original and expanded queries
  - Concatenated queries with optional cross-encoder reranking
- **Evaluation Framework**: Compares expansion effectiveness against baselines

### 3. Document Sampling (`src/sampling.py`)
- **Multiple Sampling Methods**:
  - Random sampling
  - Uniform sampling (stratified by document length)
  - Retrieval-based sampling
  - Query expansion-based sampling
  - Clustering-based sampling
- **Flexible Pipeline**: Supports chaining methods (e.g., retrieve then cluster)

### 4. Clustering-Based Sampling (`src/clustering.py`)
- **Dimensionality Reduction**: UMAP or PCA for embedding compression
- **Clustering Algorithms**: KMeans (balanced) or HDBSCAN (density-based)
- **Representative Selection**: Centroid, random, density-based, or max-probability selection

### 5. Topic Modeling (`src/topic_modelling.py`)
- **BERTopic Integration**: Advanced topic modeling with multiple configuration options
- **Guided Topic Modeling**: Query-informed topic discovery
- **Multiple Sampling Strategies**: Supports all sampling methods for comparative analysis
- **Visualization**: Automatic generation of topic distribution and similarity plots

### 6. Evaluation Framework (`src/topic_evaluation.py`)
- **Document-Level Metrics**:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Omega Index
  - B-Cubed Precision, Recall, and F1
- **Topic-Level Metrics**:
  - Topic word overlap (Jaccard similarity)
  - Semantic similarity using embeddings
  - Greedy and Hungarian algorithm matching
- **Comprehensive Visualizations**: Radar charts, heatmaps, comparative plots

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd retrieval-guided-coreset

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"