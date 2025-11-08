# end_to_end_evaluation.py
"""
End-to-end evaluation of BERTopic using different sampling strategies with pairwise comparisons.

Compares 4 methods:
1. Random Uniform Sampling
2. Direct Retrieval (Hybrid BM25+SBERT)
3. Query Expansion + Retrieval
4. Simple Keyword Search (BM25 only)

METRICS IMPLEMENTED:

Intrinsic Quality Metrics (Per-Method):
- NPMI Coherence: Semantic coherence of topic words based on co-occurrence
- Embedding Coherence: Average pairwise cosine similarity of word embeddings within topics
- Semantic Diversity: Average pairwise distance between topics (embedding-based)
- Lexical Diversity: Unique word fraction across all topics (vocabulary redundancy)
- Document Coverage: Fraction of documents assigned to topics (1 - outlier_ratio)
- Number of Topics: Total topics discovered
- Average Topic Size: Mean documents per topic

Query Alignment Metrics (Per-Method):
- Topic-Query Similarity (Avg): Average similarity between all topics and the query
- Max Query Similarity: Highest topic-query similarity score
- Query-Relevant Ratio: Fraction of topics above similarity threshold (0.5)
- Top-3 Avg Similarity: Average similarity of top-3 most query-relevant topics

Pairwise Comparison Metrics:
- Topic Word Overlap (Jaccard): Lexical overlap between matched topics
- Topic Semantic Similarity: Embedding-based similarity between matched topics
- Precision @ thresholds (0.5, 0.6, 0.7): What fraction of B's topics match A's topics
- Recall @ thresholds (0.5, 0.6, 0.7): What fraction of A's topics match B's topics
- F1 @ thresholds (0.5, 0.6, 0.7): Topic matching quality at different thresholds
- NPMI Coherence Difference: Difference in coherence between methods
- Embedding Coherence Difference: Difference in embedding coherence
- Semantic Diversity Difference: Difference in semantic diversity
- Lexical Diversity Difference: Difference in lexical diversity
- ARI (Adjusted Rand Index): Agreement on document clustering (overlap only)
- NMI (Normalized Mutual Information): Shared clustering information (overlap only)

Visualizations Generated:
1. intrinsic_quality_metrics.png: 3x2 grid of NPMI, Embedding Coherence, Semantic/Lexical Diversity, Coverage, Topic Count
2. query_alignment_metrics.png: 1x3 grid of Avg/Max/Ratio query similarity metrics
3. diversity_scatter.png: Semantic vs Lexical diversity scatter plot
4. Pairwise heatmaps (14 plots): Similarity, Overlap, F1 @0.5/0.6/0.7, Precision @0.5/0.6/0.7, Recall @0.5/0.6/0.7, NPMI diff, Embedding diff, ARI
"""

import os
import sys
import json
import pickle
import logging
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# BERTopic imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Import from existing modules
from search import (
    TextPreprocessor,
    IndexManager,
    SearchEngine,
    RetrievalMethod,
    HybridStrategy
)
from evaluation import SearchEvaluationUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class EndToEndEvaluator:
    """End-to-end evaluator for topic modeling with pairwise method comparisons"""

    def __init__(
        self,
        corpus_dataset,
        queries_dataset,
        qrels_dataset,
        query_id: str,
        sample_size: Optional[int] = None,
        embedding_model_name: str = "all-mpnet-base-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dataset_name: str = "trec-covid",
        output_dir: str = "results/topic_evaluation",
        cache_dir: str = "cache",
        random_seed: int = 42,
        device: str = "cpu",
        save_topic_models: bool = False,
        force_regenerate_samples: bool = False,
        force_regenerate_topics: bool = False,
        force_regenerate_evaluation: bool = False
    ):
        """
        Initialize the end-to-end evaluator

        Args:
            corpus_dataset: Full corpus dataset
            queries_dataset: Queries dataset
            qrels_dataset: Relevance judgments dataset
            query_id: Query ID to evaluate
            sample_size: Sample size (if None, determined by qrels count)
            embedding_model_name: Name of embedding model
            cross_encoder_model_name: Name of cross-encoder model
            dataset_name: Dataset name for caching
            output_dir: Output directory for results
            cache_dir: Cache directory
            random_seed: Random seed for reproducibility
            device: Device to use ('cpu', 'cuda', 'mps')
            save_topic_models: Whether to save full BERTopic models (420 MB each, only needed for interactive exploration)
            force_regenerate_samples: Force regenerate samples
            force_regenerate_topics: Force regenerate topic models
            force_regenerate_evaluation: Force regenerate evaluation
        """
        self.corpus_dataset = corpus_dataset
        self.queries_dataset = queries_dataset
        self.qrels_dataset = qrels_dataset
        self.query_id = str(query_id)
        self.embedding_model_name = embedding_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.random_seed = random_seed
        self.device = device
        self.save_topic_models = save_topic_models
        self.force_regenerate_samples = force_regenerate_samples
        self.force_regenerate_topics = force_regenerate_topics
        self.force_regenerate_evaluation = force_regenerate_evaluation

        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Get query text
        self.query_text = self._get_query_text()
        logger.info(f"Query {query_id}: {self.query_text}")

        # Set sample size
        self.sample_size = sample_size
        logger.info(f"Sample size: {self.sample_size}")

        # Create output directories (new unified structure)
        self.output_dir = os.path.join(output_dir, f"query_{query_id}")
        self.samples_dir = os.path.join(self.output_dir, "samples")
        self.topic_models_dir = os.path.join(self.output_dir, "topic_models")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.topics_summary_dir = os.path.join(self.results_dir, "topics_summary")

        for directory in [self.samples_dir, self.topic_models_dir, self.results_dir,
                          self.plots_dir, self.topics_summary_dir]:
            os.makedirs(directory, exist_ok=True)

        # Save configuration
        self._save_config()

        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.index_manager = IndexManager(self.preprocessor)

        # Build or load indices
        self._initialize_indices()

    def _get_query_text(self) -> str:
        """Get query text for given query ID"""
        for query in self.queries_dataset:
            if str(query["_id"]) == self.query_id:
                return query["text"]
        raise ValueError(f"Query ID {self.query_id} not found in queries dataset")


    def _save_config(self):
        """Save configuration to JSON"""
        config = {
            "query_id": self.query_id,
            "query_text": self.query_text if hasattr(self, 'query_text') else None,
            "sample_size": self.sample_size,
            "embedding_model": self.embedding_model_name,
            "cross_encoder_model": self.cross_encoder_model_name,
            "dataset_name": self.dataset_name,
            "random_seed": self.random_seed,
            "force_flags": {
                "samples": self.force_regenerate_samples,
                "topics": self.force_regenerate_topics,
                "evaluation": self.force_regenerate_evaluation
            }
        }

        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")

    def _initialize_indices(self):
        """Initialize or load search indices"""
        logger.info(f"Initializing search indices on device: {self.device}...")

        # Build BM25 index
        self.bm25, self.corpus_texts, self.corpus_ids = self.index_manager.build_bm25_index(
            self.corpus_dataset,
            dataset_name=self.dataset_name,
            force_reindex=False
        )

        # Build SBERT index
        self.sbert_model, self.doc_embeddings = self.index_manager.build_sbert_index(
            self.corpus_texts,
            model_name=self.embedding_model_name,
            dataset_name=self.dataset_name,
            batch_size=64,
            force_reindex=False,
            device=self.device
        )

        # Initialize search engine
        self.search_engine = SearchEngine(
            preprocessor=self.preprocessor,
            bm25=self.bm25,
            corpus_texts=self.corpus_texts,
            corpus_ids=self.corpus_ids,
            sbert_model=self.sbert_model,
            doc_embeddings=self.doc_embeddings,
            cross_encoder_model_name=self.cross_encoder_model_name,
            device=self.device
        )

        logger.info("Search indices initialized")

    def _load_or_compute(self, cache_path: str, compute_fn, force: bool = False):
        """Universal caching pattern"""
        if not force and os.path.exists(cache_path):
            logger.info(f"Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        logger.info(f"Computing (cache miss or force=True)")
        result = compute_fn()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Saved to cache: {cache_path}")

        return result

    def sample_random_uniform(self) -> Dict[str, Any]:
        """Method 1: Random uniform sampling"""
        logger.info(f"Method 1: Random uniform sampling ({self.sample_size} docs)")

        cache_path = os.path.join(self.samples_dir, "random_uniform.pkl")

        def compute():
            # Pure random sampling
            all_indices = list(range(len(self.corpus_dataset)))
            sample_indices = random.sample(all_indices, self.sample_size)

            docs = []
            doc_ids = []
            for idx in tqdm(sample_indices, desc="Extracting random docs"):
                doc = self.corpus_dataset[idx]
                docs.append(doc["title"] + "\n\n" + doc["text"])
                doc_ids.append(doc["_id"])

            return {
                "method": "random_uniform",
                "doc_ids": doc_ids,
                "doc_texts": docs,
                "sample_size": len(docs)
            }

        return self._load_or_compute(cache_path, compute, self.force_regenerate_samples)

    def sample_direct_retrieval(self) -> Dict[str, Any]:
        """Method 2: Direct retrieval (Hybrid BM25+SBERT)"""
        logger.info(f"Method 2: Direct retrieval ({self.sample_size} docs)")

        cache_path = os.path.join(self.samples_dir, "direct_retrieval.pkl")

        def compute():
            # Perform hybrid search
            results = self.search_engine.search(
                query=self.query_text,
                top_k=self.sample_size,
                method=RetrievalMethod.HYBRID,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=False
            )

            doc_ids = [doc_id for doc_id, _ in results]
            doc_texts = []

            for doc_id in tqdm(doc_ids, desc="Extracting retrieved docs"):
                doc_text = self.search_engine.get_document_by_id(doc_id)
                doc_texts.append(doc_text)

            return {
                "method": "direct_retrieval",
                "doc_ids": doc_ids,
                "doc_texts": doc_texts,
                "sample_size": len(doc_ids)
            }

        return self._load_or_compute(cache_path, compute, self.force_regenerate_samples)

    def sample_query_expansion(self) -> Dict[str, Any]:
        """Method 3: Query expansion + retrieval"""
        logger.info(f"Method 3: Query expansion + retrieval ({self.sample_size} docs)")

        cache_path = os.path.join(self.samples_dir, "query_expansion.pkl")

        def compute():
            # Load cached keywords
            keyword_cache_path = "/home/srangre1/cache/keywords/keybert_k10_div0.7_top10docs_mpnet_k1000_ngram1-2.json"

            if not os.path.exists(keyword_cache_path):
                raise FileNotFoundError(f"Keyword cache not found: {keyword_cache_path}")

            with open(keyword_cache_path, 'r') as f:
                all_keywords = json.load(f)

            query_keywords = all_keywords.get(self.query_id, [])

            if not query_keywords:
                logger.warning(f"No keywords found for query {self.query_id}, using direct retrieval")
                return self.sample_direct_retrieval()

            logger.info(f"Using keywords: {query_keywords}")

            # Perform baseline search
            baseline_results = self.search_engine.search(
                query=self.query_text,
                top_k=self.sample_size * 2,  # Retrieve more for fusion
                method=RetrievalMethod.HYBRID,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=False
            )

            # Search with each keyword
            keyword_results = []
            for keyword in tqdm(query_keywords, desc="Searching with keywords"):
                kw_results = self.search_engine.search(
                    query=keyword,
                    top_k=self.sample_size * 2,
                    method=RetrievalMethod.HYBRID,
                    hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                    use_mmr=False,
                    use_cross_encoder=False
                )
                keyword_results.append(kw_results)

            # Weighted RRF fusion
            all_rankings = [baseline_results] + keyword_results
            weights = [0.7] + [0.3 / len(query_keywords)] * len(query_keywords)

            combined = self._reciprocal_rank_fusion(all_rankings, weights)

            # Take top sample_size
            doc_ids = [doc_id for doc_id, _ in combined[:self.sample_size]]
            doc_texts = []

            for doc_id in tqdm(doc_ids, desc="Extracting expanded docs"):
                doc_text = self.search_engine.get_document_by_id(doc_id)
                doc_texts.append(doc_text)

            return {
                "method": "query_expansion",
                "doc_ids": doc_ids,
                "doc_texts": doc_texts,
                "sample_size": len(doc_ids),
                "keywords": query_keywords
            }

        return self._load_or_compute(cache_path, compute, self.force_regenerate_samples)

    def sample_keyword_search(self) -> Dict[str, Any]:
        """Method 4: Simple keyword search (BM25 only)"""
        logger.info(f"Method 4: Simple keyword search ({self.sample_size} docs)")

        cache_path = os.path.join(self.samples_dir, "keyword_search.pkl")

        def compute():
            # Perform BM25-only search
            results = self.search_engine.search(
                query=self.query_text,
                top_k=self.sample_size,
                method=RetrievalMethod.BM25,
                use_mmr=False,
                use_cross_encoder=False
            )

            doc_ids = [doc_id for doc_id, _ in results]
            doc_texts = []

            for doc_id in tqdm(doc_ids, desc="Extracting keyword search docs"):
                doc_text = self.search_engine.get_document_by_id(doc_id)
                doc_texts.append(doc_text)

            return {
                "method": "keyword_search",
                "doc_ids": doc_ids,
                "doc_texts": doc_texts,
                "sample_size": len(doc_ids)
            }

        return self._load_or_compute(cache_path, compute, self.force_regenerate_samples)


    def _reciprocal_rank_fusion(
        self,
        rankings: List[List[Tuple[str, float]]],
        weights: List[float],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """Reciprocal rank fusion for combining rankings"""
        scores = defaultdict(float)

        for i, rank_list in enumerate(rankings):
            weight = weights[i]
            for rank, (doc_id, _) in enumerate(rank_list):
                scores[doc_id] += weight * (1.0 / (k + rank + 1))

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def run_topic_modeling(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run BERTopic on a document sample"""
        method = sample["method"]
        logger.info(f"Running BERTopic on {method} ({sample['sample_size']} docs)")

        model_cache_path = os.path.join(self.topic_models_dir, f"{method}_model.pkl")
        results_cache_path = os.path.join(self.topic_models_dir, f"{method}_results.pkl")

        def compute():
            docs = sample["doc_texts"]
            doc_ids = sample["doc_ids"]

            # Initialize SentenceTransformer with explicit device
            from sentence_transformers import SentenceTransformer
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer
            embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)

            # Initialize HDBSCAN with fixed min_cluster_size for consistency
            hdbscan_model = HDBSCAN(
                min_cluster_size=5,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True
            )

            # Configure CountVectorizer to remove stopwords and filter short words
            vectorizer_model = CountVectorizer(
                stop_words='english',
                min_df=2,  # Minimum document frequency
                ngram_range=(1, 2),  # Unigrams and bigrams
                max_features=10000  # Limit vocabulary size
            )

            # Initialize BERTopic with pre-loaded embedding model and fixed parameters
            topic_model = BERTopic(
                embedding_model=embedding_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                verbose=True,
                calculate_probabilities=True
            )

            # Fit model
            logger.info(f"Fitting BERTopic on {len(docs)} documents...")
            topics, probs = topic_model.fit_transform(docs)

            # Get topic info
            topic_info = topic_model.get_topic_info()

            # Get topic representations with scores
            topic_words = {}
            topic_words_with_scores = {}
            topic_labels = {}

            for topic_id in topic_info['Topic']:
                if topic_id != -1:  # Skip outlier topic
                    words = topic_model.get_topic(topic_id)
                    topic_words[topic_id] = [word for word, _ in words]
                    topic_words_with_scores[topic_id] = [(word, float(score)) for word, score in words]

                    # Generate readable topic label from top words
                    top_words = "_".join([word for word, _ in words[:3]])
                    topic_labels[topic_id] = top_words

            # Get topic names from BERTopic (auto-generated labels)
            topic_names = {}
            for topic_id in topic_info['Topic']:
                if topic_id != -1:
                    # BERTopic stores names in topic_info
                    topic_row = topic_info[topic_info['Topic'] == topic_id]
                    if not topic_row.empty and 'Name' in topic_row.columns:
                        topic_names[topic_id] = topic_row['Name'].values[0]
                    else:
                        # Fallback to label we created
                        topic_names[topic_id] = topic_labels.get(topic_id, f"Topic_{topic_id}")

            # Save model (optional - only if save_topic_models=True)
            # Models are 420 MB each and only needed for interactive exploration
            if self.save_topic_models:
                topic_model.save(model_cache_path, serialization="pickle")
                logger.info(f"Saved topic model to {model_cache_path}")
            else:
                logger.info(f"Skipping model save (save_topic_models=False) - saving ~420 MB")

            # Convert to list if needed (topics might be list or numpy array)
            topics_list = topics.tolist() if hasattr(topics, 'tolist') else topics
            probs_list = probs.tolist() if probs is not None and hasattr(probs, 'tolist') else probs

            results = {
                "method": method,
                "doc_ids": doc_ids,
                "topics": topics_list,
                "probabilities": probs_list,
                "topic_info": topic_info.to_dict(),
                "topic_words": topic_words,
                "topic_words_with_scores": topic_words_with_scores,
                "topic_labels": topic_labels,
                "topic_names": topic_names,
                "n_topics": len(topic_words),
                "n_docs": len(docs)
            }

            return results

        return self._load_or_compute(results_cache_path, compute, self.force_regenerate_topics)

    def compute_pairwise_metrics(
        self,
        method_a: str,
        method_b: str,
        results_a: Dict[str, Any],
        results_b: Dict[str, Any],
        sample_a: Dict[str, Any],
        sample_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute all metrics for a pair of methods

        Args:
            method_a: First method name
            method_b: Second method name
            results_a: Topic results for method A
            results_b: Topic results for method B
            sample_a: Sample for method A
            sample_b: Sample for method B

        Returns:
            Dictionary of pairwise metrics
        """
        metrics = {
            "method_a": method_a,
            "method_b": method_b,
        }

        # Extract topic information
        topics_a = results_a["topic_words"]
        topics_b = results_b["topic_words"]

        # Topic coverage comparison
        metrics["n_topics_a"] = len(topics_a)
        metrics["n_topics_b"] = len(topics_b)
        metrics["topic_count_ratio"] = len(topics_a) / len(topics_b) if len(topics_b) > 0 else 0.0

        # Topic diversity comparison (semantic and lexical)
        diversity_semantic_a = self._compute_topic_diversity_semantic(topics_a)
        diversity_semantic_b = self._compute_topic_diversity_semantic(topics_b)
        metrics["diversity_semantic_a"] = diversity_semantic_a
        metrics["diversity_semantic_b"] = diversity_semantic_b
        metrics["diversity_semantic_diff"] = diversity_semantic_a - diversity_semantic_b

        diversity_lexical_a = self._compute_topic_diversity_lexical(topics_a)
        diversity_lexical_b = self._compute_topic_diversity_lexical(topics_b)
        metrics["diversity_lexical_a"] = diversity_lexical_a
        metrics["diversity_lexical_b"] = diversity_lexical_b
        metrics["diversity_lexical_diff"] = diversity_lexical_a - diversity_lexical_b

        # Intrinsic quality metrics
        # NPMI Coherence
        npmi_a = self._compute_npmi_coherence(topics_a, sample_a["doc_texts"])
        npmi_b = self._compute_npmi_coherence(topics_b, sample_b["doc_texts"])
        metrics["npmi_coherence_a"] = npmi_a
        metrics["npmi_coherence_b"] = npmi_b
        metrics["npmi_coherence_diff"] = npmi_a - npmi_b

        # Embedding Coherence
        emb_coh_a = self._compute_embedding_coherence(topics_a)
        emb_coh_b = self._compute_embedding_coherence(topics_b)
        metrics["embedding_coherence_a"] = emb_coh_a
        metrics["embedding_coherence_b"] = emb_coh_b
        metrics["embedding_coherence_diff"] = emb_coh_a - emb_coh_b

        # Per-topic query alignment metrics
        alignment_a = self._compute_per_topic_query_alignment(results_a)
        alignment_b = self._compute_per_topic_query_alignment(results_b)
        metrics["max_query_similarity_a"] = alignment_a["max_query_similarity"]
        metrics["max_query_similarity_b"] = alignment_b["max_query_similarity"]
        metrics["query_relevant_ratio_a"] = alignment_a["query_relevant_ratio"]
        metrics["query_relevant_ratio_b"] = alignment_b["query_relevant_ratio"]
        metrics["top3_avg_similarity_a"] = alignment_a["top3_avg_similarity"]
        metrics["top3_avg_similarity_b"] = alignment_b["top3_avg_similarity"]

        # Topic matching using Hungarian algorithm
        word_overlaps, semantic_sims = self._match_topics(topics_a, topics_b)

        if word_overlaps and semantic_sims:
            metrics["topic_word_overlap_mean"] = float(np.mean(word_overlaps))
            metrics["topic_word_overlap_std"] = float(np.std(word_overlaps))
            metrics["topic_semantic_similarity_mean"] = float(np.mean(semantic_sims))
            metrics["topic_semantic_similarity_std"] = float(np.std(semantic_sims))

            # Topic matching at different thresholds
            for threshold in [0.5, 0.6, 0.7]:
                matched = sum(1 for s in semantic_sims if s >= threshold)

                # Precision: how many of B's topics match A's topics
                precision_b = matched / len(topics_b) if len(topics_b) > 0 else 0.0
                # Recall: how many of A's topics match B's topics
                recall_a = matched / len(topics_a) if len(topics_a) > 0 else 0.0
                # F1
                f1 = 2 * precision_b * recall_a / (precision_b + recall_a) if (precision_b + recall_a) > 0 else 0.0

                threshold_str = str(threshold).replace('.', '')
                metrics[f"precision_b_@{threshold_str}"] = precision_b
                metrics[f"recall_a_@{threshold_str}"] = recall_a
                metrics[f"f1_@{threshold_str}"] = f1
        else:
            # No topics to compare
            metrics["topic_word_overlap_mean"] = 0.0
            metrics["topic_word_overlap_std"] = 0.0
            metrics["topic_semantic_similarity_mean"] = 0.0
            metrics["topic_semantic_similarity_std"] = 0.0
            for threshold in [0.5, 0.6, 0.7]:
                threshold_str = str(threshold).replace('.', '')
                metrics[f"precision_b_@{threshold_str}"] = 0.0
                metrics[f"recall_a_@{threshold_str}"] = 0.0
                metrics[f"f1_@{threshold_str}"] = 0.0

        # Topic-Query similarity for both methods
        metrics["topic_query_similarity_a"] = self._compute_topic_query_similarity(results_a)
        metrics["topic_query_similarity_b"] = self._compute_topic_query_similarity(results_b)
        metrics["topic_query_similarity_diff"] = metrics["topic_query_similarity_a"] - metrics["topic_query_similarity_b"]

        # Document distribution comparison
        outlier_a = np.sum(np.array(results_a["topics"]) == -1) / len(results_a["topics"])
        outlier_b = np.sum(np.array(results_b["topics"]) == -1) / len(results_b["topics"])
        metrics["outlier_ratio_a"] = float(outlier_a)
        metrics["outlier_ratio_b"] = float(outlier_b)
        metrics["outlier_ratio_diff"] = float(outlier_a - outlier_b)

        # Average topic size comparison
        topics_arr_a = np.array(results_a["topics"])
        topics_arr_b = np.array(results_b["topics"])

        unique_a, counts_a = np.unique(topics_arr_a[topics_arr_a != -1], return_counts=True)
        unique_b, counts_b = np.unique(topics_arr_b[topics_arr_b != -1], return_counts=True)

        avg_size_a = float(np.mean(counts_a)) if len(counts_a) > 0 else 0.0
        avg_size_b = float(np.mean(counts_b)) if len(counts_b) > 0 else 0.0

        metrics["avg_topic_size_a"] = avg_size_a
        metrics["avg_topic_size_b"] = avg_size_b
        metrics["avg_topic_size_diff"] = avg_size_a - avg_size_b

        # Document-level clustering metrics (if overlap exists)
        doc_ids_a = set(sample_a["doc_ids"])
        doc_ids_b = set(sample_b["doc_ids"])
        overlap = doc_ids_a & doc_ids_b

        metrics["overlap_count"] = len(overlap)
        metrics["overlap_percent_a"] = len(overlap) / len(doc_ids_a) * 100 if len(doc_ids_a) > 0 else 0.0
        metrics["overlap_percent_b"] = len(overlap) / len(doc_ids_b) * 100 if len(doc_ids_b) > 0 else 0.0

        if len(overlap) >= 2:
            doc_to_idx_a = {doc_id: i for i, doc_id in enumerate(sample_a["doc_ids"])}
            doc_to_idx_b = {doc_id: i for i, doc_id in enumerate(sample_b["doc_ids"])}

            topics_overlap_a = [results_a["topics"][doc_to_idx_a[doc_id]] for doc_id in overlap]
            topics_overlap_b = [results_b["topics"][doc_to_idx_b[doc_id]] for doc_id in overlap]

            try:
                ari = adjusted_rand_score(topics_overlap_a, topics_overlap_b)
                nmi = normalized_mutual_info_score(topics_overlap_a, topics_overlap_b)
                metrics["ari"] = float(ari)
                metrics["nmi"] = float(nmi)
            except:
                metrics["ari"] = None
                metrics["nmi"] = None
        else:
            metrics["ari"] = None
            metrics["nmi"] = None

        return metrics

    def _compute_topic_diversity_semantic(self, topic_words: Dict[int, List[str]]) -> float:
        """
        Compute semantic diversity: average pairwise distance between topics in embedding space

        Higher values indicate topics are more semantically distinct from each other.

        Args:
            topic_words: Dictionary mapping topic IDs to lists of words

        Returns:
            Average pairwise semantic distance (0-1, higher = more diverse)
        """
        if len(topic_words) < 2:
            return 0.0

        # Load embedding model
        model = SentenceTransformer(self.embedding_model_name, device=self.device)

        # Get embeddings for topics
        topic_embeddings = []
        for topic_id, words in topic_words.items():
            top_words = " ".join(words[:5])
            embedding = model.encode(top_words, convert_to_tensor=False, show_progress_bar=False)
            topic_embeddings.append(embedding)

        topic_embeddings = np.array(topic_embeddings)

        # Compute pairwise distances
        similarities = cosine_similarity(topic_embeddings)
        n = len(topic_embeddings)
        distances = 1 - similarities
        upper_triangle = distances[np.triu_indices(n, k=1)]

        return float(np.mean(upper_triangle))

    def _compute_topic_diversity_lexical(self, topic_words: Dict[int, List[str]], top_k: int = 10) -> float:
        """
        Compute lexical diversity: unique word fraction across all topics

        Measures vocabulary redundancy. Higher values indicate less word overlap across topics.

        Args:
            topic_words: Dictionary mapping topic IDs to lists of words
            top_k: Number of top words to consider per topic

        Returns:
            Ratio of unique words to total words (0-1, higher = less redundancy)
        """
        if not topic_words:
            return 0.0

        unique_words = set()
        total_words = 0

        for topic_id, words in topic_words.items():
            unique_words.update(words[:top_k])
            total_words += min(len(words), top_k)

        return len(unique_words) / total_words if total_words > 0 else 0.0

    def _match_topics(
        self,
        topics_a: Dict[int, List[str]],
        topics_b: Dict[int, List[str]],
        top_n: int = 10
    ) -> Tuple[List[float], List[float]]:
        """Match topics using Hungarian algorithm"""
        if not topics_a or not topics_b:
            return [], []

        # Load embedding model
        model = SentenceTransformer(self.embedding_model_name, device=self.device)

        # Get topic IDs
        topic_ids_a = list(topics_a.keys())
        topic_ids_b = list(topics_b.keys())

        # Compute embeddings
        embeddings_a = []
        for topic_id in topic_ids_a:
            words = " ".join(topics_a[topic_id][:top_n])
            emb = model.encode(words, convert_to_tensor=False, show_progress_bar=False)
            embeddings_a.append(emb)

        embeddings_b = []
        for topic_id in topic_ids_b:
            words = " ".join(topics_b[topic_id][:top_n])
            emb = model.encode(words, convert_to_tensor=False, show_progress_bar=False)
            embeddings_b.append(emb)

        embeddings_a = np.array(embeddings_a)
        embeddings_b = np.array(embeddings_b)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings_a, embeddings_b)

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

        # Get matched similarities and word overlaps
        semantic_sims = []
        word_overlaps = []

        for i, j in zip(row_ind, col_ind):
            topic_id_a = topic_ids_a[i]
            topic_id_b = topic_ids_b[j]

            # Semantic similarity
            semantic_sims.append(similarity_matrix[i, j])

            # Word overlap (Jaccard)
            words_a = set(topics_a[topic_id_a][:top_n])
            words_b = set(topics_b[topic_id_b][:top_n])

            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            jaccard = intersection / union if union > 0 else 0.0

            word_overlaps.append(jaccard)

        return word_overlaps, semantic_sims

    def _compute_topic_query_similarity(self, results: Dict[str, Any]) -> float:
        """
        Compute average similarity between topics and the query

        Args:
            results: Topic modeling results

        Returns:
            Average cosine similarity between all topics and the query
        """
        topic_words = results["topic_words"]

        if not topic_words:
            return 0.0

        # Load embedding model
        model = SentenceTransformer(self.embedding_model_name, device=self.device)

        # Encode query
        query_embedding = model.encode(self.query_text, convert_to_tensor=False, show_progress_bar=False)

        # Encode all topics (top 10 words each)
        topic_embeddings = []
        for topic_id, words in topic_words.items():
            top_words = " ".join(words[:10])
            emb = model.encode(top_words, convert_to_tensor=False, show_progress_bar=False)
            topic_embeddings.append(emb)

        if not topic_embeddings:
            return 0.0

        topic_embeddings = np.array(topic_embeddings)
        query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, topic_embeddings)[0]

        # Return average similarity
        return float(np.mean(similarities))

    def _compute_per_topic_query_alignment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute per-topic query alignment metrics

        Returns detailed metrics about how individual topics align with the query.

        Args:
            results: Topic modeling results

        Returns:
            Dictionary with:
            - max_query_similarity: Highest topic-query similarity
            - query_relevant_ratio: Fraction of topics above similarity threshold (0.5)
            - top3_avg_similarity: Average similarity of top-3 most query-relevant topics
            - per_topic_similarities: List of (topic_id, similarity) tuples
        """
        topic_words = results["topic_words"]

        if not topic_words:
            return {
                "max_query_similarity": 0.0,
                "query_relevant_ratio": 0.0,
                "top3_avg_similarity": 0.0,
                "per_topic_similarities": []
            }

        # Load embedding model
        model = SentenceTransformer(self.embedding_model_name, device=self.device)

        # Encode query
        query_embedding = model.encode(self.query_text, convert_to_tensor=False, show_progress_bar=False)

        # Encode all topics and compute similarities
        similarities = []
        for topic_id, words in topic_words.items():
            top_words = " ".join(words[:10])
            topic_emb = model.encode(top_words, convert_to_tensor=False, show_progress_bar=False)
            sim = float(cosine_similarity([query_embedding], [topic_emb])[0][0])
            similarities.append((topic_id, sim))

        # Sort by similarity (descending)
        similarities_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Compute metrics
        max_sim = similarities_sorted[0][1] if similarities_sorted else 0.0

        # Count topics above threshold
        threshold = 0.5
        relevant_count = sum(1 for _, sim in similarities if sim >= threshold)
        relevant_ratio = relevant_count / len(similarities) if similarities else 0.0

        # Top-3 average
        top3_sims = [sim for _, sim in similarities_sorted[:3]]
        top3_avg = float(np.mean(top3_sims)) if top3_sims else 0.0

        return {
            "max_query_similarity": max_sim,
            "query_relevant_ratio": relevant_ratio,
            "top3_avg_similarity": top3_avg,
            "per_topic_similarities": similarities_sorted
        }

    def _compute_npmi_coherence(self, topic_words: Dict[int, List[str]], doc_texts: List[str], top_k: int = 10) -> float:
        """
        Compute average NPMI (Normalized Pointwise Mutual Information) coherence across topics

        NPMI measures semantic coherence of topic words based on co-occurrence in documents.
        Higher values indicate more interpretable/coherent topics.

        Args:
            topic_words: Dictionary mapping topic IDs to lists of words
            doc_texts: List of document texts for computing co-occurrences
            top_k: Number of top words to consider per topic

        Returns:
            Average NPMI coherence across all topics (range: -1 to 1, higher = more coherent)
        """
        if not topic_words or not doc_texts:
            return 0.0

        from collections import Counter
        import math

        # Tokenize documents (simple whitespace + lowercase)
        tokenized_docs = []
        for doc in doc_texts:
            tokens = set(doc.lower().split())
            tokenized_docs.append(tokens)

        n_docs = len(tokenized_docs)

        # Compute document frequencies for all words
        word_doc_freq = Counter()
        for tokens in tokenized_docs:
            for word in tokens:
                word_doc_freq[word] += 1

        # Compute NPMI for each topic
        topic_npmis = []

        for topic_id, words in topic_words.items():
            top_words = words[:top_k]

            # Compute NPMI for all word pairs
            pair_npmis = []
            for i in range(len(top_words)):
                for j in range(i + 1, len(top_words)):
                    word_i = top_words[i].lower()
                    word_j = top_words[j].lower()

                    # Count co-occurrences
                    co_occur = sum(1 for tokens in tokenized_docs if word_i in tokens and word_j in tokens)

                    # Compute PMI
                    if co_occur > 0:
                        p_i = word_doc_freq.get(word_i, 0) / n_docs
                        p_j = word_doc_freq.get(word_j, 0) / n_docs
                        p_ij = co_occur / n_docs

                        if p_i > 0 and p_j > 0 and p_ij > 0:
                            pmi = math.log(p_ij / (p_i * p_j))
                            npmi = pmi / (-math.log(p_ij))  # Normalize by -log(p(i,j))
                            pair_npmis.append(npmi)

            # Average NPMI for this topic
            if pair_npmis:
                topic_npmi = float(np.mean(pair_npmis))
                topic_npmis.append(topic_npmi)

        # Average across all topics
        return float(np.mean(topic_npmis)) if topic_npmis else 0.0

    def _compute_embedding_coherence(self, topic_words: Dict[int, List[str]], top_k: int = 10) -> float:
        """
        Compute average embedding coherence across topics

        Measures how semantically tight topic words are by computing average pairwise
        cosine similarity of word embeddings within each topic.

        Args:
            topic_words: Dictionary mapping topic IDs to lists of words
            top_k: Number of top words to consider per topic

        Returns:
            Average embedding coherence across all topics (0-1, higher = tighter topics)
        """
        if not topic_words:
            return 0.0

        # Load embedding model
        model = SentenceTransformer(self.embedding_model_name, device=self.device)

        topic_coherences = []

        for topic_id, words in topic_words.items():
            top_words = words[:top_k]

            if len(top_words) < 2:
                continue

            # Get embeddings for words
            word_embeddings = model.encode(top_words, convert_to_tensor=False, show_progress_bar=False)

            # Compute pairwise cosine similarities
            similarities = cosine_similarity(word_embeddings)

            # Get upper triangle (exclude diagonal)
            n = len(word_embeddings)
            upper_triangle = similarities[np.triu_indices(n, k=1)]

            # Average similarity for this topic
            topic_coherence = float(np.mean(upper_triangle))
            topic_coherences.append(topic_coherence)

        # Average across all topics
        return float(np.mean(topic_coherences)) if topic_coherences else 0.0

    def _save_topic_summaries(self, topic_results: Dict[str, Dict[str, Any]], samples: Dict[str, Dict[str, Any]]):
        """Save human-readable topic summaries for qualitative analysis"""
        logger.info("Saving human-readable topic summaries...")

        for method_name, results in topic_results.items():
            if results is None:
                continue

            # Create text summary
            summary_path = os.path.join(self.topics_summary_dir, f"{method_name}_topics.txt")
            json_path = os.path.join(self.topics_summary_dir, f"{method_name}_topics.json")

            # Text summary
            with open(summary_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"TOPIC SUMMARY: {method_name.upper()}\n")
                f.write(f"Query ID: {self.query_id}\n")
                f.write(f"Query: {self.query_text}\n")
                f.write(f"Number of Topics: {results['n_topics']}\n")
                f.write(f"Number of Documents: {results['n_docs']}\n")
                f.write("="*80 + "\n\n")

                # Get topic info
                topic_labels = results.get('topic_labels', {})
                topic_names = results.get('topic_names', {})
                topic_words_with_scores = results.get('topic_words_with_scores', {})

                # Count documents per topic
                topics_array = np.array(results['topics'])
                unique_topics, counts = np.unique(topics_array[topics_array != -1], return_counts=True)
                topic_counts = dict(zip(unique_topics, counts))

                # Sort topics by size (largest first)
                sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

                for topic_id, count in sorted_topics:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"TOPIC {topic_id}: {topic_labels.get(topic_id, 'N/A')}\n")
                    f.write(f"Name: {topic_names.get(topic_id, 'N/A')}\n")
                    f.write(f"Document Count: {count}\n")
                    f.write(f"-"*80 + "\n")

                    # Top words with scores
                    if topic_id in topic_words_with_scores:
                        f.write("Top Words (with c-TF-IDF scores):\n")
                        for i, (word, score) in enumerate(topic_words_with_scores[topic_id][:10], 1):
                            f.write(f"  {i:2d}. {word:20s} {score:.4f}\n")
                    f.write("\n")

                # Outliers
                n_outliers = np.sum(topics_array == -1)
                if n_outliers > 0:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"OUTLIERS (Topic -1)\n")
                    f.write(f"Document Count: {n_outliers}\n")
                    f.write(f"Percentage: {100 * n_outliers / len(topics_array):.2f}%\n")

            # JSON summary for programmatic access
            json_summary = {
                "method": method_name,
                "query_id": self.query_id,
                "query_text": self.query_text,
                "n_topics": results['n_topics'],
                "n_docs": results['n_docs'],
                "topics": []
            }

            # Get topic info
            topic_labels = results.get('topic_labels', {})
            topic_names = results.get('topic_names', {})
            topic_words_with_scores = results.get('topic_words_with_scores', {})

            # Count documents per topic
            topics_array = np.array(results['topics'])
            unique_topics, counts = np.unique(topics_array[topics_array != -1], return_counts=True)
            topic_counts = dict(zip(unique_topics, counts))
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

            for topic_id, count in sorted_topics:
                topic_data = {
                    "topic_id": int(topic_id),
                    "label": topic_labels.get(topic_id, f"Topic_{topic_id}"),
                    "name": topic_names.get(topic_id, f"Topic_{topic_id}"),
                    "doc_count": int(count),
                    "top_words": topic_words_with_scores.get(topic_id, [])[:10]
                }
                json_summary["topics"].append(topic_data)

            # Add outlier info
            n_outliers = np.sum(topics_array == -1)
            if n_outliers > 0:
                json_summary["outliers"] = {
                    "count": int(n_outliers),
                    "percentage": float(100 * n_outliers / len(topics_array))
                }

            with open(json_path, 'w') as f:
                json.dump(json_summary, f, indent=2)

            logger.info(f"Saved topic summary for {method_name}: {summary_path}")

        logger.info(f"All topic summaries saved to {self.topics_summary_dir}")

    def run_all_pairwise_comparisons(
        self,
        topic_results: Dict[str, Dict[str, Any]],
        samples: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Run pairwise comparisons for all method pairs

        Returns:
            DataFrame with all pairwise metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Running pairwise comparisons...")
        logger.info("="*80)

        methods = list(topic_results.keys())

        if len(methods) < 2:
            logger.error(f"Need at least 2 methods, found {len(methods)}")
            return pd.DataFrame()

        all_metrics = []

        # Compare each pair
        pairs = list(combinations(methods, 2))
        logger.info(f"Computing {len(pairs)} pairwise comparisons...\n")

        for method_a, method_b in tqdm(pairs, desc="Pairwise comparisons"):
            try:
                metrics = self.compute_pairwise_metrics(
                    method_a=method_a,
                    method_b=method_b,
                    results_a=topic_results[method_a],
                    results_b=topic_results[method_b],
                    sample_a=samples[method_a],
                    sample_b=samples[method_b]
                )
                all_metrics.append(metrics)

                logger.debug(f"Compared {method_a} vs {method_b}")
            except Exception as e:
                logger.error(f"Error comparing {method_a} vs {method_b}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)

        # Save results
        csv_path = os.path.join(self.results_dir, "pairwise_metrics.csv")
        json_path = os.path.join(self.results_dir, "pairwise_metrics.json")

        df.to_csv(csv_path, index=False)
        logger.info(f"\n✓ Saved pairwise metrics to {csv_path}")

        with open(json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"✓ Saved pairwise metrics to {json_path}")

        return df

    def create_per_method_summary(
        self,
        pairwise_df: pd.DataFrame,
        topic_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create per-method summary statistics

        Args:
            pairwise_df: DataFrame with pairwise metrics
            topic_results: Dict of topic modeling results

        Returns:
            DataFrame with per-method statistics
        """
        logger.info("Creating per-method summary...")

        methods = list(topic_results.keys())
        summary_data = []

        for method in methods:
            # Get all comparisons involving this method
            method_comparisons = pairwise_df[
                (pairwise_df['method_a'] == method) | (pairwise_df['method_b'] == method)
            ]

            # Extract metrics when this method is method_a or method_b
            topic_counts = []
            diversity_semantic = []
            diversity_lexical = []
            npmi_coherence = []
            embedding_coherence = []
            outlier_ratios = []
            avg_topic_sizes = []
            topic_query_sims = []
            max_query_sims = []
            query_relevant_ratios = []
            top3_avg_sims = []

            for _, row in method_comparisons.iterrows():
                if row['method_a'] == method:
                    topic_counts.append(row['n_topics_a'])
                    diversity_semantic.append(row['diversity_semantic_a'])
                    diversity_lexical.append(row['diversity_lexical_a'])
                    npmi_coherence.append(row['npmi_coherence_a'])
                    embedding_coherence.append(row['embedding_coherence_a'])
                    outlier_ratios.append(row['outlier_ratio_a'])
                    avg_topic_sizes.append(row['avg_topic_size_a'])
                    topic_query_sims.append(row['topic_query_similarity_a'])
                    max_query_sims.append(row['max_query_similarity_a'])
                    query_relevant_ratios.append(row['query_relevant_ratio_a'])
                    top3_avg_sims.append(row['top3_avg_similarity_a'])
                else:  # method_b == method
                    topic_counts.append(row['n_topics_b'])
                    diversity_semantic.append(row['diversity_semantic_b'])
                    diversity_lexical.append(row['diversity_lexical_b'])
                    npmi_coherence.append(row['npmi_coherence_b'])
                    embedding_coherence.append(row['embedding_coherence_b'])
                    outlier_ratios.append(row['outlier_ratio_b'])
                    avg_topic_sizes.append(row['avg_topic_size_b'])
                    topic_query_sims.append(row['topic_query_similarity_b'])
                    max_query_sims.append(row['max_query_similarity_b'])
                    query_relevant_ratios.append(row['query_relevant_ratio_b'])
                    top3_avg_sims.append(row['top3_avg_similarity_b'])

            summary_data.append({
                "method": method,
                "n_topics": topic_counts[0] if topic_counts else 0,
                "diversity_semantic": diversity_semantic[0] if diversity_semantic else 0,
                "diversity_lexical": diversity_lexical[0] if diversity_lexical else 0,
                "npmi_coherence": npmi_coherence[0] if npmi_coherence else 0,
                "embedding_coherence": embedding_coherence[0] if embedding_coherence else 0,
                "outlier_ratio": outlier_ratios[0] if outlier_ratios else 0,
                "document_coverage": (1 - outlier_ratios[0]) if outlier_ratios else 1.0,
                "avg_topic_size": avg_topic_sizes[0] if avg_topic_sizes else 0,
                "topic_query_similarity": topic_query_sims[0] if topic_query_sims else 0,
                "max_query_similarity": max_query_sims[0] if max_query_sims else 0,
                "query_relevant_ratio": query_relevant_ratios[0] if query_relevant_ratios else 0,
                "top3_avg_similarity": top3_avg_sims[0] if top3_avg_sims else 0,
                "n_docs": topic_results[method]['n_docs']
            })

        summary_df = pd.DataFrame(summary_data)

        # Save summary
        summary_path = os.path.join(self.results_dir, "per_method_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Saved per-method summary to {summary_path}")

        return summary_df

    def create_comprehensive_plots(
        self,
        pairwise_df: pd.DataFrame,
        per_method_df: pd.DataFrame
    ):
        """
        Create comprehensive visualization plots for all metrics

        Args:
            pairwise_df: DataFrame with pairwise metrics
            per_method_df: DataFrame with per-method summary
        """
        logger.info("\n" + "="*80)
        logger.info("Creating comprehensive visualizations...")
        logger.info("="*80)

        # Set style
        sns.set_style("whitegrid")
        methods = per_method_df['method'].tolist()
        n_methods = len(methods)

        # Plot 1: Intrinsic Quality Metrics (3x2 grid)
        fig, axes = plt.subplots(3, 2, figsize=(14, 16))
        fig.suptitle(f'Intrinsic Topic Quality Metrics - Query {self.query_id}', fontsize=16, fontweight='bold')

        quality_metrics = [
            ('npmi_coherence', 'NPMI Coherence', 'Blues_d'),
            ('embedding_coherence', 'Embedding Coherence', 'Greens_d'),
            ('diversity_semantic', 'Semantic Diversity', 'Purples_d'),
            ('diversity_lexical', 'Lexical Diversity', 'Oranges_d'),
            ('document_coverage', 'Document Coverage', 'YlGn'),
            ('n_topics', 'Number of Topics', 'Greys_d')
        ]

        for idx, (metric, title, color) in enumerate(quality_metrics):
            ax = axes[idx // 2, idx % 2]
            values = per_method_df[metric].values
            colors_list = sns.color_palette(color, n_methods)

            bars = ax.bar(methods, values, color=colors_list, edgecolor='black', linewidth=1.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.set_xlabel('Method', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}' if isinstance(value, float) and value < 10 else f'{value:.0f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "intrinsic_quality_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved intrinsic quality metrics plot")

        # Plot 2: Query Alignment Metrics (1x3 grid)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Query Alignment Metrics - Query {self.query_id}', fontsize=16, fontweight='bold')

        query_metrics = [
            ('topic_query_similarity', 'Avg Topic-Query Similarity', 'viridis'),
            ('max_query_similarity', 'Max Topic-Query Similarity', 'plasma'),
            ('query_relevant_ratio', 'Query-Relevant Topic Ratio', 'cividis')
        ]

        for idx, (metric, title, cmap) in enumerate(query_metrics):
            ax = axes[idx]
            values = per_method_df[metric].values
            colors_list = sns.color_palette(cmap, n_methods)

            bars = ax.bar(methods, values, color=colors_list, edgecolor='black', linewidth=1.5)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.set_xlabel('Method', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.0)

            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "query_alignment_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved query alignment metrics plot")

        # Plot 3: Diversity Scatter (Semantic vs Lexical)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors_list = sns.color_palette("husl", n_methods)

        for idx, method in enumerate(methods):
            semantic = per_method_df[per_method_df['method'] == method]['diversity_semantic'].values[0]
            lexical = per_method_df[per_method_df['method'] == method]['diversity_lexical'].values[0]
            ax.scatter(semantic, lexical, s=200, color=colors_list[idx],
                      edgecolor='black', linewidth=2, label=method, alpha=0.7)
            ax.annotate(method, (semantic, lexical),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)

        ax.set_xlabel('Semantic Diversity (higher = more distinct concepts)', fontsize=12)
        ax.set_ylabel('Lexical Diversity (higher = less word overlap)', fontsize=12)
        ax.set_title(f'Topic Diversity: Semantic vs Lexical - Query {self.query_id}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Add quadrant lines (optional interpretation guides)
        ax.axhline(0.8, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(0.6, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "diversity_scatter.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved diversity scatter plot")

        # Plot 4: Pairwise Similarity Heatmaps
        pairwise_metrics_to_plot = [
            ("topic_semantic_similarity_mean", "Topic Semantic Similarity", "Blues", (0, 1)),
            ("topic_word_overlap_mean", "Topic Word Overlap (Jaccard)", "Greens", (0, 1)),
            ("f1_@05", "Topic Matching F1 @0.5", "RdPu", (0, 1)),
            ("f1_@06", "Topic Matching F1 @0.6", "RdPu", (0, 1)),
            ("f1_@07", "Topic Matching F1 @0.7", "RdPu", (0, 1)),
            ("precision_b_@05", "Topic Precision (B) @0.5", "Purples", (0, 1)),
            ("precision_b_@06", "Topic Precision (B) @0.6", "Purples", (0, 1)),
            ("precision_b_@07", "Topic Precision (B) @0.7", "Purples", (0, 1)),
            ("recall_a_@05", "Topic Recall (A) @0.5", "Oranges", (0, 1)),
            ("recall_a_@06", "Topic Recall (A) @0.6", "Oranges", (0, 1)),
            ("recall_a_@07", "Topic Recall (A) @0.7", "Oranges", (0, 1)),
            ("npmi_coherence_diff", "NPMI Coherence Difference (A-B)", "RdBu_r", (-0.5, 0.5)),
            ("embedding_coherence_diff", "Embedding Coherence Difference (A-B)", "RdBu_r", (-0.2, 0.2)),
            ("ari", "Adjusted Rand Index (ARI)", "YlGn", (-0.5, 1)),
        ]

        for metric_key, title, cmap, vrange in pairwise_metrics_to_plot:
            # Create matrix
            matrix = np.zeros((n_methods, n_methods))

            for i, method_a in enumerate(methods):
                for j, method_b in enumerate(methods):
                    if i == j:
                        # Diagonal: self-comparison (perfect score for similarity metrics)
                        if "ari" in metric_key.lower():
                            matrix[i, j] = 1.0
                        elif "diff" in metric_key.lower():
                            matrix[i, j] = 0.0
                        else:
                            matrix[i, j] = 1.0
                    else:
                        # Find the metric value
                        row = pairwise_df[(pairwise_df['method_a'] == method_a) & (pairwise_df['method_b'] == method_b)]
                        if row.empty:
                            # Try reverse order
                            row = pairwise_df[(pairwise_df['method_a'] == method_b) & (pairwise_df['method_b'] == method_a)]
                            if not row.empty and metric_key in row.columns:
                                value = row[metric_key].values[0]
                                matrix[i, j] = value if value is not None and not np.isnan(value) else 0.0
                        elif metric_key in row.columns:
                            value = row[metric_key].values[0]
                            matrix[i, j] = value if value is not None and not np.isnan(value) else 0.0

            # Create heatmap
            plt.figure(figsize=(10, 8))

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                xticklabels=methods,
                yticklabels=methods,
                vmin=vrange[0],
                vmax=vrange[1],
                cbar_kws={'label': title},
                square=True,
                linewidths=0.5
            )

            plt.title(f"{title}\n(Method A vs Method B) - Query {self.query_id}", fontsize=14, fontweight='bold')
            plt.xlabel("Method B", fontsize=12)
            plt.ylabel("Method A", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Save plot
            filename = metric_key.replace('@', '_at_').replace('_', '-') + "_heatmap.png"
            plot_path = os.path.join(self.plots_dir, filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Saved {title} heatmap")

        logger.info(f"\n✓ All plots saved to {self.plots_dir}")

    def run_full_evaluation(self):
        """Run full end-to-end evaluation pipeline with pairwise comparisons"""

        logger.info("="*80)
        logger.info(f"Starting End-to-End Evaluation for Query {self.query_id}")
        logger.info(f"Sample size: {self.sample_size}")
        logger.info("="*80)

        # Step 1: Generate samples
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Generating Document Samples")
        logger.info("="*80)

        samples = {}
        samples["random_uniform"] = self.sample_random_uniform()
        samples["direct_retrieval"] = self.sample_direct_retrieval()
        samples["query_expansion"] = self.sample_query_expansion()
        samples["keyword_search"] = self.sample_keyword_search()

        # Step 2: Run topic modeling
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Running Topic Modeling")
        logger.info("="*80)

        topic_results = {}
        for method_name, sample in samples.items():
            try:
                topic_results[method_name] = self.run_topic_modeling(sample)
            except Exception as e:
                logger.error(f"Error in topic modeling for {method_name}: {e}")
                topic_results[method_name] = None

        # Remove failed methods
        topic_results = {k: v for k, v in topic_results.items() if v is not None}
        samples = {k: v for k, v in samples.items() if k in topic_results}

        if len(topic_results) < 2:
            logger.error("Need at least 2 successful topic models for pairwise comparison")
            return None

        # Step 3: Save topic summaries
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Saving Topic Summaries")
        logger.info("="*80)

        try:
            self._save_topic_summaries(topic_results, samples)
        except Exception as e:
            logger.error(f"Error saving topic summaries: {e}")
            logger.warning("Continuing with evaluation despite topic summary error")

        # Step 4: Run pairwise comparisons
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Running Pairwise Comparisons")
        logger.info("="*80)

        pairwise_df = self.run_all_pairwise_comparisons(topic_results, samples)

        if pairwise_df.empty:
            logger.error("No pairwise comparisons completed. Exiting.")
            return None

        # Step 5: Create per-method summary
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Creating Per-Method Summary")
        logger.info("="*80)

        per_method_df = self.create_per_method_summary(pairwise_df, topic_results)

        # Step 6: Create comprehensive visualizations
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Creating Comprehensive Visualizations")
        logger.info("="*80)

        self.create_comprehensive_plots(pairwise_df, per_method_df)

        # Step 7: Print summary
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)

        print("\nPer-Method Summary:")
        print(per_method_df.to_string(index=False))

        logger.info("\n" + "="*80)
        logger.info("Evaluation Complete!")
        logger.info(f"All results saved to: {self.output_dir}")
        logger.info("="*80)

        return {
            "samples": samples,
            "topic_results": topic_results,
            "pairwise_metrics": pairwise_df,
            "per_method_summary": per_method_df
        }


def aggregate_cross_query_results(
    results_base_dir: str,
    query_ids: List[str],
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Aggregate results across all queries to understand overall method performance.

    Args:
        results_base_dir: Base directory containing query result folders
        query_ids: List of query IDs to aggregate
        output_dir: Directory to save aggregated results (default: results_base_dir/aggregate_results)

    Returns:
        Dictionary containing aggregated metrics
    """
    logger.info("="*80)
    logger.info("Aggregating Results Across Queries")
    logger.info("="*80)

    if output_dir is None:
        output_dir = os.path.join(results_base_dir, "aggregate_results")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Collect all per-method summaries
    per_method_data = []
    pairwise_data = []

    logger.info(f"Loading results from {len(query_ids)} queries...")

    for query_id in query_ids:
        query_dir = os.path.join(results_base_dir, f"query_{query_id}", "results")

        # Load per-method summary
        per_method_csv = os.path.join(query_dir, "per_method_summary.csv")
        if os.path.exists(per_method_csv):
            df = pd.read_csv(per_method_csv)
            df['query_id'] = query_id
            per_method_data.append(df)
        else:
            logger.warning(f"Missing per_method_summary.csv for query {query_id}")

        # Load pairwise metrics
        pairwise_json = os.path.join(query_dir, "pairwise_metrics.json")
        if os.path.exists(pairwise_json):
            with open(pairwise_json, 'r') as f:
                pairwise = json.load(f)
                for item in pairwise:
                    item['query_id'] = query_id
                pairwise_data.extend(pairwise)
        else:
            logger.warning(f"Missing pairwise_metrics.json for query {query_id}")

    if not per_method_data:
        logger.error("No per-method data found!")
        return None

    # Combine all data
    all_per_method = pd.concat(per_method_data, ignore_index=True)
    all_pairwise = pd.DataFrame(pairwise_data)

    logger.info(f"Loaded data from {len(per_method_data)} queries")
    logger.info(f"Total per-method records: {len(all_per_method)}")
    logger.info(f"Total pairwise records: {len(all_pairwise)}")

    # ===== PER-METHOD AGGREGATION =====
    logger.info("\nAggregating per-method metrics...")

    # Metrics to aggregate with mean/std/median/min/max
    aggregatable_metrics = [
        'diversity_semantic',
        'diversity_lexical',
        'npmi_coherence',
        'embedding_coherence',
        'outlier_ratio',
        'document_coverage',
        'topic_query_similarity',
        'max_query_similarity',
        'query_relevant_ratio',
        'top3_avg_similarity'
    ]

    # Metrics to report but not aggregate (query-dependent)
    descriptive_metrics = ['n_topics', 'avg_topic_size']

    per_method_aggregates = []

    for method in all_per_method['method'].unique():
        method_data = all_per_method[all_per_method['method'] == method]

        agg_record = {'method': method, 'n_queries': len(method_data)}

        # Aggregate main metrics
        for metric in aggregatable_metrics:
            if metric in method_data.columns:
                values = method_data[metric].dropna()
                if len(values) > 0:
                    agg_record[f'{metric}_mean'] = values.mean()
                    agg_record[f'{metric}_std'] = values.std()
                    agg_record[f'{metric}_median'] = values.median()
                    agg_record[f'{metric}_min'] = values.min()
                    agg_record[f'{metric}_max'] = values.max()

        # Report descriptive metrics
        for metric in descriptive_metrics:
            if metric in method_data.columns:
                values = method_data[metric].dropna()
                if len(values) > 0:
                    agg_record[f'{metric}_mean'] = values.mean()
                    agg_record[f'{metric}_std'] = values.std()

        per_method_aggregates.append(agg_record)

    per_method_agg_df = pd.DataFrame(per_method_aggregates)

    # Save per-method aggregates
    per_method_csv_path = os.path.join(output_dir, "per_method_aggregates.csv")
    per_method_json_path = os.path.join(output_dir, "per_method_aggregates.json")

    per_method_agg_df.to_csv(per_method_csv_path, index=False)
    per_method_agg_df.to_json(per_method_json_path, orient='records', indent=2)

    logger.info(f"Saved per-method aggregates to {per_method_csv_path}")

    # ===== PAIRWISE AGGREGATION =====
    logger.info("\nAggregating pairwise metrics...")

    # Metrics to aggregate
    pairwise_aggregatable = [
        'diversity_semantic_diff',
        'diversity_lexical_diff',
        'npmi_coherence_diff',
        'embedding_coherence_diff',
        'topic_query_similarity_diff',
        'outlier_ratio_diff',
        'topic_word_overlap_mean',
        'topic_semantic_similarity_mean',
        'precision_b_@05',
        'recall_a_@05',
        'f1_@05',
        'precision_b_@06',
        'recall_a_@06',
        'f1_@06',
        'precision_b_@07',
        'recall_a_@07',
        'f1_@07',
        'ari',
        'nmi'
    ]

    pairwise_aggregates = []

    # Group by method pair
    if 'method_a' in all_pairwise.columns and 'method_b' in all_pairwise.columns:
        for (method_a, method_b), group in all_pairwise.groupby(['method_a', 'method_b']):
            agg_record = {
                'method_a': method_a,
                'method_b': method_b,
                'n_queries': len(group)
            }

            for metric in pairwise_aggregatable:
                if metric in group.columns:
                    values = group[metric].dropna()
                    if len(values) > 0:
                        agg_record[f'{metric}_mean'] = values.mean()
                        agg_record[f'{metric}_std'] = values.std()
                        agg_record[f'{metric}_median'] = values.median()
                        agg_record[f'{metric}_min'] = values.min()
                        agg_record[f'{metric}_max'] = values.max()

            pairwise_aggregates.append(agg_record)

    pairwise_agg_df = pd.DataFrame(pairwise_aggregates)

    # Save pairwise aggregates
    pairwise_csv_path = os.path.join(output_dir, "pairwise_aggregates.csv")
    pairwise_json_path = os.path.join(output_dir, "pairwise_aggregates.json")

    pairwise_agg_df.to_csv(pairwise_csv_path, index=False)
    pairwise_agg_df.to_json(pairwise_json_path, orient='records', indent=2)

    logger.info(f"Saved pairwise aggregates to {pairwise_csv_path}")

    # ===== VISUALIZATIONS =====
    logger.info("\nGenerating aggregate visualizations...")

    _generate_aggregate_visualizations(
        all_per_method,
        all_pairwise,
        per_method_agg_df,
        pairwise_agg_df,
        output_dir
    )

    logger.info("="*80)
    logger.info("Cross-Query Aggregation Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)

    return {
        'per_method_aggregates': per_method_agg_df,
        'pairwise_aggregates': pairwise_agg_df,
        'raw_per_method': all_per_method,
        'raw_pairwise': all_pairwise
    }


def _generate_aggregate_visualizations(
    all_per_method: pd.DataFrame,
    all_pairwise: pd.DataFrame,
    per_method_agg: pd.DataFrame,
    pairwise_agg: pd.DataFrame,
    output_dir: str
):
    """Generate visualizations for aggregated results"""

    plots_dir = os.path.join(output_dir, "plots")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # 1. Method Quality Comparison (Boxplots)
    logger.info("Creating method quality comparison plots...")

    quality_metrics = [
        ('npmi_coherence', 'NPMI Coherence'),
        ('embedding_coherence', 'Embedding Coherence'),
        ('diversity_semantic', 'Semantic Diversity'),
        ('diversity_lexical', 'Lexical Diversity'),
        ('document_coverage', 'Document Coverage'),
        ('outlier_ratio', 'Outlier Ratio')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(quality_metrics):
        if metric in all_per_method.columns:
            sns.boxplot(data=all_per_method, x='method', y=metric, ax=axes[idx])
            axes[idx].set_title(f'{title} Across Queries', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Method', fontsize=10)
            axes[idx].set_ylabel(title, fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "method_quality_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Method Alignment Comparison (Boxplots)
    logger.info("Creating method alignment comparison plots...")

    alignment_metrics = [
        ('topic_query_similarity', 'Avg Topic-Query Similarity'),
        ('max_query_similarity', 'Max Topic-Query Similarity'),
        ('query_relevant_ratio', 'Query-Relevant Ratio'),
        ('top3_avg_similarity', 'Top-3 Avg Similarity')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(alignment_metrics):
        if metric in all_per_method.columns:
            sns.boxplot(data=all_per_method, x='method', y=metric, ax=axes[idx])
            axes[idx].set_title(f'{title} Across Queries', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Method', fontsize=10)
            axes[idx].set_ylabel(title, fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "method_alignment_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Pairwise Coherence Difference Heatmap
    logger.info("Creating pairwise difference heatmaps...")

    pairwise_heatmap_metrics = [
        ('npmi_coherence_diff_mean', 'Avg NPMI Coherence Difference', 'RdYlGn'),
        ('embedding_coherence_diff_mean', 'Avg Embedding Coherence Difference', 'RdYlGn'),
        ('diversity_semantic_diff_mean', 'Avg Semantic Diversity Difference', 'RdYlGn'),
        ('topic_query_similarity_diff_mean', 'Avg Topic-Query Similarity Difference', 'RdYlGn'),
        ('f1_@05_mean', 'Avg F1 Score @ 0.5 Threshold', 'YlGnBu'),
        ('topic_word_overlap_mean_mean', 'Avg Topic Word Overlap', 'YlOrRd'),
        ('topic_semantic_similarity_mean_mean', 'Avg Topic Semantic Similarity', 'YlOrRd')
    ]

    for metric, title, cmap in pairwise_heatmap_metrics:
        if metric in pairwise_agg.columns:
            # Create pivot table
            methods = sorted(set(pairwise_agg['method_a'].tolist() + pairwise_agg['method_b'].tolist()))
            pivot_data = np.zeros((len(methods), len(methods)))
            pivot_data[:] = np.nan

            method_to_idx = {m: i for i, m in enumerate(methods)}

            for _, row in pairwise_agg.iterrows():
                i = method_to_idx[row['method_a']]
                j = method_to_idx[row['method_b']]
                value = row[metric]
                if pd.notna(value):
                    pivot_data[i, j] = value
                    # For symmetric metrics, mirror the value
                    if 'diff' not in metric:
                        pivot_data[j, i] = value
                    else:
                        pivot_data[j, i] = -value

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            # Determine color scale center
            if 'diff' in metric:
                vmax = np.nanmax(np.abs(pivot_data))
                vmin = -vmax
                center = 0
            else:
                vmin = np.nanmin(pivot_data)
                vmax = np.nanmax(pivot_data)
                center = None

            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                xticklabels=methods,
                yticklabels=methods,
                ax=ax,
                cbar_kws={'label': title}
            )

            ax.set_title(f'{title}\n(Averaged Across Queries)', fontsize=14, fontweight='bold')

            plt.tight_layout()

            # Create safe filename
            safe_metric = metric.replace('_mean', '').replace('@', '-at-').replace('_', '-')
            plt.savefig(os.path.join(plots_dir, f"pairwise_{safe_metric}_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()

    logger.info(f"Saved all aggregate visualizations to {plots_dir}")


def main():
    """Main function"""

    # ===== CONFIGURATION PARAMETERS =====

    # Query configuration - can be a single query ID or a list of query IDs
    QUERY_IDS = ["2", "9", "10", "13", "18", "21", "23", "24", "26", "27", "34", "43", "45", "47", "48"]  # 15 open-ended queries

    # Sample size (fixed at 1000 documents for all methods)
    SAMPLE_SIZE = 1000

    # Model configuration
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Dataset configuration
    DATASET_NAME = "trec-covid"

    # Directory configuration
    OUTPUT_DIR = "results/topic_evaluation"
    CACHE_DIR = "cache"

    # Device configuration
    DEVICE = "cpu"  # Use "cpu" to avoid CUDA errors, "cuda" if GPU is compatible

    # Topic model saving configuration
    # Set to True to save full BERTopic models (420 MB each, only needed for interactive exploration)
    # Set to False to save only results (30-500 KB each, sufficient for evaluation)
    # Default: False (saves ~25.2 GB for 15 queries × 4 methods)
    SAVE_TOPIC_MODELS = False

    # Force flags - SET TO TRUE FOR NEW SAMPLE SIZE (1000 docs)
    FORCE_REGENERATE_SAMPLES = True
    FORCE_REGENERATE_TOPICS = True
    FORCE_REGENERATE_EVALUATION = True

    # Random seed
    RANDOM_SEED = 42

    # ===== LOAD DATASETS =====

    logger.info("Loading datasets...")
    from datasets import load_dataset

    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")

    logger.info(f"Loaded {len(corpus_dataset)} documents, {len(queries_dataset)} queries, {len(qrels_dataset)} qrels")

    # ===== HANDLE SINGLE QUERY VS MULTIPLE QUERIES =====

    # Convert single query ID to list for uniform processing
    if isinstance(QUERY_IDS, str):
        query_ids_to_run = [QUERY_IDS]
    else:
        query_ids_to_run = QUERY_IDS

    # ===== RUN EVALUATION FOR EACH QUERY =====

    all_results = {}

    for query_id in query_ids_to_run:
        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING QUERY {query_id} ({query_ids_to_run.index(query_id) + 1}/{len(query_ids_to_run)})")
        logger.info("="*80 + "\n")

        evaluator = EndToEndEvaluator(
            corpus_dataset=corpus_dataset,
            queries_dataset=queries_dataset,
            qrels_dataset=qrels_dataset,
            query_id=query_id,
            sample_size=SAMPLE_SIZE,
            embedding_model_name=EMBEDDING_MODEL,
            cross_encoder_model_name=CROSS_ENCODER_MODEL,
            dataset_name=DATASET_NAME,
            output_dir=OUTPUT_DIR,
            cache_dir=CACHE_DIR,
            random_seed=RANDOM_SEED,
            device=DEVICE,
            save_topic_models=SAVE_TOPIC_MODELS,
            force_regenerate_samples=FORCE_REGENERATE_SAMPLES,
            force_regenerate_topics=FORCE_REGENERATE_TOPICS,
            force_regenerate_evaluation=FORCE_REGENERATE_EVALUATION
        )

        results = evaluator.run_full_evaluation()
        all_results[query_id] = results

    # ===== FINAL SUMMARY =====

    logger.info("\n" + "="*80)
    logger.info("ALL QUERIES COMPLETED!")
    logger.info("="*80)
    logger.info(f"Processed {len(query_ids_to_run)} queries: {', '.join(query_ids_to_run)}")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info("="*80 + "\n")

    # ===== AGGREGATE CROSS-QUERY RESULTS =====

    # Only run aggregation if we processed multiple queries
    if len(query_ids_to_run) > 1:
        logger.info("\n" + "="*80)
        logger.info("AGGREGATING RESULTS ACROSS ALL QUERIES")
        logger.info("="*80 + "\n")

        aggregate_results = aggregate_cross_query_results(
            results_base_dir=OUTPUT_DIR,
            query_ids=query_ids_to_run
        )

        if aggregate_results:
            logger.info("\n" + "="*80)
            logger.info("AGGREGATION COMPLETE!")
            logger.info("="*80)
            logger.info(f"Aggregate results saved to: {OUTPUT_DIR}/aggregate_results")
            logger.info("="*80 + "\n")
    else:
        logger.info("\nSkipping cross-query aggregation (only 1 query processed)")

    return all_results


if __name__ == "__main__":
    main()
