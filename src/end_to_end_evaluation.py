# end_to_end_evaluation.py
"""
End-to-end evaluation of BERTopic using different sampling strategies.

Compares 4 methods:
1. Random Uniform Sampling (REFERENCE)
2. Direct Retrieval
3. Query Expansion + Retrieval
4. QRELs Labeled Set

Evaluates topic quality using topic-level and document-level metrics.
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
from query_expansion import (
    QueryExpander,
    QueryCombinationStrategy
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
    """End-to-end evaluator for topic modeling with different sampling strategies"""

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
        output_dir: str = "results/end_to_end_evaluation",
        cache_dir: str = "cache",
        random_seed: int = 42,
        device: str = "cpu",
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
        self.force_regenerate_samples = force_regenerate_samples
        self.force_regenerate_topics = force_regenerate_topics
        self.force_regenerate_evaluation = force_regenerate_evaluation

        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Get query text
        self.query_text = self._get_query_text()
        logger.info(f"Query {query_id}: {self.query_text}")

        # Determine sample size from qrels if not provided
        if sample_size is None:
            self.sample_size = self._get_qrels_sample_size()
            logger.info(f"Sample size determined from qrels: {self.sample_size}")
        else:
            self.sample_size = sample_size

        # Create output directories
        self.output_dir = os.path.join(output_dir, f"query_{query_id}")
        self.samples_dir = os.path.join(self.output_dir, "samples")
        self.topic_models_dir = os.path.join(self.output_dir, "topic_models")
        self.evaluation_dir = os.path.join(self.output_dir, "evaluation")
        self.plots_dir = os.path.join(self.evaluation_dir, "plots")

        for directory in [self.samples_dir, self.topic_models_dir, self.evaluation_dir, self.plots_dir]:
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

    def _get_qrels_sample_size(self) -> int:
        """Determine sample size from qrels labeled documents"""
        _, _, overall_relevant = SearchEvaluationUtils.build_qrels_dicts(self.qrels_dataset)

        # Convert query_id to int for qrels lookup
        query_id_int = int(self.query_id)

        if query_id_int not in overall_relevant:
            raise ValueError(f"Query ID {query_id_int} has no labeled documents in qrels")

        return len(overall_relevant[query_id_int])

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
        """Method 1: Random uniform sampling (REFERENCE)"""
        logger.info(f"Method 1: Random uniform sampling ({self.sample_size} docs)")

        cache_path = os.path.join(self.samples_dir, "method1_random_uniform.pkl")

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
        """Method 2: Direct retrieval"""
        logger.info(f"Method 2: Direct retrieval ({self.sample_size} docs)")

        cache_path = os.path.join(self.samples_dir, "method2_direct_retrieval.pkl")

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

        cache_path = os.path.join(self.samples_dir, "method3_query_expansion.pkl")

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

    def sample_qrels_labeled(self) -> Dict[str, Any]:
        """Method 4: QRELs labeled set"""
        logger.info(f"Method 4: QRELs labeled set")

        cache_path = os.path.join(self.samples_dir, "method4_qrels_labeled.pkl")

        def compute():
            # Get labeled doc IDs from qrels
            _, _, overall_relevant = SearchEvaluationUtils.build_qrels_dicts(self.qrels_dataset)
            query_id_int = int(self.query_id)
            labeled_doc_ids = overall_relevant[query_id_int]

            # Extract documents from corpus
            docs = []
            doc_ids = []

            for doc in tqdm(self.corpus_dataset, desc="Extracting labeled docs"):
                if doc["_id"] in labeled_doc_ids:
                    docs.append(doc["title"] + "\n\n" + doc["text"])
                    doc_ids.append(doc["_id"])

            # Check for missing documents
            missing = labeled_doc_ids - set(doc_ids)
            if missing:
                logger.warning(f"Query {self.query_id}: {len(missing)} labeled docs not found in corpus. Using {len(doc_ids)} docs.")

            return {
                "method": "qrels_labeled",
                "doc_ids": doc_ids,
                "doc_texts": docs,
                "sample_size": len(docs),
                "missing_docs": len(missing)
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
            embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)

            # Initialize BERTopic with pre-loaded embedding model
            topic_model = BERTopic(
                embedding_model=embedding_model,
                verbose=True,
                calculate_probabilities=True
            )

            # Fit model
            logger.info(f"Fitting BERTopic on {len(docs)} documents...")
            topics, probs = topic_model.fit_transform(docs)

            # Get topic info
            topic_info = topic_model.get_topic_info()

            # Get topic representations
            topic_words = {}
            for topic_id in topic_info['Topic']:
                if topic_id != -1:  # Skip outlier topic
                    words = topic_model.get_topic(topic_id)
                    topic_words[topic_id] = [word for word, _ in words]

            # Save model
            topic_model.save(model_cache_path, serialization="pickle")
            logger.info(f"Saved topic model to {model_cache_path}")

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
                "n_topics": len(topic_words),
                "n_docs": len(docs)
            }

            return results

        return self._load_or_compute(results_cache_path, compute, self.force_regenerate_topics)

    def evaluate_all(
        self,
        reference_results: Dict[str, Any],
        sample_results: Dict[str, Any],
        reference_sample: Dict[str, Any],
        sample_sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a sample against reference"""

        logger.info(f"Evaluating {sample_results['method']} against reference")

        metrics = {}

        # Topic-level metrics (Priority 1)
        metrics.update(self._compute_topic_level_metrics(reference_results, sample_results))

        # Document distribution metrics (Priority 2)
        metrics.update(self._compute_document_distribution_metrics(sample_results))

        # Document clustering metrics (Priority 3) - only if overlap exists
        metrics.update(self._compute_document_clustering_metrics(
            reference_results, sample_results, reference_sample, sample_sample
        ))

        return metrics

    def _compute_topic_level_metrics(
        self,
        reference_results: Dict[str, Any],
        sample_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute topic-level comparison metrics"""

        ref_topic_words = reference_results["topic_words"]
        sample_topic_words = sample_results["topic_words"]

        metrics = {
            "topic_coverage_reference": len(ref_topic_words),
            "topic_coverage_sample": len(sample_topic_words),
        }

        # Topic diversity (average pairwise distance)
        metrics["topic_diversity_reference"] = self._compute_topic_diversity(ref_topic_words)
        metrics["topic_diversity_sample"] = self._compute_topic_diversity(sample_topic_words)

        # Topic matching using semantic similarity
        word_overlap_scores, semantic_sim_scores = self._match_topics(
            ref_topic_words, sample_topic_words, top_n=10
        )

        metrics["topic_word_overlap_mean"] = float(np.mean(word_overlap_scores)) if word_overlap_scores else 0.0
        metrics["topic_word_overlap_std"] = float(np.std(word_overlap_scores)) if word_overlap_scores else 0.0
        metrics["topic_semantic_similarity_mean"] = float(np.mean(semantic_sim_scores)) if semantic_sim_scores else 0.0
        metrics["topic_semantic_similarity_std"] = float(np.std(semantic_sim_scores)) if semantic_sim_scores else 0.0

        # Topic precision, recall, F1 at multiple thresholds
        thresholds = [0.5, 0.6, 0.7]

        for threshold in thresholds:
            matched_topics = sum(1 for score in semantic_sim_scores if score >= threshold)

            precision = matched_topics / len(sample_topic_words) if len(sample_topic_words) > 0 else 0.0
            recall = matched_topics / len(ref_topic_words) if len(ref_topic_words) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Store with threshold suffix
            threshold_str = str(threshold).replace('.', '')
            metrics[f"topic_precision_@{threshold_str}"] = precision
            metrics[f"topic_recall_@{threshold_str}"] = recall
            metrics[f"topic_f1_@{threshold_str}"] = f1

        return metrics

    def _compute_topic_diversity(self, topic_words: Dict[int, List[str]]) -> float:
        """Compute average pairwise distance between topics"""
        if len(topic_words) < 2:
            return 0.0

        # Get embeddings for topic words
        embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)

        topic_embeddings = []
        for topic_id, words in topic_words.items():
            # Use top 5 words for topic representation
            top_words = " ".join(words[:5])
            embedding = embedding_model.encode(top_words, convert_to_tensor=False)
            topic_embeddings.append(embedding)

        topic_embeddings = np.array(topic_embeddings)

        # Compute pairwise cosine distances
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(topic_embeddings)

        # Get upper triangle (excluding diagonal)
        n = len(topic_embeddings)
        distances = 1 - similarities
        upper_triangle = distances[np.triu_indices(n, k=1)]

        return float(np.mean(upper_triangle))

    def _match_topics(
        self,
        ref_topic_words: Dict[int, List[str]],
        sample_topic_words: Dict[int, List[str]],
        top_n: int = 10
    ) -> Tuple[List[float], List[float]]:
        """Match topics using Hungarian algorithm and compute similarities"""

        if not ref_topic_words or not sample_topic_words:
            return [], []

        # Get embeddings for all topics
        embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)

        ref_topics = list(ref_topic_words.keys())
        sample_topics = list(sample_topic_words.keys())

        # Compute semantic similarity matrix
        ref_embeddings = []
        for topic_id in ref_topics:
            words = " ".join(ref_topic_words[topic_id][:top_n])
            emb = embedding_model.encode(words, convert_to_tensor=False)
            ref_embeddings.append(emb)

        sample_embeddings = []
        for topic_id in sample_topics:
            words = " ".join(sample_topic_words[topic_id][:top_n])
            emb = embedding_model.encode(words, convert_to_tensor=False)
            sample_embeddings.append(emb)

        ref_embeddings = np.array(ref_embeddings)
        sample_embeddings = np.array(sample_embeddings)

        # Compute similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(ref_embeddings, sample_embeddings)

        # Hungarian algorithm for optimal matching (maximize similarity = minimize -similarity)
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

        # Compute matched similarities
        semantic_similarities = []
        word_overlaps = []

        for i, j in zip(row_ind, col_ind):
            ref_topic_id = ref_topics[i]
            sample_topic_id = sample_topics[j]

            # Semantic similarity
            semantic_similarities.append(similarity_matrix[i, j])

            # Word overlap (Jaccard)
            ref_words_set = set(ref_topic_words[ref_topic_id][:top_n])
            sample_words_set = set(sample_topic_words[sample_topic_id][:top_n])

            intersection = len(ref_words_set & sample_words_set)
            union = len(ref_words_set | sample_words_set)
            jaccard = intersection / union if union > 0 else 0.0

            word_overlaps.append(jaccard)

        return word_overlaps, semantic_similarities

    def _compute_document_distribution_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metrics about document distribution across topics"""

        topics = np.array(results["topics"])

        # Count docs per topic (excluding -1)
        unique_topics, counts = np.unique(topics[topics != -1], return_counts=True)

        metrics = {}

        # Outlier ratio
        n_outliers = np.sum(topics == -1)
        metrics["outlier_ratio"] = float(n_outliers / len(topics)) if len(topics) > 0 else 0.0

        # Average topic size
        if len(counts) > 0:
            metrics["avg_topic_size"] = float(np.mean(counts))
            metrics["std_topic_size"] = float(np.std(counts))
        else:
            metrics["avg_topic_size"] = 0.0
            metrics["std_topic_size"] = 0.0

        # Singleton topic ratio
        n_singletons = np.sum(counts == 1)
        metrics["singleton_topic_ratio"] = float(n_singletons / len(counts)) if len(counts) > 0 else 0.0

        # Document-topic entropy
        if len(counts) > 0:
            probs = counts / np.sum(counts)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            metrics["document_topic_entropy"] = float(entropy)
        else:
            metrics["document_topic_entropy"] = 0.0

        return metrics

    def _compute_document_clustering_metrics(
        self,
        reference_results: Dict[str, Any],
        sample_results: Dict[str, Any],
        reference_sample: Dict[str, Any],
        sample_sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute document-level clustering metrics on overlapping documents"""

        metrics = {}

        # Find overlapping documents
        ref_doc_ids = set(reference_sample["doc_ids"])
        sample_doc_ids = set(sample_sample["doc_ids"])
        overlap = ref_doc_ids & sample_doc_ids

        metrics["overlap_count"] = len(overlap)

        if len(overlap) < 2:
            logger.info(f"Insufficient overlap ({len(overlap)} docs) for clustering metrics")
            metrics["ari"] = None
            metrics["nmi"] = None
            return metrics

        # Get topic assignments for overlapping docs
        ref_doc_to_idx = {doc_id: i for i, doc_id in enumerate(reference_sample["doc_ids"])}
        sample_doc_to_idx = {doc_id: i for i, doc_id in enumerate(sample_sample["doc_ids"])}

        ref_topics_overlap = []
        sample_topics_overlap = []

        for doc_id in overlap:
            ref_topics_overlap.append(reference_results["topics"][ref_doc_to_idx[doc_id]])
            sample_topics_overlap.append(sample_results["topics"][sample_doc_to_idx[doc_id]])

        # Compute metrics
        try:
            ari = adjusted_rand_score(ref_topics_overlap, sample_topics_overlap)
            nmi = normalized_mutual_info_score(ref_topics_overlap, sample_topics_overlap)

            metrics["ari"] = float(ari)
            metrics["nmi"] = float(nmi)
        except Exception as e:
            logger.warning(f"Error computing clustering metrics: {e}")
            metrics["ari"] = None
            metrics["nmi"] = None

        return metrics

    def create_summary_table(self, all_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Create summary table of all metrics"""

        # Define metric order and nice names
        metric_names = {
            # Topic-level (Priority 1)
            "topic_coverage_sample": "Topic Coverage",
            "topic_diversity_sample": "Topic Diversity",
            "topic_word_overlap_mean": "Word Overlap (mean)",
            "topic_semantic_similarity_mean": "Semantic Similarity (mean)",
            # Precision/Recall/F1 at multiple thresholds
            "topic_precision_@05": "Precision @0.5",
            "topic_recall_@05": "Recall @0.5",
            "topic_f1_@05": "F1 @0.5",
            "topic_precision_@06": "Precision @0.6",
            "topic_recall_@06": "Recall @0.6",
            "topic_f1_@06": "F1 @0.6",
            "topic_precision_@07": "Precision @0.7",
            "topic_recall_@07": "Recall @0.7",
            "topic_f1_@07": "F1 @0.7",
            # Document distribution (Priority 2)
            "outlier_ratio": "Outlier Ratio",
            "avg_topic_size": "Avg Topic Size",
            "singleton_topic_ratio": "Singleton Ratio",
            "document_topic_entropy": "Doc-Topic Entropy",
            # Clustering (Priority 3)
            "overlap_count": "Overlap Count",
            "ari": "ARI",
            "nmi": "NMI",
        }

        rows = []
        for method, metrics in all_metrics.items():
            row = {"Method": method}
            for metric_key, metric_name in metric_names.items():
                value = metrics.get(metric_key, None)
                if value is not None:
                    if isinstance(value, float):
                        row[metric_name] = f"{value:.4f}"
                    else:
                        row[metric_name] = str(value)
                else:
                    row[metric_name] = "N/A"
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def create_plots(self, all_metrics: Dict[str, Dict[str, Any]]):
        """Create visualization plots"""

        logger.info("Creating visualization plots...")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

        # Plot 1: Topic Coverage
        self._plot_bar_metric(
            all_metrics,
            "topic_coverage_sample",
            "Topic Coverage Comparison",
            "Number of Topics",
            "topic_coverage.png"
        )

        # Plot 2: Topic Precision, Recall, F1 at threshold 0.5
        self._plot_grouped_bars(
            all_metrics,
            ["topic_precision_@05", "topic_recall_@05", "topic_f1_@05"],
            "Topic Matching Metrics @0.5 Threshold",
            "topic_precision_recall_f1_threshold05.png"
        )

        # Plot 3: Topic Precision, Recall, F1 at threshold 0.6
        self._plot_grouped_bars(
            all_metrics,
            ["topic_precision_@06", "topic_recall_@06", "topic_f1_@06"],
            "Topic Matching Metrics @0.6 Threshold",
            "topic_precision_recall_f1_threshold06.png"
        )

        # Plot 4: Topic Precision, Recall, F1 at threshold 0.7
        self._plot_grouped_bars(
            all_metrics,
            ["topic_precision_@07", "topic_recall_@07", "topic_f1_@07"],
            "Topic Matching Metrics @0.7 Threshold",
            "topic_precision_recall_f1_threshold07.png"
        )

        # Plot 5: Topic Quality Metrics
        self._plot_grouped_bars(
            all_metrics,
            ["topic_diversity_sample", "topic_word_overlap_mean", "topic_semantic_similarity_mean"],
            "Topic Quality Metrics",
            "topic_quality.png"
        )

        # Plot 6: Document Distribution
        self._plot_grouped_bars(
            all_metrics,
            ["outlier_ratio", "singleton_topic_ratio", "document_topic_entropy"],
            "Document Distribution Metrics",
            "document_distribution.png"
        )

        logger.info(f"Plots saved to {self.plots_dir}")

    def _plot_bar_metric(self, all_metrics, metric_key, title, ylabel, filename):
        """Create a simple bar plot for a single metric"""
        methods = list(all_metrics.keys())
        values = [all_metrics[m].get(metric_key, 0) for m in methods]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel("Method", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}' if isinstance(value, float) else str(value),
                    ha='center', va='bottom', fontsize=10)

        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_grouped_bars(self, all_metrics, metric_keys, title, filename):
        """Create grouped bar plot for multiple metrics"""
        methods = list(all_metrics.keys())

        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, metric_key in enumerate(metric_keys):
            values = [all_metrics[m].get(metric_key, 0) for m in methods]
            offset = width * (i - len(metric_keys)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=metric_key.replace('_', ' ').title(),
                         color=colors[i % len(colors)])

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if value is not None and value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}',
                           ha='center', va='bottom', fontsize=8)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Method', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def run_full_evaluation(self):
        """Run full end-to-end evaluation pipeline"""

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
        samples["qrels_labeled"] = self.sample_qrels_labeled()

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

        # Step 3: Evaluate
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Computing Evaluation Metrics")
        logger.info("="*80)

        # Use random_uniform as reference
        reference_name = "random_uniform"
        reference_results = topic_results[reference_name]
        reference_sample = samples[reference_name]

        all_metrics = {}

        # Check if reference model succeeded
        if reference_results is None:
            logger.error(f"Reference method '{reference_name}' failed. Cannot proceed with evaluation.")
            logger.error("All topic modeling failed. Please check CUDA/device compatibility.")
            return {
                "samples": samples,
                "topic_results": topic_results,
                "metrics": {},
                "summary_df": None,
                "error": "Reference topic model failed"
            }

        # Evaluate all methods against reference (including reference against itself)
        for method_name, results in topic_results.items():
            if results is None:
                continue

            try:
                metrics = self.evaluate_all(
                    reference_results,
                    results,
                    reference_sample,
                    samples[method_name]
                )
                metrics["method"] = method_name
                all_metrics[method_name] = metrics
            except Exception as e:
                logger.error(f"Error evaluating {method_name}: {e}")
                continue

        # Step 4: Create summary and plots
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Creating Summary and Visualizations")
        logger.info("="*80)

        # Create summary table
        summary_df = self.create_summary_table(all_metrics)

        # Save results
        summary_csv_path = os.path.join(self.evaluation_dir, "metrics_summary.csv")
        summary_json_path = os.path.join(self.evaluation_dir, "metrics_summary.json")
        metrics_raw_path = os.path.join(self.evaluation_dir, "metrics_raw.pkl")

        summary_df.to_csv(summary_csv_path, index=False)

        with open(summary_json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        with open(metrics_raw_path, 'wb') as f:
            pickle.dump({
                "samples": samples,
                "topic_results": topic_results,
                "metrics": all_metrics
            }, f)

        logger.info(f"Summary saved to {summary_csv_path}")
        logger.info(f"Metrics saved to {summary_json_path}")
        logger.info(f"Raw data saved to {metrics_raw_path}")

        # Create plots
        self.create_plots(all_metrics)

        # Print summary
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        print("\n" + summary_df.to_string(index=False))

        logger.info("\n" + "="*80)
        logger.info("Evaluation Complete!")
        logger.info(f"All results saved to: {self.output_dir}")
        logger.info("="*80)

        return {
            "samples": samples,
            "topic_results": topic_results,
            "metrics": all_metrics,
            "summary_df": summary_df
        }

    def _compute_topic_diversity_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Helper to compute diversity for reference"""
        topic_words = results["topic_words"]
        diversity = self._compute_topic_diversity(topic_words)
        return {"topic_diversity_sample": diversity}


def main():
    """Main function"""

    # ===== CONFIGURATION PARAMETERS =====

    # Query configuration
    QUERY_ID = "43"

    # Sample size (None = auto-determine from qrels)
    SAMPLE_SIZE = None

    # Model configuration
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Dataset configuration
    DATASET_NAME = "trec-covid"

    # Directory configuration
    OUTPUT_DIR = "results/end_to_end_evaluation"
    CACHE_DIR = "cache"

    # Device configuration
    DEVICE = "cpu"  # Use "cpu" to avoid CUDA errors, "cuda" if GPU is compatible

    # Force flags
    FORCE_REGENERATE_SAMPLES = False
    FORCE_REGENERATE_TOPICS = False
    FORCE_REGENERATE_EVALUATION = False

    # Random seed
    RANDOM_SEED = 42

    # ===== LOAD DATASETS =====

    logger.info("Loading datasets...")
    from datasets import load_dataset

    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")

    logger.info(f"Loaded {len(corpus_dataset)} documents, {len(queries_dataset)} queries, {len(qrels_dataset)} qrels")

    # ===== INITIALIZE EVALUATOR =====

    evaluator = EndToEndEvaluator(
        corpus_dataset=corpus_dataset,
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        query_id=QUERY_ID,
        sample_size=SAMPLE_SIZE,
        embedding_model_name=EMBEDDING_MODEL,
        cross_encoder_model_name=CROSS_ENCODER_MODEL,
        dataset_name=DATASET_NAME,
        output_dir=OUTPUT_DIR,
        cache_dir=CACHE_DIR,
        random_seed=RANDOM_SEED,
        device=DEVICE,
        force_regenerate_samples=FORCE_REGENERATE_SAMPLES,
        force_regenerate_topics=FORCE_REGENERATE_TOPICS,
        force_regenerate_evaluation=FORCE_REGENERATE_EVALUATION
    )

    # ===== RUN EVALUATION =====

    results = evaluator.run_full_evaluation()

    return results


if __name__ == "__main__":
    main()