# topic_evaluation.py
import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any, Set
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TopicEvaluator:
    """Class for evaluating topic models against a reference model"""
    
    def __init__(self, cache_dir: str = "cache/topic_evaluation"):
        """
        Initialize topic evaluator
        
        Args:
            cache_dir: Directory for caching results
        """
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, reference_method: str, sample_method: str, prefix: str = "eval") -> str:
        """
        Generate a cache path based on reference and sample method names
        
        Args:
            reference_method: Name of the reference method
            sample_method: Name of the evaluated method
            prefix: Optional prefix for cache file
            
        Returns:
            Cache file path
        """
        # Create safe names
        safe_ref = reference_method.replace(" ", "_").lower()
        safe_sample = sample_method.replace(" ", "_").lower()
        
        # Create the filename
        filename = f"{prefix}_{safe_ref}_vs_{safe_sample}.pkl"
        
        return os.path.join(self.cache_dir, filename)
    
    def calculate_adjusted_rand_index(
        self,
        reference_topics: List[int],
        sample_topics: List[int],
        reference_docs: List[str],
        sample_docs: List[str]
    ) -> float:
        """
        Calculate Adjusted Rand Index (ARI) between two topic assignments
        
        Args:
            reference_topics: Topic assignments for reference documents
            sample_topics: Topic assignments for sample documents
            reference_docs: Reference document IDs or texts
            sample_docs: Sample document IDs or texts
            
        Returns:
            ARI score
        """
        # Find common documents
        common_docs = list(set(reference_docs).intersection(set(sample_docs)))
        
        if not common_docs:
            logger.warning("No common documents found for ARI calculation")
            return 0.0
        
        # Create mappings from documents to indices
        ref_doc_to_idx = {doc: i for i, doc in enumerate(reference_docs)}
        sample_doc_to_idx = {doc: i for i, doc in enumerate(sample_docs)}
        
        # Get topic assignments for common documents
        ref_topics = [reference_topics[ref_doc_to_idx[doc]] for doc in common_docs]
        sample_topics = [sample_topics[sample_doc_to_idx[doc]] for doc in common_docs]
        
        # Calculate ARI
        return adjusted_rand_score(ref_topics, sample_topics)
    
    def calculate_normalized_mutual_info(
        self,
        reference_topics: List[int],
        sample_topics: List[int],
        reference_docs: List[str],
        sample_docs: List[str],
        average_method: str = 'arithmetic'
    ) -> float:
        """
        Calculate Normalized Mutual Information (NMI) between two topic assignments
        
        Args:
            reference_topics: Topic assignments for reference documents
            sample_topics: Topic assignments for sample documents
            reference_docs: Reference document IDs or texts
            sample_docs: Sample document IDs or texts
            average_method: Method for averaging ('arithmetic', 'geometric', 'min', 'max')
            
        Returns:
            NMI score
        """
        # Find common documents
        common_docs = list(set(reference_docs).intersection(set(sample_docs)))
        
        if not common_docs:
            logger.warning("No common documents found for NMI calculation")
            return 0.0
        
        # Create mappings from documents to indices
        ref_doc_to_idx = {doc: i for i, doc in enumerate(reference_docs)}
        sample_doc_to_idx = {doc: i for i, doc in enumerate(sample_docs)}
        
        # Get topic assignments for common documents
        ref_topics = [reference_topics[ref_doc_to_idx[doc]] for doc in common_docs]
        sample_topics = [sample_topics[sample_doc_to_idx[doc]] for doc in common_docs]
        
        # Calculate NMI
        return normalized_mutual_info_score(
            ref_topics, 
            sample_topics, 
            average_method=average_method
        )
    
    def calculate_omega_index(
        self,
        reference_topics: List[int],
        sample_topics: List[int],
        reference_docs: List[str],
        sample_docs: List[str]
    ) -> float:
        """
        Calculate Omega Index between two topic assignments
        Omega Index measures agreement on both same and different-cluster assignments
        
        Args:
            reference_topics: Topic assignments for reference documents
            sample_topics: Topic assignments for sample documents
            reference_docs: Reference document IDs or texts
            sample_docs: Sample document IDs or texts
            
        Returns:
            Omega Index score
        """
        # Find common documents
        common_docs = list(set(reference_docs).intersection(set(sample_docs)))
        
        if not common_docs:
            logger.warning("No common documents found for Omega Index calculation")
            return 0.0
        
        # Create mappings from documents to indices
        ref_doc_to_idx = {doc: i for i, doc in enumerate(reference_docs)}
        sample_doc_to_idx = {doc: i for i, doc in enumerate(sample_docs)}
        
        # Get topic assignments for common documents
        ref_topics = [reference_topics[ref_doc_to_idx[doc]] for doc in common_docs]
        sample_topics = [sample_topics[sample_doc_to_idx[doc]] for doc in common_docs]
        
        # Calculate all document pairs that are in the same cluster in each clustering
        n = len(common_docs)
        
        # Get pairs in same cluster for reference
        ref_same_cluster = defaultdict(set)
        for i in range(n):
            for j in range(i+1, n):
                if ref_topics[i] == ref_topics[j] and ref_topics[i] != -1:
                    ref_same_cluster[ref_topics[i]].add((i, j))
        
        # Get pairs in same cluster for sample
        sample_same_cluster = defaultdict(set)
        for i in range(n):
            for j in range(i+1, n):
                if sample_topics[i] == sample_topics[j] and sample_topics[i] != -1:
                    sample_same_cluster[sample_topics[i]].add((i, j))
        
        # Count pairs with same relationship (both in same cluster or both in different clusters)
        total_pairs = n * (n - 1) // 2
        
        # Flatten sets of pairs
        ref_pairs = set()
        for pairs in ref_same_cluster.values():
            ref_pairs.update(pairs)
            
        sample_pairs = set()
        for pairs in sample_same_cluster.values():
            sample_pairs.update(pairs)
        
        # Count pairs with same relationship
        same_relationship = len(ref_pairs.intersection(sample_pairs)) + (total_pairs - len(ref_pairs.union(sample_pairs)))
        
        # Calculate expected same relationship by chance
        ref_same_count = len(ref_pairs)
        sample_same_count = len(sample_pairs)
        
        expected_same = (ref_same_count * sample_same_count + (total_pairs - ref_same_count) * (total_pairs - sample_same_count)) / total_pairs
        
        # Calculate Omega Index
        if total_pairs == expected_same:
            return 1.0  # Perfect agreement with expected
        else:
            return (same_relationship - expected_same) / (total_pairs - expected_same)
    
    def calculate_bcubed_precision_recall(
        self,
        reference_topics: List[int],
        sample_topics: List[int],
        reference_docs: List[str],
        sample_docs: List[str]
    ) -> Tuple[float, float, float]:
        """
        Calculate B-Cubed Precision, Recall, and F1-Score
        B-Cubed metrics are designed for overlapping clustering evaluation
        
        Args:
            reference_topics: Topic assignments for reference documents
            sample_topics: Topic assignments for sample documents
            reference_docs: Reference document IDs or texts
            sample_docs: Sample document IDs or texts
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        # Find common documents
        common_docs = list(set(reference_docs).intersection(set(sample_docs)))
        
        if not common_docs:
            logger.warning("No common documents found for B-Cubed calculation")
            return 0.0, 0.0, 0.0
        
        # Create mappings from documents to indices
        ref_doc_to_idx = {doc: i for i, doc in enumerate(reference_docs)}
        sample_doc_to_idx = {doc: i for i, doc in enumerate(sample_docs)}
        
        # Get topic assignments for common documents
        ref_topics = [reference_topics[ref_doc_to_idx[doc]] for doc in common_docs]
        sample_topics = [sample_topics[sample_doc_to_idx[doc]] for doc in common_docs]
        
        # Calculate B-Cubed Precision and Recall
        precision_sum = 0.0
        recall_sum = 0.0
        n = len(common_docs)
        
        for i in range(n):
            # Count documents in same cluster in sample
            sample_same_cluster = [j for j in range(n) if sample_topics[j] == sample_topics[i] and sample_topics[i] != -1]
            sample_cluster_size = len(sample_same_cluster)
            
            # Count documents in same cluster in reference
            ref_same_cluster = [j for j in range(n) if ref_topics[j] == ref_topics[i] and ref_topics[i] != -1]
            ref_cluster_size = len(ref_same_cluster)
            
            # Count correctly clustered documents
            correctly_clustered = len([j for j in sample_same_cluster if ref_topics[j] == ref_topics[i] and ref_topics[i] != -1])
            
            # Calculate precision and recall for this document
            if sample_cluster_size > 0:
                precision_sum += correctly_clustered / sample_cluster_size
            
            if ref_cluster_size > 0:
                recall_sum += correctly_clustered / ref_cluster_size
        
        # Average precision and recall
        precision = precision_sum / n if n > 0 else 0.0
        recall = recall_sum / n if n > 0 else 0.0
        
        # Calculate F1-score
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        return precision, recall, f1_score
    
    def calculate_topic_word_overlap(
        self,
        reference_topic_words: Dict[int, List[Tuple[str, float]]],
        sample_topic_words: Dict[int, List[Tuple[str, float]]],
        top_n: int = 10
    ) -> Dict[int, Dict[int, float]]:
        """
        Calculate overlap between topic words across two models
        
        Args:
            reference_topic_words: Reference topic words dictionary
            sample_topic_words: Sample topic words dictionary
            top_n: Number of top words to consider
            
        Returns:
            Dictionary mapping reference topics to dictionaries mapping sample topics to overlap scores
        """
        # Skip outlier topic (-1)
        ref_topics = {k: v for k, v in reference_topic_words.items() if k != -1}
        sample_topics = {k: v for k, v in sample_topic_words.items() if k != -1}
        
        # Extract top-n words for each topic
        ref_top_words = {
            topic: set(word for word, _ in words[:top_n]) 
            for topic, words in ref_topics.items()
        }
        
        sample_top_words = {
            topic: set(word for word, _ in words[:top_n]) 
            for topic, words in sample_topics.items()
        }
        
        # Calculate Jaccard similarity between all pairs of topics
        overlap_matrix = {}
        
        for ref_topic, ref_words in ref_top_words.items():
            overlap_matrix[ref_topic] = {}
            
            for sample_topic, sample_words in sample_top_words.items():
                # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
                intersection = len(ref_words.intersection(sample_words))
                union = len(ref_words.union(sample_words))
                
                if union > 0:
                    jaccard = intersection / union
                else:
                    jaccard = 0.0
                
                overlap_matrix[ref_topic][sample_topic] = jaccard
        
        return overlap_matrix
    
    def calculate_topic_word_precision_recall_greedy(
        self,
        reference_topic_words: Dict[int, List[Tuple[str, float]]],
        sample_topic_words: Dict[int, List[Tuple[str, float]]],
        top_n: int = 10
    ) -> Tuple[float, float, float]:
        """
        Calculate precision and recall for topic words using greedy matching
        
        Args:
            reference_topic_words: Reference topic words dictionary
            sample_topic_words: Sample topic words dictionary
            top_n: Number of top words to consider
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        # Calculate overlap matrix
        overlap_matrix = self.calculate_topic_word_overlap(
            reference_topic_words,
            sample_topic_words,
            top_n=top_n
        )
        
        # Skip outlier topic (-1)
        ref_topics = list(k for k in reference_topic_words.keys() if k != -1)
        sample_topics = list(k for k in sample_topic_words.keys() if k != -1)
        
        if not ref_topics or not sample_topics:
            logger.warning("No valid topics found for precision/recall calculation")
            return 0.0, 0.0, 0.0
        
        # Perform greedy matching
        matched_pairs = []
        matched_ref_topics = set()
        matched_sample_topics = set()
        
        # Extract all scores
        all_scores = []
        for ref_topic, scores in overlap_matrix.items():
            for sample_topic, score in scores.items():
                all_scores.append((ref_topic, sample_topic, score))
        
        # Sort by score (descending)
        all_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy matching
        for ref_topic, sample_topic, score in all_scores:
            if ref_topic not in matched_ref_topics and sample_topic not in matched_sample_topics:
                matched_pairs.append((ref_topic, sample_topic, score))
                matched_ref_topics.add(ref_topic)
                matched_sample_topics.add(sample_topic)
        
        # Calculate precision and recall
        total_overlap = sum(score for _, _, score in matched_pairs)
        precision = total_overlap / len(sample_topics) if sample_topics else 0.0
        recall = total_overlap / len(ref_topics) if ref_topics else 0.0
        
        # Calculate F1-score
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        return precision, recall, f1_score
    
    def calculate_topic_word_precision_recall_hungarian(
        self,
        reference_topic_words: Dict[int, List[Tuple[str, float]]],
        sample_topic_words: Dict[int, List[Tuple[str, float]]],
        top_n: int = 10
    ) -> Tuple[float, float, float, List[Tuple[int, int, float]]]:
        """
        Calculate precision and recall for topic words using Hungarian algorithm
        This provides the optimal matching between topics
        
        Args:
            reference_topic_words: Reference topic words dictionary
            sample_topic_words: Sample topic words dictionary
            top_n: Number of top words to consider
            
        Returns:
            Tuple of (precision, recall, f1_score, matched_pairs)
        """
        # Calculate overlap matrix
        overlap_matrix = self.calculate_topic_word_overlap(
            reference_topic_words,
            sample_topic_words,
            top_n=top_n
        )
        
        # Skip outlier topic (-1)
        ref_topics = list(k for k in reference_topic_words.keys() if k != -1)
        sample_topics = list(k for k in sample_topic_words.keys() if k != -1)
        
        if not ref_topics or not sample_topics:
            logger.warning("No valid topics found for precision/recall calculation")
            return 0.0, 0.0, 0.0, []
        
        # Create cost matrix for Hungarian algorithm
        cost_matrix = np.zeros((len(ref_topics), len(sample_topics)))
        
        for i, ref_topic in enumerate(ref_topics):
            for j, sample_topic in enumerate(sample_topics):
                # Convert similarity to cost (1 - similarity)
                cost_matrix[i, j] = 1.0 - overlap_matrix[ref_topic][sample_topic]
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract matched pairs
        matched_pairs = []
        total_overlap = 0.0
        
        for i, j in zip(row_ind, col_ind):
            ref_topic = ref_topics[i]
            sample_topic = sample_topics[j]
            overlap = 1.0 - cost_matrix[i, j]  # Convert cost back to similarity
            
            matched_pairs.append((ref_topic, sample_topic, overlap))
            total_overlap += overlap
        
        # Calculate precision and recall
        precision = total_overlap / len(sample_topics) if sample_topics else 0.0
        recall = total_overlap / len(ref_topics) if ref_topics else 0.0
        
        # Calculate F1-score
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        return precision, recall, f1_score, matched_pairs
    
    def calculate_topic_semantic_similarity(
        self,
        reference_topic_words: Dict[int, List[Tuple[str, float]]],
        sample_topic_words: Dict[int, List[Tuple[str, float]]],
        top_n: int = 10
    ) -> Dict[int, Dict[int, float]]:
        """
        Calculate semantic similarity between topic words using embeddings
        
        Args:
            reference_topic_words: Reference topic words dictionary
            sample_topic_words: Sample topic words dictionary
            top_n: Number of top words to consider
            
        Returns:
            Dictionary mapping reference topics to dictionaries mapping sample topics to similarity scores
        """
        # Skip outlier topic (-1)
        ref_topics = {k: v for k, v in reference_topic_words.items() if k != -1}
        sample_topics = {k: v for k, v in sample_topic_words.items() if k != -1}
        
        # Ensure we have topics to compare
        if not ref_topics or not sample_topics:
            return {}
        
        # Extract top-n words for each topic
        ref_top_words = {
            topic: [word for word, _ in words[:top_n]] 
            for topic, words in ref_topics.items()
        }
        
        sample_top_words = {
            topic: [word for word, _ in words[:top_n]] 
            for topic, words in sample_topics.items()
        }
        
        # Calculate similarity matrix
        similarity_matrix = {}
        
        # Get or create sentence transformer model
        if not hasattr(self, 'sbert_model'):
            from sentence_transformers import SentenceTransformer
            self.sbert_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Calculate embeddings for all topic word lists
        ref_embeddings = {}
        for topic, words in ref_top_words.items():
            # Join words into a space-separated string
            topic_text = " ".join(words)
            ref_embeddings[topic] = self.sbert_model.encode(topic_text, convert_to_tensor=True)
        
        sample_embeddings = {}
        for topic, words in sample_top_words.items():
            topic_text = " ".join(words)
            sample_embeddings[topic] = self.sbert_model.encode(topic_text, convert_to_tensor=True)
        
        # Calculate cosine similarity between all pairs
        from sentence_transformers import util
        
        for ref_topic, ref_embedding in ref_embeddings.items():
            similarity_matrix[ref_topic] = {}
            
            for sample_topic, sample_embedding in sample_embeddings.items():
                # Calculate cosine similarity
                cos_sim = util.cos_sim(ref_embedding, sample_embedding).item()
                similarity_matrix[ref_topic][sample_topic] = cos_sim
        
        return similarity_matrix

    def visualize_topic_overlap_matrix(
        self,
        evaluation_result: Dict,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize topic similarity matrix as a heatmap
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract overlap matrix
        overlap_matrix = evaluation_result.get("topic_word_overlap", {})
        
        # Check if we have data to visualize
        if not overlap_matrix:
            logger.warning("No topic overlap data to visualize")
            # Create an empty figure
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No topic overlap data available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            
            # Save if requested
            if output_path:
                plt.savefig(output_path, dpi=300)
                logger.info(f"Saved empty topic overlap matrix to: {output_path}")
            
            return plt.gcf()
        
        # Convert to numpy array
        ref_topics = sorted(overlap_matrix.keys())
        
        # Check if we have reference topics
        if not ref_topics:
            logger.warning("No reference topics found in overlap matrix")
            # Create an empty figure
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No reference topics available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            
            # Save if requested
            if output_path:
                plt.savefig(output_path, dpi=300)
            
            return plt.gcf()
        
        # Get sample topics from the first reference topic's scores
        # (all reference topics should have the same sample topics)
        first_ref = ref_topics[0]
        if first_ref not in overlap_matrix or not overlap_matrix[first_ref]:
            logger.warning("No sample topics found in overlap matrix")
            # Create an empty figure
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No sample topics available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            
            # Save if requested
            if output_path:
                plt.savefig(output_path, dpi=300)
            
            return plt.gcf()
        
        sample_topics = sorted(overlap_matrix[first_ref].keys())
        
        # Create matrix array
        matrix_array = np.zeros((len(ref_topics), len(sample_topics)))
        
        for i, ref_topic in enumerate(ref_topics):
            for j, sample_topic in enumerate(sample_topics):
                if sample_topic in overlap_matrix[ref_topic]:
                    matrix_array[i, j] = overlap_matrix[ref_topic][sample_topic]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot heatmap
        ax = sns.heatmap(
            matrix_array,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=[f"T{t}" for t in sample_topics],
            yticklabels=[f"T{t}" for t in ref_topics]
        )
        
        # Add labels
        plt.xlabel(f"Topics in {evaluation_result['sample_method']}")
        plt.ylabel(f"Topics in {evaluation_result['reference_method']}")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Topic Similarity: {evaluation_result['reference_method']} vs {evaluation_result['sample_method']}")
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved topic similarity matrix to: {output_path}")
        
        return plt.gcf()
    
    def evaluate_topic_model(
        self,
        reference_result: Dict,
        sample_result: Dict,
        top_n_words: int = 10,
        force_recompute: bool = False
    ) -> Dict:
        """
        Evaluate a topic model against a reference model
        
        Args:
            reference_result: Reference topic model result
            sample_result: Sample topic model result
            top_n_words: Number of top words to consider for word-level metrics
            force_recompute: Whether to force recomputing evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract method names
        reference_method = reference_result["method"]
        sample_method = sample_result["method"]
        
        # Generate cache path
        cache_path = self._get_cache_path(reference_method, sample_method)
        
        # Check cache
        if not force_recompute and os.path.exists(cache_path):
            logger.info(f"Loading evaluation results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Evaluating topic model: {sample_method} against {reference_method}...")
        
        # Extract topic assignments and documents
        reference_topics = reference_result["topics"]
        sample_topics = sample_result["topics"]
        
        # Extract document info (using document text as identifier)
        reference_docs = reference_result["document_info"]["Document"].tolist()
        sample_docs = sample_result["document_info"]["Document"].tolist()
        
        # Calculate document-level metrics
        ari = self.calculate_adjusted_rand_index(
            reference_topics,
            sample_topics,
            reference_docs,
            sample_docs
        )
        
        nmi = self.calculate_normalized_mutual_info(
            reference_topics,
            sample_topics,
            reference_docs,
            sample_docs
        )
        
        omega = self.calculate_omega_index(
            reference_topics,
            sample_topics,
            reference_docs,
            sample_docs
        )
        
        bcubed_precision, bcubed_recall, bcubed_f1 = self.calculate_bcubed_precision_recall(
            reference_topics,
            sample_topics,
            reference_docs,
            sample_docs
        )
        
        # Calculate semantic similarity between topics
        topic_similarity = self.calculate_topic_semantic_similarity(
            reference_result["topic_words"],
            sample_result["topic_words"],
            top_n=top_n_words
        )
        
        # Calculate topic word level metrics using topic similarity
        # Greedy matching
        word_precision_greedy, word_recall_greedy, word_f1_greedy = self.calculate_topic_word_precision_recall_greedy(
            reference_result["topic_words"],
            sample_result["topic_words"],
            top_n=top_n_words
        )
        
        # Hungarian matching
        word_precision_hungarian, word_recall_hungarian, word_f1_hungarian, matched_pairs = self.calculate_topic_word_precision_recall_hungarian(
            reference_result["topic_words"],
            sample_result["topic_words"],
            top_n=top_n_words
        )
        
        # Build result dictionary
        result = {
            "reference_method": reference_method,
            "sample_method": sample_method,
            "document_metrics": {
                "adjusted_rand_index": ari,
                "normalized_mutual_info": nmi,
                "omega_index": omega,
                "bcubed_precision": bcubed_precision,
                "bcubed_recall": bcubed_recall,
                "bcubed_f1": bcubed_f1
            },
            "topic_word_metrics": {
                "greedy": {
                    "precision": word_precision_greedy,
                    "recall": word_recall_greedy,
                    "f1_score": word_f1_greedy
                },
                "hungarian": {
                    "precision": word_precision_hungarian,
                    "recall": word_recall_hungarian,
                    "f1_score": word_f1_hungarian,
                    "matched_pairs": matched_pairs
                }
            },
            "topic_word_overlap": topic_similarity,  # Use semantic similarity instead of word overlap
            "common_docs_count": len(set(reference_docs).intersection(set(sample_docs))),
            "reference_docs_count": len(reference_docs),
            "sample_docs_count": len(sample_docs)
        }
        
        # Cache results
        logger.info(f"Caching evaluation results to: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def evaluate_multiple_models(
        self,
        reference_result: Dict,
        sample_results: Dict[str, Dict],
        top_n_words: int = 10,
        force_recompute: bool = False
    ) -> Dict[str, Dict]:
        """
        Evaluate multiple topic models against a reference model
        
        Args:
            reference_result: Reference topic model result
            sample_results: Dictionary mapping sample names to results
            top_n_words: Number of top words to consider for word-level metrics
            force_recompute: Whether to force recomputing evaluation
            
        Returns:
            Dictionary mapping sample names to evaluation results
        """
        evaluation_results = {}
        
        for name, result in sample_results.items():
            logger.info(f"Evaluating model: {name}")
            
            # Add the method name to the result for identification
            result["method"] = name
            
            # Run evaluation
            eval_result = self.evaluate_topic_model(
                reference_result=reference_result,
                sample_result=result,
                top_n_words=top_n_words,
                force_recompute=force_recompute
            )
            
            evaluation_results[name] = eval_result
        
        return evaluation_results
    
    def visualize_document_metrics_radar(
        self,
        evaluation_result: Dict,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a radar chart for document-level metrics
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract metrics
        doc_metrics = evaluation_result["document_metrics"]
        
        # Define metrics to include
        metrics = [
            "adjusted_rand_index", "normalized_mutual_info", "omega_index",
            "bcubed_precision", "bcubed_recall", "bcubed_f1"
        ]
        
        # Define nice labels
        metric_labels = {
            "adjusted_rand_index": "ARI",
            "normalized_mutual_info": "NMI",
            "omega_index": "Omega",
            "bcubed_precision": "B³ Precision",
            "bcubed_recall": "B³ Recall",
            "bcubed_f1": "B³ F1"
        }
        
        # Extract values
        values = [doc_metrics[m] for m in metrics]
        labels = [metric_labels[m] for m in metrics]
        
        # Create radar chart
        plt.figure(figsize=(10, 8))
        
        # Set up radial axes
        ax = plt.subplot(111, polar=True)
        
        # Number of metrics
        N = len(metrics)
        
        # Ensure values are normalized between 0 and 1
        values = np.clip(values, 0, 1)
        
        # Create angles for each metric (evenly spaced on unit circle)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        # Make the plot circular by repeating the first value and angle
        values = np.append(values, values[0])
        angles = np.append(angles, angles[0])
        labels = labels + [labels[0]]  # Add the first label again to close the circle
        
        # Plot data
        ax.plot(angles, values, 'b-', linewidth=2)
        ax.fill(angles, values, 'b', alpha=0.2)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set labels at the correct angles
        ax.set_xticks(angles[:-1])  # Use the angles without the repeated one
        ax.set_xticklabels(labels[:-1])  # Use labels without the repeated one
        
        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(f"Document-Level Metrics: {evaluation_result['sample_method']}")
        
        # Draw y-axis markers
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved document metrics radar chart to: {output_path}")
        
        return plt.gcf()

    def visualize_topic_word_metrics_comparison(
        self,
        evaluation_result: Dict,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a bar chart comparing greedy and Hungarian matching for topic word metrics
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract metrics
        greedy_metrics = evaluation_result["topic_word_metrics"]["greedy"]
        hungarian_metrics = evaluation_result["topic_word_metrics"]["hungarian"]
        
        # Define metrics to include
        metrics = ["precision", "recall", "f1_score"]
        
        # Extract values
        greedy_values = [greedy_metrics[m] for m in metrics]
        hungarian_values = [hungarian_metrics[m] for m in metrics]
        
        # Setup plot
        plt.figure(figsize=(10, 6))
        
        # Set width of bars
        barWidth = 0.35
        
        # Set position of bars
        r1 = np.arange(len(metrics))
        r2 = [x + barWidth for x in r1]
        
        # Create bars
        plt.bar(r1, greedy_values, width=barWidth, edgecolor='white', label='Greedy Matching')
        plt.bar(r2, hungarian_values, width=barWidth, edgecolor='white', label='Hungarian Matching')
        
        # Add value labels on top of each bar
        for i, v in enumerate(greedy_values):
            plt.text(r1[i], v + 0.01, f"{v:.3f}", ha='center')
        
        for i, v in enumerate(hungarian_values):
            plt.text(r2[i], v + 0.01, f"{v:.3f}", ha='center')
        
        # Add labels and legend
        plt.xlabel('Metric', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.xticks([r + barWidth/2 for r in range(len(metrics))], [m.capitalize() for m in metrics])
        plt.legend()
        
        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(f"Topic Word Metrics: {evaluation_result['sample_method']}")
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved topic word metrics comparison chart to: {output_path}")
        
        return plt.gcf()

    def visualize_bcubed_metrics(
        self,
        evaluation_result: Dict,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a pie chart visualizing B-Cubed metrics
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract metrics
        bcubed_precision = evaluation_result["document_metrics"]["bcubed_precision"]
        bcubed_recall = evaluation_result["document_metrics"]["bcubed_recall"]
        bcubed_f1 = evaluation_result["document_metrics"]["bcubed_f1"]
        
        # Setup plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Create precision pie chart
        precision_sizes = [bcubed_precision, 1 - bcubed_precision]
        ax1.pie(precision_sizes, explode=(0.1, 0), 
                labels=['Correct', 'Incorrect'],
                colors=['#66b3ff', '#e6e6e6'],
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.set_title(f'B³ Precision: {bcubed_precision:.4f}')
        
        # Create recall pie chart
        recall_sizes = [bcubed_recall, 1 - bcubed_recall]
        ax2.pie(recall_sizes, explode=(0.1, 0), 
                labels=['Found', 'Missed'],
                colors=['#99ff99', '#e6e6e6'],
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax2.set_title(f'B³ Recall: {bcubed_recall:.4f}')
        
        # Create F1 pie chart
        f1_sizes = [bcubed_f1, 1 - bcubed_f1]
        ax3.pie(f1_sizes, explode=(0.1, 0), 
                labels=['F1 Score', 'Remainder'],
                colors=['#ffcc99', '#e6e6e6'],
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax3.set_title(f'B³ F1 Score: {bcubed_f1:.4f}')
        
        # Set title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"B-Cubed Metrics: {evaluation_result['sample_method']}", fontsize=16)
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved B-Cubed metrics visualization to: {output_path}")
        
        return plt.gcf()

    def visualize_comparative_metrics(
        self, 
        evaluation_results: Dict[str, Dict], 
        metrics_type: str = "document",
        metrics: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comparison chart for multiple models and selected metrics
        
        Args:
            evaluation_results: Dictionary mapping method names to evaluation results
            metrics_type: Type of metrics to compare ('document' or 'topic_word')
            metrics: List of specific metrics to include (None for all)
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Create data for plot
        methods = list(evaluation_results.keys())
        
        # Define default metrics based on type
        if metrics_type == "document":
            if metrics is None:
                metrics = ["adjusted_rand_index", "normalized_mutual_info", "omega_index", 
                        "bcubed_precision", "bcubed_recall", "bcubed_f1"]
            metric_source = "document_metrics"
        else:  # topic_word
            if metrics is None:
                metrics = ["precision", "recall", "f1_score"]
            metric_source = "topic_word_metrics.hungarian"  # Default to Hungarian
        
        # Define nice labels
        metric_labels = {
            "adjusted_rand_index": "ARI",
            "normalized_mutual_info": "NMI",
            "omega_index": "Omega",
            "bcubed_precision": "B³ Precision",
            "bcubed_recall": "B³ Recall",
            "bcubed_f1": "B³ F1",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1 Score"
        }
        
        # Create DataFrame for plotting
        data = []
        for method in methods:
            result = evaluation_results[method]
            
            # Handle nested access for topic_word metrics
            if "." in metric_source:
                parts = metric_source.split(".")
                source = result
                for part in parts:
                    source = source.get(part, {})
            else:
                source = result.get(metric_source, {})
            
            # Extract metric values
            for metric in metrics:
                value = source.get(metric, 0.0)
                data.append({
                    "Method": method,
                    "Metric": metric_labels.get(metric, metric),
                    "Value": value
                })
        
        df = pd.DataFrame(data)
        
        # Setup plot
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        ax = sns.barplot(x="Metric", y="Value", hue="Method", data=df)
        
        # Add value labels on top of each bar
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        # Add labels and legend
        plt.xlabel('Metric', fontweight='bold')
        plt.ylabel('Score', fontweight='bold')
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(f"{metrics_type.capitalize()} Metrics Comparison")
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved comparative metrics visualization to: {output_path}")
        
        return plt.gcf()

    def visualize_coverage_ratio(
        self,
        evaluation_result: Dict,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a visualization showing coverage ratio of sample vs. reference
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract data
        common_count = evaluation_result["common_docs_count"]
        reference_count = evaluation_result["reference_docs_count"]
        sample_count = evaluation_result["sample_docs_count"]
        
        # Calculate percentages
        coverage_pct = (common_count / reference_count) * 100
        sample_uniqueness = ((sample_count - common_count) / sample_count) * 100 if sample_count > 0 else 0
        
        # Setup plot
        plt.figure(figsize=(10, 6))
        
        # Create stacked bar for reference dataset
        plt.bar(['Reference Dataset'], [reference_count], label='Reference Documents',
                color='#8c96c6', edgecolor='white')
        
        # Create stacked bar for sample dataset
        plt.bar(['Sample Dataset'], [common_count], label='Common Documents',
                color='#4daf4a', edgecolor='white')
        plt.bar(['Sample Dataset'], [sample_count - common_count], bottom=[common_count],
                label='Sample-Only Documents', color='#fc8d59', edgecolor='white')
        
        # Add labels
        for i, v in enumerate([reference_count]):
            plt.text(0, v + 100, f"{v:,}", ha='center')
            
        plt.text(1, common_count/2, f"{common_count:,}\n({coverage_pct:.1f}%)", ha='center', va='center')
        if sample_count - common_count > 0:
            plt.text(1, common_count + (sample_count - common_count)/2, 
                    f"{sample_count - common_count:,}\n({sample_uniqueness:.1f}%)", 
                    ha='center', va='center')
        
        # Set title and labels
        if title:
            plt.title(title)
        else:
            plt.title(f"Document Coverage: {evaluation_result['sample_method']}")
        
        plt.ylabel('Number of Documents')
        plt.xticks(rotation=0)
        
        # Add legend
        plt.legend(loc='upper right')
        
        # Use log scale if reference is much larger
        if reference_count > sample_count * 10:
            plt.yscale('log')
            plt.ylabel('Number of Documents (log scale)')
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved coverage ratio visualization to: {output_path}")
        
        return plt.gcf()
    
    def calculate_topic_semantic_similarity(
        self,
        reference_topic_words: Dict[int, List[Tuple[str, float]]],
        sample_topic_words: Dict[int, List[Tuple[str, float]]],
        top_n: int = 10
    ) -> Dict[int, Dict[int, float]]:
        """
        Calculate semantic similarity between topic words using embeddings
        
        Args:
            reference_topic_words: Reference topic words dictionary
            sample_topic_words: Sample topic words dictionary
            top_n: Number of top words to consider
            
        Returns:
            Dictionary mapping reference topics to dictionaries mapping sample topics to similarity scores
        """
        # Skip outlier topic (-1)
        ref_topics = {k: v for k, v in reference_topic_words.items() if k != -1}
        sample_topics = {k: v for k, v in sample_topic_words.items() if k != -1}
        
        # Check if we have any valid topics to compare
        if not ref_topics or not sample_topics:
            logger.warning("No valid topics found for semantic similarity calculation")
            # Return a default structure with zero similarity
            if ref_topics:
                return {t: {st: 0.0 for st in (list(sample_topics.keys()) or [0])} 
                    for t in ref_topics.keys()}
            elif sample_topics:
                return {0: {t: 0.0 for t in sample_topics.keys()}}
            else:
                return {0: {0: 0.0}}  # Fallback for no topics
        
        # Extract top-n words for each topic
        ref_top_words = {
            topic: [word for word, _ in words[:top_n] if word] 
            for topic, words in ref_topics.items()
        }
        
        sample_top_words = {
            topic: [word for word, _ in words[:top_n] if word] 
            for topic, words in sample_topics.items()
        }
        
        # Calculate similarity matrix
        similarity_matrix = {}
        
        try:
            # Get or create sentence transformer model
            if not hasattr(self, 'sbert_model'):
                from sentence_transformers import SentenceTransformer
                logger.info("Initializing SentenceTransformer model for semantic similarity")
                self.sbert_model = SentenceTransformer('all-mpnet-base-v2')
            
            # Calculate embeddings for all topic word lists
            ref_embeddings = {}
            for topic, words in ref_top_words.items():
                if not words:  # Handle empty word list
                    ref_embeddings[topic] = None
                    continue
                    
                # Join words into a space-separated string
                topic_text = " ".join(words)
                try:
                    ref_embeddings[topic] = self.sbert_model.encode(topic_text, convert_to_tensor=True)
                except Exception as e:
                    logger.error(f"Error encoding reference topic {topic}: {str(e)}")
                    ref_embeddings[topic] = None
            
            sample_embeddings = {}
            for topic, words in sample_top_words.items():
                if not words:  # Handle empty word list
                    sample_embeddings[topic] = None
                    continue
                    
                topic_text = " ".join(words)
                try:
                    sample_embeddings[topic] = self.sbert_model.encode(topic_text, convert_to_tensor=True)
                except Exception as e:
                    logger.error(f"Error encoding sample topic {topic}: {str(e)}")
                    sample_embeddings[topic] = None
            
            # Calculate cosine similarity between all pairs
            from sentence_transformers import util
            
            for ref_topic, ref_embedding in ref_embeddings.items():
                similarity_matrix[ref_topic] = {}
                
                for sample_topic, sample_embedding in sample_embeddings.items():
                    # Skip if either embedding is None
                    if ref_embedding is None or sample_embedding is None:
                        similarity_matrix[ref_topic][sample_topic] = 0.0
                        continue
                        
                    # Calculate cosine similarity
                    try:
                        cos_sim = util.cos_sim(ref_embedding, sample_embedding).item()
                        similarity_matrix[ref_topic][sample_topic] = float(cos_sim)
                    except Exception as e:
                        logger.error(f"Error calculating similarity between topics {ref_topic} and {sample_topic}: {str(e)}")
                        similarity_matrix[ref_topic][sample_topic] = 0.0
        
        except Exception as e:
            logger.error(f"Error in semantic similarity calculation: {str(e)}")
            # Provide a fallback similarity matrix with zeros
            similarity_matrix = {t: {st: 0.0 for st in sample_topics.keys()} 
                            for t in ref_topics.keys()}
        
        return similarity_matrix

    def visualize_topic_overlap_matrix(
        self,
        evaluation_result: Dict,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize topic similarity matrix as a heatmap
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract overlap matrix
        overlap_matrix = evaluation_result.get("topic_word_overlap", {})
        
        # Check if we have data to visualize
        if not overlap_matrix:
            logger.warning("No topic overlap data to visualize")
            # Create an empty figure
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No topic overlap data available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            
            # Save if requested
            if output_path:
                plt.savefig(output_path, dpi=300)
                logger.info(f"Saved empty topic overlap matrix to: {output_path}")
            
            return plt.gcf()
        
        # Convert to numpy array - safely
        ref_topics = sorted(overlap_matrix.keys())
        
        # Check if we have reference topics
        if not ref_topics:
            logger.warning("No reference topics found in overlap matrix")
            # Create an empty figure
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No reference topics available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            
            # Save if requested
            if output_path:
                plt.savefig(output_path, dpi=300)
            
            return plt.gcf()
        
        # Get sample topics from the first reference topic's scores
        # (all reference topics should have the same sample topics)
        first_ref = ref_topics[0]
        
        # Handle case where first_ref doesn't exist in overlap_matrix
        if first_ref not in overlap_matrix:
            logger.warning(f"Reference topic {first_ref} not found in overlap matrix")
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Reference topic {first_ref} not found", 
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300)
            
            return plt.gcf()
        
        sample_topics = sorted(overlap_matrix[first_ref].keys()) if overlap_matrix[first_ref] else []
        
        if not sample_topics:
            logger.warning("No sample topics found in overlap matrix")
            # Create an empty figure showing reference topics but no samples
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No sample topics available", 
                    horizontalalignment='center', verticalalignment='center')
            plt.tight_layout()
            
            # Save if requested
            if output_path:
                plt.savefig(output_path, dpi=300)
            
            return plt.gcf()
        
        # Create matrix array - with safety check
        if len(ref_topics) == 0 or len(sample_topics) == 0:
            # Handle the case of empty topics - just create a 1x1 matrix with zero
            matrix_array = np.zeros((1, 1))
            ref_topics = [0]
            sample_topics = [0]
        else:
            matrix_array = np.zeros((len(ref_topics), len(sample_topics)))
            
            for i, ref_topic in enumerate(ref_topics):
                for j, sample_topic in enumerate(sample_topics):
                    if ref_topic in overlap_matrix and sample_topic in overlap_matrix[ref_topic]:
                        matrix_array[i, j] = overlap_matrix[ref_topic][sample_topic]
        
        # Create figure
        plt.figure(figsize=(max(8, len(sample_topics) * 0.7), max(6, len(ref_topics) * 0.5)))
        
        try:
            # Plot heatmap - with safe option for zero-size arrays
            if matrix_array.size > 0:
                ax = sns.heatmap(
                    matrix_array,
                    annot=True,
                    fmt=".2f",
                    cmap="YlGnBu",
                    xticklabels=[f"T{t}" for t in sample_topics],
                    yticklabels=[f"T{t}" for t in ref_topics],
                    vmin=0.0,  # Explicitly set min value
                    vmax=1.0   # Explicitly set max value
                )
            else:
                plt.text(0.5, 0.5, "No data to visualize", 
                        horizontalalignment='center', verticalalignment='center')
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            plt.text(0.5, 0.5, f"Error creating heatmap: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
        
        # Add labels
        plt.xlabel(f"Topics in {evaluation_result['sample_method']}")
        plt.ylabel(f"Topics in {evaluation_result['reference_method']}")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Topic Similarity: {evaluation_result['reference_method']} vs {evaluation_result['sample_method']}")
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved topic similarity matrix to: {output_path}")
        
        return plt.gcf()
    
    def visualize_document_metrics(
        self,
        evaluation_results: Dict[str, Dict],
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize document-level metrics for multiple evaluations
        
        Args:
            evaluation_results: Dictionary mapping names to evaluation results
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract metrics
        names = list(evaluation_results.keys())
        metrics = list(evaluation_results[names[0]]["document_metrics"].keys())
        
        # Create data
        data = []
        for name in names:
            for metric in metrics:
                value = evaluation_results[name]["document_metrics"][metric]
                data.append({
                    "Method": name,
                    "Metric": metric,
                    "Value": value
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Plot grouped bar chart
        ax = sns.barplot(x="Metric", y="Value", hue="Method", data=df)
        
        # Add labels
        plt.xlabel("Metric")
        plt.ylabel("Score")
        
        if title:
            plt.title(title)
        else:
            plt.title("Document-Level Topic Evaluation Metrics")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Add legend
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved document metrics visualization to: {output_path}")
        
        return plt.gcf()
    
    def visualize_topic_word_metrics(
        self,
        evaluation_results: Dict[str, Dict],
        matching_method: str = "hungarian",
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize topic word metrics for multiple evaluations
        
        Args:
            evaluation_results: Dictionary mapping names to evaluation results
            matching_method: Method for matching ('greedy' or 'hungarian')
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract metrics
        names = list(evaluation_results.keys())
        metrics = ["precision", "recall", "f1_score"]
        
        # Create data
        data = []
        for name in names:
            for metric in metrics:
                value = evaluation_results[name]["topic_word_metrics"][matching_method][metric]
                data.append({
                    "Method": name,
                    "Metric": metric,
                    "Value": value
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot grouped bar chart
        ax = sns.barplot(x="Metric", y="Value", hue="Method", data=df)
        
        # Add labels
        plt.xlabel("Metric")
        plt.ylabel("Score")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Topic Word Metrics ({matching_method.title()} Matching)")
        
        # Add legend
        plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved topic word metrics visualization to: {output_path}")
        
        return plt.gcf()
    
    def calculate_topic_distribution_similarity(
        self,
        reference_result: Dict,
        sample_result: Dict
    ) -> float:
        """
        Calculate Jensen-Shannon divergence between topic distributions
        
        Args:
            reference_result: Reference topic model result
            sample_result: Sample topic model result
            
        Returns:
            Similarity score (1 - JS divergence)
        """
        from scipy.spatial.distance import jensenshannon
        from scipy.stats import entropy
        import numpy as np
        
        # Get topic distributions (frequency of documents per topic)
        ref_topics = reference_result["topics"]
        sample_topics = sample_result["topics"]
        
        # Count documents per topic
        ref_counts = {}
        for topic in ref_topics:
            if topic != -1:  # Skip noise/outlier topic
                ref_counts[topic] = ref_counts.get(topic, 0) + 1
        
        sample_counts = {}
        for topic in sample_topics:
            if topic != -1:  # Skip noise/outlier topic
                sample_counts[topic] = sample_counts.get(topic, 0) + 1
        
        # Get total counts (excluding noise)
        ref_total = sum(ref_counts.values())
        sample_total = sum(sample_counts.values())
        
        # Check if we have valid data
        if ref_total == 0 or sample_total == 0:
            return 0.0
        
        # Get the set of all topic IDs
        all_topics = sorted(set(list(ref_counts.keys()) + list(sample_counts.keys())))
        
        # Create probability distributions
        ref_dist = np.array([ref_counts.get(t, 0) / ref_total for t in all_topics])
        sample_dist = np.array([sample_counts.get(t, 0) / sample_total for t in all_topics])
        
        # Calculate Jensen-Shannon divergence
        js_distance = jensenshannon(ref_dist, sample_dist)
        
        # Convert to similarity (1 - distance)
        return 1.0 - js_distance

    def visualize_topic_distribution_comparison(
        self,
        reference_result: Dict,
        sample_result: Dict,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a visualization comparing topic distributions
        
        Args:
            reference_result: Reference topic model result
            sample_result: Sample topic model result
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Get topic distributions
        ref_topics = reference_result.get("topics", [])
        sample_topics = sample_result.get("topics", [])
        
        # Get method names safely
        reference_method = reference_result.get("sample_method", 
                        reference_result.get("reference_method", "Reference"))
        sample_method = sample_result.get("sample_method", "Sample")
        
        # Count documents per topic, excluding noise
        ref_counts = {}
        for topic in ref_topics:
            if topic != -1:
                ref_counts[topic] = ref_counts.get(topic, 0) + 1
        
        sample_counts = {}
        for topic in sample_topics:
            if topic != -1:
                sample_counts[topic] = sample_counts.get(topic, 0) + 1
        
        # Get total counts
        ref_total = sum(ref_counts.values()) if ref_counts else 1
        sample_total = sum(sample_counts.values()) if sample_counts else 1
        
        # Get topic percentages
        ref_pcts = {t: (c / ref_total * 100) for t, c in ref_counts.items()}
        sample_pcts = {t: (c / sample_total * 100) for t, c in sample_counts.items()}
        
        # Get a list of all topics
        all_topics = sorted(set(list(ref_counts.keys()) + list(sample_counts.keys())))
        
        # Create data for plotting
        x = list(range(len(all_topics)))
        ref_values = [ref_pcts.get(t, 0) for t in all_topics]
        sample_values = [sample_pcts.get(t, 0) for t in all_topics]
        
        # Setup plot
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        width = 0.35
        ax = plt.gca()
        ref_bars = ax.bar([i - width/2 for i in x], ref_values, width, label=reference_method)
        sample_bars = ax.bar([i + width/2 for i in x], sample_values, width, label=sample_method)
        
        # Add labels and title
        plt.xlabel('Topic ID')
        plt.ylabel('Percentage of Documents (%)')
        
        if title:
            plt.title(title)
        else:
            # Calculate similarity using Jensen-Shannon divergence if possible
            try:
                from scipy.spatial.distance import jensenshannon
                ref_dist = np.array(ref_values) / 100
                sample_dist = np.array(sample_values) / 100
                
                # Check for valid distributions
                if np.sum(ref_dist) > 0 and np.sum(sample_dist) > 0:
                    # Normalize distributions
                    ref_dist = ref_dist / np.sum(ref_dist)
                    sample_dist = sample_dist / np.sum(sample_dist)
                    
                    # Calculate JS divergence
                    js_distance = jensenshannon(ref_dist, sample_dist)
                    similarity = 1.0 - js_distance
                    
                    plt.title(f'Topic Distribution Comparison\nSimilarity: {similarity:.3f}')
                else:
                    plt.title('Topic Distribution Comparison')
            except Exception:
                plt.title('Topic Distribution Comparison')
        
        # Set x-ticks
        plt.xticks(x, [f'T{t}' for t in all_topics])
        if len(all_topics) > 15:
            plt.xticks(rotation=90)
        
        # Add legend
        plt.legend()
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved topic distribution comparison to: {output_path}")
        
        return plt.gcf()

    def visualize_sampling_efficiency(
        self,
        evaluation_results: Dict[str, Dict],
        metric: str = "bcubed_f1",
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a visualization showing sampling efficiency (metric vs sample size)
        
        Args:
            evaluation_results: Dictionary mapping method names to evaluation results
            metric: Metric to use for efficiency calculation
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract data
        methods = []
        sample_sizes = []
        metric_values = []
        ref_sizes = []
        
        for name, result in evaluation_results.items():
            # Skip methods with zero metric value
            if result["document_metrics"].get(metric, 0) == 0:
                continue
                
            methods.append(name)
            sample_sizes.append(result["sample_docs_count"])
            ref_sizes.append(result["reference_docs_count"])
            
            # Get metric value based on specified metric
            if "." in metric:
                # Handle nested metrics like topic_word_metrics.hungarian.f1_score
                parts = metric.split(".")
                value = result
                for part in parts:
                    value = value.get(part, {})
                metric_values.append(value if isinstance(value, (int, float)) else 0)
            else:
                # Handle top-level metrics like bcubed_f1
                metric_values.append(result["document_metrics"].get(metric, 0))
        
        # Calculate sampling ratio and efficiency
        sampling_ratios = [s / r for s, r in zip(sample_sizes, ref_sizes)]
        sampling_efficiency = [v / r for v, r in zip(metric_values, sampling_ratios)]
        
        # Setup plot
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(sampling_ratios, metric_values, s=[e * 100 for e in sampling_efficiency], 
                    alpha=0.7, c=range(len(methods)), cmap='viridis')
        
        # Add method labels
        for i, method in enumerate(methods):
            plt.annotate(method, (sampling_ratios[i], metric_values[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        # Add labels and title
        plt.xlabel('Sampling Ratio (Sample Size / Reference Size)')
        
        # Get nice metric name
        metric_labels = {
            "adjusted_rand_index": "Adjusted Rand Index",
            "normalized_mutual_info": "Normalized Mutual Information",
            "omega_index": "Omega Index",
            "bcubed_precision": "B-Cubed Precision",
            "bcubed_recall": "B-Cubed Recall", 
            "bcubed_f1": "B-Cubed F1 Score",
            "topic_word_metrics.hungarian.f1_score": "Topic Word F1 Score (Hungarian)",
            "topic_word_metrics.greedy.f1_score": "Topic Word F1 Score (Greedy)"
        }
        metric_name = metric_labels.get(metric, metric)
        plt.ylabel(metric_name)
        
        if title:
            plt.title(title)
        else:
            plt.title(f'Sampling Efficiency: {metric_name} vs Sampling Ratio')
        
        # Add size legend
        sizes = [0.001, 0.01, 0.1]
        labels = ["0.1%", "1%", "10%"]
        
        for size, label in zip(sizes, labels):
            plt.scatter([], [], s=size * 100, alpha=0.7, c='gray', label=f'Efficiency: {label}')
        
        plt.legend(title="Sampling Efficiency\n(Score/Ratio)")
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Use log scale for x-axis if large differences
        if max(sampling_ratios) / min(sampling_ratios) > 100:
            plt.xscale('log')
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved sampling efficiency visualization to: {output_path}")
        
        return plt.gcf()

    def visualize_information_density(
        self,
        evaluation_results: Dict[str, Dict],
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a visualization showing information density in different samples
        
        Args:
            evaluation_results: Dictionary mapping method names to evaluation results
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract data
        methods = []
        sample_sizes = []
        bcubed_f1 = []
        topic_f1 = []
        
        for name, result in evaluation_results.items():
            # Skip methods with zero metric values
            doc_f1 = result["document_metrics"].get("bcubed_f1", 0)
            word_f1 = result["topic_word_metrics"]["hungarian"].get("f1_score", 0)
            
            if doc_f1 == 0 and word_f1 == 0:
                continue
                
            methods.append(name)
            sample_sizes.append(result["sample_docs_count"])
            bcubed_f1.append(doc_f1)
            topic_f1.append(word_f1)
        
        # Calculate density
        density = [(b + t) / 2 for b, t in zip(bcubed_f1, topic_f1)]
        
        # Setup plot - vertical bar chart
        plt.figure(figsize=(12, 8))
        
        # Create positions for bars
        x = range(len(methods))
        
        # Create stacked bars
        plt.bar(x, bcubed_f1, label='Document Structure (B³ F1)', color='#66b3ff')
        plt.bar(x, topic_f1, bottom=bcubed_f1, label='Topic Content (Word F1)', color='#99ff99')
        
        # Add labels and title
        plt.xlabel('Sampling Method')
        plt.ylabel('Metric Score (Higher is Better)')
        plt.xticks(x, methods, rotation=45, ha='right')
        
        # Add sample size annotations
        for i, (size, d) in enumerate(zip(sample_sizes, density)):
            plt.text(i, bcubed_f1[i] + topic_f1[i] + 0.03, f'Size: {size:,}', 
                    ha='center', va='bottom', fontsize=9)
            plt.text(i, bcubed_f1[i] + topic_f1[i]/2, f'Density: {d:.3f}', 
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        
        if title:
            plt.title(title)
        else:
            plt.title('Information Density Comparison')
        
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved information density visualization to: {output_path}")
        
        return plt.gcf()

    def export_evaluation_results(
        self,
        evaluation_results: Dict[str, Dict],
        reference_method: str,
        output_dir: str = "results/evaluation",
        create_visualizations: bool = True
    ) -> Dict[str, str]:
        """
        Export evaluation results to files
        
        Args:
            evaluation_results: Dictionary mapping sample methods to evaluation results
            reference_method: Name of the reference method
            output_dir: Directory to save results
            create_visualizations: Whether to create visualization plots
            
        Returns:
            Dictionary mapping sample methods to output directories
        """
        logger.info(f"Exporting evaluation results to {output_dir}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a safe version of the reference name
        safe_ref = reference_method.replace(" ", "_").lower()
        
        # Create a directory for this reference
        ref_dir = os.path.join(output_dir, safe_ref)
        os.makedirs(ref_dir, exist_ok=True)
        
        output_paths = {}
        
        # Create summary file with all metrics
        summary_path = os.path.join(ref_dir, "evaluation_summary.csv")
        
        # Prepare summary data
        summary_data = []
        for name, result in evaluation_results.items():
            # Extract metrics
            doc_metrics = result["document_metrics"]
            word_metrics_greedy = result["topic_word_metrics"]["greedy"]
            word_metrics_hungarian = result["topic_word_metrics"]["hungarian"]
            
            # Create row
            row = {
                "Method": name,
                "Common Documents": result["common_docs_count"],
                "Reference Documents": result["reference_docs_count"],
                "Sample Documents": result["sample_docs_count"],
                "ARI": doc_metrics["adjusted_rand_index"],
                "NMI": doc_metrics["normalized_mutual_info"],
                "Omega": doc_metrics["omega_index"],
                "B-Cubed Precision": doc_metrics["bcubed_precision"],
                "B-Cubed Recall": doc_metrics["bcubed_recall"],
                "B-Cubed F1": doc_metrics["bcubed_f1"],
                "Word Precision (Greedy)": word_metrics_greedy["precision"],
                "Word Recall (Greedy)": word_metrics_greedy["recall"],
                "Word F1 (Greedy)": word_metrics_greedy["f1_score"],
                "Word Precision (Hungarian)": word_metrics_hungarian["precision"],
                "Word Recall (Hungarian)": word_metrics_hungarian["recall"],
                "Word F1 (Hungarian)": word_metrics_hungarian["f1_score"]
            }
            
            summary_data.append(row)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
        
        # Find reference result data for topic distribution comparison
        ref_result = None
        for method_name, r in evaluation_results.items():
            if method_name == reference_method:
                ref_result = r
                break
        
        # Process each sample
        for name, result in evaluation_results.items():
            # Create a safe version of the sample name
            safe_name = name.replace(" ", "_").lower()
            
            # Create directory for this sample
            sample_dir = os.path.join(ref_dir, safe_name)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save detailed evaluation results
            detail_path = os.path.join(sample_dir, "evaluation_details.json")
            
            # Safely compute average and max overlaps
            overlap_summary = {"average_overlap": 0.0, "max_overlap": 0.0}
            
            if result.get("topic_word_overlap"):
                # Collect all individual overlap values
                all_overlaps = []
                for ref_topic, sample_scores in result["topic_word_overlap"].items():
                    if sample_scores:  # Check if not empty
                        all_overlaps.extend(list(sample_scores.values()))
                
                # Only compute statistics if we have overlaps
                if all_overlaps:
                    overlap_summary["average_overlap"] = float(np.mean(all_overlaps))
                    overlap_summary["max_overlap"] = float(np.max(all_overlaps))
            
            # Create serializable version of the result
            serializable_result = {
                "reference_method": result["reference_method"],
                "sample_method": result["sample_method"],
                "document_metrics": result["document_metrics"],
                "topic_word_metrics": {
                    "greedy": result["topic_word_metrics"]["greedy"],
                    "hungarian": {k: v for k, v in result["topic_word_metrics"]["hungarian"].items() if k != "matched_pairs"}
                },
                "topic_word_overlap_summary": overlap_summary,
                "common_docs_count": result["common_docs_count"],
                "reference_docs_count": result["reference_docs_count"],
                "sample_docs_count": result["sample_docs_count"]
            }
            
            # Save to JSON
            with open(detail_path, "w") as f:
                import json
                json.dump(serializable_result, f, indent=2)
            
            # Save matched pairs
            matched_pairs = result["topic_word_metrics"]["hungarian"]["matched_pairs"]
            pairs_path = os.path.join(sample_dir, "matched_topics.csv")
            
            if matched_pairs:
                pairs_df = pd.DataFrame(matched_pairs, columns=["Reference Topic", "Sample Topic", "Overlap"])
                pairs_df.to_csv(pairs_path, index=False)
            
            # Create visualizations if requested
            if create_visualizations:
                # Try each visualization separately to avoid one failure affecting others
                try:
                    # Topic overlap matrix
                    matrix_path = os.path.join(sample_dir, "topic_similarity_matrix.png")
                    self.visualize_topic_overlap_matrix(
                        result,
                        output_path=matrix_path,
                        title=f"Topic Similarity: {result['reference_method']} vs {result['sample_method']}"
                    )
                    plt.close()
                except Exception as e:
                    logger.error(f"Error generating topic similarity matrix for {name}: {str(e)}")
                
                try:
                    # Document metrics radar chart
                    radar_path = os.path.join(sample_dir, "document_metrics_radar.png")
                    self.visualize_document_metrics_radar(
                        result,
                        output_path=radar_path,
                        title=f"Document Metrics: {result['sample_method']}"
                    )
                    plt.close()
                except Exception as e:
                    logger.error(f"Error generating radar chart for {name}: {str(e)}")
                
                try:
                    # Topic word metrics comparison
                    topic_compare_path = os.path.join(sample_dir, "topic_word_metrics_comparison.png")
                    self.visualize_topic_word_metrics_comparison(
                        result,
                        output_path=topic_compare_path,
                        title=f"Topic Word Metrics: {result['sample_method']}"
                    )
                    plt.close()
                except Exception as e:
                    logger.error(f"Error generating topic metrics comparison for {name}: {str(e)}")
                
                try:
                    # B-Cubed metrics visualization
                    bcubed_path = os.path.join(sample_dir, "bcubed_metrics.png")
                    self.visualize_bcubed_metrics(
                        result,
                        output_path=bcubed_path,
                        title=f"B-Cubed Metrics: {result['sample_method']}"
                    )
                    plt.close()
                except Exception as e:
                    logger.error(f"Error generating B-Cubed metrics for {name}: {str(e)}")
                
                try:
                    # Coverage ratio visualization
                    coverage_path = os.path.join(sample_dir, "coverage_ratio.png")
                    self.visualize_coverage_ratio(
                        result,
                        output_path=coverage_path,
                        title=f"Document Coverage: {result['sample_method']}"
                    )
                    plt.close()
                except Exception as e:
                    logger.error(f"Error generating coverage ratio for {name}: {str(e)}")
                    
                try:
                    # Topic distribution comparison
                    topic_dist_path = os.path.join(sample_dir, "topic_distribution.png")
                    
                    if ref_result and "topics" in result and "topics" in ref_result:
                        self.visualize_topic_distribution_comparison(
                            ref_result,
                            result,
                            output_path=topic_dist_path,
                            title=f"Topic Distribution: {ref_result.get('sample_method', reference_method)} vs {result.get('sample_method', name)}"
                        )
                        plt.close()
                    else:
                        logger.warning(f"Skipping topic distribution comparison for {name} (missing topic data)")
                except Exception as e:
                    logger.error(f"Error generating topic distribution comparison for {name}: {str(e)}")
            
            output_paths[name] = sample_dir
        
        # Create comparative visualizations if multiple samples
        if len(evaluation_results) > 1 and create_visualizations:
            try:
                # Document metrics
                doc_metrics_path = os.path.join(ref_dir, "document_metrics.png")
                self.visualize_document_metrics(
                    evaluation_results,
                    output_path=doc_metrics_path,
                    title=f"Document-Level Topic Evaluation Metrics (Reference: {reference_method})"
                )
                plt.close()
            except Exception as e:
                logger.error(f"Error generating document metrics visualization: {str(e)}")
                
            try:
                # Comparative document metrics (NEW)
                comparative_doc_path = os.path.join(ref_dir, "comparative_document_metrics.png")
                self.visualize_comparative_metrics(
                    evaluation_results,
                    metrics_type="document",
                    metrics=["adjusted_rand_index", "normalized_mutual_info", "omega_index"],
                    output_path=comparative_doc_path,
                    title=f"Document Clustering Metrics Comparison (Reference: {reference_method})"
                )
                plt.close()
            except Exception as e:
                logger.error(f"Error generating comparative document metrics: {str(e)}")
                
            try:
                # Comparative B-Cubed metrics (NEW)
                comparative_bcubed_path = os.path.join(ref_dir, "comparative_bcubed_metrics.png")
                self.visualize_comparative_metrics(
                    evaluation_results,
                    metrics_type="document",
                    metrics=["bcubed_precision", "bcubed_recall", "bcubed_f1"],
                    output_path=comparative_bcubed_path,
                    title=f"B-Cubed Metrics Comparison (Reference: {reference_method})"
                )
                plt.close()
            except Exception as e:
                logger.error(f"Error generating comparative B-Cubed metrics: {str(e)}")
                
            try:
                # Topic word metrics (Greedy)
                word_metrics_greedy_path = os.path.join(ref_dir, "topic_word_metrics_greedy.png")
                self.visualize_topic_word_metrics(
                    evaluation_results,
                    matching_method="greedy",
                    output_path=word_metrics_greedy_path,
                    title=f"Topic Word Metrics with Greedy Matching (Reference: {reference_method})"
                )
                plt.close()
            except Exception as e:
                logger.error(f"Error generating greedy metrics visualization: {str(e)}")
                
            try:
                # Topic word metrics (Hungarian)
                word_metrics_hungarian_path = os.path.join(ref_dir, "topic_word_metrics_hungarian.png")
                self.visualize_topic_word_metrics(
                    evaluation_results,
                    matching_method="hungarian",
                    output_path=word_metrics_hungarian_path,
                    title=f"Topic Word Metrics with Hungarian Matching (Reference: {reference_method})"
                )
                plt.close()
            except Exception as e:
                logger.error(f"Error generating Hungarian metrics visualization: {str(e)}")
                
            try:
                # Sampling efficiency visualization (NEW)
                sampling_efficiency_path = os.path.join(ref_dir, "sampling_efficiency.png")
                self.visualize_sampling_efficiency(
                    evaluation_results,
                    metric="bcubed_f1",  # Use B-Cubed F1 as default metric
                    output_path=sampling_efficiency_path,
                    title=f"Topic Modeling Efficiency with Different Sampling Methods"
                )
                plt.close()
            except Exception as e:
                logger.error(f"Error generating sampling efficiency visualization: {str(e)}")
                
            try:
                # Information density visualization (NEW)
                density_path = os.path.join(ref_dir, "information_density.png")
                self.visualize_information_density(
                    evaluation_results,
                    output_path=density_path,
                    title=f"Information Density in Different Sampling Methods"
                )
                plt.close()
            except Exception as e:
                logger.error(f"Error generating information density visualization: {str(e)}")
        
        return output_paths

def main():
    """Main function to evaluate topic models using FULL_DATASET as the gold standard"""
    import os
    import pickle
    from topic_modelling import TopicModelingMethod
    
    # Define constants
    CACHE_DIR = "cache"
    TOPIC_MODELING_CACHE_DIR = CACHE_DIR
    OUTPUT_DIR = "results/evaluation"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load topic modeling results from cache - use the specific path from topic_modeling.py
    topic_modeling_results_path = os.path.join(TOPIC_MODELING_CACHE_DIR, "all_topic_model_results.pkl")
    
    logger.info(f"Loading topic modeling results from: {topic_modeling_results_path}")
    
    if os.path.exists(topic_modeling_results_path):
        with open(topic_modeling_results_path, "rb") as f:
            topic_models_by_query = pickle.load(f)
    else:
        logger.warning(f"Topic modeling results not found at {topic_modeling_results_path}")
        logger.info("Searching for individual topic model caches...")
        
        # Fall back to individual model caches if needed
        topic_model_files = []
        for root, dirs, files in os.walk(TOPIC_MODELING_CACHE_DIR):
            for file in files:
                if file.startswith("topic_model_") and file.endswith(".pkl"):
                    topic_model_files.append(os.path.join(root, file))
        
        if topic_model_files:
            logger.info(f"Found {len(topic_model_files)} individual topic model files")
            
            # Load individual files
            topic_models_by_query = {"query_level": {}}
            for file_path in topic_model_files:
                with open(file_path, "rb") as f:
                    model_result = pickle.load(f)
                
                # Extract method and query info
                method = os.path.basename(file_path).replace("topic_model_", "").replace(".pkl", "")
                query_id = model_result.get("query_id", "unknown")
                
                if query_id not in topic_models_by_query["query_level"]:
                    topic_models_by_query["query_level"][query_id] = {}
                
                topic_models_by_query["query_level"][query_id][method] = model_result
        else:
            logger.error("No topic modeling results found. Please run topic_modeling.py first.")
            return None
    
    # Initialize evaluator
    evaluator = TopicEvaluator(cache_dir=os.path.join(CACHE_DIR, "topic_evaluation"))
    
    # Process each query
    all_evaluation_results = {}
    
    for query_id, methods in topic_models_by_query.items():
        if query_id == "corpus_level":
            # Skip corpus level for now - we'll process it separately
            continue
        
        logger.info(f"Evaluating topic models for query: {query_id}")
        
        # Filter to topic model results
        topic_results = {}
        for method_name, result in methods.items():
            # Extract the topic model result
            if "topic_model_result" in result:
                topic_results[method_name] = result["topic_model_result"]
            else:
                topic_results[method_name] = result
        
        if not topic_results:
            logger.warning(f"No topic model results found for query {query_id}")
            continue
        
        # Always use FULL_DATASET as the reference (gold standard) if available
        if TopicModelingMethod.FULL_DATASET.value in topic_results:
            reference_method = TopicModelingMethod.FULL_DATASET.value
            reference_result = topic_results[reference_method]
            
            # Create samples dictionary with all but reference
            samples = {name: result for name, result in topic_results.items() 
                      if name != reference_method}
            
            if not samples:
                logger.warning(f"Only reference model found for query {query_id}, cannot evaluate")
                continue
            
            # Run evaluation
            evaluation_results = evaluator.evaluate_multiple_models(
                reference_result=reference_result,
                sample_results=samples,
                top_n_words=10,
                force_recompute=False
            )
            
            # Export results
            query_output_dir = os.path.join(OUTPUT_DIR, f"query_{query_id}")
            evaluator.export_evaluation_results(
                evaluation_results=evaluation_results,
                reference_method=reference_method,
                output_dir=query_output_dir,
                create_visualizations=True
            )
            
            # Store results
            all_evaluation_results[query_id] = {
                "reference_method": reference_method,
                "evaluation_results": evaluation_results
            }
        else:
            logger.warning(f"FULL_DATASET model not found for query {query_id}, skipping evaluation")
    
    # Process corpus level if available
    if "corpus_level" in topic_models_by_query:
        logger.info("Evaluating topic models at corpus level")
        
        corpus_methods = topic_models_by_query["corpus_level"]
        
        # Extract topic model results
        corpus_topic_results = {}
        for method_name, result in corpus_methods.items():
            if "topic_model_result" in result:
                corpus_topic_results[method_name] = result["topic_model_result"]
            else:
                corpus_topic_results[method_name] = result
        
        # Always use FULL_DATASET as reference for corpus-level evaluation
        if TopicModelingMethod.FULL_DATASET.value in corpus_topic_results:
            corpus_reference_method = TopicModelingMethod.FULL_DATASET.value
            corpus_reference_result = corpus_topic_results[corpus_reference_method]
            
            # Create samples dictionary with all but reference
            corpus_samples = {name: result for name, result in corpus_topic_results.items() 
                             if name != corpus_reference_method}
            
            if corpus_samples:
                # Run evaluation
                corpus_evaluation_results = evaluator.evaluate_multiple_models(
                    reference_result=corpus_reference_result,
                    sample_results=corpus_samples,
                    top_n_words=10,
                    force_recompute=False
                )
                
                # Export results
                corpus_output_dir = os.path.join(OUTPUT_DIR, "corpus_level")
                evaluator.export_evaluation_results(
                    evaluation_results=corpus_evaluation_results,
                    reference_method=corpus_reference_method,
                    output_dir=corpus_output_dir,
                    create_visualizations=True
                )
                
                # Store results
                all_evaluation_results["corpus_level"] = {
                    "reference_method": corpus_reference_method,
                    "evaluation_results": corpus_evaluation_results
                }
            else:
                logger.warning("Only FULL_DATASET model found at corpus level, cannot evaluate")
        else:
            logger.warning("FULL_DATASET model not found at corpus level, skipping evaluation")
    
    # Print summary of all results
    logger.info("\n===== TOPIC EVALUATION SUMMARY =====")
    for query_id, result in all_evaluation_results.items():
        reference_method = result["reference_method"]
        evaluation_results = result["evaluation_results"]
        
        logger.info(f"\nResults for query: {query_id} (Reference: {reference_method})")
        logger.info(f"{'Method':<30} {'ARI':<8} {'Omega':<8} {'Word F1':<8}")
        logger.info("-" * 55)
        
        for method, eval_result in evaluation_results.items():
            ari = eval_result["document_metrics"]["adjusted_rand_index"]
            omega = eval_result["document_metrics"]["omega_index"]
            word_f1 = eval_result["topic_word_metrics"]["hungarian"]["f1_score"]
            
            logger.info(f"{method:<30} {ari:.4f}  {omega:.4f}  {word_f1:.4f}")
    
    logger.info(f"\nDetailed evaluation results exported to {OUTPUT_DIR}")
    
    return all_evaluation_results


if __name__ == "__main__":
    main()
