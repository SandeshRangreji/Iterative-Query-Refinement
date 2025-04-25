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
    
    def _get_cache_path(self, reference_name: str, sample_name: str, prefix: str = "eval") -> str:
        """
        Generate a cache path based on reference and sample names
        
        Args:
            reference_name: Name of the reference sample
            sample_name: Name of the evaluated sample
            prefix: Optional prefix for cache file
            
        Returns:
            Cache file path
        """
        # Create safe names
        safe_ref = reference_name.replace(" ", "_").lower()
        safe_sample = sample_name.replace(" ", "_").lower()
        
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
        same_relationship = 0
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
        if n > 0:
            precision = precision_sum / n
            recall = recall_sum / n
        else:
            precision = 0.0
            recall = 0.0
        
        # Calculate F1-score
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
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
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        return precision, recall, f1_score
    
    def calculate_topic_word_precision_recall_hungarian(
        self,
        reference_topic_words: Dict[int, List[Tuple[str, float]]],
        sample_topic_words: Dict[int, List[Tuple[str, float]]],
        top_n: int = 10
    ) -> Tuple[float, float, float, List[Tuple[int, int, float]]]:
        """
        Calculate precision and recall for topic words using Hungarian algorithm
        
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
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0
        
        return precision, recall, f1_score, matched_pairs
    
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
        # Extract names
        reference_name = reference_result["sample_name"]
        sample_name = sample_result["sample_name"]
        
        # Generate cache path
        cache_path = self._get_cache_path(reference_name, sample_name)
        
        # Check cache
        if not force_recompute and os.path.exists(cache_path):
            logger.info(f"Loading evaluation results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Evaluating topic model: {sample_name} against {reference_name}...")
        
        # Extract topic assignments and documents
        reference_topics = reference_result["topics"]
        sample_topics = sample_result["topics"]
        
        # Use document texts as identifiers
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
        
        # Calculate topic word level metrics
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
        
        # Create overlap matrix for visualization
        overlap_matrix = self.calculate_topic_word_overlap(
            reference_result["topic_words"],
            sample_result["topic_words"],
            top_n=top_n_words
        )
        
        # Build result dictionary
        result = {
            "reference_name": reference_name,
            "sample_name": sample_name,
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
            "topic_word_overlap": overlap_matrix,
            "common_docs_count": len(set(reference_docs).intersection(set(sample_docs))),
            "reference_docs_count": len(reference_docs),
            "sample_docs_count": len(sample_docs)
        }
        
        # Cache results
        logger.info(f"Caching evaluation results to: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def evaluate_multiple_topic_models(
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
            
            # Run evaluation
            eval_result = self.evaluate_topic_model(
                reference_result=reference_result,
                sample_result=result,
                top_n_words=top_n_words,
                force_recompute=force_recompute
            )
            
            evaluation_results[name] = eval_result
        
        return evaluation_results
    
    def visualize_topic_overlap_matrix(
        self,
        evaluation_result: Dict,
        output_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize topic word overlap matrix as a heatmap
        
        Args:
            evaluation_result: Evaluation result dictionary
            output_path: Optional path to save visualization
            title: Optional title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract overlap matrix
        overlap_matrix = evaluation_result["topic_word_overlap"]
        
        # Convert to numpy array
        ref_topics = sorted(overlap_matrix.keys())
        sample_topics = sorted(set(k for d in overlap_matrix.values() for k in d.keys()))
        
        matrix_array = np.zeros((len(ref_topics), len(sample_topics)))
        
        for i, ref_topic in enumerate(ref_topics):
            for j, sample_topic in enumerate(sample_topics):
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
            yticklabels=[f"T{t}" for t in ref_topics],
            cbar_kws={"label": "Jaccard Similarity"}
        )
        
        # Add labels
        plt.xlabel(f"Topics in {evaluation_result['sample_name']}")
        plt.ylabel(f"Topics in {evaluation_result['reference_name']}")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Topic Word Overlap: {evaluation_result['reference_name']} vs {evaluation_result['sample_name']}")
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved topic overlap matrix to: {output_path}")
        
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
                    "Sample": name,
                    "Metric": metric,
                    "Value": value
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot grouped bar chart
        ax = sns.barplot(x="Metric", y="Value", hue="Sample", data=df)
        
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
        plt.legend(title="Sample", bbox_to_anchor=(1.05, 1), loc='upper left')
        
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
                    "Sample": name,
                    "Metric": metric,
                    "Value": value
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot grouped bar chart
        ax = sns.barplot(x="Metric", y="Value", hue="Sample", data=df)
        
        # Add labels
        plt.xlabel("Metric")
        plt.ylabel("Score")
        
        if title:
            plt.title(title)
        else:
            plt.title(f"Topic Word Metrics ({matching_method.title()} Matching)")
        
        # Add legend
        plt.legend(title="Sample", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved topic word metrics visualization to: {output_path}")
        
        return plt.gcf()


    def export_evaluation_results(
        self,
        evaluation_results: Dict[str, Dict],
        reference_name: str,
        output_dir: str = "results/evaluation",
        create_visualizations: bool = True
    ) -> Dict[str, str]:
        """
        Export evaluation results to files
        
        Args:
            evaluation_results: Dictionary mapping sample names to evaluation results
            reference_name: Name of the reference sample
            output_dir: Directory to save results
            create_visualizations: Whether to create visualization plots
            
        Returns:
            Dictionary mapping sample names to output directories
        """
        logger.info(f"Exporting evaluation results to {output_dir}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a safe version of the reference name
        safe_ref = reference_name.replace(" ", "_").lower()
        
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
                "Sample": name,
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
        
        # Process each sample
        for name, result in evaluation_results.items():
            # Create a safe version of the sample name
            safe_name = name.replace(" ", "_").lower()
            
            # Create directory for this sample
            sample_dir = os.path.join(ref_dir, safe_name)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save detailed evaluation results
            detail_path = os.path.join(sample_dir, "evaluation_details.json")
            
            # Create serializable version of the result
            serializable_result = {
                "reference_name": result["reference_name"],
                "sample_name": result["sample_name"],
                "document_metrics": result["document_metrics"],
                "topic_word_metrics": {
                    "greedy": result["topic_word_metrics"]["greedy"],
                    "hungarian": {k: v for k, v in result["topic_word_metrics"]["hungarian"].items() if k != "matched_pairs"}
                },
                # Cannot serialize nested dictionaries with integer keys to JSON
                "topic_word_overlap_summary": {
                    "average_overlap": np.mean([
                        np.mean(list(scores.values())) 
                        for scores in result["topic_word_overlap"].values()
                    ]),
                    "max_overlap": np.max([
                        np.max(list(scores.values())) 
                        for scores in result["topic_word_overlap"].values()
                    ]) if result["topic_word_overlap"] else 0
                },
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
                # Topic overlap matrix
                matrix_path = os.path.join(sample_dir, "topic_overlap_matrix.png")
                self.visualize_topic_overlap_matrix(
                    result,
                    output_path=matrix_path,
                    title=f"Topic Word Overlap: {result['reference_name']} vs {result['sample_name']}"
                )
                plt.close()
            
            output_paths[name] = sample_dir
        
        # Create comparative visualizations if multiple samples
        if len(evaluation_results) > 1 and create_visualizations:
            # Document metrics
            doc_metrics_path = os.path.join(ref_dir, "document_metrics.png")
            self.visualize_document_metrics(
                evaluation_results,
                output_path=doc_metrics_path,
                title=f"Document-Level Topic Evaluation Metrics (Reference: {reference_name})"
            )
            plt.close()
            
            # Topic word metrics (Greedy)
            word_metrics_greedy_path = os.path.join(ref_dir, "topic_word_metrics_greedy.png")
            self.visualize_topic_word_metrics(
                evaluation_results,
                matching_method="greedy",
                output_path=word_metrics_greedy_path,
                title=f"Topic Word Metrics with Greedy Matching (Reference: {reference_name})"
            )
            plt.close()
            
            # Topic word metrics (Hungarian)
            word_metrics_hungarian_path = os.path.join(ref_dir, "topic_word_metrics_hungarian.png")
            self.visualize_topic_word_metrics(
                evaluation_results,
                matching_method="hungarian",
                output_path=word_metrics_hungarian_path,
                title=f"Topic Word Metrics with Hungarian Matching (Reference: {reference_name})"
            )
            plt.close()
        
        return output_paths

def main():
    """Main function to demonstrate topic evaluation functionality"""
    import pickle
    
    # Define constants
    CACHE_DIR = "cache"
    OUTPUT_DIR = "results/evaluation"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Try to load topic results from cache
    topic_results_path = os.path.join(CACHE_DIR, "topic_results.pkl")
    
    if os.path.exists(topic_results_path):
        logger.info(f"Loading topic results from cache: {topic_results_path}")
        with open(topic_results_path, "rb") as f:
            topic_results = pickle.load(f)
    else:
        logger.warning("No cached topic results found. Please run topic_modeling.py first.")
        return
    
    # Extract reference result (assuming it's the first one)
    reference_name = list(topic_results.keys())[0]
    reference_result = topic_results[reference_name]
    
    # Create samples dictionary with all but reference
    samples = {name: result for name, result in topic_results.items() if name != reference_name}
    
    # Initialize evaluator
    evaluator = TopicEvaluator(cache_dir=os.path.join(CACHE_DIR, "topic_evaluation"))
    
    # Run evaluation
    logger.info(f"Evaluating topic models against reference: {reference_name}")
    evaluation_results = evaluator.evaluate_multiple_topic_models(
        reference_result=reference_result,
        sample_results=samples,
        top_n_words=10,
        force_recompute=False
    )
    
    # Export results
    logger.info("Exporting evaluation results...")
    output_paths = evaluator.export_evaluation_results(
        evaluation_results=evaluation_results,
        reference_name=reference_name,
        output_dir=OUTPUT_DIR,
        create_visualizations=True
    )
    
    # Print summary of results
    logger.info("\n===== TOPIC EVALUATION RESULTS SUMMARY =====")
    logger.info(f"{'Sample':<30} {'ARI':<8} {'NMI':<8} {'Word F1':<8}")
    logger.info("-" * 55)
    
    for name, result in evaluation_results.items():
        ari = result["document_metrics"]["adjusted_rand_index"]
        nmi = result["document_metrics"]["normalized_mutual_info"]
        word_f1 = result["topic_word_metrics"]["hungarian"]["f1_score"]
        
        logger.info(f"{name:<30} {ari:.4f}  {nmi:.4f}  {word_f1:.4f}")
    
    logger.info(f"\nResults exported to {OUTPUT_DIR}")
    
    return evaluation_results


if __name__ == "__main__":
    main()