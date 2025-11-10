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

# Import from topic_modelling.py for consistent enums and cache handling
from topic_modelling import TopicModelingMethod, TopicModeler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TopicEvaluator:
    """Class for evaluating topic models against a reference model"""
    
    def __init__(
        self, 
        cache_dir: str = "cache",
        corpus_subset_size: Optional[int] = None
    ):
        """
        Initialize topic evaluator
        
        Args:
            cache_dir: Directory for caching results
            corpus_subset_size: Size of corpus subset (None for full corpus)
        """
        self.cache_dir = cache_dir
        self.corpus_subset_size = corpus_subset_size
        
        # Generate corpus-specific cache directory
        self.corpus_cache_dir = self._generate_corpus_cache_dir()
        self.evaluation_cache_dir = os.path.join(self.corpus_cache_dir, "topic_evaluation")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.evaluation_cache_dir, exist_ok=True)
    
    def _generate_corpus_cache_dir(self) -> str:
        """Generate corpus-specific cache directory - consistent with other modules"""
        if self.corpus_subset_size is None:
            corpus_size_str = "full"
        else:
            corpus_size_str = f"{self.corpus_subset_size}"
        
        return os.path.join(self.cache_dir, f"corpus{corpus_size_str}")
    
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
        
        return os.path.join(self.evaluation_cache_dir, filename)
    
    def load_topic_model_registry(self) -> Dict:
        """Load topic model registry from cache"""
        registry_path = os.path.join(self.corpus_cache_dir, "topic_models", "all_topic_model_results.pkl")
        
        if os.path.exists(registry_path):
            logger.info(f"Loading topic model registry from: {registry_path}")
            with open(registry_path, "rb") as f:
                return pickle.load(f)
        else:
            logger.warning(f"Topic model registry not found at: {registry_path}")
            return {}
    
    def load_topic_model_result(
        self,
        method: str,
        query_id: Optional[str] = None
    ) -> Dict:
        """
        Load a specific topic model result from cache
        
        Args:
            method: Method name (canonical)
            query_id: Query ID (for query-specific models)
            
        Returns:
            Topic model result dictionary
        """
        # Load registry
        registry = self.load_topic_model_registry()
        
        if not registry:
            raise ValueError("No topic model registry found. Please run topic_modelling.py first.")
        
        # Determine query level
        if query_id is not None:
            query_level = query_id
        else:
            query_level = "corpus_level"
        
        # Check if query level exists
        if query_level not in registry:
            available_levels = list(registry.keys())
            raise ValueError(f"Query level '{query_level}' not found in registry. Available: {available_levels}")
        
        # Check if method exists for this query level
        if method not in registry[query_level]:
            available_methods = list(registry[query_level].keys())
            raise ValueError(f"Method '{method}' not found for query level '{query_level}'. Available: {available_methods}")
        
        # Return the topic model result
        return registry[query_level][method]
    
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
        reference_topics = reference_result["topic_model_result"]["topics"]
        sample_topics = sample_result["topic_model_result"]["topics"]
        
        # Extract document info (using document text as identifier)
        reference_docs = reference_result["topic_model_result"]["document_info"]["Document"].tolist()
        sample_docs = sample_result["topic_model_result"]["document_info"]["Document"].tolist()
        
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
            reference_result["topic_model_result"]["topic_words"],
            sample_result["topic_model_result"]["topic_words"],
            top_n=top_n_words
        )
        
        # Calculate topic word level metrics using topic similarity
        # Greedy matching
        word_precision_greedy, word_recall_greedy, word_f1_greedy = self.calculate_topic_word_precision_recall_greedy(
            reference_result["topic_model_result"]["topic_words"],
            sample_result["topic_model_result"]["topic_words"],
            top_n=top_n_words
        )
        
        # Hungarian matching
        word_precision_hungarian, word_recall_hungarian, word_f1_hungarian, matched_pairs = self.calculate_topic_word_precision_recall_hungarian(
            reference_result["topic_model_result"]["topic_words"],
            sample_result["topic_model_result"]["topic_words"],
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
            
            # Run evaluation
            eval_result = self.evaluate_topic_model(
                reference_result=reference_result,
                sample_result=result,
                top_n_words=top_n_words,
                force_recompute=force_recompute
            )
            
            evaluation_results[name] = eval_result
        
        return evaluation_results
    
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
            
            output_paths[name] = sample_dir
        
        return output_paths


def main():
    """Main function to evaluate topic models using FULL_DATASET as the gold standard"""
    from datasets import load_dataset
    
    # ===== CONFIGURATION PARAMETERS =====
    
    # Corpus and cache configuration - should match topic_modelling.py
    CORPUS_SUBSET_SIZE = 10000  # Set to None for full corpus
    CACHE_DIR = "cache"
    OUTPUT_DIR = "results/evaluation"
    
    # Evaluation parameters
    TOP_N_WORDS = 10
    FORCE_RECOMPUTE = False
    
    # Reference method (gold standard)
    REFERENCE_METHOD = TopicModelingMethod.FULL_DATASET.value
    
    # Log level
    LOG_LEVEL = 'INFO'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load datasets for metadata
    logger.info("Loading datasets for metadata...")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")
    logger.info(f"Loaded {len(corpus_dataset)} documents, {len(queries_dataset)} queries, {len(qrels_dataset)} relevance judgments")
    
    # Initialize evaluator
    evaluator = TopicEvaluator(
        cache_dir=CACHE_DIR,
        corpus_subset_size=CORPUS_SUBSET_SIZE
    )
    
    # Load topic modeling results from cache
    logger.info("Loading topic modeling results from cache...")
    topic_models_registry = evaluator.load_topic_model_registry()
    
    if not topic_models_registry:
        logger.error("No topic modeling results found. Please run topic_modelling.py first.")
        return None
    
    logger.info(f"Found topic models for {len(topic_models_registry)} query/corpus levels")
    
    # Process each query/corpus level
    all_evaluation_results = {}
    
    for query_level, methods_results in topic_models_registry.items():
        logger.info(f"Evaluating topic models for: {query_level}")
        
        # Check if reference method is available
        if REFERENCE_METHOD not in methods_results:
            logger.warning(f"Reference method '{REFERENCE_METHOD}' not found for {query_level}. Available: {list(methods_results.keys())}")
            continue
        
        # Get reference result
        reference_result = methods_results[REFERENCE_METHOD]
        
        # Create samples dictionary with all but reference
        samples = {name: result for name, result in methods_results.items() 
                  if name != REFERENCE_METHOD}
        
        if not samples:
            logger.warning(f"Only reference model found for {query_level}, cannot evaluate")
            continue
        
        # Run evaluation
        evaluation_results = evaluator.evaluate_multiple_models(
            reference_result=reference_result,
            sample_results=samples,
            top_n_words=TOP_N_WORDS,
            force_recompute=FORCE_RECOMPUTE
        )
        
        # Export results
        level_output_dir = os.path.join(OUTPUT_DIR, query_level)
        evaluator.export_evaluation_results(
            evaluation_results=evaluation_results,
            reference_method=REFERENCE_METHOD,
            output_dir=level_output_dir,
            create_visualizations=True
        )
        
        # Store results
        all_evaluation_results[query_level] = {
            "reference_method": REFERENCE_METHOD,
            "evaluation_results": evaluation_results
        }
    
    # Print summary of all results
    logger.info("\n===== TOPIC EVALUATION SUMMARY =====")
    logger.info(f"Reference Method: {REFERENCE_METHOD}")
    logger.info(f"Corpus Subset Size: {CORPUS_SUBSET_SIZE if CORPUS_SUBSET_SIZE else 'full'}")
    
    for query_level, result in all_evaluation_results.items():
        evaluation_results = result["evaluation_results"]
        
        logger.info(f"\nResults for {query_level}:")
        logger.info(f"{'Method':<30} {'ARI':<8} {'Omega':<8} {'Word F1':<8} {'Docs':<8}")
        logger.info("-" * 65)
        
        for method, eval_result in evaluation_results.items():
            ari = eval_result["document_metrics"]["adjusted_rand_index"]
            omega = eval_result["document_metrics"]["omega_index"]
            word_f1 = eval_result["topic_word_metrics"]["hungarian"]["f1_score"]
            docs = eval_result["sample_docs_count"]
            
            logger.info(f"{method:<30} {ari:.4f}  {omega:.4f}  {word_f1:.4f}  {docs:<8}")
    
    logger.info(f"\nDetailed evaluation results exported to {OUTPUT_DIR}")
    
    # Print corpus cache info
    logger.info(f"\n===== CACHE INFORMATION =====")
    logger.info(f"Corpus cache directory: {evaluator.corpus_cache_dir}")
    logger.info(f"Topic models cache: {os.path.join(evaluator.corpus_cache_dir, 'topic_models')}")
    logger.info(f"Evaluation cache: {evaluator.evaluation_cache_dir}")
    
    return all_evaluation_results


if __name__ == "__main__":
    main()