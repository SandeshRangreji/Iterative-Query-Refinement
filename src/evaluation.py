# evaluation.py
import os
import logging
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional
from sklearn.metrics import adjusted_rand_score, normalized_mutual_information_score
from sklearn.metrics.cluster import contingency_matrix
import torch
from tqdm import tqdm
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Import functionality from other files
from search import TextPreprocessor, IndexManager
from clustering import ClusteringSampler, RetrievalBasedSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TopicModelEvaluator:
    """Class for evaluating topic models and clustering results"""
    
    def __init__(
        self,
        corpus_dataset,
        cache_dir: str = "evaluation_cache",
        embedding_model_name: str = "all-mpnet-base-v2",
        bertopic_model_batch_size: int = 64
    ):
        """
        Initialize evaluator
        
        Args:
            corpus_dataset: Dataset containing corpus documents
            cache_dir: Directory for caching results
            embedding_model_name: Model name for embeddings
            bertopic_model_batch_size: Batch size for BERTopic
        """
        self.corpus_dataset = corpus_dataset
        self.cache_dir = cache_dir
        self.embedding_model_name = embedding_model_name
        self.bertopic_model_batch_size = bertopic_model_batch_size
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize corpus texts, preprocessor and indices
        self._initialize_corpus()
    
    def _initialize_corpus(self):
        """Initialize corpus texts and basic components"""
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        self.index_manager = IndexManager(self.preprocessor)
        
        # Get corpus texts and ids
        logger.info("Loading corpus texts...")
        _, self.corpus_texts, self.corpus_ids = self.index_manager.build_bm25_index(
            self.corpus_dataset,
            cache_path=os.path.join(self.cache_dir, "eval_bm25_index.pkl"),
            force_reindex=False
        )
        
        # Initialize topic model embeddings (loaded on demand)
        self.topic_model_embeddings = None
    
    def _get_topic_model_embeddings(self, force_recompute: bool = False) -> torch.Tensor:
        """Get or compute embeddings for topic modeling"""
        if self.topic_model_embeddings is not None and not force_recompute:
            return self.topic_model_embeddings
        
        cache_path = os.path.join(self.cache_dir, f"topic_model_embeddings_{self.embedding_model_name.replace('/', '_')}.pt")
        
        if os.path.exists(cache_path) and not force_recompute:
            logger.info(f"Loading topic model embeddings from cache: {cache_path}")
            data = torch.load(cache_path)
            self.topic_model_embeddings = data["embeddings"]
            return self.topic_model_embeddings
        
        # Build SBERT index to get embeddings
        logger.info(f"Computing embeddings for topic modeling with {self.embedding_model_name}...")
        _, self.topic_model_embeddings = self.index_manager.build_sbert_index(
            self.corpus_texts,
            model_name=self.embedding_model_name,
            batch_size=self.bertopic_model_batch_size,
            cache_path=cache_path,
            force_reindex=True
        )
        
        return self.topic_model_embeddings
    
    def create_bertopic_model(
        self, 
        corpus_texts: List[str], 
        embeddings: Optional[torch.Tensor] = None,
        embedding_model_name: Optional[str] = None,
        min_topic_size: int = 10,
        diversity: float = 0.2,
        n_gram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42
    ) -> BERTopic:
        """
        Create a BERTopic model with the given parameters
        
        Args:
            corpus_texts: List of document texts
            embeddings: Pre-computed document embeddings (optional)
            embedding_model_name: Model name for embeddings if not provided
            min_topic_size: Minimum size of topics
            diversity: Diversity parameter for MMR
            n_gram_range: N-gram range for CountVectorizer
            random_state: Random state for reproducibility
            
        Returns:
            BERTopic model
        """
        # Determine embedding model to use
        if embedding_model_name is None:
            embedding_model_name = self.embedding_model_name
        
        # Create representation model with MMR for diversity
        representation_model = MaximalMarginalRelevance(diversity=diversity)
        
        # Create vectorizer for words
        vectorizer_model = CountVectorizer(
            stop_words='english',
            ngram_range=n_gram_range,
            min_df=2,
            max_df=0.95
        )
        
        # Create enhanced TF-IDF model
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        
        # Create BERTopic model
        topic_model = BERTopic(
            language="english",
            min_topic_size=min_topic_size,
            embedding_model=embedding_model_name,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            calculate_probabilities=True,
            verbose=True,
            random_state=random_state
        )
        
        return topic_model
    
    def fit_topic_model(
        self,
        corpus_texts: List[str],
        embeddings: Optional[torch.Tensor] = None,
        model_name: str = "full_corpus",
        min_topic_size: int = 10,
        diversity: float = 0.2,
        force_recompute: bool = False
    ) -> Tuple[BERTopic, List[int], List[List[Tuple[str, float]]]]:
        """
        Fit a BERTopic model on the given corpus
        
        Args:
            corpus_texts: List of document texts
            embeddings: Pre-computed document embeddings (optional)
            model_name: Name for the model (for caching)
            min_topic_size: Minimum size of topics
            diversity: Diversity parameter for MMR
            force_recompute: Whether to force recomputation
            
        Returns:
            Tuple of (BERTopic model, topic assignments, topic words)
        """
        # Check if cached model exists
        cache_path = os.path.join(self.cache_dir, f"bertopic_{model_name}.pkl")
        
        if os.path.exists(cache_path) and not force_recompute:
            logger.info(f"Loading BERTopic model from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
            return result["model"], result["topics"], result["topic_words"]
        
        # Convert embeddings to numpy if provided
        if embeddings is not None:
            embeddings_np = embeddings.cpu().numpy()
        else:
            # Get embeddings if not provided
            all_embeddings = self._get_topic_model_embeddings()
            
            # Match embeddings to corpus texts if using a subset
            if len(corpus_texts) < len(self.corpus_texts):
                # Find indices of texts in the full corpus
                indices = []
                corpus_texts_set = set(corpus_texts)
                for i, text in enumerate(self.corpus_texts):
                    if text in corpus_texts_set:
                        indices.append(i)
                
                embeddings_np = all_embeddings[indices].cpu().numpy()
            else:
                embeddings_np = all_embeddings.cpu().numpy()
        
        # Create model
        topic_model = self.create_bertopic_model(
            corpus_texts=corpus_texts,
            min_topic_size=min_topic_size,
            diversity=diversity
        )
        
        # Fit model
        logger.info(f"Fitting BERTopic model on {len(corpus_texts)} documents...")
        topics, probs = topic_model.fit_transform(corpus_texts, embeddings=embeddings_np)
        
        # Get topic words
        topic_words = topic_model.get_topics()
        
        # Save model
        logger.info(f"Saving BERTopic model to cache: {cache_path}")
        result = {
            "model": topic_model,
            "topics": topics,
            "topic_words": topic_words
        }
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return topic_model, topics, topic_words
    
    def compute_topic_overlap(
        self,
        full_model: BERTopic,
        full_topics: List[int],
        subset_model: BERTopic,
        subset_topics: List[int],
        subset_indices: List[int],
        metric: str = "ari"
    ) -> float:
        """
        Compute topic overlap between full and subset models
        
        Args:
            full_model: BERTopic model fitted on full corpus
            full_topics: Topic assignments for full corpus
            subset_model: BERTopic model fitted on subset
            subset_topics: Topic assignments for subset
            subset_indices: Indices of subset documents in full corpus
            metric: Metric to use ('ari', 'nmi', 'omega', 'bcubed')
            
        Returns:
            Overlap score
        """
        # Get topic assignments for subset documents in full model
        full_subset_topics = [full_topics[i] for i in subset_indices]
        
        # Filter out outlier topics (-1)
        valid_indices = []
        filtered_full_topics = []
        filtered_subset_topics = []
        
        for i, (full_t, subset_t) in enumerate(zip(full_subset_topics, subset_topics)):
            if full_t != -1 and subset_t != -1:
                valid_indices.append(i)
                filtered_full_topics.append(full_t)
                filtered_subset_topics.append(subset_t)
        
        # Check if we have enough valid topics
        if len(filtered_full_topics) < 2:
            logger.warning("Not enough valid topics for overlap calculation")
            return 0.0
        
        # Compute overlap based on metric
        if metric.lower() == "ari":
            # Adjusted Rand Index
            score = adjusted_rand_score(filtered_full_topics, filtered_subset_topics)
            
        elif metric.lower() == "nmi":
            # Normalized Mutual Information
            score = normalized_mutual_information_score(filtered_full_topics, filtered_subset_topics)
            
        elif metric.lower() == "bcubed":
            # B-Cubed Precision and Recall
            # Compute contingency matrix
            cont_matrix = contingency_matrix(filtered_full_topics, filtered_subset_topics)
            
            # Calculate B-cubed precision and recall
            precision = 0
            for i, subset_t in enumerate(filtered_subset_topics):
                cluster_size = np.sum(np.array(filtered_subset_topics) == subset_t)
                precision += cont_matrix[filtered_full_topics[i], subset_t] / cluster_size
            precision /= len(filtered_subset_topics)
            
            recall = 0
            for i, full_t in enumerate(filtered_full_topics):
                cluster_size = np.sum(np.array(filtered_full_topics) == full_t)
                recall += cont_matrix[full_t, filtered_subset_topics[i]] / cluster_size
            recall /= len(filtered_full_topics)
            
            # F1 score
            if precision + recall > 0:
                score = 2 * precision * recall / (precision + recall)
            else:
                score = 0
                
        else:
            raise ValueError(f"Unknown overlap metric: {metric}")
        
        return score
    
    def compare_topic_words(
        self,
        full_model: BERTopic,
        subset_model: BERTopic,
        top_n: int = 10,
        jaccard: bool = True
    ) -> Tuple[float, Dict]:
        """
        Compare topic words between full and subset models
        
        Args:
            full_model: BERTopic model fitted on full corpus
            subset_model: BERTopic model fitted on subset
            top_n: Number of top words to compare
            jaccard: Whether to use Jaccard similarity
            
        Returns:
            Tuple of (average similarity, topic mapping)
        """
        # Get topic words from both models
        full_topic_words = {}
        for topic_id, words in full_model.get_topics().items():
            if topic_id != -1:  # Skip outlier topic
                full_topic_words[topic_id] = [word for word, _ in words[:top_n]]
        
        subset_topic_words = {}
        for topic_id, words in subset_model.get_topics().items():
            if topic_id != -1:  # Skip outlier topic
                subset_topic_words[topic_id] = [word for word, _ in words[:top_n]]
        
        # Compute similarities between all topic pairs
        similarities = {}
        for full_id, full_words in full_topic_words.items():
            similarities[full_id] = {}
            for subset_id, subset_words in subset_topic_words.items():
                if jaccard:
                    # Jaccard similarity
                    intersection = len(set(full_words) & set(subset_words))
                    union = len(set(full_words) | set(subset_words))
                    if union > 0:
                        similarity = intersection / union
                    else:
                        similarity = 0
                else:
                    # Overlap coefficient
                    intersection = len(set(full_words) & set(subset_words))
                    min_size = min(len(full_words), len(subset_words))
                    if min_size > 0:
                        similarity = intersection / min_size
                    else:
                        similarity = 0
                
                similarities[full_id][subset_id] = similarity
        
        # Create mapping of best matching topics
        topic_mapping = {}
        used_subset_topics = set()
        
        # Sort full topics by number of words (descending)
        sorted_full_topics = sorted(
            full_topic_words.keys(),
            key=lambda x: len(full_topic_words[x]),
            reverse=True
        )
        
        for full_id in sorted_full_topics:
            best_subset_id = None
            best_similarity = -1
            
            for subset_id, similarity in similarities[full_id].items():
                if subset_id not in used_subset_topics and similarity > best_similarity:
                    best_subset_id = subset_id
                    best_similarity = similarity
            
            if best_subset_id is not None and best_similarity > 0:
                topic_mapping[full_id] = {
                    "subset_id": best_subset_id,
                    "similarity": best_similarity,
                    "full_words": full_topic_words[full_id],
                    "subset_words": subset_topic_words[best_subset_id]
                }
                used_subset_topics.add(best_subset_id)
        
        # Calculate average similarity
        if topic_mapping:
            avg_similarity = sum(info["similarity"] for info in topic_mapping.values()) / len(topic_mapping)
        else:
            avg_similarity = 0
        
        return avg_similarity, topic_mapping
    
    def evaluate_topic_coverage(
        self,
        full_model: BERTopic,
        subset_model: BERTopic,
        top_n_words: int = 10
    ) -> Dict:
        """
        Evaluate how well the subset topics cover the full corpus topics
        
        Args:
            full_model: BERTopic model fitted on full corpus
            subset_model: BERTopic model fitted on subset
            top_n_words: Number of top words to compare
            
        Returns:
            Dictionary with coverage metrics
        """
        # Get average word similarity and topic mapping
        avg_similarity, topic_mapping = self.compare_topic_words(
            full_model=full_model,
            subset_model=subset_model,
            top_n=top_n_words,
            jaccard=True
        )
        
        # Calculate coverage metrics
        full_topics = set(t for t in full_model.get_topics().keys() if t != -1)
        mapped_full_topics = set(topic_mapping.keys())
        
        # Coverage = % of full topics that have a matching subset topic
        coverage = len(mapped_full_topics) / len(full_topics) if full_topics else 0
        
        # Calculate sizes of topics
        full_topic_sizes = full_model.get_topic_sizes()
        full_topic_sizes_dict = {row[0]: row[1] for row in full_topic_sizes}
        
        # Calculate coverage weighted by topic size
        total_docs = sum(size for t, size in full_topic_sizes_dict.items() if t != -1)
        covered_docs = sum(full_topic_sizes_dict.get(t, 0) for t in mapped_full_topics)
        
        weighted_coverage = covered_docs / total_docs if total_docs > 0 else 0
        
        # Group mapped topics by similarity ranges
        similarity_ranges = {
            "90-100%": 0,
            "70-90%": 0,
            "50-70%": 0,
            "30-50%": 0,
            "0-30%": 0
        }
        
        for info in topic_mapping.values():
            sim = info["similarity"]
            if sim >= 0.9:
                similarity_ranges["90-100%"] += 1
            elif sim >= 0.7:
                similarity_ranges["70-90%"] += 1
            elif sim >= 0.5:
                similarity_ranges["50-70%"] += 1
            elif sim >= 0.3:
                similarity_ranges["30-50%"] += 1
            else:
                similarity_ranges["0-30%"] += 1
        
        # Result
        result = {
            "total_full_topics": len(full_topics),
            "mapped_topics": len(mapped_full_topics),
            "coverage": coverage,
            "weighted_coverage": weighted_coverage,
            "avg_similarity": avg_similarity,
            "similarity_ranges": similarity_ranges,
            "topic_mapping": topic_mapping
        }
        
        return result

    def evaluate_sample(
        self,
        sample_result: Dict,
        model_name: str,
        min_topic_size: int = 10,
        top_n_words: int = 10,
        force_recompute: bool = False
    ) -> Dict:
        """
        Evaluate a sample by fitting a topic model and comparing to full corpus
        
        Args:
            sample_result: Sampling result dictionary
            model_name: Name for the model (for caching)
            min_topic_size: Minimum size of topics
            top_n_words: Number of top words to compare
            force_recompute: Whether to force recomputation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract sample texts and indices
        sample_texts = sample_result["sample_texts"]
        sample_indices = sample_result["sample_indices"]
        
        # Get embeddings
        all_embeddings = self._get_topic_model_embeddings()
        sample_embeddings = all_embeddings[sample_indices]
        
        # Fit topic models
        full_model, full_topics, _ = self.fit_topic_model(
            corpus_texts=self.corpus_texts,
            embeddings=all_embeddings,
            model_name="full_corpus",
            min_topic_size=min_topic_size,
            force_recompute=force_recompute
        )
        
        subset_model, subset_topics, _ = self.fit_topic_model(
            corpus_texts=sample_texts,
            embeddings=sample_embeddings,
            model_name=model_name,
            min_topic_size=min_topic_size,
            force_recompute=force_recompute
        )
        
        # Evaluate topic overlap
        ari_score = self.compute_topic_overlap(
            full_model=full_model,
            full_topics=full_topics,
            subset_model=subset_model,
            subset_topics=subset_topics,
            subset_indices=sample_indices,
            metric="ari"
        )
        
        nmi_score = self.compute_topic_overlap(
            full_model=full_model,
            full_topics=full_topics,
            subset_model=subset_model,
            subset_topics=subset_topics,
            subset_indices=sample_indices,
            metric="nmi"
        )
        
        bcubed_score = self.compute_topic_overlap(
            full_model=full_model,
            full_topics=full_topics,
            subset_model=subset_model,
            subset_topics=subset_topics,
            subset_indices=sample_indices,
            metric="bcubed"
        )
        
        # Evaluate topic coverage
        coverage_metrics = self.evaluate_topic_coverage(
            full_model=full_model,
            subset_model=subset_model,
            top_n_words=top_n_words
        )
        
        # Collect topic information
        full_topic_sizes = full_model.get_topic_sizes()
        full_topic_sizes = {row[0]: row[1] for row in full_topic_sizes}
        
        subset_topic_sizes = subset_model.get_topic_sizes()
        subset_topic_sizes = {row[0]: row[1] for row in subset_topic_sizes}
        
        # Calculate topic distributions
        full_topic_dist = {t: size / len(self.corpus_texts) for t, size in full_topic_sizes.items() if t != -1}
        subset_topic_dist = {t: size / len(sample_texts) for t, size in subset_topic_sizes.items() if t != -1}
        
        # Calculate similarity of topic distributions using Jensen-Shannon divergence
        # (simplified approximation using mapped topics)
        topic_mapping = coverage_metrics["topic_mapping"]
        mapped_subset_to_full = {info["subset_id"]: full_id for full_id, info in topic_mapping.items()}
        
        dist_similarity = 0
        total_weight = 0
        
        for subset_id, subset_prob in subset_topic_dist.items():
            if subset_id in mapped_subset_to_full:
                full_id = mapped_subset_to_full[subset_id]
                full_prob = full_topic_dist.get(full_id, 0)
                
                # Simple approximation of distribution similarity
                min_prob = min(subset_prob, full_prob)
                max_prob = max(subset_prob, full_prob)
                
                if max_prob > 0:
                    similarity = min_prob / max_prob
                    weight = (subset_prob + full_prob) / 2
                    dist_similarity += similarity * weight
                    total_weight += weight
        
        if total_weight > 0:
            dist_similarity /= total_weight
        
        # Compile results
        result = {
            "model_name": model_name,
            "sample_size": len(sample_texts),
            "sampling_rate": len(sample_texts) / len(self.corpus_texts),
            "overlap_metrics": {
                "ari": ari_score,
                "nmi": nmi_score,
                "bcubed": bcubed_score
            },
            "topic_coverage": coverage_metrics,
            "topic_dist_similarity": dist_similarity,
            "full_model_info": {
                "num_topics": len([t for t in full_topic_sizes if t != -1]),
                "largest_topic": max(full_topic_sizes.values() if full_topic_sizes else [0]),
                "smallest_topic": min([size for t, size in full_topic_sizes.items() if t != -1], default=0),
                "outliers": full_topic_sizes.get(-1, 0)
            },
            "subset_model_info": {
                "num_topics": len([t for t in subset_topic_sizes if t != -1]),
                "largest_topic": max(subset_topic_sizes.values() if subset_topic_sizes else [0]),
                "smallest_topic": min([size for t, size in subset_topic_sizes.items() if t != -1], default=0),
                "outliers": subset_topic_sizes.get(-1, 0)
            }
        }
        
        return result
    
    def plot_evaluation_results(self, results: List[Dict]):
        """
        Create visualizations of evaluation results
        
        Args:
            results: List of evaluation result dictionaries
        """
        # Prepare data
        models = [r['model_name'] for r in results]
        sample_sizes = [r['sample_size'] for r in results]
        sampling_rates = [r['sampling_rate'] * 100 for r in results]
        
        ari_scores = [r['overlap_metrics']['ari'] for r in results]
        nmi_scores = [r['overlap_metrics']['nmi'] for r in results]
        bcubed_scores = [r['overlap_metrics']['bcubed'] for r in results]
        
        coverages = [r['topic_coverage']['coverage'] * 100 for r in results]
        weighted_coverages = [r['topic_coverage']['weighted_coverage'] * 100 for r in results]
        avg_similarities = [r['topic_coverage']['avg_similarity'] * 100 for r in results]
        dist_similarities = [r['topic_dist_similarity'] * 100 for r in results]
        
        # Set style
        plt.style.use('ggplot')
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Sample sizes and rates
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.bar(models, sampling_rates, color='skyblue')
        ax1.set_title('Sampling Rate (%)')
        ax1.set_ylabel('Percentage of Corpus')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Topic Overlap Metrics
        ax2 = fig.add_subplot(2, 3, 2)
        x = np.arange(len(models))
        width = 0.25
        
        ax2.bar(x - width, ari_scores, width, label='ARI')
        ax2.bar(x, nmi_scores, width, label='NMI')
        ax2.bar(x + width, bcubed_scores, width, label='B-Cubed')
        
        ax2.set_title('Topic Overlap Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45)
        ax2.set_ylabel('Score')
        ax2.legend()
        
        # 3. Topic Coverage Metrics
        ax3 = fig.add_subplot(2, 3, 3)
        x = np.arange(len(models))
        width = 0.3
        
        ax3.bar(x - width/2, coverages, width, label='Coverage')
        ax3.bar(x + width/2, weighted_coverages, width, label='Weighted Coverage')
        
        ax3.set_title('Topic Coverage (%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.set_ylabel('Percentage')
        ax3.legend()
        
        # 4. Word Similarity and Distribution Similarity
        ax4 = fig.add_subplot(2, 3, 4)
        x = np.arange(len(models))
        width = 0.3
        
        ax4.bar(x - width/2, avg_similarities, width, label='Word Similarity')
        ax4.bar(x + width/2, dist_similarities, width, label='Distribution Similarity')
        
        ax4.set_title('Topic Word and Distribution Similarity (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45)
        ax4.set_ylabel('Percentage')
        ax4.legend()
        
        # 5. Topic Similarity Distribution
        ax5 = fig.add_subplot(2, 3, 5)
        
        # Prepare data for stacked bar chart
        ranges = ["90-100%", "70-90%", "50-70%", "30-50%", "0-30%"]
        data = {}
        
        for r in range(len(results)):
            similarity_ranges = results[r]['topic_coverage']['similarity_ranges']
            for sim_range in ranges:
                if sim_range not in data:
                    data[sim_range] = []
                data[sim_range].append(similarity_ranges[sim_range])
        
        bottom = np.zeros(len(models))
        for sim_range in ranges:
            ax5.bar(models, data[sim_range], bottom=bottom, label=sim_range)
            bottom += np.array(data[sim_range])
        
        ax5.set_title('Topic Similarity Distribution')
        ax5.set_ylabel('Number of Topics')
        ax5.tick_params(axis='x', rotation=45)
        ax5.legend()
        
        # 6. Topic Counts
        ax6 = fig.add_subplot(2, 3, 6)
        full_topics = [r['full_model_info']['num_topics'] for r in results]
        subset_topics = [r['subset_model_info']['num_topics'] for r in results]
        
        x = np.arange(len(models))
        width = 0.3
        
        ax6.bar(x - width/2, full_topics, width, label='Full Corpus')
        ax6.bar(x + width/2, subset_topics, width, label='Sample')
        
        ax6.set_title('Number of Topics')
        ax6.set_xticks(x)
        ax6.set_xticklabels(models, rotation=45)
        ax6.set_ylabel('Count')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.cache_dir, 'evaluation_results.png'), dpi=300)
        plt.close()
        
        # Create detailed plots for each model
        for i, result in enumerate(results):
            model_name = result['model_name']
            topic_mapping = result['topic_coverage']['topic_mapping']
            
            # Skip if no topic mapping
            if not topic_mapping:
                continue
            
            # Plot word similarity matrix for top topics
            top_topics = sorted(
                topic_mapping.items(), 
                key=lambda x: x[1]['similarity'], 
                reverse=True
            )[:20]  # Show top 20 topics
            
            if len(top_topics) <= 1:
                continue
            
            # Extract topic IDs and similarities
            full_ids = [t[0] for t in top_topics]
            subset_ids = [t[1]['subset_id'] for t in top_topics]
            similarities = [t[1]['similarity'] for t in top_topics]
            
            # Create topic words comparison table
            plt.figure(figsize=(14, len(top_topics) * 0.8))
            
            # Display as a table
            cell_text = []
            for full_id, info in top_topics:
                subset_id = info['subset_id']
                similarity = info['similarity']
                full_words = ', '.join(info['full_words'][:5])  # Show top 5 words
                subset_words = ', '.join(info['subset_words'][:5])
                
                cell_text.append([
                    f"T{full_id}", 
                    f"T{subset_id}", 
                    f"{similarity:.2f}", 
                    full_words, 
                    subset_words
                ])
            
            plt.table(
                cellText=cell_text,
                colLabels=['Full ID', 'Subset ID', 'Similarity', 'Full Topic Words', 'Subset Topic Words'],
                loc='center',
                cellLoc='left',
                colWidths=[0.1, 0.1, 0.1, 0.35, 0.35]
            )
            plt.axis('off')
            plt.title(f'Top Topic Mappings for {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.cache_dir, f'topic_mapping_{model_name}.png'), dpi=300)
            plt.close()


    def load_and_evaluate_samples(self, sample_configs: List[Dict], force_recompute: bool = False) -> List[Dict]:
        """
        Load and evaluate multiple samples
        
        Args:
            sample_configs: List of sample configurations
            force_recompute: Whether to force recomputation
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for config in sample_configs:
            model_name = config["name"]
            sample_path = config["sample_path"]
            
            logger.info(f"Evaluating sample: {model_name}")
            
            # Check if evaluation cache exists
            eval_cache_path = os.path.join(self.cache_dir, f"eval_{model_name}.pkl")
            
            if os.path.exists(eval_cache_path) and not force_recompute:
                logger.info(f"Loading evaluation from cache: {eval_cache_path}")
                with open(eval_cache_path, "rb") as f:
                    evaluation = pickle.load(f)
                    results.append(evaluation)
                continue
            
            # Load sample
            if os.path.exists(sample_path):
                logger.info(f"Loading sample from: {sample_path}")
                with open(sample_path, "rb") as f:
                    sample_result = pickle.load(f)
            else:
                logger.error(f"Sample file not found: {sample_path}")
                continue
            
            # Evaluate sample
            evaluation = self.evaluate_sample(
                sample_result=sample_result,
                model_name=model_name,
                min_topic_size=config.get("min_topic_size", 10),
                top_n_words=config.get("top_n_words", 10),
                force_recompute=force_recompute
            )
            
            # Save evaluation to cache
            logger.info(f"Saving evaluation to cache: {eval_cache_path}")
            with open(eval_cache_path, "wb") as f:
                pickle.dump(evaluation, f)
            
            results.append(evaluation)
        
        return results


def main():
    """Main function for sample evaluation"""
    # Define constants
    CACHE_DIR = "evaluation_cache"
    EMBEDDING_MODEL = 'all-mpnet-base-v2'
    LOG_LEVEL = 'INFO'
    FORCE_RECOMPUTE = False
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Load datasets
    logger.info("Loading datasets...")
    from datasets import load_dataset
    
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    logger.info(f"Loaded {len(corpus_dataset)} documents")
    
    # Initialize evaluator
    evaluator = TopicModelEvaluator(
        corpus_dataset=corpus_dataset,
        cache_dir=CACHE_DIR,
        embedding_model_name=EMBEDDING_MODEL
    )
    
    # Define sample configurations to evaluate
    sample_configs = [
        {
            "name": "subset_retrieval",
            "sample_path": "sampling_cache/retrieval_based_none_HYBRID_1000.pkl",
            "min_topic_size": 10,
            "top_n_words": 10
        },
        {
            "name": "subset_retrieval_expansion",
            "sample_path": "sampling_cache/retrieval_based_keybert_pmi_HYBRID_1000.pkl",
            "min_topic_size": 10,
            "top_n_words": 10
        },
        {
            "name": "subset_retrieval_clustering",
            "sample_path": "sampling_cache/sample_retrieval_umap_kmeans_500.pkl",
            "min_topic_size": 10,
            "top_n_words": 10
        },
        {
            "name": "subset_retrieval_expansion_clustering",
            "sample_path": "sampling_cache/sample_retrieval_expansion_umap_kmeans_500.pkl",
            "min_topic_size": 10,
            "top_n_words": 10
        },
        {
            "name": "full_corpus_sample",
            "sample_path": "sampling_cache/full_corpus_umap_kmeans_1000.pkl",
            "min_topic_size": 10,
            "top_n_words": 10
        }
    ]
    
    # Load and evaluate samples
    results = evaluator.load_and_evaluate_samples(
        sample_configs=sample_configs,
        force_recompute=FORCE_RECOMPUTE
    )
    
    # Plot results
    if results:
        logger.info("Plotting evaluation results...")
        evaluator.plot_evaluation_results(results)
        logger.info(f"Plots saved to {CACHE_DIR}")
    
    # Print summary results
    logger.info("\n===== EVALUATION SUMMARY =====")
    logger.info(f"{'Model':<30} {'Topics':<10} {'Coverage':<10} {'ARI':<10} {'NMI':<10}")
    logger.info("-" * 70)
    
    for result in results:
        model_name = result['model_name']
        num_topics = result['subset_model_info']['num_topics']
        coverage = result['topic_coverage']['coverage'] * 100
        ari = result['overlap_metrics']['ari']
        nmi = result['overlap_metrics']['nmi']
        
        logger.info(f"{model_name:<30} {num_topics:<10} {coverage:<10.1f}% {ari:<10.3f} {nmi:<10.3f}")
    
    return results


# Execute main function if called directly
if __name__ == "__main__":
    main()