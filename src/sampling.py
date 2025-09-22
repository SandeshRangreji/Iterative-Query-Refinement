# sampling.py
import os
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
from collections import defaultdict
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import hdbscan
from tqdm import tqdm
from enum import Enum
import random

# Import from other modules
from search import (
    TextPreprocessor, 
    IndexManager,
    SearchEngine,
    RetrievalMethod,
    HybridStrategy,
    run_search_for_multiple_queries
)
from keyword_extraction import KeywordExtractor
from query_expansion import (
    QueryExpander, 
    QueryExpansionMethod, 
    QueryCombinationStrategy
)
from evaluation import SearchEvaluationUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DimensionReductionMethod(str, Enum):
    """Enum for dimension reduction methods"""
    NONE = "none"
    UMAP = "umap"
    PCA = "pca"

class ClusteringMethod(str, Enum):
    """Enum for clustering methods"""
    KMEANS = "kmeans"  # Now uses Mini-Batch KMeans by default
    HDBSCAN = "hdbscan"

class RepresentativeSelectionMethod(str, Enum):
    """Enum for representative selection methods"""
    CENTROID = "centroid"
    RANDOM = "random"
    DENSITY = "density"
    MAX_PROBABILITY = "max_probability"

class SamplingMethod(str, Enum):
    """Enum for sampling methods"""
    FULL_DATASET = "full_dataset"
    RETRIEVAL = "retrieval"
    QUERY_EXPANSION = "query_expansion"
    UNIFORM = "uniform"
    RANDOM = "random"
    CLUSTERING = "clustering"

class Sampler:
    """Class for sampling documents from a corpus using various methods"""
    
    def __init__(
        self,
        corpus_dataset,
        queries_dataset=None,
        qrels_dataset=None,
        preprocessor: Optional[TextPreprocessor] = None,
        index_manager: Optional[IndexManager] = None,
        cache_dir: str = "cache/sampling",
        embedding_model_name: str = "all-mpnet-base-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        random_seed: int = 42
    ):
        """
        Initialize the sampler
        
        Args:
            corpus_dataset: Dataset containing corpus documents
            queries_dataset: Dataset containing queries (optional, required for query-based sampling)
            qrels_dataset: Dataset containing relevance judgments (optional, for evaluation)
            preprocessor: Text preprocessor instance (will create one if None)
            index_manager: Index manager instance (will create one if None)
            cache_dir: Directory for caching results
            embedding_model_name: Name of the embedding model to use
            cross_encoder_model_name: Name of the cross-encoder model to use
            random_seed: Random seed for reproducibility
        """
        self.corpus_dataset = corpus_dataset
        self.queries_dataset = queries_dataset
        self.qrels_dataset = qrels_dataset
        self.embedding_model_name = embedding_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cache_dir = cache_dir
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize preprocessor if not provided
        if preprocessor is None:
            self.preprocessor = TextPreprocessor()
        else:
            self.preprocessor = preprocessor
        
        # Initialize index manager if not provided
        if index_manager is None:
            self.index_manager = IndexManager(self.preprocessor)
        else:
            self.index_manager = index_manager
        
        # Initialize indices and models
        self._initialize_indices_and_models()
        
        # Initialize search engine
        self.search_engine = SearchEngine(
            preprocessor=self.preprocessor,
            bm25=self.bm25,
            corpus_texts=self.corpus_texts,
            corpus_ids=self.corpus_ids,
            sbert_model=self.sbert_model,
            doc_embeddings=self.doc_embeddings,
            cross_encoder_model_name=cross_encoder_model_name
        )
    
    def _initialize_indices_and_models(self, force_reindex: bool = False):
        """
        Initialize or load indices and models
        
        Args:
            force_reindex: Whether to force rebuilding indices
        """
        logger.info("Initializing indices and models...")
        
        # Build BM25 index
        self.bm25, self.corpus_texts, self.corpus_ids = self.index_manager.build_bm25_index(
            self.corpus_dataset,
            cache_path=os.path.join(self.cache_dir, "bm25_index.pkl"),
            force_reindex=force_reindex
        )
        
        # Build SBERT index
        self.sbert_model, self.doc_embeddings = self.index_manager.build_sbert_index(
            self.corpus_texts,
            model_name=self.embedding_model_name,
            batch_size=64,
            cache_path=os.path.join(self.cache_dir, "sbert_index.pt"),
            force_reindex=force_reindex
        )
        
        # Create mapping from corpus IDs to indices
        self.corpus_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.corpus_ids)}
    
    def _get_doc_texts_from_ids(self, doc_ids: List[str]) -> List[str]:
        """
        Get document texts from document IDs
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            List of document texts
        """
        return [self.corpus_texts[self.corpus_id_to_idx[doc_id]] for doc_id in doc_ids]
    
    def _get_doc_embeddings_from_ids(self, doc_ids: List[str]) -> torch.Tensor:
        """
        Get document embeddings from document IDs
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            Tensor of document embeddings
        """
        indices = [self.corpus_id_to_idx[doc_id] for doc_id in doc_ids]
        return self.doc_embeddings[indices]
    
    def _get_cache_path(self, method: str, config: Dict[str, Any], prefix: str = "") -> str:
        """
        Generate a cache path based on method and config
        
        Args:
            method: Sampling method name
            config: Configuration dictionary
            prefix: Optional prefix for cache file
            
        Returns:
            Cache file path
        """
        # Create a string representation of key config parameters
        config_str = "_".join([f"{k}={v}" for k, v in sorted(config.items()) 
                              if k in ['sample_size', 'n_clusters', 'dim_reduction_method', 
                                      'clustering_method', 'top_k', 'retrieval_method']])
        
        # Create the filename
        filename = f"{prefix}_{method}_{config_str}.pkl" if prefix else f"{method}_{config_str}.pkl"
        
        return os.path.join(self.cache_dir, filename)
    
    def sample_full_dataset(self) -> Dict:
        """
        Return the full dataset with no sampling
        
        Returns:
            Dictionary with all documents in the corpus
        """
        logger.info(f"Returning full dataset ({len(self.corpus_ids)} documents)...")
        
        # Return the entire corpus
        result = {
            "method": SamplingMethod.FULL_DATASET,
            "config": {"sample_size": len(self.corpus_ids)},
            "total_docs": len(self.corpus_ids),
            "sampled_docs": len(self.corpus_ids),
            "sampling_rate": 1.0,
            "sample_ids": self.corpus_ids,
            "sample_texts": self.corpus_texts
        }
    
        return result
    
    def sample_random(self, sample_size: int) -> Dict:
        """
        Perform simple random sampling of documents
        
        Args:
            sample_size: Number of documents to sample
            
        Returns:
            Dictionary with sampling results
        """
        logger.info(f"Performing random sampling (size: {sample_size})...")
        
        # Ensure sample size doesn't exceed corpus size
        actual_sample_size = min(sample_size, len(self.corpus_ids))
        
        # Randomly select indices
        indices = sorted(random.sample(range(len(self.corpus_ids)), actual_sample_size))
        
        # Get corresponding document IDs and texts
        sampled_ids = [self.corpus_ids[i] for i in indices]
        sampled_texts = [self.corpus_texts[i] for i in indices]
        
        # Prepare result
        result = {
            "method": SamplingMethod.RANDOM,
            "config": {"sample_size": sample_size},
            "total_docs": len(self.corpus_ids),
            "sampled_docs": len(sampled_ids),
            "sampling_rate": len(sampled_ids) / len(self.corpus_ids),
            "sample_ids": sampled_ids,
            "sample_texts": sampled_texts,
            "sample_indices": indices
        }
        
        return result
    
    def sample_uniform(self, sample_size: int) -> Dict:
        """
        Perform uniform sampling (stratified by document length)
        
        Args:
            sample_size: Number of documents to sample
            
        Returns:
            Dictionary with sampling results
        """
        logger.info(f"Performing uniform sampling (size: {sample_size})...")
        
        # Calculate document lengths
        doc_lengths = [len(text.split()) for text in self.corpus_texts]
        
        # Create bins (quantiles) for stratification
        num_bins = 10
        bin_edges = np.quantile(doc_lengths, np.linspace(0, 1, num_bins + 1))
        
        # Assign documents to bins
        bins = [[] for _ in range(num_bins)]
        for i, length in enumerate(doc_lengths):
            for bin_idx in range(num_bins):
                if bin_edges[bin_idx] <= length <= bin_edges[bin_idx + 1]:
                    bins[bin_idx].append(i)
                    break
        
        # Ensure sample size doesn't exceed corpus size
        actual_sample_size = min(sample_size, len(self.corpus_ids))
        
        # Calculate samples per bin
        samples_per_bin = [max(1, int(len(bin_docs) / len(self.corpus_ids) * actual_sample_size)) 
                           for bin_docs in bins]
        
        # Adjust to ensure we get the exact sample size
        total_allocated = sum(samples_per_bin)
        if total_allocated < actual_sample_size:
            # Distribute remaining samples
            diff = actual_sample_size - total_allocated
            for i in range(diff):
                # Add to bins with most documents first
                bin_idx = sorted(range(num_bins), key=lambda i: len(bins[i]) / samples_per_bin[i], reverse=True)[i % num_bins]
                samples_per_bin[bin_idx] += 1
        elif total_allocated > actual_sample_size:
            # Remove excess samples
            diff = total_allocated - actual_sample_size
            for i in range(diff):
                # Remove from bins with fewest documents first
                bin_idx = sorted(range(num_bins), key=lambda i: len(bins[i]) / samples_per_bin[i])[i % num_bins]
                if samples_per_bin[bin_idx] > 1:  # Ensure at least one sample per bin
                    samples_per_bin[bin_idx] -= 1
        
        # Sample from each bin
        sampled_indices = []
        for bin_idx, bin_docs in enumerate(bins):
            if bin_docs:  # Ensure bin is not empty
                bin_samples = min(samples_per_bin[bin_idx], len(bin_docs))
                if bin_samples > 0:
                    sampled_indices.extend(random.sample(bin_docs, bin_samples))
        
        # Get corresponding document IDs and texts
        sampled_ids = [self.corpus_ids[i] for i in sampled_indices]
        sampled_texts = [self.corpus_texts[i] for i in sampled_indices]
        
        # Prepare result
        result = {
            "method": SamplingMethod.UNIFORM,
            "config": {"sample_size": sample_size},
            "total_docs": len(self.corpus_ids),
            "sampled_docs": len(sampled_ids),
            "sampling_rate": len(sampled_ids) / len(self.corpus_ids),
            "sample_ids": sampled_ids,
            "sample_texts": sampled_texts,
            "sample_indices": sampled_indices,
            "bin_info": {
                "num_bins": num_bins,
                "bin_edges": bin_edges.tolist(),
                "bin_sizes": [len(bin_docs) for bin_docs in bins],
                "samples_per_bin": samples_per_bin
            }
        }
        
        return result
    
    def sample_retrieval(
        self,
        sample_size: int,
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        top_k: int = 1000,
        use_mmr: bool = False,
        use_cross_encoder: bool = False,
        mmr_lambda: float = 0.5,
        force_regenerate: bool = False
    ) -> Dict:
        """
        Sample documents based on retrieval results
        
        Args:
            sample_size: Target sample size
            retrieval_method: Method for document retrieval
            hybrid_strategy: Strategy for hybrid retrieval
            top_k: Number of documents to retrieve per query
            use_mmr: Whether to use MMR for diversity
            use_cross_encoder: Whether to use cross-encoder reranking
            mmr_lambda: Lambda parameter for MMR
            force_regenerate: Whether to force regenerating results
            
        Returns:
            Dictionary with sampling results
        """
        if self.queries_dataset is None:
            raise ValueError("Queries dataset is required for retrieval-based sampling")
        
        # Create config dictionary for cache path
        config = {
            "sample_size": sample_size,
            "retrieval_method": retrieval_method.value,
            "hybrid_strategy": hybrid_strategy.value,
            "top_k": top_k,
            "use_mmr": use_mmr,
            "use_cross_encoder": use_cross_encoder,
            "mmr_lambda": mmr_lambda
        }
        
        # Generate cache path
        cache_path = self._get_cache_path(SamplingMethod.RETRIEVAL.value, config)
        
        # Check cache
        if not force_regenerate and os.path.exists(cache_path):
            logger.info(f"Loading retrieval sampling results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Performing retrieval-based sampling (size: {sample_size})...")
        
        # Run search for all queries
        query_results = run_search_for_multiple_queries(
            search_engine=self.search_engine,
            queries_dataset=self.queries_dataset,
            top_k=top_k,
            method=retrieval_method,
            hybrid_strategy=hybrid_strategy,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            use_cross_encoder=use_cross_encoder
        )
        
        # Collect all unique document IDs from results
        all_doc_ids = set()
        doc_scores = defaultdict(float)  # Store highest score for each document
        
        for query_id, results in query_results.items():
            for doc_id, score in results:
                all_doc_ids.add(doc_id)
                # Keep the highest score across all queries
                doc_scores[doc_id] = max(doc_scores[doc_id], score)
        
        # Sort documents by score
        sorted_doc_ids = sorted(all_doc_ids, key=lambda x: doc_scores[x], reverse=True)
        
        # Select top-k documents if we retrieved more than requested
        if len(sorted_doc_ids) > sample_size:
            sampled_ids = sorted_doc_ids[:sample_size]
        else:
            # If we didn't retrieve enough documents, use all retrieved docs
            sampled_ids = sorted_doc_ids
        
        # Get texts for sampled documents
        sampled_texts = self._get_doc_texts_from_ids(sampled_ids)
        
        # Prepare result
        result = {
            "method": SamplingMethod.RETRIEVAL,
            "config": config,
            "total_docs": len(self.corpus_ids),
            "total_queries": len(self.queries_dataset),
            "retrieved_docs": len(all_doc_ids),
            "sampled_docs": len(sampled_ids),
            "sampling_rate": len(sampled_ids) / len(self.corpus_ids),
            "sample_ids": sampled_ids,
            "sample_texts": sampled_texts,
            "raw_query_results": query_results
        }
        
        # Cache results
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def sample_query_expansion(
        self,
        sample_size: int,
        expansion_method: QueryExpansionMethod = QueryExpansionMethod.KEYBERT,
        combination_strategy: QueryCombinationStrategy = QueryCombinationStrategy.WEIGHTED_RRF,
        num_keywords: int = 5,
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        top_k: int = 1000,
        use_mmr: bool = False,
        use_cross_encoder: bool = False,
        original_query_weight: float = 0.7,
        force_regenerate: bool = False,
        force_regenerate_keywords: bool = False
    ) -> Dict:
        """
        Sample documents based on query expansion results
        
        Args:
            sample_size: Target sample size
            expansion_method: Method for query expansion
            combination_strategy: Strategy for combining original and expanded queries
            num_keywords: Number of keywords to extract per query
            retrieval_method: Method for document retrieval
            hybrid_strategy: Strategy for hybrid retrieval
            top_k: Number of documents to retrieve per query
            use_mmr: Whether to use MMR for diversity
            use_cross_encoder: Whether to use cross-encoder reranking
            original_query_weight: Weight for original query in RRF combination
            force_regenerate: Whether to force regenerating expansion results
            force_regenerate_keywords: Whether to force regenerating keywords
            
        Returns:
            Dictionary with sampling results
        """
        if self.queries_dataset is None:
            raise ValueError("Queries dataset is required for query expansion sampling")
        
        # Create config dictionary for cache path
        config = {
            "sample_size": sample_size,
            "expansion_method": expansion_method.value,
            "combination_strategy": combination_strategy.value,
            "num_keywords": num_keywords,
            "retrieval_method": retrieval_method.value,
            "hybrid_strategy": hybrid_strategy.value,
            "top_k": top_k,
            "use_mmr": use_mmr,
            "use_cross_encoder": use_cross_encoder,
            "original_query_weight": original_query_weight
        }
        
        # Generate cache path
        cache_path = self._get_cache_path(SamplingMethod.QUERY_EXPANSION.value, config)
        
        # Check cache
        if not force_regenerate and os.path.exists(cache_path):
            logger.info(f"Loading query expansion sampling results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Performing query expansion sampling (size: {sample_size})...")
        
        # Initialize keyword extractor and pass force flag
        keyword_extractor = KeywordExtractor(
            keybert_model=self.sbert_model,
            cache_dir=self.cache_dir,
            top_k_docs=top_k,
            top_n_docs_for_extraction=10
        )
        
        # Extract keywords with force flag
        all_keywords = keyword_extractor.extract_keywords_for_queries(
            queries_dataset=self.queries_dataset,
            corpus_dataset=self.corpus_dataset,
            num_keywords=num_keywords,
            diversity=0.7,
            force_regenerate=force_regenerate_keywords  # PASS FLAG
        )
        
        # Initialize query expander
        query_expander = QueryExpander(
            preprocessor=self.preprocessor,
            index_manager=self.index_manager,
            cache_dir=self.cache_dir,
            sbert_model_name=self.embedding_model_name,
            cross_encoder_model_name=self.cross_encoder_model_name
        )
        
        # Run baseline search with original queries for RRF combination
        baseline_results = run_search_for_multiple_queries(
            search_engine=self.search_engine,
            queries_dataset=self.queries_dataset,
            top_k=top_k,
            method=retrieval_method,
            hybrid_strategy=hybrid_strategy,
            use_mmr=use_mmr,
            mmr_lambda=0.5,
            use_cross_encoder=use_cross_encoder
        )
        
        # Run query expansion
        expansion_results = query_expander.expand_queries(
            queries_dataset=self.queries_dataset,
            corpus_dataset=self.corpus_dataset,
            baseline_results=baseline_results,
            search_engine=self.search_engine,
            query_keywords=all_keywords,
            expansion_method=expansion_method,
            combination_strategy=combination_strategy,
            num_keywords=num_keywords,
            top_k_results=top_k,
            original_query_weight=original_query_weight,
            force_regenerate_expansion=force_regenerate  # PASS FLAG
        )
        
        # Extract document IDs from expanded results
        all_doc_ids = set()
        doc_scores = defaultdict(float)
        
        for item in expansion_results["expanded_results"]:
            for doc_id, score in item["results"]:
                all_doc_ids.add(doc_id)
                # Keep the highest score
                doc_scores[doc_id] = max(doc_scores[doc_id], score)
        
        # Sort documents by score
        sorted_doc_ids = sorted(all_doc_ids, key=lambda x: doc_scores[x], reverse=True)
        
        # Select top-k documents if we retrieved more than requested
        if len(sorted_doc_ids) > sample_size:
            sampled_ids = sorted_doc_ids[:sample_size]
        else:
            # If we didn't retrieve enough documents, use all retrieved docs
            sampled_ids = sorted_doc_ids
        
        # Get texts for sampled documents
        sampled_texts = self._get_doc_texts_from_ids(sampled_ids)
        
        # Prepare result
        result = {
            "method": SamplingMethod.QUERY_EXPANSION,
            "config": config,
            "total_docs": len(self.corpus_ids),
            "total_queries": len(self.queries_dataset),
            "retrieved_docs": len(all_doc_ids),
            "sampled_docs": len(sampled_ids),
            "sampling_rate": len(sampled_ids) / len(self.corpus_ids),
            "sample_ids": sampled_ids,
            "sample_texts": sampled_texts,
            "expansion_results": expansion_results
        }
        
        # Cache results
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def sample_clustering(
        self,
        sample_size: int,
        source_method: SamplingMethod = SamplingMethod.FULL_DATASET,
        source_config: Optional[Dict] = None,
        dim_reduction_method: DimensionReductionMethod = DimensionReductionMethod.UMAP,
        n_components: int = 50,
        clustering_method: ClusteringMethod = ClusteringMethod.KMEANS,
        rep_selection_method: RepresentativeSelectionMethod = RepresentativeSelectionMethod.CENTROID,
        n_per_cluster: int = 1,
        min_cluster_size: int = 5,
        batch_size: int = 1000,
        max_iter: int = 100,
        n_init: int = 3,
        max_no_improvement: int = 10,
        reassignment_ratio: float = 0.01,
        force_recompute: bool = False,
        force_source_regenerate: bool = False
    ) -> Dict:
        """
        Sample documents using clustering with Mini-Batch KMeans optimization
        
        Args:
            sample_size: Target sample size
            source_method: Method for obtaining documents to cluster
            source_config: Configuration for source method
            dim_reduction_method: Method for dimensionality reduction
            n_components: Number of components for reduced space
            clustering_method: Method for clustering
            rep_selection_method: Method for selecting representatives
            n_per_cluster: Number of documents to select per cluster
            min_cluster_size: Minimum cluster size (for HDBSCAN)
            batch_size: Batch size for Mini-Batch KMeans
            max_iter: Maximum iterations for Mini-Batch KMeans
            n_init: Number of initializations for Mini-Batch KMeans
            max_no_improvement: Early stopping parameter for Mini-Batch KMeans
            reassignment_ratio: Controls reassignment frequency for Mini-Batch KMeans
            force_recompute: Whether to force recomputing clustering
            force_source_regenerate: Whether to force regenerating source documents
            
        Returns:
            Dictionary with sampling results
        """
        # Default source config if not provided
        if source_method == SamplingMethod.FULL_DATASET:
            source_result = self.sample_full_dataset()
            
        elif source_method == SamplingMethod.RETRIEVAL:
            source_result = self.sample_retrieval(
                sample_size=source_config["sample_size"],
                retrieval_method=source_config["retrieval_method"],
                hybrid_strategy=source_config["hybrid_strategy"],
                top_k=source_config["top_k"],
                use_mmr=source_config["use_mmr"],
                use_cross_encoder=source_config["use_cross_encoder"],
                force_regenerate=force_source_regenerate  # PASS FLAG
            )
            
        elif source_method == SamplingMethod.QUERY_EXPANSION:
            source_result = self.sample_query_expansion(
                sample_size=source_config["sample_size"],
                expansion_method=source_config["expansion_method"],
                combination_strategy=source_config["combination_strategy"],
                num_keywords=source_config["num_keywords"],
                retrieval_method=source_config["retrieval_method"],
                hybrid_strategy=source_config["hybrid_strategy"],
                top_k=source_config["top_k"],
                use_mmr=source_config["use_mmr"],
                use_cross_encoder=source_config["use_cross_encoder"],
                force_regenerate=force_source_regenerate,  # PASS FLAG
                force_regenerate_keywords=force_source_regenerate  # PASS FLAG
            )
        
        # Create config dictionary for cache path
        config = {
            "sample_size": sample_size,
            "source_method": source_method.value,
            "dim_reduction_method": dim_reduction_method.value,
            "n_components": n_components,
            "clustering_method": clustering_method.value,
            "rep_selection_method": rep_selection_method.value,
            "n_per_cluster": n_per_cluster,
            "min_cluster_size": min_cluster_size
        }
        
        # Generate cache path
        cache_path = self._get_cache_path(SamplingMethod.CLUSTERING.value, config)
        
        # Check cache
        if not force_recompute and os.path.exists(cache_path):
            logger.info(f"Loading clustering sampling results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Performing clustering-based sampling (size: {sample_size})...")
        
        # Get source documents based on method
        if source_method == SamplingMethod.FULL_DATASET:
            source_result = self.sample_full_dataset(source_config["sample_size"])
        elif source_method == SamplingMethod.RETRIEVAL:
            source_result = self.sample_retrieval(
                sample_size=source_config["sample_size"],
                retrieval_method=source_config["retrieval_method"],
                hybrid_strategy=source_config["hybrid_strategy"],
                top_k=source_config["top_k"],
                use_mmr=source_config["use_mmr"],
                use_cross_encoder=source_config["use_cross_encoder"],
                force_regenerate=force_source_regenerate
            )
        elif source_method == SamplingMethod.QUERY_EXPANSION:
            source_result = self.sample_query_expansion(
                sample_size=source_config["sample_size"],
                expansion_method=source_config["expansion_method"],
                combination_strategy=source_config["combination_strategy"],
                num_keywords=source_config["num_keywords"],
                retrieval_method=source_config["retrieval_method"],
                hybrid_strategy=source_config["hybrid_strategy"],
                top_k=source_config["top_k"],
                use_mmr=source_config["use_mmr"],
                use_cross_encoder=source_config["use_cross_encoder"],
                force_regenerate=force_source_regenerate
            )
        else:
            raise ValueError(f"Unsupported source method: {source_method}")
        
        # Get document IDs and embeddings from source
        source_doc_ids = source_result["sample_ids"]
        source_doc_texts = source_result["sample_texts"]
        source_embeddings = self._get_doc_embeddings_from_ids(source_doc_ids)
        
        # Convert to numpy for processing
        embeddings_np = source_embeddings.cpu().numpy()
        
        # Calculate number of clusters based on sample size
        if clustering_method == ClusteringMethod.KMEANS:
            n_clusters = min(sample_size, len(source_doc_ids))
        else:  # HDBSCAN
            n_clusters = sample_size // n_per_cluster
        
        # Perform dimensionality reduction if requested
        if dim_reduction_method != DimensionReductionMethod.NONE:
            logger.info(f"Reducing dimensions with {dim_reduction_method.value} to {n_components} components...")
            reduced_embeddings = self._reduce_dimensions(
                embeddings_np, 
                method=dim_reduction_method.value, 
                n_components=n_components
            )
        else:
            reduced_embeddings = embeddings_np
        
        # Perform clustering
        logger.info(f"Clustering with {clustering_method.value}...")
        labels, cluster_info = self._cluster_documents(
            reduced_embeddings, 
            method=clustering_method.value, 
            n_clusters=n_clusters, 
            min_cluster_size=min_cluster_size,
            batch_size=batch_size,          # NOW ACTUALLY USED
            max_iter=max_iter,              # NOW ACTUALLY USED
            n_init=n_init,                  # NOW ACTUALLY USED
            max_no_improvement=max_no_improvement,  # NOW ACTUALLY USED
            reassignment_ratio=reassignment_ratio   # NOW ACTUALLY USED
        )
        
        # Select representatives
        logger.info(f"Selecting representatives with {rep_selection_method.value}...")
        selected_indices = self._select_cluster_representatives(
            reduced_embeddings, 
            labels, 
            method=rep_selection_method.value, 
            n_per_cluster=n_per_cluster,
            probabilities=cluster_info.get("probabilities")
        )
        
        # Get selected document IDs and texts
        sampled_ids = [source_doc_ids[i] for i in selected_indices]
        sampled_texts = [source_doc_texts[i] for i in selected_indices]
        
        # Prepare result
        result = {
            "method": SamplingMethod.CLUSTERING,
            "config": config,
            "source_method": source_method.value,
            "source_config": source_config,
            "total_docs": len(self.corpus_ids),
            "source_docs": len(source_doc_ids),
            "sampled_docs": len(sampled_ids),
            "sampling_rate": len(sampled_ids) / len(self.corpus_ids),
            "sample_ids": sampled_ids,
            "sample_texts": sampled_texts,
            "cluster_info": cluster_info,
            "dim_reduction_info": {
                "method": dim_reduction_method.value,
                "n_components": n_components
            }
        }
        
        # Cache results
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 50,
        random_state: int = 42,
        cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Reduce dimensions of embeddings
        
        Args:
            embeddings: Input embeddings
            method: Reduction method ('umap' or 'pca')
            n_components: Number of components in reduced space
            random_state: Random state for reproducibility
            cache_key: Optional key for caching results
            
        Returns:
            Reduced embeddings
        """
        # Generate cache path if a key is provided
        cache_path = None
        if cache_key is not None:
            cache_path = os.path.join(
                self.cache_dir, 
                f"reduced_{method}_{n_components}_{cache_key}.npy"
            )
            
            # Check cache
            if cache_path and os.path.exists(cache_path):
                logger.info(f"Loading reduced embeddings from cache: {cache_path}")
                return np.load(cache_path)
        
        if method.lower() == "umap":
            # UMAP for non-linear dimensionality reduction
            reducer = umap.UMAP(
                n_components=n_components,
                metric='cosine',
                n_neighbors=15,
                min_dist=0.1,
                random_state=random_state
            )
            reduced_embeddings = reducer.fit_transform(embeddings)
            
        elif method.lower() == "pca":
            # PCA for linear dimensionality reduction
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced_embeddings = reducer.fit_transform(embeddings)
            
        else:
            raise ValueError(f"Unknown dimension reduction method: {method}")
        
        # Save to cache if a path is provided
        if cache_path:
            logger.info(f"Saving reduced embeddings to cache: {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, reduced_embeddings)
        
        return reduced_embeddings
    
    def _cluster_documents(
        self,
        embeddings: np.ndarray,
        method: str = "kmeans",
        n_clusters: int = 100,
        min_cluster_size: int = 5,
        batch_size: int = 1000,
        max_iter: int = 100,
        n_init: int = 3,
        max_no_improvement: int = 10,
        reassignment_ratio: float = 0.01,
        random_state: int = 42,
        cache_key: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cluster documents using Mini-Batch KMeans (faster) or HDBSCAN
        
        Args:
            embeddings: Document embeddings
            method: Clustering method ('kmeans' or 'hdbscan')
            n_clusters: Number of clusters for Mini-Batch KMeans
            min_cluster_size: Minimum cluster size for HDBSCAN
            batch_size: Batch size for Mini-Batch KMeans
            max_iter: Maximum iterations for Mini-Batch KMeans
            n_init: Number of initializations for Mini-Batch KMeans
            max_no_improvement: Early stopping parameter for Mini-Batch KMeans
            reassignment_ratio: Controls reassignment frequency for Mini-Batch KMeans
            random_state: Random state for reproducibility
            cache_key: Optional key for caching results
            
        Returns:
            Tuple of (cluster labels, cluster info dictionary)
        """
        # Generate cache path if a key is provided
        cache_path = None
        if cache_key is not None:
            cache_path = os.path.join(
                self.cache_dir, 
                f"clusters_{method}_{n_clusters}_{min_cluster_size}_{batch_size}_{cache_key}.pkl"
            )
            
            # Check cache
            if cache_path and os.path.exists(cache_path):
                logger.info(f"Loading clustering results from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    result = pickle.load(f)
                return result["labels"], result["cluster_info"]
        
        if method.lower() == "kmeans":
            # Mini-Batch KMeans - optimized for speed
            from sklearn.cluster import MiniBatchKMeans
            
            # Adjust batch size based on dataset size for optimal performance
            actual_batch_size = min(batch_size, len(embeddings))
            
            logger.info(f"Running Mini-Batch KMeans with {n_clusters} clusters, batch_size={actual_batch_size}")
            
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=actual_batch_size,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
                max_no_improvement=max_no_improvement,
                reassignment_ratio=reassignment_ratio,
                compute_labels=True
            )
            
            labels = clusterer.fit_predict(embeddings)
            
            # Calculate cluster assignment probabilities (distance-based, inverse)
            distances = clusterer.transform(embeddings)
            probabilities = 1.0 / (1.0 + distances)
            
            # Normalize to sum to 1 per row
            row_sums = probabilities.sum(axis=1, keepdims=True)
            probabilities = probabilities / (row_sums + 1e-9)  # Add small epsilon to avoid division by zero
            
            # Prepare cluster info
            cluster_info = {
                "method": "mini_batch_kmeans",
                "n_clusters": n_clusters,
                "batch_size": actual_batch_size,
                "max_iter": max_iter,
                "n_init": n_init,
                "max_no_improvement": max_no_improvement,
                "reassignment_ratio": reassignment_ratio,
                "inertia": clusterer.inertia_,
                "cluster_centers": clusterer.cluster_centers_,
                "cluster_sizes": np.bincount(labels, minlength=n_clusters).tolist(),
                "probabilities": probabilities,
                "n_iter": clusterer.n_iter_
            }
            
        elif method.lower() == "hdbscan":
            # HDBSCAN implementation remains the same
            min_samples = min(min_cluster_size // 2, 2)
                
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(embeddings)
            
            # Get probabilities if available
            if hasattr(clusterer, 'probabilities_'):
                probabilities = clusterer.probabilities_
            else:
                probabilities = None
            
            # Count cluster sizes (excluding noise points labeled as -1)
            valid_labels = labels[labels >= 0]
            unique_labels = np.unique(labels)
            unique_valid_labels = unique_labels[unique_labels >= 0]
            
            if len(valid_labels) > 0:
                cluster_sizes = np.bincount(valid_labels, minlength=len(unique_valid_labels)).tolist()
            else:
                cluster_sizes = []
            
            # Prepare cluster info
            cluster_info = {
                "method": "hdbscan",
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "num_clusters": len(unique_valid_labels),
                "noise_points": np.sum(labels == -1),
                "cluster_sizes": cluster_sizes,
                "probabilities": probabilities,
                "outlier_scores": clusterer.outlier_scores_ if hasattr(clusterer, 'outlier_scores_') else None
            }
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Save to cache if a path is provided
        if cache_path:
            logger.info(f"Saving clustering results to cache: {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            result = {
                "labels": labels,
                "cluster_info": cluster_info
            }
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        
        return labels, cluster_info
    
    def _select_cluster_representatives(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = "centroid",
        n_per_cluster: int = 1,
        probabilities: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Select representative documents from each cluster
        
        Args:
            embeddings: Document embeddings
            labels: Cluster labels
            method: Method for selecting representatives
            n_per_cluster: Number of representatives per cluster
            probabilities: Cluster assignment probabilities (for max_probability method)
            
        Returns:
            List of selected document indices
        """
        selected_indices = []
        
        # Get unique cluster labels (exclude noise points with label -1)
        unique_clusters = np.unique(labels)
        if -1 in unique_clusters:
            unique_clusters = unique_clusters[unique_clusters != -1]
        
        for cluster_id in unique_clusters:
            # Get indices of documents in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Handle different selection methods
            if method.lower() == "random":
                # Random selection
                np.random.shuffle(cluster_indices)
                selected = cluster_indices[:n_per_cluster]
                
            elif method.lower() == "centroid":
                # Select documents closest to cluster centroid
                cluster_points = embeddings[cluster_indices]
                centroid = np.mean(cluster_points, axis=0)
                
                # Calculate distances to centroid
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                
                # Sort by distance (ascending)
                sorted_indices = np.argsort(distances)
                selected = cluster_indices[sorted_indices[:n_per_cluster]]
                
            elif method.lower() == "density":
                # Select documents in densest regions (if outlier scores available)
                if "cluster_info" in dir(self) and "outlier_scores" in self.cluster_info:
                    # Lower outlier scores = higher density
                    outlier_scores = self.cluster_info["outlier_scores"][cluster_indices]
                    sorted_indices = np.argsort(outlier_scores)
                    selected = cluster_indices[sorted_indices[:n_per_cluster]]
                else:
                    # Fallback to centroid method
                    cluster_points = embeddings[cluster_indices]
                    centroid = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    sorted_indices = np.argsort(distances)
                    selected = cluster_indices[sorted_indices[:n_per_cluster]]
            
            elif method.lower() == "max_probability":
                # Select documents with highest cluster assignment probability
                if probabilities is not None:
                    # Get probabilities for this cluster
                    cluster_probs = probabilities[cluster_indices, cluster_id]
                    
                    # Sort by probability (descending)
                    sorted_indices = np.argsort(-cluster_probs)
                    selected = cluster_indices[sorted_indices[:n_per_cluster]]
                else:
                    # Fallback to centroid method
                    cluster_points = embeddings[cluster_indices]
                    centroid = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    sorted_indices = np.argsort(distances)
                    selected = cluster_indices[sorted_indices[:n_per_cluster]]
            
            else:
                raise ValueError(f"Unknown representative selection method: {method}")
            
            selected_indices.extend(selected)
        
        return selected_indices
    
    def sample_multi_config(
        self,
        configs: List[Dict],
        force_regenerate: bool = False
    ) -> Dict[str, Dict]:
        """
        Sample documents using multiple configurations
        
        Args:
            configs: List of sampling configurations
            force_regenerate: Whether to force regenerating results
            
        Returns:
            Dictionary mapping configuration names to sampling results
        """
        results = {}
        
        for config in configs:
            config_name = config.get("name", f"config_{len(results)}")
            method = config.get("method", SamplingMethod.RANDOM)
            
            logger.info(f"Processing {config_name} with method {method}...")
            
            if method == SamplingMethod.FULL_DATASET:
                result = self.sample_full_dataset(
                    sample_size=config.get("sample_size", len(self.corpus_ids))
                )
            
            elif method == SamplingMethod.RANDOM:
                result = self.sample_random(
                    sample_size=config.get("sample_size", 1000)
                )
            
            elif method == SamplingMethod.UNIFORM:
                result = self.sample_uniform(
                    sample_size=config.get("sample_size", 1000)
                )
            
            elif method == SamplingMethod.RETRIEVAL:
                result = self.sample_retrieval(
                    sample_size=config.get("sample_size", 1000),
                    retrieval_method=config.get("retrieval_method", RetrievalMethod.HYBRID),
                    hybrid_strategy=config.get("hybrid_strategy", HybridStrategy.SIMPLE_SUM),
                    top_k=config.get("top_k", 1000),
                    use_mmr=config.get("use_mmr", False),
                    use_cross_encoder=config.get("use_cross_encoder", False),
                    mmr_lambda=config.get("mmr_lambda", 0.5),
                    force_regenerate=force_regenerate
                )
            
            elif method == SamplingMethod.QUERY_EXPANSION:
                result = self.sample_query_expansion(
                    sample_size=config.get("sample_size", 1000),
                    expansion_method=config.get("expansion_method", QueryExpansionMethod.KEYBERT),
                    combination_strategy=config.get("combination_strategy", QueryCombinationStrategy.WEIGHTED_RRF),
                    num_keywords=config.get("num_keywords", 5),
                    retrieval_method=config.get("retrieval_method", RetrievalMethod.HYBRID),
                    hybrid_strategy=config.get("hybrid_strategy", HybridStrategy.SIMPLE_SUM),
                    top_k=config.get("top_k", 1000),
                    use_mmr=config.get("use_mmr", False),
                    use_cross_encoder=config.get("use_cross_encoder", False),
                    original_query_weight=config.get("original_query_weight", 0.7),
                    force_regenerate=force_regenerate,
                    force_regenerate_keywords=force_regenerate
                )
            
            elif method == SamplingMethod.CLUSTERING:
                result = self.sample_clustering(
                    sample_size=config.get("sample_size", 1000),
                    source_method=config.get("source_method", SamplingMethod.FULL_DATASET),
                    source_config=config.get("source_config"),
                    dim_reduction_method=config.get("dim_reduction_method", DimensionReductionMethod.UMAP),
                    n_components=config.get("n_components", 50),
                    clustering_method=config.get("clustering_method", ClusteringMethod.KMEANS),
                    rep_selection_method=config.get("rep_selection_method", RepresentativeSelectionMethod.CENTROID),
                    n_per_cluster=config.get("n_per_cluster", 1),
                    min_cluster_size=config.get("min_cluster_size", 5),
                    force_recompute=force_regenerate,
                    force_source_regenerate=force_regenerate
                )
            
            else:
                raise ValueError(f"Unknown sampling method: {method}")
            
            results[config_name] = result
        
        return results
    
    def compare_samples(
        self,
        sample_results: Dict[str, Dict],
        evaluation_metrics: List[str] = ["diversity", "coverage", "overlap"],
        reference_result: Optional[str] = None
    ) -> Dict:
        """
        Compare multiple document samples
        
        Args:
            sample_results: Dictionary mapping names to sampling results
            evaluation_metrics: List of metrics to evaluate
            reference_result: Optional name of result to use as reference
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Comparing samples...")
        
        comparison = {
            "sample_sizes": {},
            "sample_rates": {},
            "metrics": {}
        }
        
        # Extract basic statistics
        for name, result in sample_results.items():
            comparison["sample_sizes"][name] = result["sampled_docs"]
            comparison["sample_rates"][name] = result["sampling_rate"]
        
        # Calculate overlap between samples
        if "overlap" in evaluation_metrics:
            overlap_matrix = {}
            
            for name1, result1 in sample_results.items():
                overlap_matrix[name1] = {}
                set1 = set(result1["sample_ids"])
                
                for name2, result2 in sample_results.items():
                    set2 = set(result2["sample_ids"])
                    
                    # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
                    overlap = len(set1.intersection(set2)) / len(set1.union(set2)) if set1 or set2 else 0
                    overlap_matrix[name1][name2] = overlap
            
            comparison["metrics"]["overlap"] = overlap_matrix
        
        # Calculate embedding-based diversity
        if "diversity" in evaluation_metrics:
            diversity_scores = {}
            
            for name, result in sample_results.items():
                # Get embeddings for sampled documents
                sample_embeddings = self._get_doc_embeddings_from_ids(result["sample_ids"])
                
                # Calculate pairwise distances (cosine distance = 1 - cosine similarity)
                sample_embeddings_np = sample_embeddings.cpu().numpy()
                
                # Normalize embeddings for cosine similarity
                norms = np.linalg.norm(sample_embeddings_np, axis=1, keepdims=True)
                normalized_embeddings = sample_embeddings_np / norms
                
                # Calculate pairwise similarities
                similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
                
                # Convert to distances
                distances = 1 - similarities
                
                # Calculate diversity metrics
                diversity_scores[name] = {
                    "mean_distance": float(np.mean(distances)),
                    "min_distance": float(np.min(distances[np.triu_indices_from(distances, k=1)])) if len(distances) > 1 else 0,
                    "max_distance": float(np.max(distances))
                }
            
            comparison["metrics"]["diversity"] = diversity_scores
        
        # Calculate coverage (if qrels dataset is available)
        if "coverage" in evaluation_metrics and self.qrels_dataset is not None:
            coverage_scores = {}
            
            # Build relevance judgements
            _, _, relevance_docs_by_query = SearchEvaluationUtils.build_qrels_dicts(self.qrels_dataset)
            
            for name, result in sample_results.items():
                sample_ids_set = set(result["sample_ids"])
                
                # Calculate coverage of relevant documents
                covered_relevant = 0
                total_relevant = 0
                
                for query_id, relevant_docs in relevance_docs_by_query.items():
                    intersection = relevant_docs.intersection(sample_ids_set)
                    covered_relevant += len(intersection)
                    total_relevant += len(relevant_docs)
                
                # Calculate coverage score
                coverage_score = covered_relevant / total_relevant if total_relevant > 0 else 0
                
                coverage_scores[name] = {
                    "covered_relevant": covered_relevant,
                    "total_relevant": total_relevant,
                    "coverage_score": coverage_score
                }
            
            comparison["metrics"]["coverage"] = coverage_scores
        
        return comparison
    
    def save_samples(
        self,
        samples: Dict[str, Dict],
        output_dir: str = "results/samples",
        format: str = "json"
    ) -> Dict[str, str]:
        """
        Save samples to disk
        
        Args:
            samples: Dictionary mapping names to sampling results
            output_dir: Directory to save results
            format: Output format ('json', 'pickle', or 'both')
            
        Returns:
            Dictionary mapping names to file paths
        """
        logger.info(f"Saving samples to {output_dir}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = {}
        
        for name, result in samples.items():
            # Create a filename-safe version of the name
            safe_name = name.replace(" ", "_").lower()
            
            # Create subdirectory for this sample
            sample_dir = os.path.join(output_dir, safe_name)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save metadata
            metadata = {
                "name": name,
                "method": result["method"].value if isinstance(result["method"], Enum) else result["method"],
                "config": {k: (v.value if isinstance(v, Enum) else v) 
                          for k, v in result["config"].items()},
                "total_docs": result["total_docs"],
                "sampled_docs": result["sampled_docs"],
                "sampling_rate": result["sampling_rate"]
            }
            
            # Save IDs
            ids_path = os.path.join(sample_dir, "sample_ids.txt")
            with open(ids_path, "w") as f:
                for doc_id in result["sample_ids"]:
                    f.write(f"{doc_id}\n")
            
            # Save documents
            docs_path = os.path.join(sample_dir, "sample_documents.txt")
            with open(docs_path, "w") as f:
                for i, (doc_id, text) in enumerate(zip(result["sample_ids"], result["sample_texts"])):
                    f.write(f"--- Document {i+1} (ID: {doc_id}) ---\n")
                    f.write(f"{text}\n\n")
            
            # Save in requested format(s)
            if format.lower() in ["json", "both"]:
                import json
                
                # Create a JSON-serializable version
                serializable_result = {
                    "name": name,
                    "method": result["method"].value if isinstance(result["method"], Enum) else result["method"],
                    "config": {k: (v.value if isinstance(v, Enum) else v) 
                              for k, v in result["config"].items()},
                    "total_docs": result["total_docs"],
                    "sampled_docs": result["sampled_docs"],
                    "sampling_rate": result["sampling_rate"],
                    "sample_ids": result["sample_ids"]
                }
                
                json_path = os.path.join(sample_dir, "metadata.json")
                with open(json_path, "w") as f:
                    json.dump(serializable_result, f, indent=2)
            
            if format.lower() in ["pickle", "both"]:
                pickle_path = os.path.join(sample_dir, "full_result.pkl")
                with open(pickle_path, "wb") as f:
                    pickle.dump(result, f)
            
            saved_paths[name] = sample_dir
        
        return saved_paths


def main():
    """Main function to demonstrate sampling functionality"""
    from datasets import load_dataset
    
    # Define constants
    CACHE_DIR = "cache"
    LOG_LEVEL = 'INFO'
    OUTPUT_DIR = "results/samples"
    SAMPLE_SIZE = 10000  # Increased from 1000
    
    # Force flags - individual control for each module
    FORCE_REINDEX = False                      # For BM25/SBERT indices (search.py)
    FORCE_REGENERATE_KEYWORDS = False          # For KeyBERT extraction (keyword_extraction.py)
    FORCE_REGENERATE_EXPANSION = False         # For query expansion (query_expansion.py)
    FORCE_REGENERATE_SAMPLING = False          # For sampling method results (sampling.py)
    FORCE_RECOMPUTE_CLUSTERING = False         # For clustering specifically (sampling.py)
    FORCE_REGENERATE_SOURCE = False            # For source docs in clustering (sampling.py)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")
    logger.info(f"Loaded {len(corpus_dataset)} documents, {len(queries_dataset)} queries, {len(qrels_dataset)} relevance judgments")
    
    # Initialize sampler - pass force_reindex for underlying indices
    logger.info("Initializing sampler...")
    sampler = Sampler(
        corpus_dataset=corpus_dataset,
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        cache_dir=CACHE_DIR,
        embedding_model_name="all-mpnet-base-v2",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        random_seed=42
    )
    
    # Force reindex if needed - call internal method to rebuild indices
    if FORCE_REINDEX:
        logger.info("Force reindexing enabled - rebuilding indices...")
        sampler._initialize_indices_and_models(force_reindex=True)
    
    # Define multiple sampling configurations with all methods
    configs = [
        {
            "name": "Random Sample",
            "method": SamplingMethod.RANDOM,
            "sample_size": SAMPLE_SIZE
        },
        {
            "name": "Uniform Sample",
            "method": SamplingMethod.UNIFORM,
            "sample_size": SAMPLE_SIZE
        },
        {
            "name": "Retrieval Sample",
            "method": SamplingMethod.RETRIEVAL,
            "sample_size": SAMPLE_SIZE,
            "retrieval_method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "top_k": SAMPLE_SIZE,
            "use_mmr": False,
            "use_cross_encoder": False
        },
        {
            "name": "Query Expansion Sample",
            "method": SamplingMethod.QUERY_EXPANSION,
            "sample_size": SAMPLE_SIZE,
            "expansion_method": QueryExpansionMethod.KEYBERT,
            "combination_strategy": QueryCombinationStrategy.WEIGHTED_RRF,
            "num_keywords": 5,
            "retrieval_method": RetrievalMethod.HYBRID,
            "top_k": SAMPLE_SIZE
        },
        {
            "name": "Clustering (Full Dataset)",
            "method": SamplingMethod.CLUSTERING,
            "sample_size": SAMPLE_SIZE,
            "source_method": SamplingMethod.FULL_DATASET,
            "dim_reduction_method": DimensionReductionMethod.UMAP,
            "n_components": 50,
            "clustering_method": ClusteringMethod.KMEANS,
            "rep_selection_method": RepresentativeSelectionMethod.CENTROID,
            "batch_size": 2000,
            "max_iter": 50,
            "n_init": 2
        },
        {
            "name": "Clustering (Retrieval)",
            "method": SamplingMethod.CLUSTERING,
            "sample_size": SAMPLE_SIZE,
            "source_method": SamplingMethod.RETRIEVAL,
            "source_config": {
                "sample_size": SAMPLE_SIZE * 5,  # Retrieval multiplier for clustering
                "retrieval_method": RetrievalMethod.HYBRID,
                "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
                "top_k": SAMPLE_SIZE * 5,
                "use_mmr": False,
                "use_cross_encoder": False
            },
            "dim_reduction_method": DimensionReductionMethod.UMAP,
            "n_components": 50,
            "clustering_method": ClusteringMethod.KMEANS,
            "rep_selection_method": RepresentativeSelectionMethod.CENTROID,
            "batch_size": 2000,
            "max_iter": 50,
            "n_init": 2
        },
        {
            "name": "Query Expansion + Clustering",
            "method": SamplingMethod.CLUSTERING,
            "sample_size": SAMPLE_SIZE,
            "source_method": SamplingMethod.QUERY_EXPANSION,
            "source_config": {
                "sample_size": SAMPLE_SIZE * 3,  # Multiplier for clustering
                "expansion_method": QueryExpansionMethod.KEYBERT,
                "combination_strategy": QueryCombinationStrategy.WEIGHTED_RRF,
                "num_keywords": 5,
                "retrieval_method": RetrievalMethod.HYBRID,
                "top_k": SAMPLE_SIZE * 3
            },
            "dim_reduction_method": DimensionReductionMethod.UMAP,
            "n_components": 50,
            "clustering_method": ClusteringMethod.KMEANS,
            "rep_selection_method": RepresentativeSelectionMethod.CENTROID,
            "batch_size": 2000,
            "max_iter": 50,
            "n_init": 2
        }
    ]
    
    # Run all sampling configurations with proper force flag passing
    logger.info("Running all sampling configurations...")
    all_samples = {}
    
    for config in configs:
        config_name = config.get("name", f"config_{len(all_samples)}")
        method = config.get("method", SamplingMethod.RANDOM)
        
        logger.info(f"Processing {config_name} with method {method}...")
        
        try:
            if method == SamplingMethod.RANDOM:
                result = sampler.sample_random(
                    sample_size=config.get("sample_size", SAMPLE_SIZE)
                )
            
            elif method == SamplingMethod.UNIFORM:
                result = sampler.sample_uniform(
                    sample_size=config.get("sample_size", SAMPLE_SIZE)
                )
            
            elif method == SamplingMethod.RETRIEVAL:
                result = sampler.sample_retrieval(
                    sample_size=config.get("sample_size", SAMPLE_SIZE),
                    retrieval_method=config.get("retrieval_method", RetrievalMethod.HYBRID),
                    hybrid_strategy=config.get("hybrid_strategy", HybridStrategy.SIMPLE_SUM),
                    top_k=config.get("top_k", SAMPLE_SIZE),
                    use_mmr=config.get("use_mmr", False),
                    use_cross_encoder=config.get("use_cross_encoder", False),
                    mmr_lambda=config.get("mmr_lambda", 0.5),
                    force_regenerate=FORCE_REGENERATE_SAMPLING
                )
            
            elif method == SamplingMethod.QUERY_EXPANSION:
                result = sampler.sample_query_expansion(
                    sample_size=config.get("sample_size", SAMPLE_SIZE),
                    expansion_method=config.get("expansion_method", QueryExpansionMethod.KEYBERT),
                    combination_strategy=config.get("combination_strategy", QueryCombinationStrategy.WEIGHTED_RRF),
                    num_keywords=config.get("num_keywords", 5),
                    retrieval_method=config.get("retrieval_method", RetrievalMethod.HYBRID),
                    hybrid_strategy=config.get("hybrid_strategy", HybridStrategy.SIMPLE_SUM),
                    top_k=config.get("top_k", SAMPLE_SIZE),
                    use_mmr=config.get("use_mmr", False),
                    use_cross_encoder=config.get("use_cross_encoder", False),
                    original_query_weight=config.get("original_query_weight", 0.7),
                    force_regenerate=FORCE_REGENERATE_SAMPLING,
                    force_regenerate_keywords=FORCE_REGENERATE_KEYWORDS
                )
            
            elif method == SamplingMethod.CLUSTERING:
                result = sampler.sample_clustering(
                    sample_size=config.get("sample_size", SAMPLE_SIZE),
                    source_method=config.get("source_method", SamplingMethod.FULL_DATASET),
                    source_config=config.get("source_config"),
                    dim_reduction_method=config.get("dim_reduction_method", DimensionReductionMethod.UMAP),
                    n_components=config.get("n_components", 50),
                    clustering_method=config.get("clustering_method", ClusteringMethod.KMEANS),
                    rep_selection_method=config.get("rep_selection_method", RepresentativeSelectionMethod.CENTROID),
                    n_per_cluster=config.get("n_per_cluster", 1),
                    min_cluster_size=config.get("min_cluster_size", 5),
                    batch_size=config.get("batch_size", 1000),
                    max_iter=config.get("max_iter", 100),
                    n_init=config.get("n_init", 3),
                    max_no_improvement=config.get("max_no_improvement", 10),
                    reassignment_ratio=config.get("reassignment_ratio", 0.01),
                    force_recompute=FORCE_RECOMPUTE_CLUSTERING,
                    force_source_regenerate=FORCE_REGENERATE_SOURCE
                )
            
            else:
                raise ValueError(f"Unknown sampling method: {method}")
            
            all_samples[config_name] = result
            logger.info(f"Successfully processed {config_name}: {result['sampled_docs']} documents sampled")
            
        except Exception as e:
            logger.error(f"Error processing {config_name}: {str(e)}")
            continue
    
    # Compare samples
    logger.info("Comparing samples...")
    comparison = sampler.compare_samples(
        all_samples,
        evaluation_metrics=["diversity", "coverage", "overlap"]
    )
    
    # Print results summary
    logger.info("\n===== SAMPLING RESULTS SUMMARY =====")
    logger.info(f"{'Method':<40} {'Size':<10} {'Coverage':<10} {'Diversity':<10}")
    logger.info("-" * 70)
    
    for name, result in all_samples.items():
        size = result["sampled_docs"]
        
        # Get coverage score if available
        coverage = comparison.get("metrics", {}).get("coverage", {}).get(name, {}).get("coverage_score", 0)
        
        # Get diversity score if available
        diversity = comparison.get("metrics", {}).get("diversity", {}).get(name, {}).get("mean_distance", 0)
        
        logger.info(f"{name:<40} {size:<10} {coverage:.4f}     {diversity:.4f}")
    
    # Save samples
    logger.info("Saving samples...")
    saved_paths = sampler.save_samples(all_samples, output_dir=OUTPUT_DIR)
    
    logger.info(f"Samples saved to {OUTPUT_DIR}")
    
    return all_samples, comparison


# Execute main function if called directly
if __name__ == "__main__":
    main()