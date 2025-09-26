# sampling.py
import os
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
from collections import defaultdict
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
from enum import Enum
import random
import hashlib

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
    KMEANS = "kmeans"  # Uses Mini-Batch KMeans for exact sample size control

class RepresentativeSelectionMethod(str, Enum):
    """Enum for representative selection methods"""
    CENTROID = "centroid"
    RANDOM = "random"
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
        cache_dir: str = "cache",
        embedding_model_name: str = "all-mpnet-base-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dataset_name: str = "trec-covid",
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
            dataset_name: Name of the dataset for caching
            random_seed: Random seed for reproducibility
        """
        self.corpus_dataset = corpus_dataset
        self.queries_dataset = queries_dataset
        self.qrels_dataset = qrels_dataset
        self.embedding_model_name = embedding_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
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
            dataset_name=self.dataset_name,
            force_reindex=force_reindex
        )
        
        # Build SBERT index
        self.sbert_model, self.doc_embeddings = self.index_manager.build_sbert_index(
            self.corpus_texts,
            model_name=self.embedding_model_name,
            dataset_name=self.dataset_name,
            batch_size=64,
            force_reindex=force_reindex
        )
        
        # Create mapping from corpus IDs to indices
        self.corpus_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.corpus_ids)}
    
    def _get_doc_texts_from_ids(self, doc_ids: List[str]) -> List[str]:
        """Get document texts from document IDs"""
        return [self.corpus_texts[self.corpus_id_to_idx[doc_id]] for doc_id in doc_ids]
    
    def _get_doc_embeddings_from_ids(self, doc_ids: List[str]) -> torch.Tensor:
        """Get document embeddings from document IDs"""
        indices = [self.corpus_id_to_idx[doc_id] for doc_id in doc_ids]
        return self.doc_embeddings[indices]
    
    def _generate_source_cache_path(
        self,
        method: SamplingMethod,
        config: Dict[str, Any]
    ) -> str:
        """Generate cache path for source document sampling"""
        cache_dir = os.path.join("cache", "sampling", "source_docs")
        
        if method == SamplingMethod.FULL_DATASET:
            filename = "full_dataset_all.pkl"
        elif method == SamplingMethod.RANDOM:
            filename = f"random_k{config['sample_size']}.pkl"
        elif method == SamplingMethod.UNIFORM:
            filename = f"uniform_k{config['sample_size']}.pkl"
        elif method == SamplingMethod.RETRIEVAL:
            retrieval_method = config.get('retrieval_method', 'hybrid')
            if hasattr(retrieval_method, 'value'):
                retrieval_method = retrieval_method.value
            top_k = config.get('top_k', 1000)
            use_mmr = config.get('use_mmr', False)
            use_cross_encoder = config.get('use_cross_encoder', False)
            filename = f"retrieval_{retrieval_method}_k{top_k}_mmr-{str(use_mmr).lower()}_cross-{str(use_cross_encoder).lower()}.pkl"
        elif method == SamplingMethod.QUERY_EXPANSION:
            exp_method = config.get('expansion_method', 'keybert')
            if hasattr(exp_method, 'value'):
                exp_method = exp_method.value
            num_kw = config.get('num_keywords', 5)
            diversity = config.get('diversity', 0.7)
            strategy = config.get('combination_strategy', 'weighted-rrf')
            if hasattr(strategy, 'value'):
                strategy = strategy.value.replace('_', '-')
            top_k = config.get('top_k', 1000)
            filename = f"expansion_{exp_method}-k{num_kw}-div{diversity}_{strategy}_k{top_k}.pkl"
        else:
            # Fallback
            method_str = method.value if hasattr(method, 'value') else str(method)
            sample_size = config.get('sample_size', 1000)
            filename = f"{method_str}_k{sample_size}.pkl"
        
        return os.path.join(cache_dir, filename)
    
    def _generate_reduction_cache_path(
        self,
        method: str,
        n_components: int,
        source_hash: str
    ) -> str:
        """Generate cache path for dimensionality reduction"""
        cache_dir = os.path.join("cache", "sampling", "embeddings")
        
        if method.lower() == "umap":
            filename = f"reduced_umap_n{n_components}_neighbors15_cosine_{source_hash}.npy"
        elif method.lower() == "pca":
            filename = f"reduced_pca_n{n_components}_{source_hash}.npy"
        else:
            filename = f"reduced_{method}_n{n_components}_{source_hash}.npy"
        
        return os.path.join(cache_dir, filename)
    
    def _generate_clustering_cache_path(
        self,
        n_clusters: int,
        batch_size: int,
        max_iter: int,
        n_init: int,
        source_hash: str
    ) -> str:
        """Generate cache path for clustering results"""
        cache_dir = os.path.join("cache", "sampling", "clusters")
        filename = f"kmeans_n{n_clusters}_batch{batch_size}_iter{max_iter}_init{n_init}_{source_hash}.pkl"
        return os.path.join(cache_dir, filename)
    
    def _create_source_hash(self, source_doc_ids: List[str]) -> str:
        """Create a hash from source document IDs for cache naming"""
        # Sort IDs for consistent hashing regardless of order
        sorted_ids = sorted(source_doc_ids)
        # Take a sample if too many documents (for hash efficiency)
        if len(sorted_ids) > 1000:
            step = len(sorted_ids) // 1000
            sorted_ids = sorted_ids[::step]
        
        # Create hash
        ids_str = ''.join(sorted_ids)
        hash_obj = hashlib.md5(ids_str.encode())
        return hash_obj.hexdigest()[:8]
    
    def sample_full_dataset(self) -> Dict:
        """Return the full dataset with no sampling"""
        logger.info(f"Returning full dataset ({len(self.corpus_ids)} documents)...")
        
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
    
    def sample_random(self, sample_size: int, force_regenerate: bool = False) -> Dict:
        """Perform simple random sampling of documents"""
        logger.info(f"Performing random sampling (size: {sample_size})...")
        
        config = {"sample_size": sample_size}
        cache_path = self._generate_source_cache_path(SamplingMethod.RANDOM, config)
        
        # Check cache
        if not force_regenerate and os.path.exists(cache_path):
            logger.info(f"Loading random sample from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
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
            "config": config,
            "total_docs": len(self.corpus_ids),
            "sampled_docs": len(sampled_ids),
            "sampling_rate": len(sampled_ids) / len(self.corpus_ids),
            "sample_ids": sampled_ids,
            "sample_texts": sampled_texts,
            "sample_indices": indices
        }
        
        # Cache result
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def sample_uniform(self, sample_size: int, force_regenerate: bool = False) -> Dict:
        """Perform uniform sampling (stratified by document length)"""
        logger.info(f"Performing uniform sampling (size: {sample_size})...")
        
        config = {"sample_size": sample_size}
        cache_path = self._generate_source_cache_path(SamplingMethod.UNIFORM, config)
        
        # Check cache
        if not force_regenerate and os.path.exists(cache_path):
            logger.info(f"Loading uniform sample from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
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
            "config": config,
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
        
        # Cache result
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
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
        """Sample documents based on retrieval results"""
        if self.queries_dataset is None:
            raise ValueError("Queries dataset is required for retrieval-based sampling")
        
        # Create config dictionary for cache path
        config = {
            "sample_size": sample_size,
            "retrieval_method": retrieval_method,
            "hybrid_strategy": hybrid_strategy,
            "top_k": top_k,
            "use_mmr": use_mmr,
            "use_cross_encoder": use_cross_encoder,
            "mmr_lambda": mmr_lambda
        }
        
        # Generate cache path
        cache_path = self._generate_source_cache_path(SamplingMethod.RETRIEVAL, config)
        
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
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def sample_query_expansion(
        self,
        sample_size: int,
        expansion_method: QueryExpansionMethod = QueryExpansionMethod.KEYBERT,
        combination_strategy: QueryCombinationStrategy = QueryCombinationStrategy.WEIGHTED_RRF,
        num_keywords: int = 5,
        diversity: float = 0.7,
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        top_k: int = 1000,
        use_mmr: bool = False,
        use_cross_encoder: bool = False,
        original_query_weight: float = 0.7,
        force_regenerate: bool = False,
        force_regenerate_keywords: bool = False
    ) -> Dict:
        """Sample documents based on query expansion results"""
        if self.queries_dataset is None:
            raise ValueError("Queries dataset is required for query expansion sampling")
        
        # Create config dictionary for cache path
        config = {
            "sample_size": sample_size,
            "expansion_method": expansion_method,
            "combination_strategy": combination_strategy,
            "num_keywords": num_keywords,
            "diversity": diversity,
            "retrieval_method": retrieval_method,
            "hybrid_strategy": hybrid_strategy,
            "top_k": top_k,
            "use_mmr": use_mmr,
            "use_cross_encoder": use_cross_encoder,
            "original_query_weight": original_query_weight
        }
        
        # Generate cache path
        cache_path = self._generate_source_cache_path(SamplingMethod.QUERY_EXPANSION, config)
        
        # Check cache
        if not force_regenerate and os.path.exists(cache_path):
            logger.info(f"Loading query expansion sampling results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Performing query expansion sampling (size: {sample_size})...")
        
        # Initialize keyword extractor
        keyword_extractor = KeywordExtractor(
            keybert_model=self.sbert_model,
            cache_dir=self.cache_dir,
            top_k_docs=top_k,
            top_n_docs_for_extraction=10
        )
        
        # Extract keywords
        all_keywords = keyword_extractor.extract_keywords_for_queries(
            queries_dataset=self.queries_dataset,
            corpus_dataset=self.corpus_dataset,
            num_keywords=num_keywords,
            diversity=diversity,
            force_regenerate=force_regenerate_keywords,
            sbert_model_name=self.embedding_model_name,
            dataset_name=self.dataset_name
        )
        
        # Initialize query expander
        query_expander = QueryExpander(
            preprocessor=self.preprocessor,
            index_manager=self.index_manager,
            cache_dir=self.cache_dir,
            sbert_model_name=self.embedding_model_name,
            cross_encoder_model_name=self.cross_encoder_model_name
        )
        
        # Run baseline search
        baseline_results = query_expander.run_baseline_search(
            queries_dataset=self.queries_dataset,
            corpus_dataset=self.corpus_dataset,
            search_engine=self.search_engine,
            retrieval_method=retrieval_method,
            hybrid_strategy=hybrid_strategy,
            top_k=top_k,
            use_mmr=use_mmr,
            use_cross_encoder=use_cross_encoder,
            force_regenerate=force_regenerate
        )
        
        # Run query expansion
        expansion_results = query_expander.expand_queries(
            queries_dataset=self.queries_dataset,
            corpus_dataset=self.corpus_dataset,
            baseline_results=baseline_results,
            search_engine=self.search_engine,
            query_keywords=all_keywords,
            kw_method="keybert",
            kw_num_keywords=num_keywords,
            kw_diversity=diversity,
            expansion_method=expansion_method,
            combination_strategy=combination_strategy,
            num_keywords=num_keywords,
            top_k_results=top_k,
            retrieval_method=retrieval_method,
            use_cross_encoder=combination_strategy == QueryCombinationStrategy.CONCATENATED_RERANKED,
            force_regenerate_expansion=force_regenerate
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
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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
        rep_selection_method: RepresentativeSelectionMethod = RepresentativeSelectionMethod.CENTROID,
        batch_size: int = 1000,
        max_iter: int = 100,
        n_init: int = 3,
        max_no_improvement: int = 10,
        reassignment_ratio: float = 0.01,
        force_recompute: bool = False,
        force_source_regenerate: bool = False,
        force_reduction: bool = False
    ) -> Dict:
        """
        Sample documents using KMeans clustering with exact sample size control
        
        Args:
            sample_size: Target sample size (will be exact number of clusters)
            source_method: Method for obtaining documents to cluster
            source_config: Configuration for source method
            dim_reduction_method: Method for dimensionality reduction
            n_components: Number of components for reduced space
            rep_selection_method: Method for selecting representatives
            batch_size: Batch size for Mini-Batch KMeans
            max_iter: Maximum iterations for Mini-Batch KMeans
            n_init: Number of initializations for Mini-Batch KMeans
            max_no_improvement: Early stopping parameter for Mini-Batch KMeans
            reassignment_ratio: Controls reassignment frequency for Mini-Batch KMeans
            force_recompute: Whether to force recomputing clustering
            force_source_regenerate: Whether to force regenerating source documents
            force_reduction: Whether to force recomputing dimensionality reduction
            
        Returns:
            Dictionary with sampling results
        """
        # Get source documents based on method
        logger.info(f"Getting source documents using {source_method.value}...")
        
        if source_method == SamplingMethod.FULL_DATASET:
            source_result = self.sample_full_dataset()
        elif source_method == SamplingMethod.RETRIEVAL:
            source_result = self.sample_retrieval(
                sample_size=source_config.get("sample_size", sample_size * 5),
                retrieval_method=source_config.get("retrieval_method", RetrievalMethod.HYBRID),
                hybrid_strategy=source_config.get("hybrid_strategy", HybridStrategy.SIMPLE_SUM),
                top_k=source_config.get("top_k", sample_size * 5),
                use_mmr=source_config.get("use_mmr", False),
                use_cross_encoder=source_config.get("use_cross_encoder", False),
                force_regenerate=force_source_regenerate
            )
        elif source_method == SamplingMethod.QUERY_EXPANSION:
            source_result = self.sample_query_expansion(
                sample_size=source_config.get("sample_size", sample_size * 3),
                expansion_method=source_config.get("expansion_method", QueryExpansionMethod.KEYBERT),
                combination_strategy=source_config.get("combination_strategy", QueryCombinationStrategy.WEIGHTED_RRF),
                num_keywords=source_config.get("num_keywords", 5),
                diversity=source_config.get("diversity", 0.7),
                retrieval_method=source_config.get("retrieval_method", RetrievalMethod.HYBRID),
                top_k=source_config.get("top_k", sample_size * 3),
                force_regenerate=force_source_regenerate,
                force_regenerate_keywords=force_source_regenerate
            )
        else:
            raise ValueError(f"Unsupported source method: {source_method}")
        
        # Get document IDs and embeddings from source
        source_doc_ids = source_result["sample_ids"]
        source_doc_texts = source_result["sample_texts"]
        source_embeddings = self._get_doc_embeddings_from_ids(source_doc_ids)
        
        # Convert to numpy for processing
        embeddings_np = source_embeddings.cpu().numpy()
        
        # Create source hash for cache naming
        source_hash = self._create_source_hash(source_doc_ids)
        
        # Set exact number of clusters = sample_size for precise control
        n_clusters = sample_size
        
        # Perform dimensionality reduction if requested
        if dim_reduction_method != DimensionReductionMethod.NONE:
            logger.info(f"Reducing dimensions with {dim_reduction_method.value} to {n_components} components...")
            
            reduction_cache_path = self._generate_reduction_cache_path(
                dim_reduction_method.value, n_components, source_hash
            )
            
            if not force_reduction and os.path.exists(reduction_cache_path):
                logger.info(f"Loading reduced embeddings from cache: {reduction_cache_path}")
                reduced_embeddings = np.load(reduction_cache_path)
            else:
                reduced_embeddings = self._reduce_dimensions(
                    embeddings_np, 
                    method=dim_reduction_method.value, 
                    n_components=n_components
                )
                # Save to cache
                os.makedirs(os.path.dirname(reduction_cache_path), exist_ok=True)
                np.save(reduction_cache_path, reduced_embeddings)
        else:
            reduced_embeddings = embeddings_np
        
        # Perform clustering
        logger.info(f"Clustering with KMeans (n_clusters={n_clusters})...")
        
        clustering_cache_path = self._generate_clustering_cache_path(
            n_clusters, batch_size, max_iter, n_init, source_hash
        )
        
        if not force_recompute and os.path.exists(clustering_cache_path):
            logger.info(f"Loading clustering results from cache: {clustering_cache_path}")
            with open(clustering_cache_path, "rb") as f:
                cluster_data = pickle.load(f)
                labels = cluster_data["labels"]
                cluster_info = cluster_data["cluster_info"]
        else:
            labels, cluster_info = self._cluster_documents(
                reduced_embeddings, 
                n_clusters=n_clusters, 
                batch_size=batch_size,
                max_iter=max_iter,
                n_init=n_init,
                max_no_improvement=max_no_improvement,
                reassignment_ratio=reassignment_ratio
            )
            # Save to cache
            os.makedirs(os.path.dirname(clustering_cache_path), exist_ok=True)
            cluster_data = {"labels": labels, "cluster_info": cluster_info}
            with open(clustering_cache_path, "wb") as f:
                pickle.dump(cluster_data, f)
        
        # Select representatives (exactly 1 per cluster = sample_size documents)
        logger.info(f"Selecting representatives with {rep_selection_method.value}...")
        selected_indices = self._select_cluster_representatives(
            reduced_embeddings, 
            labels, 
            method=rep_selection_method.value, 
            probabilities=cluster_info.get("probabilities")
        )
        
        # Get selected document IDs and texts
        sampled_ids = [source_doc_ids[i] for i in selected_indices]
        sampled_texts = [source_doc_texts[i] for i in selected_indices]
        
        # Prepare result
        result = {
            "method": SamplingMethod.CLUSTERING,
            "config": {
                "sample_size": sample_size,
                "source_method": source_method.value,
                "source_config": source_config,
                "dim_reduction_method": dim_reduction_method.value,
                "n_components": n_components,
                "clustering_method": ClusteringMethod.KMEANS.value,
                "rep_selection_method": rep_selection_method.value,
                "n_clusters": n_clusters,
                "batch_size": batch_size,
                "max_iter": max_iter,
                "n_init": n_init
            },
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
        
        return result
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 50,
        random_state: int = 42
    ) -> np.ndarray:
        """Reduce dimensions of embeddings"""
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
        
        return reduced_embeddings
    
    def _cluster_documents(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 100,
        batch_size: int = 1000,
        max_iter: int = 100,
        n_init: int = 3,
        max_no_improvement: int = 10,
        reassignment_ratio: float = 0.01,
        random_state: int = 42
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cluster documents using Mini-Batch KMeans for exact cluster count control
        
        Args:
            embeddings: Document embeddings
            n_clusters: Number of clusters (exact)
            batch_size: Batch size for Mini-Batch KMeans
            max_iter: Maximum iterations for Mini-Batch KMeans
            n_init: Number of initializations for Mini-Batch KMeans
            max_no_improvement: Early stopping parameter
            reassignment_ratio: Controls reassignment frequency
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (cluster labels, cluster info dictionary)
        """
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
        
        return labels, cluster_info
    
    def _select_cluster_representatives(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = "centroid",
        probabilities: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Select exactly one representative document from each cluster
        
        Args:
            embeddings: Document embeddings
            labels: Cluster labels
            method: Method for selecting representatives
            probabilities: Cluster assignment probabilities (for max_probability method)
            
        Returns:
            List of selected document indices (exactly n_clusters documents)
        """
        selected_indices = []
        
        # Get unique cluster labels
        unique_clusters = np.unique(labels)
        
        for cluster_id in unique_clusters:
            # Get indices of documents in this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Handle different selection methods
            if method.lower() == "random":
                # Random selection
                selected_idx = np.random.choice(cluster_indices)
                
            elif method.lower() == "centroid":
                # Select document closest to cluster centroid
                cluster_points = embeddings[cluster_indices]
                centroid = np.mean(cluster_points, axis=0)
                
                # Calculate distances to centroid
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                
                # Select closest document
                closest_idx = np.argmin(distances)
                selected_idx = cluster_indices[closest_idx]
                
            elif method.lower() == "max_probability":
                # Select document with highest cluster assignment probability
                if probabilities is not None:
                    # Get probabilities for this cluster
                    cluster_probs = probabilities[cluster_indices, cluster_id]
                    
                    # Select document with highest probability
                    max_prob_idx = np.argmax(cluster_probs)
                    selected_idx = cluster_indices[max_prob_idx]
                else:
                    # Fallback to centroid method
                    cluster_points = embeddings[cluster_indices]
                    centroid = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    closest_idx = np.argmin(distances)
                    selected_idx = cluster_indices[closest_idx]
            
            else:
                raise ValueError(f"Unknown representative selection method: {method}")
            
            selected_indices.append(selected_idx)
        
        return selected_indices
    
    def compare_samples(
        self,
        sample_results: Dict[str, Dict],
        evaluation_metrics: List[str] = ["diversity", "coverage", "overlap"],
        reference_result: Optional[str] = None
    ) -> Dict:
        """Compare multiple document samples"""
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


def main():
    """Main function to demonstrate sampling functionality"""
    from datasets import load_dataset
    
    # ===== CACHING CONTROL FLAGS =====
    FORCE_REINDEX = False                      # For search indices
    FORCE_REGENERATE_KEYWORDS = False          # For keyword extraction
    FORCE_REGENERATE_RETRIEVAL = False         # For retrieval (baseline + expansion)
    FORCE_REGENERATE_SOURCE = False            # For source document sampling
    FORCE_RECOMPUTE_REDUCTION = False          # For dimensionality reduction
    FORCE_RECOMPUTE_CLUSTERING = False         # For clustering specifically
    
    # ===== DATASET PARAMETERS =====
    DATASET_NAME = 'trec-covid'
    
    # ===== SAMPLING PARAMETERS =====
    SAMPLE_SIZE = 1000
    
    # ===== KEYWORD EXTRACTION PARAMETERS =====
    NUM_KEYWORDS = 10
    DIVERSITY = 0.7
    
    # ===== RETRIEVAL PARAMETERS =====
    RETRIEVAL_METHOD = RetrievalMethod.HYBRID
    HYBRID_STRATEGY = HybridStrategy.SIMPLE_SUM
    TOP_K_DOCS = 1000
    USE_MMR = False
    USE_CROSS_ENCODER = False
    
    # ===== CLUSTERING PARAMETERS =====
    DIM_REDUCTION_METHOD = DimensionReductionMethod.UMAP
    N_COMPONENTS = 50
    REP_SELECTION_METHOD = RepresentativeSelectionMethod.CENTROID
    
    # Mini-Batch KMeans parameters (optimized for speed)
    BATCH_SIZE = 2000
    MAX_ITER = 50
    N_INIT = 2
    MAX_NO_IMPROVEMENT = 5
    REASSIGNMENT_RATIO = 0.005
    
    # ===== MODEL PARAMETERS =====
    EMBEDDING_MODEL = 'all-mpnet-base-v2'
    CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    
    # ===== OTHER PARAMETERS =====
    CACHE_DIR = "cache"
    OUTPUT_DIR = "results/samples"
    LOG_LEVEL = 'INFO'
    
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
    
    # Initialize sampler
    logger.info("Initializing sampler...")
    sampler = Sampler(
        corpus_dataset=corpus_dataset,
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        cache_dir=CACHE_DIR,
        embedding_model_name=EMBEDDING_MODEL,
        cross_encoder_model_name=CROSS_ENCODER_MODEL,
        dataset_name=DATASET_NAME,
        random_seed=42
    )
    
    # Force reindex if needed
    if FORCE_REINDEX:
        logger.info("Force reindexing enabled - rebuilding indices...")
        sampler._initialize_indices_and_models(force_reindex=True)
    
    # Define sampling configurations
    sampling_configs = [
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
            "retrieval_method": RETRIEVAL_METHOD,
            "hybrid_strategy": HYBRID_STRATEGY,
            "top_k": TOP_K_DOCS,
            "use_mmr": USE_MMR,
            "use_cross_encoder": USE_CROSS_ENCODER
        },
        {
            "name": "Query Expansion Sample",
            "method": SamplingMethod.QUERY_EXPANSION,
            "sample_size": SAMPLE_SIZE,
            "expansion_method": QueryExpansionMethod.KEYBERT,
            "combination_strategy": QueryCombinationStrategy.WEIGHTED_RRF,
            "num_keywords": NUM_KEYWORDS,
            "diversity": DIVERSITY,
            "retrieval_method": RETRIEVAL_METHOD,
            "top_k": TOP_K_DOCS
        },
        {
            "name": "Clustering (Full Dataset)",
            "method": SamplingMethod.CLUSTERING,
            "sample_size": SAMPLE_SIZE,
            "source_method": SamplingMethod.FULL_DATASET,
            "dim_reduction_method": DIM_REDUCTION_METHOD,
            "n_components": N_COMPONENTS,
            "rep_selection_method": REP_SELECTION_METHOD,
            "batch_size": BATCH_SIZE,
            "max_iter": MAX_ITER,
            "n_init": N_INIT
        },
        {
            "name": "Clustering (Retrieval)",
            "method": SamplingMethod.CLUSTERING,
            "sample_size": SAMPLE_SIZE,
            "source_method": SamplingMethod.RETRIEVAL,
            "source_config": {
                "sample_size": SAMPLE_SIZE * 5,
                "retrieval_method": RETRIEVAL_METHOD,
                "hybrid_strategy": HYBRID_STRATEGY,
                "top_k": SAMPLE_SIZE * 5,
                "use_mmr": USE_MMR,
                "use_cross_encoder": USE_CROSS_ENCODER
            },
            "dim_reduction_method": DIM_REDUCTION_METHOD,
            "n_components": N_COMPONENTS,
            "rep_selection_method": REP_SELECTION_METHOD,
            "batch_size": BATCH_SIZE,
            "max_iter": MAX_ITER,
            "n_init": N_INIT
        }
    ]
    
    # Run all sampling configurations
    logger.info("Running all sampling configurations...")
    all_samples = {}
    
    for config in sampling_configs:
        config_name = config.get("name", f"config_{len(all_samples)}")
        method = config.get("method", SamplingMethod.RANDOM)
        
        logger.info(f"Processing {config_name} with method {method}...")
        
        try:
            if method == SamplingMethod.RANDOM:
                result = sampler.sample_random(
                    sample_size=config.get("sample_size", SAMPLE_SIZE),
                    force_regenerate=FORCE_REGENERATE_SOURCE
                )
            
            elif method == SamplingMethod.UNIFORM:
                result = sampler.sample_uniform(
                    sample_size=config.get("sample_size", SAMPLE_SIZE),
                    force_regenerate=FORCE_REGENERATE_SOURCE
                )
            
            elif method == SamplingMethod.RETRIEVAL:
                result = sampler.sample_retrieval(
                    sample_size=config.get("sample_size", SAMPLE_SIZE),
                    retrieval_method=config.get("retrieval_method", RETRIEVAL_METHOD),
                    hybrid_strategy=config.get("hybrid_strategy", HYBRID_STRATEGY),
                    top_k=config.get("top_k", TOP_K_DOCS),
                    use_mmr=config.get("use_mmr", USE_MMR),
                    use_cross_encoder=config.get("use_cross_encoder", USE_CROSS_ENCODER),
                    force_regenerate=FORCE_REGENERATE_RETRIEVAL
                )
            
            elif method == SamplingMethod.QUERY_EXPANSION:
                result = sampler.sample_query_expansion(
                    sample_size=config.get("sample_size", SAMPLE_SIZE),
                    expansion_method=config.get("expansion_method", QueryExpansionMethod.KEYBERT),
                    combination_strategy=config.get("combination_strategy", QueryCombinationStrategy.WEIGHTED_RRF),
                    num_keywords=config.get("num_keywords", NUM_KEYWORDS),
                    diversity=config.get("diversity", DIVERSITY),
                    retrieval_method=config.get("retrieval_method", RETRIEVAL_METHOD),
                    top_k=config.get("top_k", TOP_K_DOCS),
                    force_regenerate=FORCE_REGENERATE_RETRIEVAL,
                    force_regenerate_keywords=FORCE_REGENERATE_KEYWORDS
                )
            
            elif method == SamplingMethod.CLUSTERING:
                result = sampler.sample_clustering(
                    sample_size=config.get("sample_size", SAMPLE_SIZE),
                    source_method=config.get("source_method", SamplingMethod.FULL_DATASET),
                    source_config=config.get("source_config"),
                    dim_reduction_method=config.get("dim_reduction_method", DIM_REDUCTION_METHOD),
                    n_components=config.get("n_components", N_COMPONENTS),
                    rep_selection_method=config.get("rep_selection_method", REP_SELECTION_METHOD),
                    batch_size=config.get("batch_size", BATCH_SIZE),
                    max_iter=config.get("max_iter", MAX_ITER),
                    n_init=config.get("n_init", N_INIT),
                    max_no_improvement=config.get("max_no_improvement", MAX_NO_IMPROVEMENT),
                    reassignment_ratio=config.get("reassignment_ratio", REASSIGNMENT_RATIO),
                    force_recompute=FORCE_RECOMPUTE_CLUSTERING,
                    force_source_regenerate=FORCE_REGENERATE_SOURCE,
                    force_reduction=FORCE_RECOMPUTE_REDUCTION
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
    
    # Print cache structure info
    logger.info(f"\n===== CACHE STRUCTURE =====")
    logger.info(f"Search indices: cache/indices/")
    logger.info(f"Keywords: cache/keywords/")
    logger.info(f"Retrieval: cache/retrieval/")
    logger.info(f"Source sampling: cache/sampling/source_docs/")
    logger.info(f"Embeddings: cache/sampling/embeddings/")
    logger.info(f"Clustering: cache/sampling/clusters/")
    
    logger.info(f"\nSampling complete. Cache organized by module and parameters.")
    
    return all_samples, comparison


# Execute main function if called directly
if __name__ == "__main__":
    main()