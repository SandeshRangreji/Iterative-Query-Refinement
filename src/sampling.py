# sampling.py
import os
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
from collections import defaultdict
import pickle
import json
import random
import hashlib
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
from enum import Enum

# GPU imports with fallback
try:
    import cupy as cp
    import cuml
    from cuml.cluster import KMeans as cuMLKMeans
    from cuml.manifold import UMAP as cuMLUMAP
    GPU_AVAILABLE = True
    logger_gpu = logging.getLogger(__name__ + ".gpu")
    logger_gpu.info("GPU libraries (CuPy, cuML) successfully imported")
except ImportError as e:
    GPU_AVAILABLE = False
    cp = None
    cuml = None
    logger_gpu = logging.getLogger(__name__ + ".gpu")
    logger_gpu.warning(f"GPU libraries not available: {e}")

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
    """Enum for sampling methods - aligned with TopicModelingMethod"""
    FULL_DATASET = "full_dataset"
    DIRECT_RETRIEVAL = "direct_retrieval"  # Aligned with TopicModelingMethod
    QUERY_EXPANSION = "query_expansion"
    CLUSTER_AFTER_RETRIEVAL = "cluster_after_retrieval"
    CLUSTER_AFTER_EXPANSION = "cluster_after_expansion"
    UNIFORM_SAMPLING = "uniform_sampling"

# Mapping for backward compatibility and method name consistency
SAMPLING_METHOD_MAPPING = {
    # Old names -> New canonical names
    "retrieval": "direct_retrieval",
    "random": "uniform_sampling",  # Random is essentially uniform sampling
    "clustering": "cluster_after_retrieval",  # Default clustering source
}

class GPUUtils:
    """Utility class for GPU operations and memory management"""
    
    @staticmethod
    def is_gpu_available() -> bool:
        """Check if GPU and required libraries are available"""
        return GPU_AVAILABLE and cp is not None and cuml is not None
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get GPU memory information in GB"""
        if not GPUUtils.is_gpu_available():
            return {"total": 0, "used": 0, "free": 0}
        
        try:
            mempool = cp.get_default_memory_pool()
            device = cp.cuda.Device()
            total_bytes = device.mem_info[1]
            used_bytes = mempool.used_bytes()
            free_bytes = total_bytes - used_bytes
            
            return {
                "total": total_bytes / (1024**3),
                "used": used_bytes / (1024**3), 
                "free": free_bytes / (1024**3)
            }
        except Exception as e:
            logger_gpu.warning(f"Error getting GPU memory info: {e}")
            return {"total": 0, "used": 0, "free": 0}
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU memory cache"""
        if GPUUtils.is_gpu_available():
            try:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                logger_gpu.info("GPU memory cache cleared")
            except Exception as e:
                logger_gpu.warning(f"Error clearing GPU cache: {e}")

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
        corpus_subset_size: Optional[int] = None,
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
            corpus_subset_size: Optional size for corpus subset (None for full corpus)
            random_seed: Random seed for reproducibility
        """
        self.original_corpus_dataset = corpus_dataset
        self.corpus_subset_size = corpus_subset_size
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
        
        # Create corpus subset if specified
        if corpus_subset_size is not None and corpus_subset_size < len(corpus_dataset):
            logger.info(f"Creating corpus subset of size {corpus_subset_size} from {len(corpus_dataset)} documents")
            indices = random.sample(range(len(corpus_dataset)), corpus_subset_size)
            indices.sort()  # Sort for reproducibility
            self.corpus_dataset = [corpus_dataset[i] for i in indices]
            self.corpus_subset_indices = indices
        else:
            self.corpus_dataset = corpus_dataset
            self.corpus_subset_indices = None
        
        # Determine corpus size for cache paths
        if corpus_subset_size is None:
            self.corpus_cache_size = len(corpus_dataset)
        else:
            self.corpus_cache_size = min(corpus_subset_size, len(corpus_dataset))
        
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
        
        # Initialize query expander and keyword extractor
        self._initialize_query_tools()
    
    def _generate_corpus_cache_dir(self) -> str:
        """Generate corpus-specific cache directory"""
        if self.corpus_subset_size is None:
            corpus_size_str = "full"
        else:
            corpus_size_str = f"{self.corpus_cache_size}"
        
        return os.path.join(self.cache_dir, f"corpus{corpus_size_str}")
    
    def _initialize_indices_and_models(self, force_reindex: bool = False):
        """
        Initialize or load indices and models for the corpus subset
        
        Args:
            force_reindex: Whether to force rebuilding indices
        """
        logger.info(f"Initializing indices and models for corpus size: {len(self.corpus_dataset)}...")
        
        # Create a corpus-specific dataset name for automatic cache separation
        if self.corpus_subset_size is None:
            effective_dataset_name = self.dataset_name
        else:
            effective_dataset_name = f"{self.dataset_name}-subset{self.corpus_cache_size}"
        
        # Build BM25 index
        self.bm25, self.corpus_texts, self.corpus_ids = self.index_manager.build_bm25_index(
            self.corpus_dataset,
            dataset_name=effective_dataset_name,
            force_reindex=force_reindex
        )
        
        # Build SBERT index
        self.sbert_model, self.doc_embeddings = self.index_manager.build_sbert_index(
            self.corpus_texts,
            model_name=self.embedding_model_name,
            dataset_name=effective_dataset_name,
            batch_size=64,
            force_reindex=force_reindex
        )
        
        # Create mapping from corpus IDs to indices
        self.corpus_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.corpus_ids)}
    
    def _initialize_query_tools(self):
        """Initialize query expander and keyword extractor"""
        # Initialize keyword extractor
        self.keyword_extractor = KeywordExtractor(
            keybert_model=self.sbert_model,
            cache_dir=self._generate_corpus_cache_dir(),
            top_k_docs=1000,
            top_n_docs_for_extraction=10
        )
        
        # Initialize query expander
        self.query_expander = QueryExpander(
            preprocessor=self.preprocessor,
            index_manager=self.index_manager,
            cache_dir=self._generate_corpus_cache_dir(),
            sbert_model_name=self.embedding_model_name,
            cross_encoder_model_name=self.cross_encoder_model_name
        )
    
    def _get_doc_texts_from_ids(self, doc_ids: List[str]) -> List[str]:
        """Get document texts from document IDs"""
        return [self.corpus_texts[self.corpus_id_to_idx[doc_id]] for doc_id in doc_ids]
    
    def _get_doc_embeddings_from_ids(self, doc_ids: List[str]) -> torch.Tensor:
        """Get document embeddings from document IDs"""
        indices = [self.corpus_id_to_idx[doc_id] for doc_id in doc_ids]
        return self.doc_embeddings[indices]
    
    def _generate_method_cache_path(
        self,
        method: SamplingMethod,
        config: Dict[str, Any],
        query_id: Optional[str] = None
    ) -> str:
        """Generate cache path for sampling method with corpus size and query info"""
        corpus_cache_dir = self._generate_corpus_cache_dir()
        
        # Determine cache subdirectory based on method type
        if query_id is not None:
            # Query-specific methods
            cache_subdir = os.path.join("sampling", "query_specific", f"query_{query_id}")
        else:
            # Corpus-level methods
            cache_subdir = os.path.join("sampling", "corpus_level")
        
        cache_dir = os.path.join(corpus_cache_dir, cache_subdir)
        
        # Generate filename based on method and config
        if method == SamplingMethod.FULL_DATASET:
            filename = "full_dataset.pkl"
        elif method == SamplingMethod.UNIFORM_SAMPLING:
            sample_size = config.get('sample_size', 1000)
            filename = f"uniform_sampling_k{sample_size}.pkl"
        elif method == SamplingMethod.DIRECT_RETRIEVAL:
            retrieval_method = config.get('retrieval_method', 'hybrid')
            if hasattr(retrieval_method, 'value'):
                retrieval_method = retrieval_method.value
            top_k = config.get('top_k', 1000)
            use_mmr = config.get('use_mmr', False)
            use_cross_encoder = config.get('use_cross_encoder', False)
            filename = f"direct_retrieval_{retrieval_method}_k{top_k}_mmr-{str(use_mmr).lower()}_cross-{str(use_cross_encoder).lower()}.pkl"
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
            filename = f"query_expansion_{exp_method}-k{num_kw}-div{diversity}_{strategy}_k{top_k}.pkl"
        elif method in [SamplingMethod.CLUSTER_AFTER_RETRIEVAL, SamplingMethod.CLUSTER_AFTER_EXPANSION]:
            source_method = config.get('source_method', 'full_dataset')
            sample_size = config.get('sample_size', 1000)
            clustering_method = config.get('clustering_method', 'kmeans')
            n_components = config.get('n_components', 50)
            filename = f"{method.value}_{source_method}_k{sample_size}_{clustering_method}_dim{n_components}.pkl"
        else:
            # Fallback
            method_str = method.value if hasattr(method, 'value') else str(method)
            sample_size = config.get('sample_size', 1000)
            filename = f"{method_str}_k{sample_size}.pkl"
        
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
    
    def save_query_centric_registry(self, all_results: Dict[str, Dict[str, Dict]]) -> str:
        """
        Save query-centric registry of all sampling results
        
        Args:
            all_results: Dictionary with structure {query_id: {method_name: result_dict}}
            
        Returns:
            Path to saved registry file
        """
        registry = {}
        
        for query_level, methods_results in all_results.items():
            registry[query_level] = {}
            
            for method_name, result_dict in methods_results.items():
                # Extract cache path from result if available
                cache_path = result_dict.get("cache_path", "")
                
                # Store essential metadata
                registry[query_level][method_name] = {
                    "cache_path": cache_path,
                    "method": result_dict.get("method", method_name),
                    "config": result_dict.get("config", {}),
                    "sampled_docs": result_dict.get("sampled_docs", 0),
                    "total_docs": result_dict.get("total_docs", len(self.corpus_dataset)),
                    "sampling_rate": result_dict.get("sampling_rate", 0.0),
                    "corpus_subset_size": self.corpus_subset_size
                }
        
        # Save registry
        corpus_cache_dir = self._generate_corpus_cache_dir()
        registry_path = os.path.join(corpus_cache_dir, "sampling_registry.json")
        
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
        
        logger.info(f"Saved sampling registry to: {registry_path}")
        return registry_path
    
    def load_query_centric_registry(self) -> Dict:
        """Load query-centric registry"""
        corpus_cache_dir = self._generate_corpus_cache_dir()
        registry_path = os.path.join(corpus_cache_dir, "sampling_registry.json")
        
        if os.path.exists(registry_path):
            with open(registry_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Registry not found at: {registry_path}")
            return {}
    
    def _reconstruct_cache_path(
        self,
        method: str,
        config: Dict,
        query_id: Optional[str] = None
    ) -> str:
        """Fallback method to reconstruct cache path if not in registry"""
        # Convert method name if needed
        canonical_method = SAMPLING_METHOD_MAPPING.get(method, method)
        
        try:
            sampling_method = SamplingMethod(canonical_method)
            return self._generate_method_cache_path(sampling_method, config, query_id)
        except ValueError:
            logger.error(f"Unknown sampling method: {method}")
            return ""
    
    def sample_full_dataset(self) -> Dict:
        """Return the full dataset with no sampling"""
        logger.info(f"Returning full dataset ({len(self.corpus_ids)} documents)...")
        
        # Generate cache path
        config = {"sample_size": len(self.corpus_ids)}
        cache_path = self._generate_method_cache_path(SamplingMethod.FULL_DATASET, config)
        
        result = {
            "method": SamplingMethod.FULL_DATASET.value,
            "config": config,
            "total_docs": len(self.corpus_ids),
            "sampled_docs": len(self.corpus_ids),
            "sampling_rate": 1.0,
            "sample_ids": self.corpus_ids,
            "sample_texts": self.corpus_texts,
            "cache_path": cache_path
        }
        
        return result
    
    def sample_uniform(self, sample_size: int, force_regenerate: bool = False) -> Dict:
        """Perform uniform sampling (stratified by document length)"""
        logger.info(f"Performing uniform sampling (size: {sample_size})...")
        
        config = {"sample_size": sample_size}
        cache_path = self._generate_method_cache_path(SamplingMethod.UNIFORM_SAMPLING, config)
        
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
            "method": SamplingMethod.UNIFORM_SAMPLING.value,
            "config": config,
            "total_docs": len(self.corpus_ids),
            "sampled_docs": len(sampled_ids),
            "sampling_rate": len(sampled_ids) / len(self.corpus_ids),
            "sample_ids": sampled_ids,
            "sample_texts": sampled_texts,
            "sample_indices": sampled_indices,
            "cache_path": cache_path,
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
    
    def sample_direct_retrieval(
        self,
        query_id: str,
        query_text: str,
        sample_size: int,
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        top_k: int = 1000,
        use_mmr: bool = False,
        use_cross_encoder: bool = False,
        mmr_lambda: float = 0.5,
        force_regenerate: bool = False
    ) -> Dict:
        """Sample documents based on direct retrieval results"""
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
        cache_path = self._generate_method_cache_path(SamplingMethod.DIRECT_RETRIEVAL, config, query_id)
        
        # Check cache
        if not force_regenerate and os.path.exists(cache_path):
            logger.info(f"Loading direct retrieval sampling results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Performing direct retrieval sampling for query {query_id} (size: {sample_size})...")
        
        # Perform search
        results = self.search_engine.search(
            query=query_text,
            top_k=top_k,
            method=retrieval_method,
            hybrid_strategy=hybrid_strategy,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            use_cross_encoder=use_cross_encoder
        )
        
        # Select top-k documents if we retrieved more than requested
        if len(results) > sample_size:
            sampled_results = results[:sample_size]
        else:
            sampled_results = results
        
        # Extract document IDs and get texts
        sampled_ids = [doc_id for doc_id, _ in sampled_results]
        sampled_texts = self._get_doc_texts_from_ids(sampled_ids)
        
        # Prepare result
        result = {
            "method": SamplingMethod.DIRECT_RETRIEVAL.value,
            "config": config,
            "query_id": query_id,
            "query_text": query_text,
            "total_docs": len(self.corpus_ids),
            "sampled_docs": len(sampled_ids),
            "sampling_rate": len(sampled_ids) / len(self.corpus_ids),
            "sample_ids": sampled_ids,
            "sample_texts": sampled_texts,
            "cache_path": cache_path,
            "raw_results": results
        }
        
        # Cache results
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def sample_query_expansion(
        self,
        query_id: str,
        query_text: str,
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
        cache_path = self._generate_method_cache_path(SamplingMethod.QUERY_EXPANSION, config, query_id)
        
        # Check cache
        if not force_regenerate and os.path.exists(cache_path):
            logger.info(f"Loading query expansion sampling results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Performing query expansion sampling for query {query_id} (size: {sample_size})...")
        
        # Create single-query dataset for the tools
        single_query_dataset = [{"_id": query_id, "text": query_text}]
        
        # Extract keywords
        all_keywords = self.keyword_extractor.extract_keywords_for_queries(
            queries_dataset=single_query_dataset,
            corpus_dataset=self.corpus_dataset,
            num_keywords=num_keywords,
            diversity=diversity,
            force_regenerate=force_regenerate_keywords,
            sbert_model_name=self.embedding_model_name,
            dataset_name=self.dataset_name
        )
        
        # Run baseline search
        baseline_results = {
            int(query_id): self.search_engine.search(
                query=query_text,
                top_k=top_k,
                method=retrieval_method,
                hybrid_strategy=hybrid_strategy,
                use_mmr=use_mmr,
                use_cross_encoder=use_cross_encoder
            )
        }
        
        # Run query expansion
        expansion_results = self.query_expander.expand_queries(
            queries_dataset=single_query_dataset,
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
            sampled_ids = sorted_doc_ids
        
        # Get texts for sampled documents
        sampled_texts = self._get_doc_texts_from_ids(sampled_ids)
        
        # Prepare result
        result = {
            "method": SamplingMethod.QUERY_EXPANSION.value,
            "config": config,
            "query_id": query_id,
            "query_text": query_text,
            "total_docs": len(self.corpus_ids),
            "retrieved_docs": len(all_doc_ids),
            "sampled_docs": len(sampled_ids),
            "sampling_rate": len(sampled_ids) / len(self.corpus_ids),
            "sample_ids": sampled_ids,
            "sample_texts": sampled_texts,
            "cache_path": cache_path,
            "expansion_results": expansion_results,
            "keywords": all_keywords.get(query_id, [])
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
        query_id: Optional[str] = None,
        query_text: Optional[str] = None,
        dim_reduction_method: DimensionReductionMethod = DimensionReductionMethod.UMAP,
        n_components: int = 50,
        rep_selection_method: RepresentativeSelectionMethod = RepresentativeSelectionMethod.CENTROID,
        use_gpu: bool = True,
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
            query_id: Query ID (for query-based source methods)
            query_text: Query text (for query-based source methods)
            dim_reduction_method: Method for dimensionality reduction
            n_components: Number of components for reduced space
            rep_selection_method: Method for selecting representatives
            use_gpu: Whether to use GPU acceleration
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
        # Create config for this clustering operation
        clustering_config = {
            "sample_size": sample_size,
            "source_method": source_method.value,
            "source_config": source_config or {},
            "dim_reduction_method": dim_reduction_method.value,
            "n_components": n_components,
            "rep_selection_method": rep_selection_method.value,
            "use_gpu": use_gpu,
            "batch_size": batch_size,
            "max_iter": max_iter,
            "n_init": n_init
        }
        
        # Determine method name for cache path
        if source_method == SamplingMethod.DIRECT_RETRIEVAL:
            method_name = SamplingMethod.CLUSTER_AFTER_RETRIEVAL
        elif source_method == SamplingMethod.QUERY_EXPANSION:
            method_name = SamplingMethod.CLUSTER_AFTER_EXPANSION
        else:
            # For other source methods, use generic clustering name
            method_name = SamplingMethod.CLUSTER_AFTER_RETRIEVAL  # Default
        
        # Generate cache path
        cache_path = self._generate_method_cache_path(method_name, clustering_config, query_id)
        
        # Check cache
        if not force_recompute and os.path.exists(cache_path):
            logger.info(f"Loading clustering sampling results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        # Get source documents based on method
        logger.info(f"Getting source documents using {source_method.value}...")
        
        if source_method == SamplingMethod.FULL_DATASET:
            source_result = self.sample_full_dataset()
        elif source_method == SamplingMethod.UNIFORM_SAMPLING:
            source_result = self.sample_uniform(
                sample_size=source_config.get("sample_size", sample_size * 5),
                force_regenerate=force_source_regenerate
            )
        elif source_method == SamplingMethod.DIRECT_RETRIEVAL:
            if query_id is None or query_text is None:
                raise ValueError("query_id and query_text are required for direct retrieval source method")
            source_result = self.sample_direct_retrieval(
                query_id=query_id,
                query_text=query_text,
                sample_size=source_config.get("sample_size", sample_size * 5),
                retrieval_method=source_config.get("retrieval_method", RetrievalMethod.HYBRID),
                hybrid_strategy=source_config.get("hybrid_strategy", HybridStrategy.SIMPLE_SUM),
                top_k=source_config.get("top_k", sample_size * 5),
                use_mmr=source_config.get("use_mmr", False),
                use_cross_encoder=source_config.get("use_cross_encoder", False),
                force_regenerate=force_source_regenerate
            )
        elif source_method == SamplingMethod.QUERY_EXPANSION:
            if query_id is None or query_text is None:
                raise ValueError("query_id and query_text are required for query expansion source method")
            source_result = self.sample_query_expansion(
                query_id=query_id,
                query_text=query_text,
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
            
            corpus_cache_dir = self._generate_corpus_cache_dir()
            reduction_cache_path = os.path.join(
                corpus_cache_dir, "embeddings", 
                f"reduced_{dim_reduction_method.value}_n{n_components}_{source_hash}.npy"
            )
            
            if not force_reduction and os.path.exists(reduction_cache_path):
                logger.info(f"Loading reduced embeddings from cache: {reduction_cache_path}")
                reduced_embeddings = np.load(reduction_cache_path)
            else:
                reduced_embeddings = self._reduce_dimensions(
                    embeddings_np, 
                    method=dim_reduction_method.value, 
                    n_components=n_components,
                    use_gpu=use_gpu
                )
                # Save to cache
                os.makedirs(os.path.dirname(reduction_cache_path), exist_ok=True)
                np.save(reduction_cache_path, reduced_embeddings)
        else:
            reduced_embeddings = embeddings_np
        
        # Perform clustering
        logger.info(f"Clustering with KMeans (n_clusters={n_clusters}, use_gpu={use_gpu})...")
        
        labels, cluster_info = self._cluster_documents(
            reduced_embeddings, 
            n_clusters=n_clusters, 
            use_gpu=use_gpu,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init=n_init,
            max_no_improvement=max_no_improvement,
            reassignment_ratio=reassignment_ratio
        )
        
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
            "method": method_name.value,
            "config": clustering_config,
            "query_id": query_id,
            "query_text": query_text,
            "source_method": source_method.value,
            "source_config": source_config,
            "total_docs": len(self.corpus_ids),
            "source_docs": len(source_doc_ids),
            "sampled_docs": len(sampled_ids),
            "sampling_rate": len(sampled_ids) / len(self.corpus_ids),
            "sample_ids": sampled_ids,
            "sample_texts": sampled_texts,
            "cache_path": cache_path,
            "cluster_info": cluster_info,
            "dim_reduction_info": {
                "method": dim_reduction_method.value,
                "n_components": n_components,
                "use_gpu": use_gpu
            }
        }
        
        # Cache results
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 50,
        use_gpu: bool = True,
        random_state: int = 42
    ) -> np.ndarray:
        """Reduce dimensions of embeddings with GPU support"""
        if method.lower() == "umap":
            if use_gpu and GPUUtils.is_gpu_available():
                try:
                    logger_gpu.info("Using GPU UMAP for dimensionality reduction")
                    # Convert to cupy array
                    embeddings_gpu = cp.asarray(embeddings)
                    
                    # Initialize cuML UMAP
                    reducer = cuMLUMAP(
                        n_components=n_components,
                        metric='cosine',
                        n_neighbors=15,
                        min_dist=0.1,
                        random_state=random_state
                    )
                    reduced_embeddings_gpu = reducer.fit_transform(embeddings_gpu)
                    
                    # Convert back to CPU numpy array
                    reduced_embeddings = cp.asnumpy(reduced_embeddings_gpu)
                    
                    # Clear GPU cache
                    GPUUtils.clear_gpu_cache()
                    
                    return reduced_embeddings
                    
                except Exception as e:
                    logger_gpu.warning(f"GPU UMAP failed, falling back to CPU: {e}")
            
            # Fallback to CPU UMAP
            logger.info("Using CPU UMAP for dimensionality reduction")
            reducer = umap.UMAP(
                n_components=n_components,
                metric='cosine',
                n_neighbors=15,
                min_dist=0.1,
                random_state=random_state
            )
            reduced_embeddings = reducer.fit_transform(embeddings)
            
        elif method.lower() == "pca":
            # PCA (CPU only for now)
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced_embeddings = reducer.fit_transform(embeddings)
            
        else:
            raise ValueError(f"Unknown dimension reduction method: {method}")
        
        return reduced_embeddings
    
    def _cluster_documents(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 100,
        use_gpu: bool = True,
        batch_size: int = 1000,
        max_iter: int = 100,
        n_init: int = 3,
        max_no_improvement: int = 10,
        reassignment_ratio: float = 0.01,
        random_state: int = 42
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cluster documents using KMeans with GPU support
        
        Args:
            embeddings: Document embeddings
            n_clusters: Number of clusters (exact)
            use_gpu: Whether to use GPU acceleration
            batch_size: Batch size for Mini-Batch KMeans
            max_iter: Maximum iterations
            n_init: Number of initializations
            max_no_improvement: Early stopping parameter
            reassignment_ratio: Controls reassignment frequency
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (cluster labels, cluster info dictionary)
        """
        # Adjust batch size based on dataset size for optimal performance
        actual_batch_size = min(batch_size, len(embeddings))
        
        if use_gpu and GPUUtils.is_gpu_available():
            try:
                logger_gpu.info(f"Using GPU KMeans with {n_clusters} clusters, batch_size={actual_batch_size}")
                
                # Convert to cupy array
                embeddings_gpu = cp.asarray(embeddings)
                
                # Initialize cuML KMeans
                clusterer = cuMLKMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    n_init=n_init,
                    max_iter=max_iter
                )
                
                labels_gpu = clusterer.fit_predict(embeddings_gpu)
                
                # Convert results back to CPU
                labels = cp.asnumpy(labels_gpu)
                cluster_centers = cp.asnumpy(clusterer.cluster_centers_)
                
                # Calculate distances for probabilities
                distances_gpu = clusterer.transform(embeddings_gpu)
                distances = cp.asnumpy(distances_gpu)
                
                # Clear GPU cache
                GPUUtils.clear_gpu_cache()
                
                # Calculate cluster assignment probabilities (distance-based, inverse)
                probabilities = 1.0 / (1.0 + distances)
                row_sums = probabilities.sum(axis=1, keepdims=True)
                probabilities = probabilities / (row_sums + 1e-9)
                
                # Prepare cluster info
                cluster_info = {
                    "method": "gpu_kmeans",
                    "n_clusters": n_clusters,
                    "max_iter": max_iter,
                    "n_init": n_init,
                    "cluster_centers": cluster_centers,
                    "cluster_sizes": np.bincount(labels, minlength=n_clusters).tolist(),
                    "probabilities": probabilities,
                    "use_gpu": True
                }
                
                return labels, cluster_info
                
            except Exception as e:
                logger_gpu.warning(f"GPU KMeans failed, falling back to CPU: {e}")
                GPUUtils.clear_gpu_cache()
        
        # Fallback to CPU Mini-Batch KMeans
        logger.info(f"Using CPU Mini-Batch KMeans with {n_clusters} clusters, batch_size={actual_batch_size}")
        
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
        probabilities = probabilities / (row_sums + 1e-9)
        
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
            "n_iter": clusterer.n_iter_,
            "use_gpu": False
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
    
    # ===== CONFIGURATION PARAMETERS =====
    
    # Corpus and query configuration
    CORPUS_SUBSET_SIZE = 10000  # Set to None for full corpus
    QUERY_IDS = ["43", "1"]     # List of query IDs to process
    SAMPLE_SIZE = 1000
    
    # GPU configuration
    USE_GPU = True
    GPU_BATCH_SIZE = 2000
    
    # Caching control flags
    FORCE_REINDEX = False
    FORCE_REGENERATE_KEYWORDS = False
    FORCE_REGENERATE_RETRIEVAL = False
    FORCE_REGENERATE_SOURCE = False
    FORCE_RECOMPUTE_REDUCTION = False
    FORCE_RECOMPUTE_CLUSTERING = False
    
    # Dataset parameters
    DATASET_NAME = 'trec-covid'
    
    # Keyword extraction parameters
    NUM_KEYWORDS = 10
    DIVERSITY = 0.7
    
    # Retrieval parameters
    RETRIEVAL_METHOD = RetrievalMethod.HYBRID
    HYBRID_STRATEGY = HybridStrategy.SIMPLE_SUM
    TOP_K_DOCS = 1000
    USE_MMR = False
    USE_CROSS_ENCODER = False
    
    # Clustering parameters
    DIM_REDUCTION_METHOD = DimensionReductionMethod.UMAP
    N_COMPONENTS = 50
    REP_SELECTION_METHOD = RepresentativeSelectionMethod.CENTROID
    
    # GPU-optimized clustering parameters
    BATCH_SIZE = GPU_BATCH_SIZE if USE_GPU else 2000
    MAX_ITER = 50
    N_INIT = 2
    MAX_NO_IMPROVEMENT = 5
    REASSIGNMENT_RATIO = 0.005
    
    # Model parameters
    EMBEDDING_MODEL = 'all-mpnet-base-v2'
    CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    
    # Other parameters
    CACHE_DIR = "cache"
    OUTPUT_DIR = "results/samples"
    LOG_LEVEL = 'INFO'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # GPU availability check
    if USE_GPU:
        if GPUUtils.is_gpu_available():
            gpu_info = GPUUtils.get_gpu_memory_info()
            logger.info(f"GPU acceleration enabled. Memory: {gpu_info['total']:.1f}GB total, {gpu_info['free']:.1f}GB free")
        else:
            logger.warning("GPU requested but not available. Falling back to CPU.")
            USE_GPU = False
    
    # Load datasets
    logger.info("Loading datasets...")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")
    logger.info(f"Loaded {len(corpus_dataset)} documents, {len(queries_dataset)} queries, {len(qrels_dataset)} relevance judgments")
    
    # Initialize sampler with corpus subset
    logger.info("Initializing sampler...")
    sampler = Sampler(
        corpus_dataset=corpus_dataset,
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        cache_dir=CACHE_DIR,
        embedding_model_name=EMBEDDING_MODEL,
        cross_encoder_model_name=CROSS_ENCODER_MODEL,
        dataset_name=DATASET_NAME,
        corpus_subset_size=CORPUS_SUBSET_SIZE,
        random_seed=42
    )
    
    # Dictionary to store all results for registry
    all_results = {}
    
    # Get query texts
    query_texts = {}
    for query_item in queries_dataset:
        if str(query_item["_id"]) in QUERY_IDS:
            query_texts[str(query_item["_id"])] = query_item["text"]
    
    # Process each query individually
    for query_id in QUERY_IDS:
        if query_id not in query_texts:
            logger.warning(f"Query ID {query_id} not found in dataset")
            continue
        
        query_text = query_texts[query_id]
        logger.info(f"\n===== PROCESSING QUERY {query_id}: '{query_text}' =====")
        
        query_results = {}
        
        # 1. Direct Retrieval
        logger.info("Running direct retrieval sampling...")
        direct_result = sampler.sample_direct_retrieval(
            query_id=query_id,
            query_text=query_text,
            sample_size=SAMPLE_SIZE,
            retrieval_method=RETRIEVAL_METHOD,
            hybrid_strategy=HYBRID_STRATEGY,
            top_k=TOP_K_DOCS,
            use_mmr=USE_MMR,
            use_cross_encoder=USE_CROSS_ENCODER,
            force_regenerate=FORCE_REGENERATE_RETRIEVAL
        )
        query_results["direct_retrieval"] = direct_result
        
        # 2. Query Expansion
        logger.info("Running query expansion sampling...")
        expansion_result = sampler.sample_query_expansion(
            query_id=query_id,
            query_text=query_text,
            sample_size=SAMPLE_SIZE,
            expansion_method=QueryExpansionMethod.KEYBERT,
            combination_strategy=QueryCombinationStrategy.WEIGHTED_RRF,
            num_keywords=NUM_KEYWORDS,
            diversity=DIVERSITY,
            retrieval_method=RETRIEVAL_METHOD,
            top_k=TOP_K_DOCS,
            force_regenerate=FORCE_REGENERATE_RETRIEVAL,
            force_regenerate_keywords=FORCE_REGENERATE_KEYWORDS
        )
        query_results["query_expansion"] = expansion_result
        
        # 3. Clustering after Retrieval
        logger.info("Running clustering after retrieval...")
        cluster_retrieval_result = sampler.sample_clustering(
            sample_size=SAMPLE_SIZE,
            source_method=SamplingMethod.DIRECT_RETRIEVAL,
            source_config={
                "sample_size": SAMPLE_SIZE * 5,
                "retrieval_method": RETRIEVAL_METHOD,
                "hybrid_strategy": HYBRID_STRATEGY,
                "top_k": SAMPLE_SIZE * 5,
                "use_mmr": USE_MMR,
                "use_cross_encoder": USE_CROSS_ENCODER
            },
            query_id=query_id,
            query_text=query_text,
            dim_reduction_method=DIM_REDUCTION_METHOD,
            n_components=N_COMPONENTS,
            rep_selection_method=REP_SELECTION_METHOD,
            use_gpu=USE_GPU,
            batch_size=BATCH_SIZE,
            max_iter=MAX_ITER,
            n_init=N_INIT,
            force_recompute=FORCE_RECOMPUTE_CLUSTERING,
            force_source_regenerate=FORCE_REGENERATE_SOURCE,
            force_reduction=FORCE_RECOMPUTE_REDUCTION
        )
        query_results["cluster_after_retrieval"] = cluster_retrieval_result
        
        # 4. Clustering after Expansion
        logger.info("Running clustering after expansion...")
        cluster_expansion_result = sampler.sample_clustering(
            sample_size=SAMPLE_SIZE,
            source_method=SamplingMethod.QUERY_EXPANSION,
            source_config={
                "sample_size": SAMPLE_SIZE * 3,
                "expansion_method": QueryExpansionMethod.KEYBERT,
                "combination_strategy": QueryCombinationStrategy.WEIGHTED_RRF,
                "num_keywords": NUM_KEYWORDS,
                "diversity": DIVERSITY,
                "retrieval_method": RETRIEVAL_METHOD,
                "top_k": SAMPLE_SIZE * 3
            },
            query_id=query_id,
            query_text=query_text,
            dim_reduction_method=DIM_REDUCTION_METHOD,
            n_components=N_COMPONENTS,
            rep_selection_method=REP_SELECTION_METHOD,
            use_gpu=USE_GPU,
            batch_size=BATCH_SIZE,
            max_iter=MAX_ITER,
            n_init=N_INIT,
            force_recompute=FORCE_RECOMPUTE_CLUSTERING,
            force_source_regenerate=FORCE_REGENERATE_SOURCE,
            force_reduction=FORCE_RECOMPUTE_REDUCTION
        )
        query_results["cluster_after_expansion"] = cluster_expansion_result
        
        # Store query results
        all_results[f"query_{query_id}"] = query_results
        
        # Print query summary
        logger.info(f"Query {query_id} completed:")
        for method, result in query_results.items():
            logger.info(f"  {method}: {result['sampled_docs']} documents")
    
    # Process corpus-level methods
    logger.info("\n===== PROCESSING CORPUS-LEVEL METHODS =====")
    corpus_results = {}
    
    # 1. Full Dataset
    logger.info("Running full dataset sampling...")
    full_result = sampler.sample_full_dataset()
    corpus_results["full_dataset"] = full_result
    
    # 2. Uniform Sampling
    logger.info("Running uniform sampling...")
    uniform_result = sampler.sample_uniform(
        sample_size=SAMPLE_SIZE,
        force_regenerate=FORCE_REGENERATE_SOURCE
    )
    corpus_results["uniform_sampling"] = uniform_result
    
    # Store corpus results
    all_results["corpus_level"] = corpus_results
    
    # Print corpus-level summary
    logger.info("Corpus-level methods completed:")
    for method, result in corpus_results.items():
        logger.info(f"  {method}: {result['sampled_docs']} documents")
    
    # Save query-centric registry
    logger.info("\n===== SAVING REGISTRY =====")
    registry_path = sampler.save_query_centric_registry(all_results)
    
    # Performance summary
    logger.info("\n===== PERFORMANCE SUMMARY =====")
    if USE_GPU and GPUUtils.is_gpu_available():
        final_gpu_info = GPUUtils.get_gpu_memory_info()
        logger.info(f"GPU Memory: {final_gpu_info['used']:.1f}GB used, {final_gpu_info['free']:.1f}GB free")
        logger.info(f"GPU clustering batch size: {BATCH_SIZE}")
        logger.info(f"GPU UMAP components: {N_COMPONENTS}")
    
    logger.info(f"Corpus subset size: {CORPUS_SUBSET_SIZE if CORPUS_SUBSET_SIZE else 'full'}")
    logger.info(f"Sample size: {SAMPLE_SIZE}")
    logger.info(f"Processed queries: {len(QUERY_IDS)}")
    logger.info(f"Total sampling methods: {sum(len(methods) for methods in all_results.values())}")
    
    # Print overall results summary
    logger.info("\n===== OVERALL RESULTS SUMMARY =====")
    for query_level, methods in all_results.items():
        logger.info(f"{query_level.upper()}:")
        for method, result in methods.items():
            size = result["sampled_docs"]
            rate = result["sampling_rate"]
            logger.info(f"  {method}: {size} docs ({rate:.4f})")
    
    logger.info(f"\nRegistry saved to: {registry_path}")
    logger.info("Sampling complete with query-centric caching and GPU acceleration!")
    
    return all_results, sampler


# Execute main function if called directly
if __name__ == "__main__":
    main()