# clustering.py
import os
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Optional
from collections import defaultdict
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap
import hdbscan
from tqdm import tqdm

# Import functionality from search.py
from search import (
    TextPreprocessor, 
    IndexManager,
    RetrievalMethod,
    HybridStrategy
)

# Import functionality from query_expansion.py
from query_expansion import QueryExpander

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClusteringSampler:
    """Class for clustering-based document sampling"""
    
    def __init__(
        self,
        corpus_texts: List[str],
        corpus_ids: List[str],
        doc_embeddings: Optional[torch.Tensor] = None,
        embedding_model_name: str = "all-mpnet-base-v2",
        cache_dir: str = "sampling_cache"
    ):
        """
        Initialize clustering sampler
        
        Args:
            corpus_texts: List of document texts
            corpus_ids: List of document IDs
            doc_embeddings: Pre-computed document embeddings (optional)
            embedding_model_name: Model name for embeddings if not provided
            cache_dir: Directory for caching results
        """
        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids
        self.doc_embeddings = doc_embeddings
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize embeddings if not provided
        if doc_embeddings is None:
            self._load_or_compute_embeddings()
    
    def _load_or_compute_embeddings(self):
        """Load cached embeddings or compute new ones"""
        cache_path = os.path.join(self.cache_dir, f"embeddings_{self.embedding_model_name.replace('/', '_')}.pt")
        
        if os.path.exists(cache_path):
            logger.info(f"Loading embeddings from cache: {cache_path}")
            data = torch.load(cache_path)
            self.doc_embeddings = data["doc_embeddings"]
        else:
            logger.info(f"Computing embeddings with model: {self.embedding_model_name}")
            from sentence_transformers import SentenceTransformer
            
            # Select device
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            model = SentenceTransformer(self.embedding_model_name, device=device)
            
            # Compute embeddings with progress bar
            logger.info("Encoding documents...")
            self.doc_embeddings = model.encode(
                self.corpus_texts, 
                batch_size=64, 
                show_progress_bar=True,
                convert_to_tensor=True
            )
            
            # Save to cache
            logger.info(f"Saving embeddings to cache: {cache_path}")
            torch.save({"doc_embeddings": self.doc_embeddings}, cache_path)
    
    def reduce_dimensions(
        self,
        method: str = "umap",
        n_components: int = 50,
        random_state: int = 42,
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Reduce dimensions of document embeddings
        
        Args:
            method: Dimensionality reduction method ('umap' or 'pca')
            n_components: Number of components for reduced space
            random_state: Random state for reproducibility
            force_recompute: Whether to force recomputation
            
        Returns:
            Reduced embeddings as numpy array
        """
        cache_path = os.path.join(
            self.cache_dir, 
            f"reduced_{method}_{n_components}_{self.embedding_model_name.replace('/', '_')}.npy"
        )
        
        if os.path.exists(cache_path) and not force_recompute:
            logger.info(f"Loading reduced embeddings from cache: {cache_path}")
            reduced_embeddings = np.load(cache_path)
            return reduced_embeddings
        
        # Convert to numpy for sklearn compatibility
        embeddings_np = self.doc_embeddings.cpu().numpy()
        
        logger.info(f"Reducing dimensions with {method} to {n_components} components...")
        
        if method.lower() == "umap":
            # UMAP for non-linear dimensionality reduction (better for topic integrity)
            reducer = umap.UMAP(
                n_components=n_components,
                metric='cosine',
                n_neighbors=15,
                min_dist=0.1,
                random_state=random_state
            )
            reduced_embeddings = reducer.fit_transform(embeddings_np)
            
        elif method.lower() == "pca":
            # PCA for linear dimensionality reduction (faster)
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced_embeddings = reducer.fit_transform(embeddings_np)
            
        else:
            raise ValueError(f"Unknown dimension reduction method: {method}")
        
        # Save to cache
        logger.info(f"Saving reduced embeddings to cache: {cache_path}")
        np.save(cache_path, reduced_embeddings)
        
        return reduced_embeddings
    
    def cluster_documents(
        self,
        reduced_embeddings: np.ndarray,
        method: str = "kmeans",
        n_clusters: int = 100,
        random_state: int = 42,
        min_cluster_size: int = 5,
        min_samples: int = None,
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, dict]:
        """
        Cluster documents using the specified algorithm
        
        Args:
            reduced_embeddings: Dimensionality-reduced embeddings
            method: Clustering method ('kmeans' or 'hdbscan')
            n_clusters: Number of clusters (for kmeans)
            random_state: Random state for reproducibility
            min_cluster_size: Minimum cluster size (for hdbscan)
            min_samples: Minimum samples (for hdbscan)
            force_recompute: Whether to force recomputation
            
        Returns:
            Tuple of (cluster labels, cluster info dictionary)
        """
        # Determine cache path based on parameters
        params_str = f"{method}_{n_clusters}_{min_cluster_size}"
        cache_path = os.path.join(
            self.cache_dir, 
            f"clusters_{params_str}_{reduced_embeddings.shape[1]}d.pkl"
        )
        
        if os.path.exists(cache_path) and not force_recompute:
            logger.info(f"Loading clustering results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
            return result["labels"], result["cluster_info"]
        
        logger.info(f"Clustering documents with {method}...")
        
        if method.lower() == "kmeans":
            # KMeans for balanced clusters
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10
            )
            labels = clusterer.fit_predict(reduced_embeddings)
            
            # Prepare cluster info
            cluster_info = {
                "method": "kmeans",
                "n_clusters": n_clusters,
                "inertia": clusterer.inertia_,
                "cluster_centers": clusterer.cluster_centers_,
                "cluster_sizes": np.bincount(labels).tolist()
            }
            
        elif method.lower() == "hdbscan":
            # HDBSCAN for density-based clustering (adaptive)
            if min_samples is None:
                min_samples = min_cluster_size // 2
                
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                gen_min_span_tree=True,
                cluster_selection_method='eom'  # Excess of Mass
            )
            labels = clusterer.fit_predict(reduced_embeddings)
            
            # Count cluster sizes (excluding noise points labeled as -1)
            valid_labels = labels[labels >= 0]
            if len(valid_labels) > 0:
                cluster_sizes = np.bincount(valid_labels).tolist()
            else:
                cluster_sizes = []
            
            # Prepare cluster info
            cluster_info = {
                "method": "hdbscan",
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "num_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "noise_points": np.sum(labels == -1),
                "cluster_sizes": cluster_sizes,
                "outlier_scores": clusterer.outlier_scores_ if hasattr(clusterer, 'outlier_scores_') else None
            }
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Save results to cache
        logger.info(f"Saving clustering results to cache: {cache_path}")
        result = {
            "labels": labels,
            "cluster_info": cluster_info
        }
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return labels, cluster_info
    
    def _get_cluster_representatives(
        self,
        reduced_embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = "centroid",
        n_per_cluster: int = 1
    ) -> List[int]:
        """
        Get representative documents from each cluster
        
        Args:
            reduced_embeddings: Dimensionality-reduced embeddings
            labels: Cluster labels for each document
            method: Method for selecting representatives ('centroid', 'random', 'density')
            n_per_cluster: Number of documents to select per cluster
            
        Returns:
            List of indices of selected documents
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
            if method == "random":
                # Random selection
                np.random.shuffle(cluster_indices)
                selected = cluster_indices[:n_per_cluster]
                
            elif method == "centroid":
                # Select documents closest to cluster centroid
                cluster_points = reduced_embeddings[cluster_indices]
                centroid = np.mean(cluster_points, axis=0)
                
                # Calculate distances to centroid
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                
                # Sort by distance (ascending)
                sorted_indices = np.argsort(distances)
                selected = cluster_indices[sorted_indices[:n_per_cluster]]
                
            elif method == "density":
                # Select documents in densest regions (only for HDBSCAN)
                if hasattr(self, 'clusterer') and hasattr(self.clusterer, 'outlier_scores_'):
                    # Lower outlier scores = higher density
                    outlier_scores = self.clusterer.outlier_scores_[cluster_indices]
                    sorted_indices = np.argsort(outlier_scores)
                    selected = cluster_indices[sorted_indices[:n_per_cluster]]
                else:
                    # Fallback to centroid method
                    cluster_points = reduced_embeddings[cluster_indices]
                    centroid = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    sorted_indices = np.argsort(distances)
                    selected = cluster_indices[sorted_indices[:n_per_cluster]]
            
            else:
                raise ValueError(f"Unknown representative selection method: {method}")
            
            selected_indices.extend(selected)
        
        return selected_indices
    
    def sample_documents(
        self,
        dim_reduction_method: str = "umap",
        n_components: int = 50,
        clustering_method: str = "kmeans",
        n_clusters: int = 100,
        rep_selection_method: str = "centroid",
        n_per_cluster: int = 1,
        min_cluster_size: int = 5,
        random_state: int = 42,
        force_recompute: bool = False
    ) -> Dict:
        """
        Sample documents using clustering
        
        Args:
            dim_reduction_method: Dimensionality reduction method
            n_components: Number of components for reduced space
            clustering_method: Clustering method
            n_clusters: Number of clusters (for kmeans)
            rep_selection_method: Method for selecting representatives
            n_per_cluster: Number of documents to select per cluster
            min_cluster_size: Minimum cluster size (for hdbscan)
            random_state: Random state for reproducibility
            force_recompute: Whether to force recomputation
            
        Returns:
            Dictionary with sampling results
        """
        # Set random seed
        np.random.seed(random_state)
        
        # Reduce dimensions
        reduced_embeddings = self.reduce_dimensions(
            method=dim_reduction_method,
            n_components=n_components,
            random_state=random_state,
            force_recompute=force_recompute
        )
        
        # Cluster documents
        labels, cluster_info = self.cluster_documents(
            reduced_embeddings=reduced_embeddings,
            method=clustering_method,
            n_clusters=n_clusters,
            random_state=random_state,
            min_cluster_size=min_cluster_size,
            force_recompute=force_recompute
        )
        
        # Get representative documents
        selected_indices = self._get_cluster_representatives(
            reduced_embeddings=reduced_embeddings,
            labels=labels,
            method=rep_selection_method,
            n_per_cluster=n_per_cluster
        )
        
        # Get corresponding document IDs and texts
        selected_ids = [self.corpus_ids[idx] for idx in selected_indices]
        selected_texts = [self.corpus_texts[idx] for idx in selected_indices]
        
        # Calculate sampling rate
        sampling_rate = len(selected_indices) / len(self.corpus_texts)
        
        # Prepare result
        result = {
            "dim_reduction_method": dim_reduction_method,
            "n_components": n_components,
            "clustering_method": clustering_method,
            "n_clusters": n_clusters,
            "rep_selection_method": rep_selection_method,
            "n_per_cluster": n_per_cluster,
            "min_cluster_size": min_cluster_size,
            "random_state": random_state,
            "total_docs": len(self.corpus_texts),
            "sampled_docs": len(selected_indices),
            "sampling_rate": sampling_rate,
            "cluster_info": cluster_info,
            "sample_indices": selected_indices,
            "sample_ids": selected_ids,
            "sample_texts": selected_texts,
            "all_labels": labels.tolist()
        }
        
        return result
    
    def save_sample(self, sample_result: Dict, name: str = "sample"):
        """
        Save sampling result to disk
        
        Args:
            sample_result: Sampling result dictionary
            name: Name for the saved sample
        """
        # Create save path
        save_path = os.path.join(
            self.cache_dir, 
            f"{name}_{sample_result['dim_reduction_method']}_{sample_result['clustering_method']}_{sample_result['sampled_docs']}.pkl"
        )
        
        # Save sample
        logger.info(f"Saving sample to: {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(sample_result, f)
    
    def load_sample(self, filepath: str) -> Dict:
        """
        Load saved sample from disk
        
        Args:
            filepath: Path to saved sample
            
        Returns:
            Sampling result dictionary
        """
        logger.info(f"Loading sample from: {filepath}")
        with open(filepath, "rb") as f:
            sample_result = pickle.load(f)
        return sample_result


class RetrievalBasedSampler:
    """Class for sampling documents based on retrieval results"""
    
    def __init__(
        self,
        corpus_dataset,
        queries_dataset,
        preprocessor: Optional[TextPreprocessor] = None,
        embedding_model_name: str = "all-mpnet-base-v2",
        cache_dir: str = "sampling_cache"
    ):
        """
        Initialize retrieval-based sampler
        
        Args:
            corpus_dataset: Dataset containing corpus documents
            queries_dataset: Dataset containing queries
            preprocessor: Text preprocessor instance
            embedding_model_name: Model name for embeddings
            cache_dir: Directory for caching results
        """
        self.corpus_dataset = corpus_dataset
        self.queries_dataset = queries_dataset
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize preprocessor if not provided
        if preprocessor is None:
            self.preprocessor = TextPreprocessor()
        else:
            self.preprocessor = preprocessor
        
        # Initialize index manager
        self.index_manager = IndexManager(self.preprocessor)
        
        # Load or build indices
        self._load_or_build_indices()
    
    def _load_or_build_indices(self):
        """Load or build indices for retrieval"""
        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25, self.corpus_texts, self.corpus_ids = self.index_manager.build_bm25_index(
            self.corpus_dataset,
            cache_path=os.path.join(self.cache_dir, "bm25_index.pkl"),
            force_reindex=False
        )
        
        # Build SBERT index
        logger.info("Building SBERT index...")
        self.sbert_model, self.doc_embeddings = self.index_manager.build_sbert_index(
            self.corpus_texts,
            model_name=self.embedding_model_name,
            batch_size=64,
            cache_path=os.path.join(self.cache_dir, "sbert_index.pt"),
            force_reindex=False
        )
    
    def retrieve_and_expand(
        self,
        query_expansion: bool = True,
        expansion_methods: List[str] = ["keybert"],
        num_terms_per_method: int = 3,
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        top_k: int = 1000,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        cache_name: str = "retrieved_docs"
    ) -> Dict:
        """
        Retrieve documents for all queries and optionally expand queries
        
        Args:
            query_expansion: Whether to use query expansion
            expansion_methods: List of expansion methods to use
            num_terms_per_method: Number of terms to add per method
            retrieval_method: Retrieval method to use
            hybrid_strategy: Strategy for hybrid retrieval
            top_k: Number of documents to retrieve per query
            use_mmr: Whether to use MMR for diversity
            mmr_lambda: Lambda parameter for MMR
            cache_name: Name for caching results
            
        Returns:
            Dictionary with retrieval results
        """
        # Determine cache path
        methods_str = "_".join(expansion_methods) if query_expansion else "none"
        cache_path = os.path.join(
            self.cache_dir, 
            f"{cache_name}_{methods_str}_{retrieval_method}_{top_k}.pkl"
        )
        
        if os.path.exists(cache_path):
            logger.info(f"Loading retrieval results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                result = pickle.load(f)
            return result
        
        # Initialize search engine
        from search import SearchEngine
        search_engine = SearchEngine(self.preprocessor)
        
        # Initialize query expander if needed
        if query_expansion:
            query_expander = QueryExpander(self.corpus_texts, keybert_model=self.embedding_model_name)
        
        # Retrieve documents for each query
        logger.info(f"Retrieving documents for {len(self.queries_dataset)} queries...")
        
        all_retrieved_ids = set()
        retrieved_docs_by_query = {}
        expanded_queries = {}
        
        for query_item in tqdm(self.queries_dataset):
            query_id = query_item["_id"]
            original_query = query_item["text"]
            
            # Expand query if enabled
            if query_expansion:
                expanded_query = query_expander.expand_query(
                    original_query,
                    methods=expansion_methods,
                    num_terms_per_method=num_terms_per_method,
                    deduplicate=True
                )
                expanded_queries[query_id] = expanded_query
            else:
                expanded_query = original_query
                expanded_queries[query_id] = original_query
            
            # Retrieve documents
            search_results = search_engine.search(
                query=expanded_query,
                bm25=self.bm25,
                corpus_texts=self.corpus_texts,
                corpus_ids=self.corpus_ids,
                sbert_model=self.sbert_model,
                doc_embeddings=self.doc_embeddings,
                top_k=top_k,
                method=retrieval_method,
                hybrid_strategy=hybrid_strategy,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda
            )
            
            # Extract document IDs and scores
            retrieved_ids = [doc_id for doc_id, _ in search_results]
            retrieved_scores = [score for _, score in search_results]
            
            # Store results for this query
            retrieved_docs_by_query[query_id] = {
                "ids": retrieved_ids,
                "scores": retrieved_scores
            }
            
            # Add to set of all retrieved IDs
            all_retrieved_ids.update(retrieved_ids)
        
        # Get texts and indices for all retrieved documents
        retrieved_indices = [self.corpus_ids.index(doc_id) for doc_id in all_retrieved_ids]
        retrieved_texts = [self.corpus_texts[idx] for idx in retrieved_indices]
        
        # Prepare result
        result = {
            "query_expansion": query_expansion,
            "expansion_methods": expansion_methods if query_expansion else [],
            "retrieval_method": retrieval_method,
            "hybrid_strategy": hybrid_strategy,
            "top_k": top_k,
            "use_mmr": use_mmr,
            "mmr_lambda": mmr_lambda,
            "total_queries": len(self.queries_dataset),
            "total_docs": len(self.corpus_texts),
            "retrieved_docs": len(all_retrieved_ids),
            "retrieval_rate": len(all_retrieved_ids) / len(self.corpus_texts),
            "retrieved_docs_by_query": retrieved_docs_by_query,
            "expanded_queries": expanded_queries,
            "all_retrieved_ids": list(all_retrieved_ids),
            "retrieved_indices": retrieved_indices,
            "retrieved_texts": retrieved_texts
        }
        
        # Save results
        logger.info(f"Saving retrieval results to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def cluster_retrieved_docs(
        self,
        retrieval_result: Dict,
        dim_reduction_method: str = "umap",
        n_components: int = 50,
        clustering_method: str = "kmeans",
        n_clusters: int = 100,
        rep_selection_method: str = "centroid",
        n_per_cluster: int = 1,
        min_cluster_size: int = 5,
        random_state: int = 42
    ) -> Dict:
        """
        Cluster and sample the retrieved documents
        
        Args:
            retrieval_result: Result from retrieve_and_expand
            dim_reduction_method: Dimensionality reduction method
            n_components: Number of components for reduced space
            clustering_method: Clustering method
            n_clusters: Number of clusters (for kmeans)
            rep_selection_method: Method for selecting representatives
            n_per_cluster: Number of documents to select per cluster
            min_cluster_size: Minimum cluster size (for hdbscan)
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with sampling results
        """
        # Extract retrieved documents
        retrieved_ids = retrieval_result["all_retrieved_ids"]
        retrieved_indices = retrieval_result["retrieved_indices"]
        retrieved_texts = retrieval_result["retrieved_texts"]
        
        logger.info(f"Clustering {len(retrieved_ids)} retrieved documents...")
        
        # Get embeddings for retrieved documents
        retrieved_embeddings = self.doc_embeddings[retrieved_indices]
        
        # Initialize clustering sampler
        sampler = ClusteringSampler(
            corpus_texts=retrieved_texts,
            corpus_ids=retrieved_ids,
            doc_embeddings=retrieved_embeddings,
            embedding_model_name=self.embedding_model_name,
            cache_dir=self.cache_dir
        )
        
        # Sample documents
        sample_result = sampler.sample_documents(
            dim_reduction_method=dim_reduction_method,
            n_components=n_components,
            clustering_method=clustering_method,
            n_clusters=n_clusters,
            rep_selection_method=rep_selection_method,
            n_per_cluster=n_per_cluster,
            min_cluster_size=min_cluster_size,
            random_state=random_state
        )
        
        # Add retrieval info to result
        sample_result["retrieval_info"] = {
            "query_expansion": retrieval_result["query_expansion"],
            "expansion_methods": retrieval_result["expansion_methods"],
            "retrieval_method": retrieval_result["retrieval_method"],
            "top_k": retrieval_result["top_k"],
            "use_mmr": retrieval_result["use_mmr"],
            "total_queries": retrieval_result["total_queries"],
            "total_corpus_docs": retrieval_result["total_docs"],
            "retrieved_docs": retrieval_result["retrieved_docs"],
            "retrieval_rate": retrieval_result["retrieval_rate"]
        }
        
        return sample_result


def main():
    """Main function to demonstrate sampling functionality"""
    # Define constants
    CACHE_DIR = "sampling_cache"
    EMBEDDING_MODEL = 'all-mpnet-base-v2'
    LOG_LEVEL = 'INFO'
    
    # Sampling configuration
    FULL_CORPUS_SAMPLING = {
        "dim_reduction": "umap",  # 'umap' or 'pca'
        "n_components": 50,
        "clustering": "kmeans",  # 'kmeans' or 'hdbscan'
        "n_clusters": 1000,  # For kmeans
        "min_cluster_size": 5,  # For hdbscan
        "selection": "centroid",  # 'centroid', 'random', or 'density'
        "n_per_cluster": 1,
        "random_state": 42
    }
    
    # Retrieval-based sampling configuration
    RETRIEVAL_SAMPLING = {
        "query_expansion": True,
        "expansion_methods": ["keybert", "pmi"],
        "num_terms": 3,
        "retrieval_method": RetrievalMethod.HYBRID,
        "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
        "top_k": 1000,
        "use_mmr": True,
        "mmr_lambda": 0.5,
        # Clustering config for retrieved docs
        "dim_reduction": "umap",
        "n_components": 30,
        "clustering": "kmeans",
        "n_clusters": 500,
        "selection": "centroid",
        "n_per_cluster": 1,
        "random_state": 42
    }
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load datasets
    logger.info("Loading datasets...")
    from datasets import load_dataset
    
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    logger.info(f"Loaded {len(corpus_dataset)} documents, {len(queries_dataset)} queries")
    
    # Initialize preprocessor and indices
    preprocessor = TextPreprocessor()
    index_manager = IndexManager(preprocessor)
    
    # Build indices
    logger.info("Building indices...")
    bm25, corpus_texts, corpus_ids = index_manager.build_bm25_index(
        corpus_dataset,
        cache_path=os.path.join(CACHE_DIR, "bm25_index.pkl"),
        force_reindex=False
    )
    
    sbert_model, doc_embeddings = index_manager.build_sbert_index(
        corpus_texts,
        model_name=EMBEDDING_MODEL,
        batch_size=64,
        cache_path=os.path.join(CACHE_DIR, "sbert_index.pt"),
        force_reindex=False
    )
    
    # Run both sampling methods
    run_full_corpus = True
    run_retrieval_based = True
    
    # 1. Full corpus sampling
    if run_full_corpus:
        logger.info("\n===== FULL CORPUS SAMPLING =====")
        full_sampler = ClusteringSampler(
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            doc_embeddings=doc_embeddings,
            embedding_model_name=EMBEDDING_MODEL,
            cache_dir=CACHE_DIR
        )
        
        full_sample = full_sampler.sample_documents(
            dim_reduction_method=FULL_CORPUS_SAMPLING["dim_reduction"],
            n_components=FULL_CORPUS_SAMPLING["n_components"],
            clustering_method=FULL_CORPUS_SAMPLING["clustering"],
            n_clusters=FULL_CORPUS_SAMPLING["n_clusters"],
            rep_selection_method=FULL_CORPUS_SAMPLING["selection"],
            n_per_cluster=FULL_CORPUS_SAMPLING["n_per_cluster"],
            min_cluster_size=FULL_CORPUS_SAMPLING["min_cluster_size"],
            random_state=FULL_CORPUS_SAMPLING["random_state"]
        )
        
        # Save sample
        full_sampler.save_sample(full_sample, name="full_corpus")
        
        logger.info(f"Full corpus sampling complete: {full_sample['sampled_docs']} documents sampled")
        logger.info(f"Sampling rate: {full_sample['sampling_rate']:.4f}")
    
    # 2. Retrieval-based sampling
    if run_retrieval_based:
        logger.info("\n===== RETRIEVAL-BASED SAMPLING =====")
        retrieval_sampler = RetrievalBasedSampler(
            corpus_dataset=corpus_dataset,
            queries_dataset=queries_dataset,
            preprocessor=preprocessor,
            embedding_model_name=EMBEDDING_MODEL,
            cache_dir=CACHE_DIR
        )
        
        # Retrieve documents
        retrieval_result = retrieval_sampler.retrieve_and_expand(
            query_expansion=RETRIEVAL_SAMPLING["query_expansion"],
            expansion_methods=RETRIEVAL_SAMPLING["expansion_methods"],
            num_terms_per_method=RETRIEVAL_SAMPLING["num_terms"],
            retrieval_method=RETRIEVAL_SAMPLING["retrieval_method"],
            hybrid_strategy=RETRIEVAL_SAMPLING["hybrid_strategy"],
            top_k=RETRIEVAL_SAMPLING["top_k"],
            use_mmr=RETRIEVAL_SAMPLING["use_mmr"],
            mmr_lambda=RETRIEVAL_SAMPLING["mmr_lambda"]
        )
        
        logger.info(f"Retrieved {retrieval_result['retrieved_docs']} documents across all queries")
        logger.info(f"Retrieval rate: {retrieval_result['retrieval_rate']:.4f}")
        
        # Cluster retrieved documents
        retrieved_sample = retrieval_sampler.cluster_retrieved_docs(
            retrieval_result=retrieval_result,
            dim_reduction_method=RETRIEVAL_SAMPLING["dim_reduction"],
            n_components=RETRIEVAL_SAMPLING["n_components"],
            clustering_method=RETRIEVAL_SAMPLING["clustering"],
            n_clusters=RETRIEVAL_SAMPLING["n_clusters"],
            rep_selection_method=RETRIEVAL_SAMPLING["selection"],
            n_per_cluster=RETRIEVAL_SAMPLING["n_per_cluster"],
            random_state=RETRIEVAL_SAMPLING["random_state"]
        )
        
        # Save sample
        retrieval_sampler.sampler.save_sample(retrieved_sample, name="retrieval_based")
        
        logger.info(f"Retrieval-based sampling complete: {retrieved_sample['sampled_docs']} documents sampled")
        logger.info(f"Final sampling rate: {retrieved_sample['sampling_rate']:.4f} of retrieved docs")
        logger.info(f"Overall sampling rate: {retrieved_sample['sampled_docs']/len(corpus_texts):.4f} of corpus")
    
    logger.info("\nSampling complete. Samples saved to cache directory.")


# Execute main function if called directly
if __name__ == "__main__":
    main()