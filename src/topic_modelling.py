# topic_modeling.py
import os
import logging
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
from enum import Enum
from tqdm import tqdm
import torch
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from datasets import load_dataset

# Import from other modules
from sampling import (
    Sampler, 
    SamplingMethod,
    DimensionReductionMethod,
    ClusteringMethod,
    RepresentativeSelectionMethod
)
from search import (
    TextPreprocessor, 
    IndexManager,
    SearchEngine,
    RetrievalMethod,
    HybridStrategy
)
from query_expansion import (
    QueryExpander, 
    QueryExpansionMethod, 
    QueryCombinationStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TopicModelingMethod(str, Enum):
    """Enum for topic modeling document sampling methods"""
    DIRECT_RETRIEVAL = "direct_retrieval"
    QUERY_EXPANSION = "query_expansion"
    CLUSTER_AFTER_RETRIEVAL = "cluster_after_retrieval"
    CLUSTER_AFTER_EXPANSION = "cluster_after_expansion"
    FULL_DATASET = "full_dataset"
    UNIFORM_SAMPLING = "uniform_sampling"

class TopicModeler:
    """Class for topic modeling using various document sampling strategies"""
    
    def __init__(
        self,
        corpus_dataset,
        queries_dataset=None,
        qrels_dataset=None,
        embedding_model_name: str = "all-mpnet-base-v2",
        cache_dir: str = "cache/topic_modeling",
        random_seed: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize the topic modeler
        
        Args:
            corpus_dataset: Dataset containing corpus documents
            queries_dataset: Dataset containing queries (optional, needed for query-based methods)
            qrels_dataset: Dataset containing relevance judgments (optional)
            embedding_model_name: Name of the embedding model to use
            cache_dir: Directory for caching results
            random_seed: Random seed for reproducibility
            device: Device to use for embeddings (None for auto-selection)
        """
        self.corpus_dataset = corpus_dataset
        self.queries_dataset = queries_dataset
        self.qrels_dataset = qrels_dataset
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        self.random_seed = random_seed
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        
        # Initialize preprocessor and index manager
        self.preprocessor = TextPreprocessor()
        self.index_manager = IndexManager(self.preprocessor)
        
        # Initialize sampler
        self.sampler = Sampler(
            corpus_dataset=corpus_dataset,
            queries_dataset=queries_dataset,
            qrels_dataset=qrels_dataset,
            preprocessor=self.preprocessor,
            index_manager=self.index_manager,
            cache_dir=cache_dir,
            embedding_model_name=embedding_model_name,
            random_seed=random_seed
        )
        
        # Initialize query expander
        self.query_expander = QueryExpander(
            preprocessor=self.preprocessor,
            index_manager=self.index_manager,
            cache_dir=cache_dir,
            sbert_model_name=embedding_model_name
        )
        
        # Initialize search indices if needed
        self._initialize_indices()
    
    def _initialize_indices(self):
        """Initialize search indices"""
        # Build BM25 index
        self.bm25, self.corpus_texts, self.corpus_ids = self.index_manager.build_bm25_index(
            self.corpus_dataset,
            cache_path=os.path.join(self.cache_dir, "bm25_index.pkl"),
            force_reindex=False
        )
        
        # Build SBERT index
        self.sbert_model, self.doc_embeddings = self.index_manager.build_sbert_index(
            self.corpus_texts,
            model_name=self.embedding_model_name,
            batch_size=64,
            cache_path=os.path.join(self.cache_dir, "sbert_index.pt"),
            force_reindex=False
        )
        
        # Initialize search engine
        self.search_engine = SearchEngine(
            preprocessor=self.preprocessor,
            bm25=self.bm25,
            corpus_texts=self.corpus_texts,
            corpus_ids=self.corpus_ids,
            sbert_model=self.sbert_model,
            doc_embeddings=self.doc_embeddings
        )
    
    def _get_cache_path(
        self, 
        prefix: str,
        method: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Generate a cache path based on prefix and configuration
        
        Args:
            prefix: Prefix for the cache file
            method: Method name
            config: Configuration dictionary
            
        Returns:
            Cache file path
        """
        # Create a string representation of key config parameters
        config_str = "_".join([f"{k}={v}" for k, v in sorted(config.items()) 
                             if isinstance(v, (str, int, float, bool))
                             and k in ['sample_size', 'min_cluster_size', 'n_components', 
                                      'mmr_diversity', 'use_guided', 'top_k']])
        
        # Create the filename
        filename = f"{prefix}_{method}_{config_str}.pkl"
        
        return os.path.join(self.cache_dir, filename)
    
    def sample_documents(
        self,
        method: TopicModelingMethod,
        query_id: Optional[str] = None,
        query_text: Optional[str] = None,
        sample_size: int = 1000,
        retrieval_config: Optional[Dict[str, Any]] = None,
        expansion_config: Optional[Dict[str, Any]] = None,
        clustering_config: Optional[Dict[str, Any]] = None,
        force_recompute: bool = False
    ) -> Dict:
        """
        Sample documents using the specified method
        
        Args:
            method: Document sampling method
            query_id: ID of the query (for query-based methods)
            query_text: Text of the query (for query-based methods)
            sample_size: Number of documents to sample
            retrieval_config: Configuration for retrieval-based methods
            expansion_config: Configuration for query expansion
            clustering_config: Configuration for clustering-based methods
            force_recompute: Whether to force recomputation
            
        Returns:
            Dictionary with sampled documents
        """
        # Default configurations
        default_retrieval_config = {
            "retrieval_method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "top_k": sample_size,
            "use_mmr": False,
            "mmr_lambda": 0.5,
            "use_cross_encoder": False
        }
        
        default_expansion_config = {
            "expansion_method": QueryExpansionMethod.KEYBERT,
            "combination_strategy": QueryCombinationStrategy.WEIGHTED_RRF,
            "num_keywords": 5,
            "original_query_weight": 0.7
        }
        
        default_clustering_config = {
            "dim_reduction_method": DimensionReductionMethod.UMAP,
            "n_components": 50,
            "clustering_method": ClusteringMethod.KMEANS,  # Now uses Mini-Batch KMeans
            "rep_selection_method": RepresentativeSelectionMethod.CENTROID,
            "n_per_cluster": 1,
            "min_cluster_size": 5,
            # Mini-Batch KMeans specific parameters
            "batch_size": 1000,
            "max_iter": 100,
            "n_init": 3,
            "max_no_improvement": 10,
            "reassignment_ratio": 0.01
        }
        
        # Merge with provided configurations
        if retrieval_config:
            default_retrieval_config.update(retrieval_config)
        retrieval_config = default_retrieval_config
        
        if expansion_config:
            default_expansion_config.update(expansion_config)
        expansion_config = default_expansion_config
        
        if clustering_config:
            default_clustering_config.update(clustering_config)
        clustering_config = default_clustering_config
        
        # Generate cache path
        cache_key = f"{method.value}"
        if query_id:
            cache_key = f"query_{query_id}_{cache_key}"
        
        config = {
            "method": method.value,
            "sample_size": sample_size
        }
        
        if method in [TopicModelingMethod.DIRECT_RETRIEVAL, TopicModelingMethod.QUERY_EXPANSION,
                     TopicModelingMethod.CLUSTER_AFTER_RETRIEVAL, TopicModelingMethod.CLUSTER_AFTER_EXPANSION]:
            config.update({
                "retrieval_method": retrieval_config["retrieval_method"].value 
                    if isinstance(retrieval_config["retrieval_method"], Enum) else retrieval_config["retrieval_method"],
                "hybrid_strategy": retrieval_config["hybrid_strategy"].value
                    if isinstance(retrieval_config["hybrid_strategy"], Enum) else retrieval_config["hybrid_strategy"],
                "top_k": retrieval_config["top_k"],
                "use_mmr": retrieval_config["use_mmr"],
                "use_cross_encoder": retrieval_config["use_cross_encoder"]
            })
        
        if method in [TopicModelingMethod.QUERY_EXPANSION, TopicModelingMethod.CLUSTER_AFTER_EXPANSION]:
            config.update({
                "expansion_method": expansion_config["expansion_method"].value
                    if isinstance(expansion_config["expansion_method"], Enum) else expansion_config["expansion_method"],
                "combination_strategy": expansion_config["combination_strategy"].value
                    if isinstance(expansion_config["combination_strategy"], Enum) else expansion_config["combination_strategy"],
                "num_keywords": expansion_config["num_keywords"],
                "original_query_weight": expansion_config["original_query_weight"]
            })
        
        if method in [TopicModelingMethod.CLUSTER_AFTER_RETRIEVAL, TopicModelingMethod.CLUSTER_AFTER_EXPANSION]:
            config.update({
                "dim_reduction_method": clustering_config["dim_reduction_method"].value
                    if isinstance(clustering_config["dim_reduction_method"], Enum) else clustering_config["dim_reduction_method"],
                "n_components": clustering_config["n_components"],
                "clustering_method": clustering_config["clustering_method"].value
                    if isinstance(clustering_config["clustering_method"], Enum) else clustering_config["clustering_method"],
                "rep_selection_method": clustering_config["rep_selection_method"].value
                    if isinstance(clustering_config["rep_selection_method"], Enum) else clustering_config["rep_selection_method"],
                "min_cluster_size": clustering_config["min_cluster_size"]
            })
        
        cache_path = self._get_cache_path("sample", cache_key, config)
        
        # Check cache
        if not force_recompute and os.path.exists(cache_path):
            logger.info(f"Loading sampled documents from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        # Handle query-based validation
        if method in [TopicModelingMethod.DIRECT_RETRIEVAL, TopicModelingMethod.QUERY_EXPANSION, 
                     TopicModelingMethod.CLUSTER_AFTER_RETRIEVAL, TopicModelingMethod.CLUSTER_AFTER_EXPANSION]:
            if query_id is None or query_text is None:
                raise ValueError("query_id and query_text must be provided for query-based methods")
        
        logger.info(f"Sampling documents with method: {method.value}")
        
        # Sample using the specified method
        if method == TopicModelingMethod.DIRECT_RETRIEVAL:
            # Direct retrieval
            result = self.sampler.sample_retrieval(
                sample_size=sample_size,
                retrieval_method=retrieval_config["retrieval_method"],
                hybrid_strategy=retrieval_config["hybrid_strategy"],
                top_k=retrieval_config["top_k"],
                use_mmr=retrieval_config["use_mmr"],
                use_cross_encoder=retrieval_config["use_cross_encoder"],
                mmr_lambda=retrieval_config["mmr_lambda"],
                force_regenerate=force_recompute
            )
            
        elif method == TopicModelingMethod.QUERY_EXPANSION:
            # Query expansion
            result = self.sampler.sample_query_expansion(
                sample_size=sample_size,
                expansion_method=expansion_config["expansion_method"],
                combination_strategy=expansion_config["combination_strategy"],
                num_keywords=expansion_config["num_keywords"],
                retrieval_method=retrieval_config["retrieval_method"],
                hybrid_strategy=retrieval_config["hybrid_strategy"],
                top_k=retrieval_config["top_k"],
                use_mmr=retrieval_config["use_mmr"],
                use_cross_encoder=retrieval_config["use_cross_encoder"],
                original_query_weight=expansion_config["original_query_weight"],
                force_regenerate=force_recompute
            )
            
        elif method == TopicModelingMethod.CLUSTER_AFTER_RETRIEVAL:
            # Cluster after retrieval
            source_config = {
                "sample_size": retrieval_config["top_k"],
                "retrieval_method": retrieval_config["retrieval_method"],
                "hybrid_strategy": retrieval_config["hybrid_strategy"],
                "top_k": retrieval_config["top_k"],
                "use_mmr": retrieval_config["use_mmr"],
                "use_cross_encoder": retrieval_config["use_cross_encoder"]
            }
            
            result = self.sampler.sample_clustering(
                sample_size=sample_size,
                source_method=SamplingMethod.RETRIEVAL,
                source_config=source_config,
                dim_reduction_method=clustering_config["dim_reduction_method"],
                n_components=clustering_config["n_components"],
                clustering_method=clustering_config["clustering_method"],
                rep_selection_method=clustering_config["rep_selection_method"],
                n_per_cluster=clustering_config["n_per_cluster"],
                min_cluster_size=clustering_config["min_cluster_size"],
                batch_size=clustering_config.get("batch_size", 1000),
                max_iter=clustering_config.get("max_iter", 100),
                n_init=clustering_config.get("n_init", 3),
                max_no_improvement=clustering_config.get("max_no_improvement", 10),
                reassignment_ratio=clustering_config.get("reassignment_ratio", 0.01),
                force_recompute=force_recompute
            )
            
        elif method == TopicModelingMethod.CLUSTER_AFTER_EXPANSION:
            # Cluster after query expansion
            source_config = {
                "sample_size": retrieval_config["top_k"],
                "expansion_method": expansion_config["expansion_method"],
                "combination_strategy": expansion_config["combination_strategy"],
                "num_keywords": expansion_config["num_keywords"],
                "retrieval_method": retrieval_config["retrieval_method"],
                "hybrid_strategy": retrieval_config["hybrid_strategy"],
                "top_k": retrieval_config["top_k"],
                "use_mmr": retrieval_config["use_mmr"],
                "use_cross_encoder": retrieval_config["use_cross_encoder"]
            }
            
            result = self.sampler.sample_clustering(
                sample_size=sample_size,
                source_method=SamplingMethod.QUERY_EXPANSION,
                source_config=source_config,
                dim_reduction_method=clustering_config["dim_reduction_method"],
                n_components=clustering_config["n_components"],
                clustering_method=clustering_config["clustering_method"],
                rep_selection_method=clustering_config["rep_selection_method"],
                n_per_cluster=clustering_config["n_per_cluster"],
                min_cluster_size=clustering_config["min_cluster_size"],
                force_recompute=force_recompute
            )
            
        elif method == TopicModelingMethod.FULL_DATASET:
            # Full dataset
            result = self.sampler.sample_full_dataset()
            
        elif method == TopicModelingMethod.UNIFORM_SAMPLING:
            # Uniform sampling
            result = self.sampler.sample_uniform(sample_size=sample_size)
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Add query information if applicable
        if query_id is not None and query_text is not None:
            result["query_id"] = query_id
            result["query_text"] = query_text
        
        # Cache result
        logger.info(f"Caching sampled documents to: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def run_bertopic(
        self,
        documents: List[str],
        method_name: str,
        query_id: Optional[str] = None,
        query_text: Optional[str] = None,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        metric: str = 'cosine',
        min_cluster_size: int = 10,
        min_samples: Optional[int] = None,
        mmr_diversity: float = 0.3,
        nr_topics: Optional[int] = None,
        use_guided: bool = True,
        hierarchical_topics: bool = False,
        reduce_frequent_words: bool = True,
        stop_words: str = 'english',
        verbose: bool = False,
        force_recompute: bool = False,
        use_svd: bool = True,
        max_documents: Optional[int] = None  # Changed to Optional, None means no limit
    ) -> Dict:
        """
        Run BERTopic on a set of documents
        
        Args:
            documents: List of document texts
            method_name: Method name for caching
            query_id: ID of the query (optional)
            query_text: Text of the query (optional, used for guided topic modeling)
            n_components: Number of components for UMAP
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            metric: Metric for UMAP
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            mmr_diversity: Diversity for MaximalMarginalRelevance
            nr_topics: Number of topics (None for auto)
            use_guided: Whether to use guided topic modeling
            hierarchical_topics: Whether to compute hierarchical topics
            reduce_frequent_words: Whether to reduce frequent words
            stop_words: Stop words for vectorizer
            verbose: Whether to show verbose output
            force_recompute: Whether to force recomputation
            use_svd: Whether to use SVD instead of UMAP (faster)
            max_documents: Maximum number of documents to process (None for no limit)
            
        Returns:
            Dictionary with topic modeling results
        """
        # Create config for cache path
        config = {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples if min_samples is not None else (min_cluster_size // 2),
            "mmr_diversity": mmr_diversity,
            "use_guided": use_guided,
            "nr_topics": nr_topics,
            "hierarchical_topics": hierarchical_topics,
            "reduce_frequent_words": reduce_frequent_words,
            "use_svd": use_svd
        }
        
        # Generate cache path
        cache_prefix = f"topic_model"
        if query_id:
            cache_prefix = f"{cache_prefix}_query_{query_id}"
        
        cache_path = self._get_cache_path(cache_prefix, method_name, config)
        
        # Check cache
        if not force_recompute and os.path.exists(cache_path):
            logger.info(f"Loading topic model results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        # Ensure we have enough documents to work with
        if len(documents) == 0:
            logger.error("No documents provided for topic modeling")
            raise ValueError("No documents provided for topic modeling")
        
        # Only limit documents if max_documents is explicitly set
        if max_documents is not None and len(documents) > max_documents:
            logger.warning(f"Limiting to {max_documents} documents out of {len(documents)}")
            # Sample systematically rather than randomly to ensure diversity
            step = len(documents) // max_documents
            documents = documents[::step][:max_documents]
        else:
            logger.info(f"Using full dataset with {len(documents)} documents")
        
        logger.info(f"Running BERTopic on {len(documents)} documents with method: {method_name}")
        
        # Force matplotlib to use non-interactive backend
        import matplotlib
        matplotlib.use('Agg')
        
        # Prepare dimensionality reduction model
        if use_svd:
            # Use TruncatedSVD (much faster than UMAP)
            from sklearn.decomposition import TruncatedSVD
            svd_components = min(min(100, len(documents) - 1), 50)  # Ensure we don't exceed document count-1
            logger.info(f"Using TruncatedSVD with {svd_components} components")
            dimensionality_reduction = TruncatedSVD(n_components=svd_components, random_state=self.random_seed)
        else:
            # Use UMAP (slower but potentially better quality)
            try:
                from umap import UMAP
                umap_model = UMAP(
                    n_components=min(n_components, len(documents) - 1),  # Ensure n_components doesn't exceed docs-1
                    n_neighbors=min(n_neighbors, len(documents) - 1),  # Ensure n_neighbors doesn't exceed docs-1
                    min_dist=min_dist,
                    metric=metric,
                    random_state=self.random_seed
                )
                dimensionality_reduction = umap_model
            except ImportError:
                logger.warning("UMAP not available, falling back to SVD")
                from sklearn.decomposition import TruncatedSVD
                svd_components = min(min(100, len(documents) - 1), 50)
                dimensionality_reduction = TruncatedSVD(n_components=svd_components, random_state=self.random_seed)
        
        # Prepare seed topic list if using guided topic modeling and query text is provided
        seed_topic_list = None
        if use_guided and query_text:
            seed_topic_list = [[query_text]]
        
        # Prepare other components
        vectorizer_model = CountVectorizer(stop_words=stop_words)
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=reduce_frequent_words)
        representation_model = MaximalMarginalRelevance(diversity=mmr_diversity)
        
        # Initialize BERTopic model
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=dimensionality_reduction,
            hdbscan_model=None,  # Use default HDBSCAN
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            min_topic_size=min_cluster_size,
            nr_topics=nr_topics,
            seed_topic_list=seed_topic_list,
            calculate_probabilities=True,
            verbose=verbose
        )
        
        # Encode documents in batches to reduce memory usage
        logger.info("Encoding documents...")
        document_embeddings = []
        documents_per_batch = 32
        
        for i in tqdm(range(0, len(documents), documents_per_batch)):
            batch = documents[i:i + documents_per_batch]
            try:
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device=self.device
                )
                # Move to CPU to save GPU memory
                document_embeddings.append(batch_embeddings.cpu().numpy())
            except Exception as e:
                logger.error(f"Error encoding batch {i}: {str(e)}")
                raise
        
        # Combine batches
        document_embeddings = np.vstack(document_embeddings)
        
        # Fit the model
        logger.info("Fitting BERTopic model...")
        try:
            topics, probs = topic_model.fit_transform(
                documents=documents,
                embeddings=document_embeddings
            )
        except Exception as e:
            logger.error(f"Error fitting BERTopic model: {str(e)}")
            raise
        
        # Handle case where probs might be None
        if probs is None:
            logger.warning("Probabilities are None, using dummy values")
            probs = np.zeros((len(documents), 1))
            
        # Get topic information
        topic_info = topic_model.get_topic_info()
        topic_words = {t: topic_model.get_topic(t) for t in set(topics) if t != -1}
        
        # Create document info dataframe with safer probability handling
        if len(probs.shape) > 1:
            # Multi-dimensional probabilities
            max_probs = probs.max(axis=1)
        else:
            # One-dimensional probabilities
            max_probs = probs
        
        document_info = pd.DataFrame({
            "Document": documents,
            "Topic": topics,
            "Probability": max_probs
        })
        
        # Generate visualizations with explicit sizes
        visualizations = {}
        try:
            import plotly.io as pio
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import io
            
            # Fix for "arrays used as indices must be of integer type" error
            # Create custom topic visualization instead of using BERTopic's visualize_topics
            if len(topic_info) > 0:
                # Get only valid topics (not -1 which is outliers)
                valid_topics = topic_info[topic_info["Topic"] >= 0]
                
                if len(valid_topics) > 0:
                    # Create a simple topics plot
                    fig = px.scatter(
                        x=range(len(valid_topics)), 
                        y=valid_topics["Count"],
                        text=valid_topics["Topic"],
                        size=valid_topics["Count"],
                        title="Topic Distribution"
                    )
                    fig.update_traces(textposition='top center')
                    
                    buf = io.BytesIO()
                    fig.write_image(buf, format='png', scale=2, width=1200, height=800)
                    buf.seek(0)
                    visualizations["topics"] = buf.getvalue()
            
            # Fix for "rows argument must be int greater than 0" error
            # Create custom barchart instead of using BERTopic's visualize_barchart
            if len(topic_info) > 0:
                valid_topics = topic_info[topic_info["Topic"] >= 0]
                
                if len(valid_topics) > 0:
                    # Sort by Count
                    valid_topics = valid_topics.sort_values("Count", ascending=False)
                    
                    # Take top 20 or fewer
                    top_n = min(20, len(valid_topics))
                    top_topics = valid_topics.head(top_n)
                    
                    # Create barchart
                    fig = px.bar(
                        top_topics,
                        x="Count",
                        y=top_topics["Topic"].astype(str),
                        orientation='h',
                        title="Top Topics by Document Count"
                    )
                    
                    buf = io.BytesIO()
                    fig.write_image(buf, format='png', scale=2, width=1200, height=800)
                    buf.seek(0)
                    visualizations["barchart"] = buf.getvalue()
            
        except Exception as e:
            # If Plotly fails, use Matplotlib as fallback
            try:
                import matplotlib.pyplot as plt
                import io
                
                if len(topic_info) > 0:
                    valid_topics = topic_info[topic_info["Topic"] >= 0]
                    
                    if len(valid_topics) > 0:
                        # Sort by Count
                        valid_topics = valid_topics.sort_values("Count", ascending=False)
                        
                        # Take top 20 or fewer
                        top_n = min(20, len(valid_topics))
                        top_topics = valid_topics.head(top_n)
                        
                        # Create barchart
                        plt.figure(figsize=(12, 8))
                        plt.barh(top_topics["Topic"].astype(str), top_topics["Count"])
                        plt.xlabel("Document Count")
                        plt.ylabel("Topic")
                        plt.title("Top Topics by Document Count")
                        
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        visualizations["barchart_matplotlib"] = buf.getvalue()
                        plt.close()
            except Exception:
                pass
        
        # Prepare result
        result = {
            "method": method_name,
            "config": config,
            "topic_model": topic_model,
            "topics": topics,
            "probabilities": probs,
            "topic_info": topic_info,
            "topic_words": topic_words,
            "document_info": document_info,
            "visualizations": visualizations,
            "num_documents": len(documents),
            "num_topics": len(topic_words)
        }
        
        # Add query information if provided
        if query_id:
            result["query_id"] = query_id
        
        if query_text:
            result["query_text"] = query_text
        
        # Cache result
        logger.info(f"Caching topic model results to: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def run_topic_modeling(
        self,
        method: TopicModelingMethod,
        query_id: Optional[str] = None,
        query_text: Optional[str] = None,
        sample_size: int = 1000,
        retrieval_config: Optional[Dict[str, Any]] = None,
        expansion_config: Optional[Dict[str, Any]] = None,
        clustering_config: Optional[Dict[str, Any]] = None,
        topic_model_config: Optional[Dict[str, Any]] = None,
        force_sample: bool = False,
        force_topic_model: bool = False
    ) -> Dict:
        """
        Run complete topic modeling pipeline - sample documents and run BERTopic
        
        Args:
            method: Document sampling method
            query_id: ID of the query (for query-based methods)
            query_text: Text of the query (for query-based methods)
            sample_size: Number of documents to sample
            retrieval_config: Configuration for retrieval-based methods
            expansion_config: Configuration for query expansion
            clustering_config: Configuration for clustering-based methods
            topic_model_config: Configuration for BERTopic
            force_sample: Whether to force recomputing document sampling
            force_topic_model: Whether to force recomputing topic modeling
            
        Returns:
            Dictionary with topic modeling results
        """
        # Default topic model configuration
        default_topic_model_config = {
            "n_components": 5,
            "n_neighbors": 15,
            "min_dist": 0.0,
            "metric": "cosine",
            "min_cluster_size": 10,
            "min_samples": None,
            "mmr_diversity": 0.3,
            "nr_topics": None,
            "use_guided": True,
            "hierarchical_topics": False,
            "reduce_frequent_words": True,
            "stop_words": "english",
            "verbose": True,
            "use_svd": True,  # Default to using SVD
            "max_documents": 10000  # Default max documents
        }
        
        # Merge with provided configuration
        if topic_model_config:
            default_topic_model_config.update(topic_model_config)
        topic_model_config = default_topic_model_config
        
        # Sample documents
        sample_result = self.sample_documents(
            method=method,
            query_id=query_id,
            query_text=query_text,
            sample_size=sample_size,
            retrieval_config=retrieval_config,
            expansion_config=expansion_config,
            clustering_config=clustering_config,
            force_recompute=force_sample
        )
        
        # Run BERTopic
        topic_model_result = self.run_bertopic(
            documents=sample_result["sample_texts"],
            method_name=method.value,
            query_id=query_id,
            query_text=query_text,
            n_components=topic_model_config["n_components"],
            n_neighbors=topic_model_config["n_neighbors"],
            min_dist=topic_model_config["min_dist"],
            metric=topic_model_config["metric"],
            min_cluster_size=topic_model_config["min_cluster_size"],
            min_samples=topic_model_config["min_samples"],
            mmr_diversity=topic_model_config["mmr_diversity"],
            nr_topics=topic_model_config["nr_topics"],
            use_guided=topic_model_config["use_guided"],
            hierarchical_topics=topic_model_config["hierarchical_topics"],
            reduce_frequent_words=topic_model_config["reduce_frequent_words"],
            stop_words=topic_model_config["stop_words"],
            verbose=topic_model_config["verbose"],
            force_recompute=force_topic_model,
            use_svd=topic_model_config.get("use_svd", True),
            max_documents=topic_model_config.get("max_documents", 10000)
        )
        
        # Combine results
        result = {
            "sample_result": sample_result,
            "topic_model_result": topic_model_result,
            "method": method.value,
            "query_id": query_id,
            "query_text": query_text
        }
        
        return result
    
    def run_multi_method_topic_modeling(
        self,
        methods: List[TopicModelingMethod],
        query_id: Optional[str] = None,
        query_text: Optional[str] = None,
        sample_size: int = 1000,
        retrieval_config: Optional[Dict[str, Any]] = None,
        expansion_config: Optional[Dict[str, Any]] = None,
        clustering_config: Optional[Dict[str, Any]] = None,
        topic_model_config: Optional[Dict[str, Any]] = None,
        force_sample: bool = False,
        force_topic_model: bool = False
    ) -> Dict[str, Dict]:
        """
        Run topic modeling with multiple methods
        
        Args:
            methods: List of document sampling methods
            query_id: ID of the query (for query-based methods)
            query_text: Text of the query (for query-based methods)
            sample_size: Number of documents to sample
            retrieval_config: Configuration for retrieval-based methods
            expansion_config: Configuration for query expansion
            clustering_config: Configuration for clustering-based methods
            topic_model_config: Configuration for BERTopic
            force_sample: Whether to force recomputing document sampling
            force_topic_model: Whether to force recomputing topic modeling
            
        Returns:
            Dictionary mapping methods to topic modeling results
        """
        results = {}
        
        for method in methods:
            logger.info(f"Running topic modeling with method: {method.value}")
            
            result = self.run_topic_modeling(
                method=method,
                query_id=query_id,
                query_text=query_text,
                sample_size=sample_size,
                retrieval_config=retrieval_config,
                expansion_config=expansion_config,
                clustering_config=clustering_config,
                topic_model_config=topic_model_config,
                force_sample=force_sample,
                force_topic_model=force_topic_model
            )
            
            results[method.value] = result
        
        return results
    
    def export_topic_models(
        self,
        topic_model_results: Dict[str, Dict],
        output_dir: str = "results/topic_models"
    ) -> Dict[str, str]:
        """
        Export topic modeling results to files with better visualization support
        
        Args:
            topic_model_results: Dictionary mapping methods to topic modeling results
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping methods to output directories
        """
        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get query information if available
        query_id = None
        query_text = None
        
        for result in topic_model_results.values():
            if "query_id" in result:
                query_id = result["query_id"]
            if "query_text" in result:
                query_text = result["query_text"]
            if query_id and query_text:
                break
        
        # Create query directory if applicable
        if query_id:
            base_dir = os.path.join(output_dir, f"query_{query_id}")
            # Export query info
            os.makedirs(base_dir, exist_ok=True)
            with open(os.path.join(base_dir, "query_info.txt"), "w") as f:
                f.write(f"Query ID: {query_id}\n")
                f.write(f"Query Text: {query_text}\n")
        else:
            base_dir = output_dir
        
        output_paths = {}
        
        for method, result in topic_model_results.items():
            # Create method directory
            method_dir = os.path.join(base_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            
            # Extract topic model result
            topic_model_result = result["topic_model_result"]
            
            # Export topic info
            topic_model_result["topic_info"].to_csv(os.path.join(method_dir, "topic_info.csv"), index=False)
            
            # Export document info
            topic_model_result["document_info"].to_csv(os.path.join(method_dir, "document_info.csv"), index=False)
            
            # Export topic words
            with open(os.path.join(method_dir, "topic_words.txt"), "w") as f:
                for topic, words in topic_model_result["topic_words"].items():
                    f.write(f"Topic {topic}:\n")
                    for word, score in words:
                        f.write(f"  {word}: {score:.4f}\n")
                    f.write("\n")
            
            # Export sample info
            sample_result = result["sample_result"]
            with open(os.path.join(method_dir, "sample_info.txt"), "w") as f:
                f.write(f"Method: {sample_result['method']}\n")
                f.write(f"Total documents in corpus: {sample_result['total_docs']}\n")
                f.write(f"Sampled documents: {sample_result['sampled_docs']}\n")
                f.write(f"Sampling rate: {sample_result['sampling_rate']:.4f}\n")
            
            # Export document IDs
            with open(os.path.join(method_dir, "document_ids.txt"), "w") as f:
                for doc_id in sample_result["sample_ids"]:
                    f.write(f"{doc_id}\n")
            
            # Export sample documents (first few)
            with open(os.path.join(method_dir, "sample_documents.txt"), "w") as f:
                for i, (doc_id, text) in enumerate(zip(sample_result["sample_ids"][:5], sample_result["sample_texts"][:5])):
                    f.write(f"Document {i+1} (ID: {doc_id}):\n")
                    f.write(f"{text[:500]}...\n\n")
            
            # Export pre-generated visualizations if available
            vis_dir = os.path.join(method_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            
            if "visualizations" in topic_model_result:
                for vis_name, vis_data in topic_model_result["visualizations"].items():
                    if vis_data:
                        with open(os.path.join(vis_dir, f"{vis_name}.png"), "wb") as f:
                            f.write(vis_data)
            
            # Save full results
            with open(os.path.join(method_dir, "full_result.pkl"), "wb") as f:
                pickle.dump(result, f)
            
            output_paths[method] = method_dir
        
        return output_paths
    
    def process_queries(
        self,
        query_ids: List[str],
        methods: List[TopicModelingMethod] = [
            TopicModelingMethod.DIRECT_RETRIEVAL,
            TopicModelingMethod.QUERY_EXPANSION,
            TopicModelingMethod.CLUSTER_AFTER_RETRIEVAL,
            TopicModelingMethod.CLUSTER_AFTER_EXPANSION,
            TopicModelingMethod.FULL_DATASET,
            TopicModelingMethod.UNIFORM_SAMPLING
        ],
        sample_size: int = 1000,
        retrieval_config: Optional[Dict[str, Any]] = None,
        expansion_config: Optional[Dict[str, Any]] = None,
        clustering_config: Optional[Dict[str, Any]] = None,
        topic_model_config: Optional[Dict[str, Any]] = None,
        force_sample: bool = False,
        force_topic_model: bool = False,
        output_dir: str = "results/topic_models"
    ) -> Dict[str, Dict[str, Dict]]:
        """
        Process multiple queries with multiple methods
        
        Args:
            query_ids: List of query IDs to process
            methods: List of topic modeling methods
            sample_size: Number of documents to sample
            retrieval_config: Configuration for retrieval-based methods
            expansion_config: Configuration for query expansion
            clustering_config: Configuration for clustering-based methods
            topic_model_config: Configuration for BERTopic
            force_sample: Whether to force recomputing document sampling
            force_topic_model: Whether to force recomputing topic modeling
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping query IDs to methods to topic modeling results
        """
        # Validate that queries dataset is available for query-based methods
        if any(method in [TopicModelingMethod.DIRECT_RETRIEVAL, 
                         TopicModelingMethod.QUERY_EXPANSION,
                         TopicModelingMethod.CLUSTER_AFTER_RETRIEVAL,
                         TopicModelingMethod.CLUSTER_AFTER_EXPANSION] for method in methods) and self.queries_dataset is None:
            raise ValueError("queries_dataset must be provided for query-based methods")
        
        # Get query texts
        query_texts = {}
        
        if self.queries_dataset is not None:
            for query_item in self.queries_dataset:
                if isinstance(query_item, dict) and "_id" in query_item and "text" in query_item:
                    # Dict access for Datasets objects
                    q_id = str(query_item["_id"])
                    query_texts[q_id] = query_item["text"]
                elif hasattr(query_item, '_id') and hasattr(query_item, 'text'):
                    # Attribute access for custom objects
                    q_id = str(query_item._id)
                    query_texts[q_id] = query_item.text
        
        # Filter to requested query IDs
        query_texts = {q_id: query_texts[q_id] for q_id in query_ids if q_id in query_texts}
        
        if not query_texts and any(method in [TopicModelingMethod.DIRECT_RETRIEVAL, 
                                            TopicModelingMethod.QUERY_EXPANSION,
                                            TopicModelingMethod.CLUSTER_AFTER_RETRIEVAL,
                                            TopicModelingMethod.CLUSTER_AFTER_EXPANSION] for method in methods):
            raise ValueError(f"No valid queries found among the specified query IDs: {query_ids}")
        
        all_results = {}
        
        # Process each query
        for query_id, query_text in query_texts.items():
            logger.info(f"Processing query {query_id}: '{query_text}'")
            
            # Run all methods for this query
            query_results = self.run_multi_method_topic_modeling(
                methods=methods,
                query_id=query_id,
                query_text=query_text,
                sample_size=sample_size,
                retrieval_config=retrieval_config,
                expansion_config=expansion_config,
                clustering_config=clustering_config,
                topic_model_config=topic_model_config,
                force_sample=force_sample,
                force_topic_model=force_topic_model
            )
            
            # Export results
            query_output_dir = os.path.join(output_dir, f"query_{query_id}")
            self.export_topic_models(query_results, query_output_dir)
            
            # Store results
            all_results[query_id] = query_results
        
        # Handle non-query methods if needed
        if TopicModelingMethod.FULL_DATASET in methods or TopicModelingMethod.UNIFORM_SAMPLING in methods:
            logger.info("Processing non-query methods")
            
            non_query_methods = [m for m in methods if m in [TopicModelingMethod.FULL_DATASET, TopicModelingMethod.UNIFORM_SAMPLING]]
            
            # Run non-query methods
            non_query_results = self.run_multi_method_topic_modeling(
                methods=non_query_methods,
                sample_size=sample_size,
                topic_model_config=topic_model_config,
                force_sample=force_sample,
                force_topic_model=force_topic_model
            )
            
            # Export results
            non_query_output_dir = os.path.join(output_dir, "corpus_level")
            self.export_topic_models(non_query_results, non_query_output_dir)
            
            # Store results
            all_results["corpus_level"] = non_query_results
        
        # Cache overall results
        cache_path = os.path.join(self.cache_dir, "all_topic_model_results.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(all_results, f)
        
        return all_results


def main():
    """Main function to demonstrate topic modeling functionality"""
    # Define constants - CONSISTENT WITH OTHER FILES
    CACHE_DIR = "cache"
    LOG_LEVEL = 'INFO'
    OUTPUT_DIR = "results/topic_models"
    
    # Query processing parameters
    QUERY_IDS = ["43"]  # Example query ID
    
    # Execution control flags - CONSISTENT NAMING
    FORCE_REINDEX = False
    FORCE_SAMPLE = False
    FORCE_TOPIC_MODEL = False
    FORCE_REGENERATE_KEYWORDS = False
    FORCE_REGENERATE_EXPANSION = False
    
    # Core parameters - CONSISTENT WITH OTHER FILES
    SAMPLE_SIZE = 1000
    NUM_KEYWORDS = 10                    # Consistent with query_expansion.py
    TOP_K_RESULTS = 1000                 # Consistent with query_expansion.py
    TOP_N_DOCS_FOR_EXTRACTION = 10       # Consistent with keyword_extraction.py
    DIVERSITY = 0.7                      # Consistent with keyword_extraction.py
    
    # Performance optimization flags
    USE_SVD = True                       # Use SVD instead of UMAP (faster)
    USE_GPU = True                       # Use GPU if available
    MAX_DOCUMENTS = None                 # No document limit
    
    # Embedding model - CONSISTENT ACROSS FILES
    EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
    CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Mini-Batch KMeans clustering parameters (optimized for speed)
    CLUSTERING_PARAMS = {
        "dim_reduction_method": DimensionReductionMethod.UMAP,
        "n_components": 30,              # Reduced for speed
        "clustering_method": ClusteringMethod.KMEANS,
        "rep_selection_method": RepresentativeSelectionMethod.CENTROID,
        "n_per_cluster": 1,
        "min_cluster_size": 5,
        # Mini-Batch KMeans specific parameters
        "batch_size": 2000,              # Larger batch for better performance
        "max_iter": 50,                  # Reduced iterations for speed
        "n_init": 2,                     # Fewer initializations for speed
        "max_no_improvement": 5,         # Early stopping for speed
        "reassignment_ratio": 0.005      # Lower ratio for stability
    }
    
    # Retrieval parameters - CONSISTENT WITH OTHER FILES
    RETRIEVAL_PARAMS = {
        "retrieval_method": RetrievalMethod.HYBRID,
        "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
        "top_k": TOP_K_RESULTS,          # Using consistent variable
        "use_mmr": False,
        "use_cross_encoder": False,
        "mmr_lambda": 0.5
    }
    
    # Query expansion parameters - CONSISTENT WITH query_expansion.py
    EXPANSION_PARAMS = {
        "expansion_method": QueryExpansionMethod.KEYBERT,
        "combination_strategy": QueryCombinationStrategy.WEIGHTED_RRF,
        "num_keywords": NUM_KEYWORDS,     # Using consistent variable
        "original_query_weight": 0.7,
        "diversity": DIVERSITY            # Using consistent variable
    }
    
    # BERTopic parameters (optimized for speed)
    TOPIC_MODEL_PARAMS = {
        "n_components": 5,
        "n_neighbors": 10,               # Reduced for speed
        "min_dist": 0.0,
        "metric": "cosine",
        "min_cluster_size": 5,
        "min_samples": None,
        "mmr_diversity": 0.3,
        "nr_topics": 15,                 # Fewer topics for faster processing
        "use_guided": True,
        "hierarchical_topics": False,
        "reduce_frequent_words": True,
        "stop_words": "english",
        "verbose": True,
        "use_svd": USE_SVD,
        "max_documents": MAX_DOCUMENTS
    }
    
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
    
    # Initialize topic modeler with consistent parameters
    logger.info("Initializing topic modeler...")
    topic_modeler = TopicModeler(
        corpus_dataset=corpus_dataset,
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        embedding_model_name=EMBEDDING_MODEL_NAME,  # Consistent variable
        cache_dir=CACHE_DIR,
        random_seed=42,
        device='cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
    )
    
    # Define methods to use
    methods = [
        TopicModelingMethod.DIRECT_RETRIEVAL,
        TopicModelingMethod.QUERY_EXPANSION,
        TopicModelingMethod.CLUSTER_AFTER_RETRIEVAL,
        TopicModelingMethod.CLUSTER_AFTER_EXPANSION,
        TopicModelingMethod.FULL_DATASET,
        TopicModelingMethod.UNIFORM_SAMPLING
    ]
    
    logger.info("Using optimized Mini-Batch KMeans clustering for speed")
    logger.info(f"Clustering batch size: {CLUSTERING_PARAMS['batch_size']}, max_iter: {CLUSTERING_PARAMS['max_iter']}")
    
    # Process queries
    logger.info("Processing queries...")
    all_results = topic_modeler.process_queries(
        query_ids=QUERY_IDS,
        methods=methods,
        sample_size=SAMPLE_SIZE,
        retrieval_config=RETRIEVAL_PARAMS,
        expansion_config=EXPANSION_PARAMS,
        clustering_config=CLUSTERING_PARAMS,      # Now properly configured
        topic_model_config=TOPIC_MODEL_PARAMS,
        force_sample=FORCE_SAMPLE,
        force_topic_model=FORCE_TOPIC_MODEL,
        output_dir=OUTPUT_DIR
    )
    
    # Performance logging
    logger.info("\n===== PERFORMANCE SUMMARY =====")
    logger.info(f"Mini-Batch KMeans batch size: {CLUSTERING_PARAMS['batch_size']}")
    logger.info(f"Max iterations: {CLUSTERING_PARAMS['max_iter']}")
    logger.info(f"Early stopping at: {CLUSTERING_PARAMS['max_no_improvement']} iterations")
    logger.info(f"Using {'SVD' if USE_SVD else 'UMAP'} for dimensionality reduction")
    
    # Print summary of results
    logger.info("\n===== TOPIC MODELING RESULTS SUMMARY =====")
    logger.info(f"{'Query':<10} {'Method':<30} {'Documents':<10} {'Topics':<10} {'Clustering':<15}")
    logger.info("-" * 80)
    
    for query_id, query_results in all_results.items():
        for method, result in query_results.items():
            sample_result = result["sample_result"]
            topic_model_result = result["topic_model_result"]
            
            num_docs = sample_result["sampled_docs"]
            num_topics = topic_model_result["num_topics"]
            
            # Get clustering info if available
            clustering_info = "N/A"
            if "cluster_info" in sample_result:
                cluster_method = sample_result["cluster_info"].get("method", "N/A")
                if cluster_method == "mini_batch_kmeans":
                    batch_size = sample_result["cluster_info"].get("batch_size", "N/A")
                    n_iter = sample_result["cluster_info"].get("n_iter", "N/A")
                    clustering_info = f"MBK(bs={batch_size},i={n_iter})"
                else:
                    clustering_info = cluster_method
            
            logger.info(f"{query_id:<10} {method:<30} {num_docs:<10} {num_topics:<10} {clustering_info:<15}")
    
    logger.info(f"\nResults exported to {OUTPUT_DIR}")
    
    return all_results

if __name__ == "__main__":
    main()