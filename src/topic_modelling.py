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
from keyword_extraction import KeywordExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryBasedTopicModel:
    """Class for running BERTopic on query-based document samples"""
    
    def __init__(
        self,
        embedding_model_name: str = "all-mpnet-base-v2",
        cache_dir: str = "cache",
        device: Optional[str] = None
    ):
        """
        Initialize query-based topic model
        
        Args:
            embedding_model_name: Name of the embedding model to use
            cache_dir: Directory for caching results
            device: Device to use for embeddings
        """
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        
        # Initialize preprocessor and index manager
        self.preprocessor = TextPreprocessor()
        self.index_manager = IndexManager(self.preprocessor)
    
    def _get_cache_path(
        self, 
        query_id: str,
        method: str,
        config: Dict[str, Any]
    ) -> str:
        """
        Generate a cache path based on query ID and configuration
        
        Args:
            query_id: ID of the query
            method: Sampling method name
            config: Configuration dictionary
            
        Returns:
            Cache file path
        """
        # Create a string representation of key config parameters
        config_str = "_".join([f"{k}={v}" for k, v in sorted(config.items()) 
                              if k in ['n_components', 'n_neighbors', 'min_cluster_size', 
                                       'mmr_diversity', 'use_guided', 'top_k']])
        
        # Create the filename
        filename = f"topic_model_query_{query_id}_{method}_{config_str}.pkl"
        
        return os.path.join(self.cache_dir, filename)
    
    def retrieve_documents_for_query(
        self,
        query_id: str,
        query_text: str,
        corpus_dataset,
        method: str = "retrieval",
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        top_k: int = 1000,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        use_cross_encoder: bool = False,
        expansion_method: Optional[QueryExpansionMethod] = None,
        combination_strategy: Optional[QueryCombinationStrategy] = None,
        num_keywords: int = 5,
        keyword_diversity: float = 0.7,
        docs_for_keywords: int = 10,
        original_query_weight: float = 0.7,
        force_recompute: bool = False
    ) -> Dict:
        """
        Retrieve documents for a specific query
        
        Args:
            query_id: ID of the query
            query_text: Text of the query
            corpus_dataset: Dataset containing corpus documents
            method: Retrieval method ("retrieval" or "query_expansion")
            retrieval_method: Method for document retrieval
            hybrid_strategy: Strategy for hybrid retrieval
            top_k: Number of documents to retrieve
            use_mmr: Whether to use MMR for diversity
            mmr_lambda: Lambda parameter for MMR (higher means more relevance, less diversity)
            use_cross_encoder: Whether to use cross-encoder reranking
            expansion_method: Method for query expansion (if method is "query_expansion")
            combination_strategy: Strategy for combining expanded queries
            num_keywords: Number of keywords to extract for query expansion
            keyword_diversity: Diversity parameter for keyword extraction (0-1)
            docs_for_keywords: Number of top documents to use for keyword extraction
            original_query_weight: Weight for original query in RRF combination
            force_recompute: Whether to force recomputation
            
        Returns:
            Dictionary with retrieved documents
        """
        # Create config for cache path
        config = {
            "method": method,
            "retrieval_method": retrieval_method.value,
            "hybrid_strategy": hybrid_strategy.value,
            "top_k": top_k,
            "use_mmr": use_mmr,
            "mmr_lambda": mmr_lambda,
            "use_cross_encoder": use_cross_encoder,
            "num_keywords": num_keywords,
            "keyword_diversity": keyword_diversity,
            "docs_for_keywords": docs_for_keywords,
            "original_query_weight": original_query_weight
        }
        
        if method == "query_expansion" and expansion_method is not None:
            config.update({
                "expansion_method": expansion_method.value,
                "combination_strategy": combination_strategy.value if combination_strategy else "none"
            })
        
        # Generate cache path
        cache_path = os.path.join(self.cache_dir, f"query_{query_id}_{method}_docs.pkl")
        
        # Check cache
        if not force_recompute and os.path.exists(cache_path):
            logger.info(f"Loading retrieved documents from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Retrieving documents for query {query_id}: '{query_text}'")
        
        # Build indices
        bm25, corpus_texts, corpus_ids = self.index_manager.build_bm25_index(
            corpus_dataset,
            cache_path=os.path.join(self.cache_dir, "bm25_index.pkl"),
            force_reindex=False
        )
        
        sbert_model, doc_embeddings = self.index_manager.build_sbert_index(
            corpus_texts,
            model_name=self.embedding_model_name,
            batch_size=64,
            cache_path=os.path.join(self.cache_dir, "sbert_index.pt"),
            force_reindex=False
        )
        
        # Initialize search engine
        search_engine = SearchEngine(
            preprocessor=self.preprocessor,
            bm25=bm25,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            sbert_model=sbert_model,
            doc_embeddings=doc_embeddings
        )
        
        if method == "retrieval":
            # Direct retrieval
            results = search_engine.search(
                query=query_text,
                top_k=top_k,
                method=retrieval_method,
                hybrid_strategy=hybrid_strategy,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
                use_cross_encoder=use_cross_encoder
            )
            
            # Get document texts
            doc_ids = [doc_id for doc_id, _ in results]
            doc_scores = [score for _, score in results]
            doc_texts = []
            
            for doc_id in doc_ids:
                doc_texts.append(search_engine.get_document_by_id(doc_id))
            
        elif method == "query_expansion" and expansion_method is not None:
            # Query expansion
            query_expander = QueryExpander(
                preprocessor=self.preprocessor,
                index_manager=self.index_manager,
                cache_dir=self.cache_dir
            )
            
            # Extract keywords for this query only
            keyword_extractor = KeywordExtractor(
                keybert_model=query_expander.sbert_model_name,
                cache_dir=query_expander.cache_dir,
                top_k_docs=top_k,                    # how many docs are eligible
                top_n_docs_for_extraction=docs_for_keywords
            )
            top_docs = []
            
            # First retrieve some docs to extract keywords from
            initial_results = search_engine.search(
                query=query_text,
                top_k=top_k,
                method=retrieval_method,
                hybrid_strategy=hybrid_strategy
            )
            
            # Get top N documents for keyword extraction
            for doc_id, _ in initial_results[:docs_for_keywords]:
                doc_text = search_engine.get_document_by_id(doc_id)
                top_docs.append(doc_text)
            
            # Extract keywords
            keywords = keyword_extractor.extract_keywords(
                query=query_text,
                docs_text=top_docs,
                num_keywords=num_keywords,
                diversity=keyword_diversity
            )
            
            # Use extracted keywords for expansion
            query_keywords = {query_id: keywords}
            baseline_results = {int(query_id): initial_results}
            
            # Run expansion
            expansion_results = query_expander.expand_queries(
                queries_dataset=[{"_id": query_id, "text": query_text}],
                corpus_dataset=corpus_dataset,
                baseline_results=baseline_results,
                search_engine=search_engine,
                query_keywords=query_keywords,
                expansion_method=expansion_method,
                combination_strategy=combination_strategy,
                num_keywords=num_keywords,
                top_k_results=top_k,
                original_query_weight=original_query_weight,
                force_regenerate_expansion=force_recompute
            )
            
            # Extract document IDs and texts
            expanded_query_results = expansion_results["expanded_results"][0]["results"]
            doc_ids = [doc_id for doc_id, _ in expanded_query_results]
            doc_scores = [score for _, score in expanded_query_results]
            doc_texts = []
            
            for doc_id in doc_ids:
                doc_texts.append(search_engine.get_document_by_id(doc_id))
                
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Build result
        result = {
            "query_id": query_id,
            "query_text": query_text,
            "method": method,
            "config": config,
            "doc_ids": doc_ids,
            "doc_scores": doc_scores,
            "doc_texts": doc_texts,
            "num_docs": len(doc_ids)
        }
        
        # Cache result
        logger.info(f"Caching retrieved documents to: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        
        return result
    
    def run_bertopic(
        self,
        query_id: str,
        query_text: str,
        documents: List[str],
        method: str,
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
        force_recompute: bool = False
    ) -> Dict:
        """
        Run BERTopic on retrieved documents for a query.
        (Docstring unchanged for brevity)
        """
        # ---------- 1. cache logic (unchanged) ----------
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
            "reduce_frequent_words": reduce_frequent_words
        }
        cache_path = self._get_cache_path(query_id, method, config)
        if not force_recompute and os.path.exists(cache_path):
            logger.info(f"Loading topic model results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        logger.info(f"Running BERTopic for query {query_id} ({len(documents)} documents)")

        # ---------- 2. build seed list BEFORE model instantiation ----------
        seed_topic_list = [[query_text]] if use_guided else None

        # ---------- 3. instantiate BERTopic WITH seed_topic_list ----------
        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            seed_topic_list=seed_topic_list,          # <- moved here
            umap_model=None,
            hdbscan_model=None,
            min_topic_size=min_cluster_size,
            nr_topics=nr_topics,
            low_memory=False,
            verbose=verbose,
            calculate_probabilities=True
        )

        # configure sub-models exactly as before
        topic_model.umap_model.n_components = n_components
        topic_model.umap_model.n_neighbors = n_neighbors
        topic_model.umap_model.min_dist = min_dist
        topic_model.umap_model.metric = metric
        topic_model.umap_model.random_state = 42
        topic_model.vectorizer_model = CountVectorizer(stop_words=stop_words)
        topic_model.representation_model = MaximalMarginalRelevance(diversity=mmr_diversity)
        topic_model.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=reduce_frequent_words)

        # ---------- 4. fit the model ----------
        if use_guided:
            document_embeddings = self.embedding_model.encode(
                documents, show_progress_bar=True, batch_size=32
            )
            topics, probs = topic_model.fit_transform(
                documents=documents,
                embeddings=document_embeddings            # <- NO seed_topic_list here
            )
        else:
            topics, probs = topic_model.fit_transform(documents)

        # ---------- 5. post-processing & caching (unchanged) ----------
        if hierarchical_topics:
            try:
                hierarchical_info = topic_model.hierarchical_topics(documents)
            except Exception as e:
                logger.warning(f"Failed to compute hierarchical topics: {e}")
                hierarchical_info = None
        else:
            hierarchical_info = None

        topic_info = topic_model.get_topic_info()
        topic_words = {t: topic_model.get_topic(t) for t in set(topics) if t != -1}
        document_info = pd.DataFrame({
            "Document": documents,
            "Topic": topics,
            "Probability": probs.max(axis=1) if len(probs.shape) > 1 else probs
        })

        result = {
            "query_id": query_id,
            "query_text": query_text,
            "method": method,
            "config": config,
            "topic_model": topic_model,
            "topics": topics,
            "probabilities": probs,
            "topic_info": topic_info,
            "topic_words": topic_words,
            "document_info": document_info,
            "hierarchical_topics": hierarchical_info,
            "num_documents": len(documents),
            "num_topics": len(topic_words)
        }

        logger.info(f"Caching topic model results to: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

        return result
    
    def process_query(
        self,
        query_id: str,
        query_text: str,
        corpus_dataset,
        methods: List[str] = ["retrieval", "query_expansion"],
        retrieval_config: Dict[str, Any] = None,
        expansion_config: Dict[str, Any] = None,
        topic_model_config: Dict[str, Any] = None,
        force_retrieve: bool = False,
        force_topic_model: bool = False
    ) -> Dict[str, Dict]:
        """
        Complete process for a query - retrieve documents and run topic modeling
        
        Args:
            query_id: ID of the query
            query_text: Text of the query
            corpus_dataset: Dataset containing corpus documents
            methods: List of methods to use
            retrieval_config: Configuration for document retrieval
            expansion_config: Configuration for query expansion
            topic_model_config: Configuration for topic modeling
            force_retrieve: Whether to force recomputing document retrieval
            force_topic_model: Whether to force recomputing topic models
            
        Returns:
            Dictionary mapping methods to topic modeling results
        """
        # Default configurations
        default_retrieval_config = {
            "retrieval_method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "top_k": 1000,
            "use_mmr": False,
            "mmr_lambda": 0.5,
            "use_cross_encoder": False
        }
        
        default_expansion_config = {
            "expansion_method": QueryExpansionMethod.KEYBERT,
            "combination_strategy": QueryCombinationStrategy.WEIGHTED_RRF,
            "num_keywords": 5,
            "keyword_diversity": 0.7,
            "docs_for_keywords": 10,
            "original_query_weight": 0.7
        }
        
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
            "verbose": True
        }
        
        # Merge with provided configurations
        if retrieval_config:
            default_retrieval_config.update(retrieval_config)
        retrieval_config = default_retrieval_config
        
        if expansion_config:
            default_expansion_config.update(expansion_config)
        expansion_config = default_expansion_config
        
        if topic_model_config:
            default_topic_model_config.update(topic_model_config)
        topic_model_config = default_topic_model_config
        
        results = {}
        
        for method in methods:
            logger.info(f"Processing query {query_id} with method '{method}'")
            
            if method == "retrieval":
                # Direct retrieval
                retrieved_docs = self.retrieve_documents_for_query(
                    query_id=query_id,
                    query_text=query_text,
                    corpus_dataset=corpus_dataset,
                    method="retrieval",
                    retrieval_method=retrieval_config["retrieval_method"],
                    hybrid_strategy=retrieval_config["hybrid_strategy"],
                    top_k=retrieval_config["top_k"],
                    use_mmr=retrieval_config["use_mmr"],
                    mmr_lambda=retrieval_config["mmr_lambda"],
                    use_cross_encoder=retrieval_config["use_cross_encoder"],
                    force_recompute=force_retrieve
                )
            
            elif method == "query_expansion":
                # Query expansion-based retrieval
                retrieved_docs = self.retrieve_documents_for_query(
                    query_id=query_id,
                    query_text=query_text,
                    corpus_dataset=corpus_dataset,
                    method="query_expansion",
                    expansion_method=expansion_config["expansion_method"],
                    combination_strategy=expansion_config["combination_strategy"],
                    top_k=retrieval_config["top_k"],
                    num_keywords=expansion_config["num_keywords"],
                    keyword_diversity=expansion_config["keyword_diversity"],
                    docs_for_keywords=expansion_config["docs_for_keywords"],
                    original_query_weight=expansion_config["original_query_weight"],
                    force_recompute=force_retrieve
                )
            
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Run topic modeling
            topics_result = self.run_bertopic(
                query_id=query_id,
                query_text=query_text,
                documents=retrieved_docs["doc_texts"],
                method=method,
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
                force_recompute=force_topic_model
            )
            
            # Store result
            results[method] = {
                "retrieved_docs": retrieved_docs,
                "topics_result": topics_result
            }
        
        return results
    
    def export_topic_results(
        self,
        query_id: str,
        topics_results: Dict[str, Dict],
        output_dir: str = "results/topics"
    ) -> Dict[str, str]:
        """
        Export topic modeling results for a query to files
        
        Args:
            query_id: ID of the query
            topics_results: Dictionary mapping methods to results
            output_dir: Directory to save results
            
        Returns:
            Dictionary mapping methods to output directories
        """
        # Get query text from first result
        query_text = next(iter(topics_results.values()))["topics_result"]["query_text"]
        
        logger.info(f"Exporting topic results for query {query_id}: '{query_text}'")
        
        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create query directory
        query_dir = os.path.join(output_dir, f"query_{query_id}")
        os.makedirs(query_dir, exist_ok=True)
        
        # Export query info
        with open(os.path.join(query_dir, "query_info.txt"), "w") as f:
            f.write(f"Query ID: {query_id}\n")
            f.write(f"Query Text: {query_text}\n")
        
        output_paths = {}
        
        for method, result in topics_results.items():
            # Create method directory
            method_dir = os.path.join(query_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            
            # Extract topics result
            topics_result = result["topics_result"]
            
            # Export topic info
            topics_result["topic_info"].to_csv(os.path.join(method_dir, "topic_info.csv"), index=False)
            
            # Export document info
            topics_result["document_info"].to_csv(os.path.join(method_dir, "document_info.csv"), index=False)
            
            # Export topic words
            with open(os.path.join(method_dir, "topic_words.txt"), "w") as f:
                for topic, words in topics_result["topic_words"].items():
                    f.write(f"Topic {topic}:\n")
                    for word, score in words:
                        f.write(f"  {word}: {score:.4f}\n")
                    f.write("\n")
            
            # Export topic model visualization if matplotlib is available
            try:
                import matplotlib.pyplot as plt
                
                # Handle topic visualization with error handling
                try:
                    fig = topics_result["topic_model"].visualize_topics()
                    plt.tight_layout()
                    plt.savefig(os.path.join(method_dir, "topic_visualization.png"), dpi=300)
                    plt.close()
                except Exception as e:
                    logger.warning(f"Could not create topic visualization: {str(e)}")
                
                # Handle topic hierarchy with error handling
                try:
                    fig = topics_result["topic_model"].visualize_hierarchy()
                    plt.tight_layout()
                    plt.savefig(os.path.join(method_dir, "topic_hierarchy.png"), dpi=300)
                    plt.close()
                except Exception as e:
                    logger.warning(f"Could not create topic hierarchy: {str(e)}")
                
                # Handle topic distribution with error handling
                try:
                    # Convert to numpy array if it's a list
                    topics = topics_result["topics"]
                    if isinstance(topics, list):
                        import numpy as np
                        topics = np.array(topics)
                    
                    fig = topics_result["topic_model"].visualize_distribution(topics)
                    plt.tight_layout()
                    plt.savefig(os.path.join(method_dir, "topic_distribution.png"), dpi=300)
                    plt.close()
                except Exception as e:
                    logger.warning(f"Could not create topic distribution: {str(e)}")
                    
            except (ImportError, Exception) as e:
                logger.warning(f"Could not create visualizations: {str(e)}")
            
            # Save retrieved document IDs
            retrieved_docs = result["retrieved_docs"]
            with open(os.path.join(method_dir, "retrieved_doc_ids.txt"), "w") as f:
                for doc_id in retrieved_docs["doc_ids"]:
                    f.write(f"{doc_id}\n")
            
            # Save full result
            with open(os.path.join(method_dir, "full_result.pkl"), "wb") as f:
                pickle.dump(result, f)
            
            output_paths[method] = method_dir
        
        return output_paths


def main():
    """Main function to demonstrate query-based topic modeling"""
    from datasets import load_dataset
    
    # Define constants
    CACHE_DIR = "cache"
    LOG_LEVEL = 'INFO'
    OUTPUT_DIR = "results/topics"
    
    # Define query IDs to process - can be modified
    QUERY_IDS = ["1", "2", "3"]  # First 3 queries
    
    # Flag to force recomputation
    FORCE_RETRIEVE = False  # Force recomputing document retrieval
    FORCE_TOPIC_MODEL = False  # Force recomputing topic models
    
    # Retrieval configuration
    RETRIEVAL_CONFIG = {
        "retrieval_method": RetrievalMethod.HYBRID,
        "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
        "top_k": 1000,
        "use_mmr": False,
        "mmr_lambda": 0.5,
        "use_cross_encoder": False
    }
    
    # Query expansion configuration
    EXPANSION_CONFIG = {
        "expansion_method": QueryExpansionMethod.KEYBERT,
        "combination_strategy": QueryCombinationStrategy.WEIGHTED_RRF,
        "num_keywords": 5,
        "keyword_diversity": 0.7,
        "docs_for_keywords": 10,
        "original_query_weight": 0.7
    }
    
    # Topic modeling configuration
    TOPIC_MODEL_CONFIG = {
        "n_components": 5,  # Number of components for UMAP
        "n_neighbors": 15,  # Number of neighbors for UMAP
        "min_dist": 0.0,  # Minimum distance for UMAP
        "metric": "cosine",  # Distance metric for UMAP
        "min_cluster_size": 10,  # Minimum cluster size for HDBSCAN
        "min_samples": None,  # Minimum samples for HDBSCAN (None defaults to min_cluster_size)
        "mmr_diversity": 0.3,  # Diversity parameter for MMR
        "nr_topics": None,  # Optional number of topics (None for automatic)
        "use_guided": True,  # Whether to use guided topic modeling
        "hierarchical_topics": False,  # Whether to compute hierarchical topics
        "reduce_frequent_words": True,  # Whether to reduce frequent words in TF-IDF
        "stop_words": "english",  # Stop words to remove
        "verbose": True  # Show verbose output
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
    
    # Extract query texts
    query_texts = {}
    for i, q in enumerate(queries_dataset):
        q_id = str(i+1)  # Convert to string ID (1-based indexing)
        
        if isinstance(q, dict) and "text" in q:
            query_texts[q_id] = q["text"]  # Dictionary access
        elif hasattr(q, 'text'):
            query_texts[q_id] = q.text  # Attribute access
        else:
            # Default approach if structure is unknown
            query_texts[q_id] = str(q)
    
    # Filter to selected query IDs
    selected_queries = {q_id: query_texts[q_id] for q_id in QUERY_IDS if q_id in query_texts}
    
    if not selected_queries:
        logger.error("No valid queries found in the specified query IDs")
        return
    
    # Initialize topic modeler
    logger.info("Initializing topic modeler...")
    topic_modeler = QueryBasedTopicModel(
        embedding_model_name="all-mpnet-base-v2",
        cache_dir=CACHE_DIR
    )
    
    all_results = {}
    
    # Process each query
    for query_id, query_text in selected_queries.items():
        logger.info(f"Processing query {query_id}: '{query_text}'")
        
        # Process the query with all methods
        results = topic_modeler.process_query(
            query_id=query_id,
            query_text=query_text,
            corpus_dataset=corpus_dataset,
            methods=["retrieval", "query_expansion"],  # Methods to use
            retrieval_config=RETRIEVAL_CONFIG,
            expansion_config=EXPANSION_CONFIG,
            topic_model_config=TOPIC_MODEL_CONFIG,
            force_retrieve=FORCE_RETRIEVE,
            force_topic_model=FORCE_TOPIC_MODEL
        )
        
        # Export results
        output_paths = topic_modeler.export_topic_results(
            query_id=query_id,
            topics_results=results,
            output_dir=OUTPUT_DIR
        )
        
        all_results[query_id] = {
            "query_text": query_text,
            "results": results,
            "output_paths": output_paths
        }
    
    # Print summary of results
    logger.info("\n===== TOPIC MODELING RESULTS SUMMARY =====")
    logger.info(f"{'Query':<10} {'Method':<20} {'Documents':<10} {'Topics':<10}")
    logger.info("-" * 55)
    
    for query_id, query_data in all_results.items():
        for method, result in query_data["results"].items():
            num_docs = result["retrieved_docs"]["num_docs"]
            num_topics = result["topics_result"]["num_topics"]
            
            logger.info(f"{query_id:<10} {method:<20} {num_docs:<10} {num_topics:<10}")
    
    logger.info(f"\nResults exported to {OUTPUT_DIR}")
    
    # Cache the summarized results
    cache_path = os.path.join(CACHE_DIR, "topic_results_by_query.pkl")
    logger.info(f"Caching topic results to: {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(all_results, f)
    
    return all_results

if __name__ == "__main__":
    main()