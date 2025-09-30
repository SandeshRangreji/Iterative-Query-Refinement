# query_expansion.py
import os
import logging
import pickle
from typing import List, Dict, Tuple, Set, Union, Optional
from enum import Enum
from collections import defaultdict
from tqdm import tqdm

# Import from other modules
from search import (
    TextPreprocessor, 
    SearchEngine, 
    IndexManager,
    RetrievalMethod,
    HybridStrategy,
    run_search_for_multiple_queries
)
from evaluation import SearchEvaluationUtils
from keyword_extraction import KeywordExtractor
from search_viz import visualize_all_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryExpansionMethod(str, Enum):
    """Enum for query expansion methods"""
    KEYBERT = "keybert"
    PMI = "pmi"
    SOPMI = "sopmi"
    COMBINED = "combined"

class QueryCombinationStrategy(str, Enum):
    """Enum for how to combine expanded query terms"""
    WEIGHTED_RRF = "weighted_rrf"  # Individual keyword queries combined with RRF
    CONCATENATED = "concatenated"   # Concatenate keywords with original query
    CONCATENATED_RERANKED = "concatenated_reranked"  # Concatenated with cross-encoder reranking

class QueryExpander:
    """Class for query expansion and evaluation"""
    
    def __init__(
        self,
        preprocessor: TextPreprocessor,
        index_manager: IndexManager,
        cache_dir: str = "cache",
        sbert_model_name: str = "all-mpnet-base-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize query expander
        
        Args:
            preprocessor: TextPreprocessor instance
            index_manager: IndexManager instance
            cache_dir: Directory for caching
            sbert_model_name: SBERT model name
            cross_encoder_model_name: Cross encoder model name
        """
        self.preprocessor = preprocessor
        self.index_manager = index_manager
        self.cache_dir = cache_dir
        self.sbert_model_name = sbert_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_baseline_cache_path(
        self,
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        top_k: int = 1000,
        use_mmr: bool = False,
        use_cross_encoder: bool = False,
        mmr_lambda: float = 0.5,
        hybrid_weight: float = 0.5
    ) -> str:
        """Generate structured cache path for baseline retrieval results"""
        cache_dir = os.path.join("cache", "retrieval", "baseline")
        
        # Build filename with key parameters
        method_str = retrieval_method.value if isinstance(retrieval_method, Enum) else str(retrieval_method)
        strategy_str = hybrid_strategy.value if isinstance(hybrid_strategy, Enum) else str(hybrid_strategy)
        strategy_str = strategy_str.replace("_", "-")  # Convert underscores to hyphens
        
        filename_parts = [method_str, strategy_str, f"k{top_k}"]
        
        # Add conditional parameters
        if retrieval_method == RetrievalMethod.HYBRID and hybrid_strategy == HybridStrategy.WEIGHTED:
            filename_parts.append(f"weight{hybrid_weight}")
        
        if use_mmr:
            filename_parts.extend([f"mmr-true", f"lambda{mmr_lambda}"])
        else:
            filename_parts.append("mmr-false")
        
        if use_cross_encoder:
            filename_parts.append("cross-true")
        else:
            filename_parts.append("cross-false")
        
        filename = "_".join(filename_parts) + ".pkl"
        return os.path.join(cache_dir, filename)
    
    def _generate_expansion_cache_path(
        self,
        # Keyword extraction parameters
        kw_method: str = "keybert",
        num_keywords: int = 5,
        diversity: float = 0.7,
        # Expansion parameters  
        expansion_method: QueryExpansionMethod = QueryExpansionMethod.KEYBERT,
        combination_strategy: QueryCombinationStrategy = QueryCombinationStrategy.WEIGHTED_RRF,
        original_query_weight: float = 0.7,
        # Retrieval parameters
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        top_k: int = 1000,
        use_cross_encoder: bool = False
    ) -> str:
        """Generate structured cache path for expanded retrieval results"""
        cache_dir = os.path.join("cache", "retrieval", "expanded")
        
        # Build filename with keyword extraction params
        kw_part = f"{kw_method}-k{num_keywords}-div{diversity}"
        
        # Add expansion strategy
        exp_method = expansion_method.value if isinstance(expansion_method, Enum) else str(expansion_method)
        comb_strategy = combination_strategy.value if isinstance(combination_strategy, Enum) else str(combination_strategy)
        comb_strategy = comb_strategy.replace("_", "-")  # Convert underscores to hyphens
        
        exp_part = f"{comb_strategy}_exp{num_keywords}_weight{original_query_weight}"
        
        # Add retrieval params
        retrieval_method_str = retrieval_method.value if isinstance(retrieval_method, Enum) else str(retrieval_method)
        retrieval_part = f"{retrieval_method_str}_k{top_k}"
        
        if use_cross_encoder:
            retrieval_part += "_cross-true"
        else:
            retrieval_part += "_cross-false"
        
        filename = f"{kw_part}_{exp_part}_{retrieval_part}.pkl"
        return os.path.join(cache_dir, filename)
    
    def _load_or_build_indices(
        self,
        corpus_dataset,
        dataset_name: str = "trec-covid",
        force_reindex: bool = False
    ):
        """Load or build search indices"""
        logger.info("Loading or building BM25 index...")
        bm25, corpus_texts, corpus_ids = self.index_manager.build_bm25_index(
            corpus_dataset,
            dataset_name=dataset_name,
            force_reindex=force_reindex
        )
        
        logger.info("Loading or building SBERT index...")
        sbert_model, doc_embeddings = self.index_manager.build_sbert_index(
            corpus_texts,
            model_name=self.sbert_model_name,
            dataset_name=dataset_name,
            batch_size=64,
            force_reindex=force_reindex
        )
        
        return bm25, corpus_texts, corpus_ids, sbert_model, doc_embeddings
    
    def _deduplicate_query_and_keywords(self, query_text: str, keywords: List[str]) -> str:
        """
        Deduplicate words between original query and keywords
        
        Args:
            query_text: Original query text
            keywords: List of keyword phrases
            
        Returns:
            Deduplicated expanded query text
        """
        seen = set()
        all_tokens = []
        
        # Add original query words first
        for word in query_text.strip().split():
            word_lower = word.lower()
            if word_lower not in seen:
                all_tokens.append(word)
                seen.add(word_lower)
        
        # Add words from keyword phrases
        for phrase in keywords:
            for word in phrase.strip().split():
                word_lower = word.lower()
                if word_lower not in seen:
                    all_tokens.append(word)
                    seen.add(word_lower)
                    
        return " ".join(all_tokens)
    
    def _reciprocal_rank_fusion(
        self,
        rankings: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None,
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """
        Combine rankings using Reciprocal Rank Fusion
        
        Args:
            rankings: List of document rankings (each a list of (doc_id, score) tuples)
            weights: Optional list of weights for each ranking
            k: RRF constant
            
        Returns:
            Combined ranking
        """
        # Default weights to equal weighting if not provided
        if weights is None:
            weights = [1.0] * len(rankings)
        
        # Ensure weights match rankings length
        assert len(weights) == len(rankings), "Length of weights must match length of rankings"
        
        scores = defaultdict(float)
        for i, rank_list in enumerate(rankings):
            weight = weights[i]
            for rank, (doc_id, _) in enumerate(rank_list):
                scores[doc_id] += weight * (1.0 / (k + rank + 1))
        
        # Sort by score
        result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return result
    
    def run_baseline_search(
        self,
        queries_dataset,
        corpus_dataset,
        search_engine: SearchEngine,
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        top_k: int = 1000,
        use_mmr: bool = False,
        use_cross_encoder: bool = False,
        mmr_lambda: float = 0.5,
        hybrid_weight: float = 0.5,
        force_regenerate: bool = False
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Run baseline search and cache results
        
        Args:
            queries_dataset: Dataset containing queries
            corpus_dataset: Dataset containing corpus documents  
            search_engine: Initialized SearchEngine instance
            retrieval_method: Method for document retrieval
            hybrid_strategy: Strategy for hybrid retrieval
            top_k: Number of documents to retrieve per query
            use_mmr: Whether to use MMR for diversity
            use_cross_encoder: Whether to use cross-encoder reranking
            mmr_lambda: Lambda parameter for MMR
            hybrid_weight: Weight for hybrid strategy
            force_regenerate: Whether to force regenerating results
            
        Returns:
            Dictionary mapping query IDs to search results
        """
        # Generate cache path
        cache_path = self._generate_baseline_cache_path(
            retrieval_method=retrieval_method,
            hybrid_strategy=hybrid_strategy,
            top_k=top_k,
            use_mmr=use_mmr,
            use_cross_encoder=use_cross_encoder,
            mmr_lambda=mmr_lambda,
            hybrid_weight=hybrid_weight
        )
        
        # Check cache
        if not force_regenerate and os.path.exists(cache_path):
            logger.info(f"Loading baseline results from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                baseline_data = pickle.load(f)
                return baseline_data.get("baseline_results", {})
        
        logger.info("Running baseline search...")
        
        # Run search for all queries
        baseline_results = run_search_for_multiple_queries(
            search_engine=search_engine,
            queries_dataset=queries_dataset,
            top_k=top_k,
            method=retrieval_method,
            hybrid_strategy=hybrid_strategy,
            hybrid_weight=hybrid_weight,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            use_cross_encoder=use_cross_encoder
        )
        
        # Format and cache results
        query_id_to_text = {int(item["_id"]): item["text"] for item in queries_dataset}
        baseline_formatted = []
        for query_id, results in baseline_results.items():
            baseline_formatted.append({
                "query_id": query_id,
                "query_text": query_id_to_text.get(query_id, ""),
                "results": results
            })
        
        baseline_data = {
            "baseline_results": baseline_results,
            "baseline_formatted": baseline_formatted,
            "config": {
                "retrieval_method": retrieval_method.value if isinstance(retrieval_method, Enum) else str(retrieval_method),
                "hybrid_strategy": hybrid_strategy.value if isinstance(hybrid_strategy, Enum) else str(hybrid_strategy),
                "top_k": top_k,
                "use_mmr": use_mmr,
                "use_cross_encoder": use_cross_encoder,
                "mmr_lambda": mmr_lambda,
                "hybrid_weight": hybrid_weight
            }
        }
        
        # Cache results
        logger.info(f"Caching baseline results to: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(baseline_data, f)
        
        return baseline_results
    
    def expand_queries(
        self,
        queries_dataset,
        corpus_dataset,
        baseline_results: Dict[int, List[Tuple[str, float]]],
        search_engine: SearchEngine,
        query_keywords: Dict[str, List[str]],
        # Keyword extraction parameters (for cache naming)
        kw_method: str = "keybert",
        kw_num_keywords: int = 5,
        kw_diversity: float = 0.7,
        # Expansion parameters
        expansion_method: QueryExpansionMethod = QueryExpansionMethod.KEYBERT,
        combination_strategy: QueryCombinationStrategy = QueryCombinationStrategy.WEIGHTED_RRF,
        num_keywords: int = 5,
        top_k_results: int = 1000,
        original_query_weight: float = 0.5,
        # Retrieval parameters
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        use_cross_encoder: bool = False,
        force_regenerate_expansion: bool = False,
        cache_path: Optional[str] = None
    ):
        """
        Expand queries and search with different strategies
        
        Args:
            queries_dataset: Dataset containing queries
            corpus_dataset: Dataset containing corpus documents
            baseline_results: Dictionary of baseline search results
            search_engine: Initialized SearchEngine instance
            query_keywords: Pre-extracted keywords for each query
            kw_method: Keyword extraction method used (for cache naming)
            kw_num_keywords: Number of keywords extracted (for cache naming)
            kw_diversity: Diversity used in keyword extraction (for cache naming)
            expansion_method: Method for keyword extraction
            combination_strategy: How to combine expanded terms
            num_keywords: Number of keywords to extract per query
            top_k_results: Number of results to retrieve per query
            original_query_weight: Weight for original query in RRF combination
            retrieval_method: Retrieval method to use
            use_cross_encoder: Whether to use cross-encoder reranking
            force_regenerate_expansion: Whether to force regenerating expansion results
            cache_path: Optional specific cache path for this configuration
            
        Returns:
            Dictionary of results, including raw search results
        """
        # Use specific cache path if provided, otherwise generate one
        if cache_path is None:
            cache_path = self._generate_expansion_cache_path(
                kw_method=kw_method,
                num_keywords=kw_num_keywords,
                diversity=kw_diversity,
                expansion_method=expansion_method,
                combination_strategy=combination_strategy,
                original_query_weight=original_query_weight,
                retrieval_method=retrieval_method,
                top_k=top_k_results,
                use_cross_encoder=use_cross_encoder
            )
        
        # Try to load cached results first
        if not force_regenerate_expansion and os.path.exists(cache_path):
            try:
                logger.info(f"Loading cached expansion results from {cache_path}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cached results: {e}")
        
        # Create a map of query_id to query_text for easier access
        query_id_to_text = {int(item["_id"]): item["text"] for item in queries_dataset}
        
        # Process expanded queries based on strategy
        logger.info(f"Processing expanded queries with {combination_strategy.value}...")
        expanded_results = {}
        
        if combination_strategy == QueryCombinationStrategy.WEIGHTED_RRF:
            # For each query, combine original results with keyword results using RRF
            for query_id, original_results in tqdm(baseline_results.items(), desc="Processing with RRF"):
                # Skip if no keywords available
                if query_keywords is None or str(query_id) not in query_keywords:
                    logger.warning(f"No keywords found for query ID {query_id}, skipping expansion")
                    continue
                
                # Get keywords for this query
                keywords = query_keywords[str(query_id)]
                
                # Search for each keyword individually
                keyword_results = []
                for keyword in keywords:
                    kw_results = search_engine.search(
                        query=keyword,
                        top_k=top_k_results,
                        method=retrieval_method,
                        hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                        use_mmr=False,
                        use_cross_encoder=use_cross_encoder
                    )
                    keyword_results.append(kw_results)
                
                # Combine with RRF, weighting original query higher
                all_rankings = [original_results] + keyword_results
                weights = [original_query_weight] + [(1.0 - original_query_weight) / len(keywords)] * len(keywords)
                
                combined_results = self._reciprocal_rank_fusion(all_rankings, weights)
                expanded_results[query_id] = combined_results
                
        elif combination_strategy == QueryCombinationStrategy.CONCATENATED:
            # Create expanded query dataset with concatenated keywords
            expanded_queries = []
            
            for query_item in queries_dataset:
                query_id = int(query_item["_id"])
                query_text = query_item["text"]
                
                # Skip if no keywords available
                if query_keywords is None or str(query_id) not in query_keywords:
                    logger.warning(f"No keywords found for query ID {query_id}, skipping expansion")
                    continue
                
                # Get keywords and concatenate with original query
                keywords = query_keywords[str(query_id)]
                expanded_text = self._deduplicate_query_and_keywords(query_text, keywords)
                
                # Add to expanded queries list
                expanded_queries.append({
                    "_id": str(query_id),
                    "text": expanded_text
                })
            
            # Run search with expanded queries
            expanded_results = run_search_for_multiple_queries(
                search_engine=search_engine,
                queries_dataset=expanded_queries,
                top_k=top_k_results,
                method=retrieval_method,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=use_cross_encoder
            )
            
        elif combination_strategy == QueryCombinationStrategy.CONCATENATED_RERANKED:
            # Create expanded query dataset with concatenated keywords
            expanded_queries = []
            
            for query_item in queries_dataset:
                query_id = int(query_item["_id"])
                query_text = query_item["text"]
                
                # Skip if no keywords available
                if query_keywords is None or str(query_id) not in query_keywords:
                    logger.warning(f"No keywords found for query ID {query_id}, skipping expansion")
                    continue
                
                # Get keywords and concatenate with original query
                keywords = query_keywords[str(query_id)]
                expanded_text = self._deduplicate_query_and_keywords(query_text, keywords)
                
                # Add to expanded queries list
                expanded_queries.append({
                    "_id": str(query_id),
                    "text": expanded_text
                })
            
            # Run search with expanded queries and cross-encoder reranking
            expanded_results = run_search_for_multiple_queries(
                search_engine=search_engine,
                queries_dataset=expanded_queries,
                top_k=top_k_results,
                method=retrieval_method,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=True  # Enable reranking
            )
            
        else:
            raise ValueError(f"Unsupported combination strategy: {combination_strategy}")
        
        # Format results for consistency with original code
        expanded_formatted = []
        for query_id, results in expanded_results.items():
            if query_id in query_id_to_text:
                expanded_formatted.append({
                    "query_id": query_id,
                    "query_text": query_id_to_text[query_id],
                    "results": results
                })
        
        result_data = {
            "expansion_method": expansion_method.value,
            "combination_strategy": combination_strategy.value,
            "num_keywords": num_keywords,
            "expanded_results": expanded_formatted,
            "config": {
                "kw_method": kw_method,
                "kw_num_keywords": kw_num_keywords,
                "kw_diversity": kw_diversity,
                "expansion_method": expansion_method.value,
                "combination_strategy": combination_strategy.value,
                "original_query_weight": original_query_weight,
                "retrieval_method": retrieval_method.value if isinstance(retrieval_method, Enum) else str(retrieval_method),
                "top_k": top_k_results,
                "use_cross_encoder": use_cross_encoder
            }
        }
        
        # Cache results
        try:
            logger.info(f"Caching expansion results to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(result_data, f)
        except Exception as e:
            logger.warning(f"Error caching results: {e}")
        
        return result_data
    
    def run_evaluation_suite(
        self,
        queries_dataset,
        corpus_dataset,
        qrels_dataset,
        expansion_configs: List[Dict],
        # Keyword extraction parameters
        kw_method: str = "keybert",
        num_keywords: int = 5,
        kw_diversity: float = 0.7,
        kw_top_k_docs: int = 1000,
        kw_top_n_docs_for_extraction: int = 10,
        # Retrieval parameters
        top_k_results: int = 1000,
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        use_mmr: bool = False,
        use_cross_encoder: bool = False,
        # Dataset parameters
        dataset_name: str = "trec-covid",
        # Force flags
        force_reindex: bool = False,
        force_regenerate_keywords: bool = False,
        force_regenerate_baseline: bool = False,
        force_regenerate_expansion: bool = False,
        output_dir: str = "results/query_expansion"
    ) -> Dict:
        """
        Run evaluation suite for different query expansion configurations
        
        Args:
            queries_dataset: Dataset containing queries
            corpus_dataset: Dataset containing corpus documents
            qrels_dataset: Dataset containing relevance judgments
            expansion_configs: List of expansion configurations to evaluate
            kw_method: Keyword extraction method
            num_keywords: Number of keywords to extract per query
            kw_diversity: Diversity for keyword extraction
            kw_top_k_docs: Number of documents to retrieve for keyword extraction
            kw_top_n_docs_for_extraction: Number of docs to use for keyword extraction
            top_k_results: Number of results to retrieve per query
            retrieval_method: Method for document retrieval
            hybrid_strategy: Strategy for hybrid retrieval
            use_mmr: Whether to use MMR
            use_cross_encoder: Whether to use cross-encoder (for baseline)
            dataset_name: Name of the dataset for caching
            force_reindex: Whether to force rebuilding indices
            force_regenerate_keywords: Whether to force regenerating keywords
            force_regenerate_baseline: Whether to force regenerating baseline
            force_regenerate_expansion: Whether to force regenerating expansion results
            output_dir: Directory for output files
            
        Returns:
            Dictionary of evaluation results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load or build indices
        bm25, corpus_texts, corpus_ids, sbert_model, doc_embeddings = self._load_or_build_indices(
            corpus_dataset,
            dataset_name=dataset_name,
            force_reindex=force_reindex
        )
        
        # Initialize search engine
        search_engine = SearchEngine(
            preprocessor=self.preprocessor, 
            bm25=bm25,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            sbert_model=sbert_model,
            doc_embeddings=doc_embeddings,
            cross_encoder_model_name=self.cross_encoder_model_name
        )
        
        # Run baseline search
        logger.info("Running baseline search...")
        baseline_results = self.run_baseline_search(
            queries_dataset=queries_dataset,
            corpus_dataset=corpus_dataset,
            search_engine=search_engine,
            retrieval_method=retrieval_method,
            hybrid_strategy=hybrid_strategy,
            top_k=top_k_results,
            use_mmr=use_mmr,
            use_cross_encoder=use_cross_encoder,
            force_regenerate=force_regenerate_baseline
        )
        
        # Initialize keyword extractor
        keyword_extractor = KeywordExtractor(
            keybert_model=self.sbert_model_name,
            cache_dir=self.cache_dir,
            top_k_docs=kw_top_k_docs,
            top_n_docs_for_extraction=kw_top_n_docs_for_extraction
        )
        
        # Extract keywords for all queries
        logger.info("Extracting keywords for all queries...")
        all_query_keywords = keyword_extractor.extract_keywords_for_queries(
            queries_dataset=queries_dataset,
            corpus_dataset=corpus_dataset,
            num_keywords=num_keywords,
            diversity=kw_diversity,
            force_regenerate=force_regenerate_keywords,
            sbert_model_name=self.sbert_model_name,
            force_reindex=force_reindex,
            dataset_name=dataset_name
        )
        
        # Track all results for visualization
        all_visualization_results = []
        results_by_config = {}
        
        # Prepare baseline result for visualization
        baseline_metrics = None
        if qrels_dataset is not None:
            # Evaluate baseline results
            avg_precisions, avg_recalls, num_evaluated = SearchEvaluationUtils.evaluate_results(
                results_by_query_id=baseline_results,
                qrels_dataset=qrels_dataset,
                top_k_p=20,
                top_k_r=top_k_results
            )

            baseline_metrics = {
                "precisions": avg_precisions,
                "recalls": avg_recalls,
                "f1_scores": {
                    level: SearchEvaluationUtils.calculate_f1_score(avg_precisions[level], avg_recalls[level])
                    for level in ['relevant', 'highly_relevant', 'overall']
                },
                "num_evaluated": num_evaluated
            }
            
            # Add baseline to visualization results
            baseline_viz_result = {
                "config": {
                    "name": "Baseline",
                    "method": retrieval_method.value,
                    "hybrid_strategy": hybrid_strategy.value,
                    "use_mmr": use_mmr,
                    "use_cross_encoder": use_cross_encoder
                },
                "avg_precisions": baseline_metrics["precisions"],
                "avg_recalls": baseline_metrics["recalls"],
                "num_evaluated": baseline_metrics["num_evaluated"]
            }
            all_visualization_results.append(baseline_viz_result)
        
        # Run each expansion configuration
        for config in expansion_configs:
            logger.info(f"Evaluating {config['name']}...")
            
            expansion_method = QueryExpansionMethod(config["expansion_method"])
            combination_strategy = QueryCombinationStrategy(config["combination_strategy"])
            
            # Run expansion and search with pre-extracted keywords
            result_data = self.expand_queries(
                queries_dataset=queries_dataset,
                corpus_dataset=corpus_dataset,
                baseline_results=baseline_results,
                search_engine=search_engine,
                query_keywords=all_query_keywords,
                kw_method=kw_method,
                kw_num_keywords=num_keywords,
                kw_diversity=kw_diversity,
                expansion_method=expansion_method,
                combination_strategy=combination_strategy,
                num_keywords=num_keywords,
                top_k_results=top_k_results,
                retrieval_method=retrieval_method,
                use_cross_encoder=combination_strategy == QueryCombinationStrategy.CONCATENATED_RERANKED,
                force_regenerate_expansion=force_regenerate_expansion
            )
            
            # Add baseline metrics if available
            if baseline_metrics is not None:
                result_data["baseline_metrics"] = baseline_metrics
            
            # Evaluate expanded results if qrels are provided
            if qrels_dataset is not None:
                # Convert to the format needed for evaluation
                expanded_results_by_query_id = {item["query_id"]: item["results"] for item in result_data["expanded_results"]}
                
                # Evaluate expanded results
                avg_precisions, avg_recalls, num_evaluated = SearchEvaluationUtils.evaluate_results(
                    results_by_query_id=expanded_results_by_query_id,
                    qrels_dataset=qrels_dataset,
                    top_k_p=20,
                    top_k_r=top_k_results
                )

                expanded_metrics = {
                    "precisions": avg_precisions,
                    "recalls": avg_recalls,
                    "f1_scores": {
                        level: SearchEvaluationUtils.calculate_f1_score(avg_precisions[level], avg_recalls[level])
                        for level in ['relevant', 'highly_relevant', 'overall']
                    },
                    "num_evaluated": num_evaluated
                }
                
                # Add metrics to result data
                result_data["expanded_metrics"] = expanded_metrics
                
                # Format for visualization
                viz_result = {
                    "config": {
                        "name": config["name"],
                        "method": retrieval_method.value,
                        "hybrid_strategy": hybrid_strategy.value,
                        "use_mmr": use_mmr,
                        "use_cross_encoder": combination_strategy == QueryCombinationStrategy.CONCATENATED_RERANKED,
                        "expansion_method": expansion_method.value,
                        "combination_strategy": combination_strategy.value
                    },
                    "avg_precisions": expanded_metrics["precisions"],
                    "avg_recalls": expanded_metrics["recalls"],
                    "num_evaluated": expanded_metrics["num_evaluated"]
                }
                all_visualization_results.append(viz_result)
            
            # Store results
            results_by_config[config["name"]] = result_data
        
        # Save combined results for visualization
        if all_visualization_results:
            output_file = os.path.join(output_dir, "query_expansion_results.json")
            SearchEvaluationUtils.save_evaluation_results(all_visualization_results, output_file)
            
            # Generate visualization plots
            logger.info("Generating visualization plots...")
            plots_dir = os.path.join(output_dir, "plots")
            plot_paths = visualize_all_results(
                all_visualization_results,
                top_k_p=20,
                top_k_r=top_k_results,
                output_dir=plots_dir
            )
        else:
            output_file = None
            plot_paths = []
        
        return {
            "visualization_results": all_visualization_results,
            "detailed_results": results_by_config,
            "output_file": output_file,
            "plot_paths": plot_paths,
            "baseline_results": baseline_results,
            "query_keywords": all_query_keywords
        }


def main():
    """Main function for query expansion evaluation"""
    from datasets import load_dataset
    
    # ===== CACHING CONTROL FLAGS =====
    FORCE_REINDEX = False                      # For search indices
    FORCE_REGENERATE_KEYWORDS = False          # For keyword extraction
    FORCE_REGENERATE_BASELINE = False          # For baseline search results
    FORCE_REGENERATE_EXPANSION = False         # For expansion results
    
    # ===== DATASET PARAMETERS =====
    DATASET_NAME = "trec-covid"
    
    # ===== KEYWORD EXTRACTION PARAMETERS =====
    KW_METHOD = "keybert"
    NUM_KEYWORDS = 10
    KW_DIVERSITY = 0.7
    KW_TOP_K_DOCS = 1000
    KW_TOP_N_DOCS = 10
    
    # ===== RETRIEVAL PARAMETERS =====
    TOP_K_RESULTS = 1000
    RETRIEVAL_METHOD = RetrievalMethod.HYBRID
    HYBRID_STRATEGY = HybridStrategy.SIMPLE_SUM
    USE_MMR = False
    USE_CROSS_ENCODER_BASELINE = False
    MMR_LAMBDA = 0.5
    HYBRID_WEIGHT = 0.5
    
    # ===== MODEL PARAMETERS =====
    SBERT_MODEL = "all-mpnet-base-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # ===== OTHER PARAMETERS =====
    OUTPUT_DIR = "results/query_expansion"
    CACHE_DIR = "cache"
    LOG_LEVEL = "INFO"
    
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
    
    # Initialize components
    preprocessor = TextPreprocessor()
    index_manager = IndexManager(preprocessor)
    
    # Initialize query expander
    query_expander = QueryExpander(
        preprocessor=preprocessor,
        index_manager=index_manager,
        cache_dir=CACHE_DIR,
        sbert_model_name=SBERT_MODEL,
        cross_encoder_model_name=CROSS_ENCODER_MODEL
    )
    
    # Define expansion configurations to evaluate
    expansion_configs = [
        {
            "name": "KeyBERT + Weighted RRF",
            "expansion_method": QueryExpansionMethod.KEYBERT.value,
            "combination_strategy": QueryCombinationStrategy.WEIGHTED_RRF.value
        },
        {
            "name": "KeyBERT + Concatenated",
            "expansion_method": QueryExpansionMethod.KEYBERT.value,
            "combination_strategy": QueryCombinationStrategy.CONCATENATED.value
        },
        {
            "name": "KeyBERT + Concatenated + CrossEncoder",
            "expansion_method": QueryExpansionMethod.KEYBERT.value,
            "combination_strategy": QueryCombinationStrategy.CONCATENATED_RERANKED.value
        }
    ]
    
    # Run evaluation suite
    results = query_expander.run_evaluation_suite(
        queries_dataset=queries_dataset,
        corpus_dataset=corpus_dataset,
        qrels_dataset=qrels_dataset,
        expansion_configs=expansion_configs,
        kw_method=KW_METHOD,
        num_keywords=NUM_KEYWORDS,
        kw_diversity=KW_DIVERSITY,
        kw_top_k_docs=KW_TOP_K_DOCS,
        kw_top_n_docs_for_extraction=KW_TOP_N_DOCS,
        top_k_results=TOP_K_RESULTS,
        retrieval_method=RETRIEVAL_METHOD,
        hybrid_strategy=HYBRID_STRATEGY,
        use_mmr=USE_MMR,
        use_cross_encoder=USE_CROSS_ENCODER_BASELINE,
        dataset_name=DATASET_NAME,
        force_reindex=FORCE_REINDEX,
        force_regenerate_keywords=FORCE_REGENERATE_KEYWORDS,
        force_regenerate_baseline=FORCE_REGENERATE_BASELINE,
        force_regenerate_expansion=FORCE_REGENERATE_EXPANSION,
        output_dir=OUTPUT_DIR
    )
    
    # Print results summary
    logger.info("\n===== QUERY EXPANSION EVALUATION RESULTS =====")
    logger.info(f"{'Method':<40} {'P@20':<10} {'R@'+str(TOP_K_RESULTS):<10} {'F1':<10}")
    logger.info("-" * 70)
    
    for result in results["visualization_results"]:
        config_name = result["config"]["name"]
        precision = result["avg_precisions"]["overall"]
        recall = result["avg_recalls"]["overall"]
        
        # Calculate F1
        f1 = SearchEvaluationUtils.calculate_f1_score(precision, recall)
        
        logger.info(f"{config_name:<40} {precision:.4f}     {recall:.4f}     {f1:.4f}")
    
    if results["output_file"]:
        logger.info(f"\nResults saved to {results['output_file']}")
    
    if results["plot_paths"]:
        logger.info(f"Visualization plots saved to {os.path.dirname(results['plot_paths'][0])}")
    
    # Print cache structure info
    logger.info(f"\n===== CACHE STRUCTURE =====")
    logger.info(f"Keyword cache: cache/keywords/")
    logger.info(f"Baseline cache: cache/retrieval/baseline/")
    logger.info(f"Expansion cache: cache/retrieval/expanded/")
    
    return results


# Execute main function if called directly
if __name__ == "__main__":
    main()