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
    
    def _load_or_build_indices(
        self,
        corpus_dataset,
        force_reindex: bool = False
    ):
        """Load or build search indices"""
        logger.info("Loading or building BM25 index...")
        bm25, corpus_texts, corpus_ids = self.index_manager.build_bm25_index(
            corpus_dataset,
            cache_path=os.path.join(self.cache_dir, "bm25_index.pkl"),
            force_reindex=force_reindex
        )
        
        logger.info("Loading or building SBERT index...")
        sbert_model, doc_embeddings = self.index_manager.build_sbert_index(
            corpus_texts,
            model_name=self.sbert_model_name,
            batch_size=64,
            cache_path=os.path.join(self.cache_dir, "sbert_index.pt"),
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
    
    def expand_queries(
        self,
        queries_dataset,
        corpus_dataset,
        baseline_results: Dict[int, List[Tuple[str, float]]],
        search_engine: SearchEngine,
        query_keywords: Dict[str, List[str]],
        expansion_method: QueryExpansionMethod = QueryExpansionMethod.KEYBERT,
        combination_strategy: QueryCombinationStrategy = QueryCombinationStrategy.WEIGHTED_RRF,
        num_keywords: int = 5,
        top_k_results: int = 1000,
        original_query_weight: float = 0.5,
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
            expansion_method: Method for keyword extraction
            combination_strategy: How to combine expanded terms
            num_keywords: Number of keywords to extract per query
            top_k_results: Number of results to retrieve per query
            original_query_weight: Weight for original query in RRF combination
            force_regenerate_expansion: Whether to force regenerating expansion results
            cache_path: Optional specific cache path for this configuration
            
        Returns:
            Dictionary of results, including raw search results
        """
        # Use specific cache path if provided, otherwise use default
        if cache_path is None:
            cache_path = os.path.join(
                self.cache_dir, 
                f"expanded_{expansion_method.value}_{combination_strategy.value}.pkl"
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
                        method=RetrievalMethod.HYBRID,
                        hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                        use_mmr=False,
                        use_cross_encoder=False
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
                method=RetrievalMethod.HYBRID,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=False
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
                method=RetrievalMethod.HYBRID,
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
        }
        
        # Cache results
        try:
            logger.info(f"Caching expansion results to {cache_path}")
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
        num_keywords: int = 5,
        top_n_docs_for_extraction: int = 10,
        top_k_results: int = 1000,
        force_reindex: bool = False,
        force_regenerate_keywords: bool = False,
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
            num_keywords: Number of keywords to extract per query
            top_n_docs_for_extraction: Number of docs to use for keyword extraction
            top_k_results: Number of results to retrieve per query
            force_reindex: Whether to force rebuilding indices
            force_regenerate_keywords: Whether to force regenerating keywords
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
        
        # Run baseline search with original queries
        logger.info("Running baseline search...")
        baseline_cache_path = os.path.join(self.cache_dir, "baseline_results.pkl")
        
        # Try to load cached baseline results
        baseline_results = None
        if not force_regenerate_expansion and os.path.exists(baseline_cache_path):
            try:
                logger.info(f"Loading cached baseline results from {baseline_cache_path}")
                with open(baseline_cache_path, "rb") as f:
                    baseline_data = pickle.load(f)
                    baseline_formatted = baseline_data.get("baseline_results", [])
                    baseline_results = {item["query_id"]: item["results"] for item in baseline_formatted}
            except Exception as e:
                logger.warning(f"Error loading cached baseline results: {e}")
        
        # Run baseline search if not loaded from cache
        if baseline_results is None:
            baseline_results = run_search_for_multiple_queries(
                search_engine=search_engine,
                queries_dataset=queries_dataset,
                top_k=top_k_results,
                method=RetrievalMethod.HYBRID,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=False
            )
            
            # Format baseline results and cache them
            query_id_to_text = {int(item["_id"]): item["text"] for item in queries_dataset}
            baseline_formatted = []
            for query_id, results in baseline_results.items():
                baseline_formatted.append({
                    "query_id": query_id,
                    "query_text": query_id_to_text.get(query_id, ""),
                    "results": results
                })
                
            baseline_data = {
                "baseline_results": baseline_formatted
            }
            
            try:
                with open(baseline_cache_path, "wb") as f:
                    pickle.dump(baseline_data, f)
            except Exception as e:
                logger.warning(f"Error caching baseline results: {e}")
        
        # Initialize keyword extractor with the appropriate parameters
        keyword_extractor = KeywordExtractor(
            keybert_model=self.sbert_model_name,
            cache_dir=self.cache_dir,
            top_k_docs=top_k_results,
            top_n_docs_for_extraction=top_n_docs_for_extraction
        )
        
        # Extract keywords for all queries once
        logger.info("Extracting keywords for all queries...")
        all_query_keywords = keyword_extractor.extract_keywords_for_queries(
            queries_dataset=queries_dataset,
            corpus_dataset=corpus_dataset,
            num_keywords=num_keywords,
            diversity=0.7,  # Default diversity parameter
            force_regenerate=force_regenerate_keywords,
            force_reindex=force_reindex
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
                    "method": RetrievalMethod.HYBRID.value,
                    "hybrid_strategy": HybridStrategy.SIMPLE_SUM.value,
                    "use_mmr": False,
                    "use_cross_encoder": False
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
            
            cache_path = os.path.join(
                self.cache_dir, 
                f"expanded_{expansion_method.value}_{combination_strategy.value}.pkl"
            )
            
            # Run expansion and search with pre-extracted keywords
            result_data = self.expand_queries(
                queries_dataset=queries_dataset,
                corpus_dataset=corpus_dataset,
                baseline_results=baseline_results,
                search_engine=search_engine,
                query_keywords=all_query_keywords,
                expansion_method=expansion_method,
                combination_strategy=combination_strategy,
                num_keywords=num_keywords,
                top_k_results=top_k_results,
                force_regenerate_expansion=force_regenerate_expansion,
                cache_path=cache_path
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
                        "method": RetrievalMethod.HYBRID.value,
                        "hybrid_strategy": HybridStrategy.SIMPLE_SUM.value,
                        "use_mmr": False,
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
            "plot_paths": plot_paths
        }


def main():
    """Main function for query expansion evaluation"""
    from datasets import load_dataset
    
    # Define constants for configuration
    FORCE_REINDEX = False
    FORCE_REGENERATE_KEYWORDS = False
    FORCE_REGENERATE_EXPANSION = False
    NUM_KEYWORDS = 10
    TOP_N_DOCS = 10
    TOP_K_RESULTS = 1000
    OUTPUT_DIR = "results/query_expansion"
    LOG_LEVEL = "INFO"
    CACHE_DIR = "cache"
    
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
        sbert_model_name="all-mpnet-base-v2",
        cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
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
        num_keywords=NUM_KEYWORDS,
        top_n_docs_for_extraction=TOP_N_DOCS,
        top_k_results=TOP_K_RESULTS,
        force_reindex=FORCE_REINDEX,
        force_regenerate_keywords=FORCE_REGENERATE_KEYWORDS,
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
    
    return results


# Execute main function if called directly
if __name__ == "__main__":
    main()