# query_expansion.py
import os
import logging
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional
from enum import Enum
from collections import Counter, defaultdict
import torch
from tqdm import tqdm
import json

# Import from other modules
from search import (
    TextPreprocessor, 
    SearchEngine, 
    IndexManager,
    RetrievalMethod,
    HybridStrategy,
    EvaluationUtils,
    save_evaluation_results
)
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
        
        # Initialize keyword extractor
        self.keyword_extractor = KeywordExtractor(
            keybert_model=sbert_model_name,
            cache_dir=cache_dir
        )
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache paths
        self.expansion_cache_path = os.path.join(cache_dir, "query_expansion_results.pkl")
    
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
    
    def expand_and_search(
        self,
        queries_dataset,
        corpus_dataset,
        qrels_dataset=None,
        expansion_method: QueryExpansionMethod = QueryExpansionMethod.KEYBERT,
        combination_strategy: QueryCombinationStrategy = QueryCombinationStrategy.WEIGHTED_RRF,
        num_keywords: int = 5,
        top_n_docs_for_extraction: int = 10,
        top_k_results: int = 1000,
        original_query_weight: float = 0.7,
        force_reindex: bool = False,
        force_regenerate: bool = False,
        cache_path: Optional[str] = None
    ):
        """
        Expand queries and search with different strategies
        
        Args:
            queries_dataset: Dataset containing queries
            corpus_dataset: Dataset containing corpus documents
            qrels_dataset: Dataset containing relevance judgments (optional)
            expansion_method: Method for keyword extraction (keybert, pmi, sopmi, combined)
            combination_strategy: How to combine expanded terms (rrf or concatenated)
            num_keywords: Number of keywords to extract per query
            top_n_docs_for_extraction: Number of docs to use for keyword extraction
            top_k_results: Number of results to retrieve per query
            original_query_weight: Weight for original query in RRF combination
            force_reindex: Whether to force rebuilding indices
            force_regenerate: Whether to force regenerating results
            cache_path: Optional specific cache path for this configuration
            
        Returns:
            Dictionary of results, including raw search results and evaluation metrics if qrels provided
        """
        # Use specific cache path if provided, otherwise use default
        if cache_path is None:
            cache_path = os.path.join(
                self.cache_dir, 
                f"expanded_{expansion_method.value}_{combination_strategy.value}.pkl"
            )
        
        # Try to load cached results first
        if not force_regenerate and os.path.exists(cache_path):
            try:
                logger.info(f"Loading cached expansion results from {cache_path}")
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cached results: {e}")
        
        # Load or build indices
        bm25, corpus_texts, corpus_ids, sbert_model, doc_embeddings = self._load_or_build_indices(
            corpus_dataset,
            force_reindex=force_reindex
        )
        
        # Initialize search engine
        search_engine = SearchEngine(self.preprocessor, self.cross_encoder_model_name)
        
        # Track results for each query
        baseline_results = []
        expanded_results = []
        
        # Process each query
        for query_item in tqdm(queries_dataset, desc="Processing queries"):
            query_id = int(query_item["_id"])
            query_text = query_item["text"]
            
            logger.info(f"Processing query ID {query_id}: {query_text}")
            
            # Baseline search (hybrid simple sum, no MMR, no cross-encoder)
            baseline_search_results = search_engine.search(
                query=query_text,
                bm25=bm25,
                corpus_texts=corpus_texts,
                corpus_ids=corpus_ids,
                sbert_model=sbert_model,
                doc_embeddings=doc_embeddings,
                top_k=top_k_results,
                method=RetrievalMethod.HYBRID,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=False
            )
            
            # Store baseline results
            baseline_results.append({
                "query_id": query_id,
                "query_text": query_text,
                "results": baseline_search_results
            })
            
            # Get top N documents for keyword extraction
            top_docs = []
            for doc_id, _ in baseline_search_results[:top_n_docs_for_extraction]:
                idx = corpus_ids.index(doc_id)
                top_docs.append(corpus_texts[idx])
            
            # Extract keywords based on selected method
            if expansion_method == QueryExpansionMethod.KEYBERT:
                keywords = self.keyword_extractor.extract_keywords(
                    query=query_text,
                    docs_text=top_docs,
                    num_keywords=num_keywords
                )
            elif expansion_method == QueryExpansionMethod.PMI:
                # TODO: Implement PMI keyword extraction
                keywords = self.keyword_extractor.extract_keywords(
                    query=query_text,
                    docs_text=top_docs,
                    num_keywords=num_keywords
                )  # Fallback to KeyBERT for now
            elif expansion_method == QueryExpansionMethod.SOPMI:
                # TODO: Implement SoPMI keyword extraction
                keywords = self.keyword_extractor.extract_keywords(
                    query=query_text,
                    docs_text=top_docs,
                    num_keywords=num_keywords
                )  # Fallback to KeyBERT for now
            elif expansion_method == QueryExpansionMethod.COMBINED:
                # Combine keywords from multiple methods (all KeyBERT for now)
                keywords = self.keyword_extractor.extract_keywords(
                    query=query_text,
                    docs_text=top_docs,
                    num_keywords=num_keywords
                )
            else:
                raise ValueError(f"Unsupported expansion method: {expansion_method}")
            
            logger.info(f"Extracted keywords: {keywords}")
            
            # Apply the selected combination strategy
            if combination_strategy == QueryCombinationStrategy.WEIGHTED_RRF:
                # Search for each keyword individually
                keyword_results = []
                for keyword in keywords:
                    kw_results = search_engine.search(
                        query=keyword,
                        bm25=bm25,
                        corpus_texts=corpus_texts,
                        corpus_ids=corpus_ids,
                        sbert_model=sbert_model,
                        doc_embeddings=doc_embeddings,
                        top_k=top_k_results,
                        method=RetrievalMethod.HYBRID,
                        hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                        use_mmr=False,
                        use_cross_encoder=False
                    )
                    keyword_results.append(kw_results)
                
                # Combine with RRF, weighting original query higher
                all_rankings = [baseline_search_results] + keyword_results
                weights = [original_query_weight] + [(1.0 - original_query_weight) / len(keywords)] * len(keywords)
                
                combined_results = self._reciprocal_rank_fusion(all_rankings, weights)
                
                expanded_results.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "expanded_text": ", ".join(keywords),  # Just for logging
                    "strategy": combination_strategy.value,
                    "results": combined_results
                })
                
            elif combination_strategy == QueryCombinationStrategy.CONCATENATED:
                # Concatenate original query with keywords
                expanded_query_text = self._deduplicate_query_and_keywords(query_text, keywords)
                logger.info(f"Expanded query: {expanded_query_text}")
                
                # Search with concatenated query
                expanded_search_results = search_engine.search(
                    query=expanded_query_text,
                    bm25=bm25,
                    corpus_texts=corpus_texts,
                    corpus_ids=corpus_ids,
                    sbert_model=sbert_model,
                    doc_embeddings=doc_embeddings,
                    top_k=top_k_results,
                    method=RetrievalMethod.HYBRID,
                    hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                    use_mmr=False,
                    use_cross_encoder=False
                )
                
                expanded_results.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "expanded_text": expanded_query_text,
                    "strategy": combination_strategy.value,
                    "results": expanded_search_results
                })
                
            elif combination_strategy == QueryCombinationStrategy.CONCATENATED_RERANKED:
                # Concatenate original query with keywords
                expanded_query_text = self._deduplicate_query_and_keywords(query_text, keywords)
                logger.info(f"Expanded query (with reranking): {expanded_query_text}")
                
                # Search with concatenated query and cross-encoder reranking
                expanded_search_results = search_engine.search(
                    query=expanded_query_text,
                    bm25=bm25,
                    corpus_texts=corpus_texts,
                    corpus_ids=corpus_ids,
                    sbert_model=sbert_model,
                    doc_embeddings=doc_embeddings,
                    top_k=top_k_results,
                    method=RetrievalMethod.HYBRID,
                    hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                    use_mmr=False,
                    use_cross_encoder=True  # Enable reranking
                )
                
                expanded_results.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "expanded_text": expanded_query_text,
                    "strategy": combination_strategy.value,
                    "results": expanded_search_results
                })
            
            else:
                raise ValueError(f"Unsupported combination strategy: {combination_strategy}")
        
        result_data = {
            "expansion_method": expansion_method.value,
            "combination_strategy": combination_strategy.value,
            "num_keywords": num_keywords,
            "baseline_results": baseline_results,
            "expanded_results": expanded_results,
        }
        
        # Perform evaluation if qrels are provided
        if qrels_dataset is not None:
            # Use EvaluationUtils from search.py
            relevant_docs_by_query, highly_relevant_docs_by_query, overall_relevant_docs_by_query = (
                EvaluationUtils.build_qrels_dicts(qrels_dataset)
            )
            
            # Evaluate baseline results
            baseline_metrics = self._evaluate_search_results(
                baseline_results, 
                relevant_docs_by_query,
                highly_relevant_docs_by_query,
                overall_relevant_docs_by_query,
                top_k_p=20,
                top_k_r=top_k_results
            )
            
            # Evaluate expanded results
            expanded_metrics = self._evaluate_search_results(
                expanded_results, 
                relevant_docs_by_query,
                highly_relevant_docs_by_query,
                overall_relevant_docs_by_query,
                top_k_p=20,
                top_k_r=top_k_results
            )
            
            # Add metrics to result data
            result_data["baseline_metrics"] = baseline_metrics
            result_data["expanded_metrics"] = expanded_metrics
        
        # Cache results
        try:
            logger.info(f"Caching expansion results to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(result_data, f)
        except Exception as e:
            logger.warning(f"Error caching results: {e}")
        
        return result_data
    
    def _evaluate_search_results(
        self,
        search_results: List[Dict],
        relevant_docs_by_query: Dict[int, Set[str]],
        highly_relevant_docs_by_query: Dict[int, Set[str]],
        overall_relevant_docs_by_query: Dict[int, Set[str]],
        top_k_p: int = 20,
        top_k_r: int = 1000
    ) -> Dict:
        """
        Evaluate search results using metrics from search.py
        
        Args:
            search_results: List of search result items
            relevant_docs_by_query: Dictionary of relevant docs by query ID
            highly_relevant_docs_by_query: Dictionary of highly relevant docs by query ID
            overall_relevant_docs_by_query: Dictionary of all relevant docs by query ID
            top_k_p: Number of results to consider for precision
            top_k_r: Number of results to consider for recall
            
        Returns:
            Dictionary of evaluation metrics
        """
        all_precisions = {'relevant': [], 'highly_relevant': [], 'overall': []}
        all_recalls = {'relevant': [], 'highly_relevant': [], 'overall': []}
        num_evaluated = 0
        
        for result_item in search_results:
            query_id = result_item["query_id"]
            retrieved_results = result_item["results"]
            
            # Extract document IDs
            retrieved_docs = [doc_id for doc_id, _ in retrieved_results[:top_k_p]]
            full_retrieved_docs = [doc_id for doc_id, _ in retrieved_results[:top_k_r]]
            
            # Helper function to calculate precision and recall
            def calculate_precision_recall(relevant_set):
                if not relevant_set:
                    return None, None
                
                # Precision@k
                retrieved_set = set(retrieved_docs)
                num_relevant_retrieved = len(relevant_set.intersection(retrieved_set))
                precision = num_relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
                
                # Recall@k
                full_retrieved_set = set(full_retrieved_docs)
                num_relevant_in_full = len(relevant_set.intersection(full_retrieved_set))
                recall = num_relevant_in_full / len(relevant_set) if relevant_set else 0
                
                return precision, recall
            
            # Calculate metrics for different relevance levels
            if query_id in relevant_docs_by_query:
                precision, recall = calculate_precision_recall(relevant_docs_by_query[query_id])
                if precision is not None:
                    all_precisions['relevant'].append(precision)
                    all_recalls['relevant'].append(recall)
            
            if query_id in highly_relevant_docs_by_query:
                precision, recall = calculate_precision_recall(highly_relevant_docs_by_query[query_id])
                if precision is not None:
                    all_precisions['highly_relevant'].append(precision)
                    all_recalls['highly_relevant'].append(recall)
            
            if query_id in overall_relevant_docs_by_query:
                precision, recall = calculate_precision_recall(overall_relevant_docs_by_query[query_id])
                if precision is not None:
                    all_precisions['overall'].append(precision)
                    all_recalls['overall'].append(recall)
            
            num_evaluated += 1
        
        # Compute averages
        avg_precisions = {
            level: (sum(precisions) / len(precisions)) if precisions else 0.0
            for level, precisions in all_precisions.items()
        }
        
        avg_recalls = {
            level: (sum(recalls) / len(recalls)) if recalls else 0.0
            for level, recalls in all_recalls.items()
        }
        
        # Calculate F1 scores
        f1_scores = {}
        for level in ['relevant', 'highly_relevant', 'overall']:
            precision = avg_precisions[level]
            recall = avg_recalls[level]
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores[level] = f1
        
        return {
            "precisions": avg_precisions,
            "recalls": avg_recalls,
            "f1_scores": f1_scores,
            "num_evaluated": num_evaluated
        }
    
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
        force_regenerate: bool = False,
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
            force_regenerate: Whether to force regenerating results
            output_dir: Directory for output files
            
        Returns:
            Dictionary of evaluation results
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Track all results for visualization
        all_visualization_results = []
        results_by_config = {}
        
        # First, run baseline (no expansion)
        logger.info("Running baseline evaluation...")
        baseline_cache_path = os.path.join(self.cache_dir, "baseline_results.pkl")
        
        baseline_result = self.expand_and_search(
            queries_dataset=queries_dataset,
            corpus_dataset=corpus_dataset,
            qrels_dataset=qrels_dataset,
            # No expansion for baseline
            cache_path=baseline_cache_path,
            force_reindex=force_reindex,
            force_regenerate=force_regenerate
        )
        
        # Format baseline result for visualization
        if "baseline_metrics" in baseline_result:
            baseline_viz_result = {
                "config": {
                    "name": "Baseline",
                    "method": RetrievalMethod.HYBRID.value,
                    "hybrid_strategy": HybridStrategy.SIMPLE_SUM.value,
                    "use_mmr": False,
                    "use_cross_encoder": False
                },
                "avg_precisions": baseline_result["baseline_metrics"]["precisions"],
                "avg_recalls": baseline_result["baseline_metrics"]["recalls"],
                "num_evaluated": baseline_result["baseline_metrics"]["num_evaluated"]
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
            
            # Run expansion and search
            result_data = self.expand_and_search(
                queries_dataset=queries_dataset,
                corpus_dataset=corpus_dataset,
                qrels_dataset=qrels_dataset,
                expansion_method=expansion_method,
                combination_strategy=combination_strategy,
                num_keywords=num_keywords,
                top_n_docs_for_extraction=top_n_docs_for_extraction,
                top_k_results=top_k_results,
                force_reindex=force_reindex,
                force_regenerate=force_regenerate,
                cache_path=cache_path
            )
            
            # Store results
            results_by_config[config["name"]] = result_data
            
            # Format for visualization
            if "expanded_metrics" in result_data:
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
                    "avg_precisions": result_data["expanded_metrics"]["precisions"],
                    "avg_recalls": result_data["expanded_metrics"]["recalls"],
                    "num_evaluated": result_data["expanded_metrics"]["num_evaluated"]
                }
                all_visualization_results.append(viz_result)
        
        # Save combined results for visualization
        output_file = os.path.join(output_dir, "query_expansion_results.json")
        save_evaluation_results(all_visualization_results, output_file)
        
        # Generate visualization plots
        logger.info("Generating visualization plots...")
        plots_dir = os.path.join(output_dir, "plots")
        plot_paths = visualize_all_results(
            all_visualization_results,
            top_k_p=20,
            top_k_r=top_k_results,
            output_dir=plots_dir
        )
        
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
    FORCE_REGENERATE = False
    NUM_KEYWORDS = 5
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
        force_regenerate=FORCE_REGENERATE,
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
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"{config_name:<40} {precision:.4f}     {recall:.4f}     {f1:.4f}")
    
    logger.info(f"\nResults saved to {results['output_file']}")
    logger.info(f"Visualization plots saved to {os.path.dirname(results['plot_paths'][0])}")
    
    return results


# Execute main function if called directly
if __name__ == "__main__":
    main()
