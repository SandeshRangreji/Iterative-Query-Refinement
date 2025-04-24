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
    EvaluationUtils
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
        qrels_dataset,
        expansion_method: QueryExpansionMethod = QueryExpansionMethod.KEYBERT,
        combination_strategy: QueryCombinationStrategy = QueryCombinationStrategy.WEIGHTED_RRF,
        num_keywords: int = 5,
        top_n_docs_for_extraction: int = 10,
        top_k_results: int = 1000,
        original_query_weight: float = 0.7,
        force_reindex: bool = False,
        force_regenerate: bool = False
    ):
        """
        Expand queries and search with different strategies
        
        Args:
            queries_dataset: Dataset containing queries
            corpus_dataset: Dataset containing corpus documents
            qrels_dataset: Dataset containing relevance judgments
            expansion_method: Method for keyword extraction (keybert, pmi, sopmi, combined)
            combination_strategy: How to combine expanded terms (rrf or concatenated)
            num_keywords: Number of keywords to extract per query
            top_n_docs_for_extraction: Number of docs to use for keyword extraction
            top_k_results: Number of results to retrieve per query
            original_query_weight: Weight for original query in RRF combination
            force_reindex: Whether to force rebuilding indices
            force_regenerate: Whether to force regenerating results
            
        Returns:
            Dictionary of evaluation results for baseline and expanded queries
        """
        # Try to load cached results first
        if not force_regenerate and os.path.exists(self.expansion_cache_path):
            try:
                logger.info(f"Loading cached expansion results from {self.expansion_cache_path}")
                with open(self.expansion_cache_path, "rb") as f:
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
                # Use QueryExpander functions from original implementation
                expander = self.keyword_extractor  # Reuse the keyword extractor
                keywords = expander.expand_with_pmi(query_text, num_terms=num_keywords)
            elif expansion_method == QueryExpansionMethod.SOPMI:
                expander = self.keyword_extractor  # Reuse the keyword extractor
                keywords = expander.expand_with_second_order_pmi(query_text, num_terms=num_keywords)
            elif expansion_method == QueryExpansionMethod.COMBINED:
                # Combine keywords from multiple methods
                keybert_keywords = self.keyword_extractor.extract_keywords(
                    query=query_text,
                    docs_text=top_docs,
                    num_keywords=num_keywords // 3 + 1
                )
                
                expander = self.keyword_extractor  # Reuse the keyword extractor
                pmi_keywords = expander.expand_with_pmi(query_text, num_terms=num_keywords // 3 + 1)
                sopmi_keywords = expander.expand_with_second_order_pmi(query_text, num_terms=num_keywords // 3 + 1)
                
                # Combine and deduplicate
                seen = set()
                keywords = []
                for kw in keybert_keywords + pmi_keywords + sopmi_keywords:
                    if kw not in seen and len(keywords) < num_keywords:
                        keywords.append(kw)
                        seen.add(kw)
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
        
        # Build qrels dictionaries for evaluation
        relevant_docs_by_query, highly_relevant_docs_by_query, overall_relevant_docs_by_query = (
            EvaluationUtils.build_qrels_dicts(qrels_dataset)
        )
        
        # Evaluate baseline results
        logger.info("Evaluating baseline search results...")
        baseline_precisions, baseline_recalls, baseline_num_evaluated = self._evaluate_results(
            results=baseline_results,
            relevant_docs_by_query=relevant_docs_by_query,
            highly_relevant_docs_by_query=highly_relevant_docs_by_query,
            overall_relevant_docs_by_query=overall_relevant_docs_by_query,
            top_k_p=20,
            top_k_r=top_k_results
        )
        
        # Evaluate expanded results
        logger.info("Evaluating expanded search results...")
        expanded_precisions, expanded_recalls, expanded_num_evaluated = self._evaluate_results(
            results=expanded_results,
            relevant_docs_by_query=relevant_docs_by_query,
            highly_relevant_docs_by_query=highly_relevant_docs_by_query,
            overall_relevant_docs_by_query=overall_relevant_docs_by_query,
            top_k_p=20,
            top_k_r=top_k_results
        )
        
        # Compile results
        result_data = {
            "baseline": {
                "config": {
                    "name": "Baseline",
                    "method": RetrievalMethod.HYBRID.value,
                    "hybrid_strategy": HybridStrategy.SIMPLE_SUM.value,
                    "use_mmr": False,
                    "use_cross_encoder": False
                },
                "avg_precisions": baseline_precisions,
                "avg_recalls": baseline_recalls,
                "num_evaluated": baseline_num_evaluated
            },
            "expanded": {
                "config": {
                    "name": f"{expansion_method.value.capitalize()} + {combination_strategy.value.replace('_', ' ').capitalize()}",
                    "method": RetrievalMethod.HYBRID.value,
                    "hybrid_strategy": HybridStrategy.SIMPLE_SUM.value,
                    "use_mmr": False,
                    "use_cross_encoder": combination_strategy == QueryCombinationStrategy.CONCATENATED_RERANKED,
                    "expansion_method": expansion_method.value,
                    "combination_strategy": combination_strategy.value,
                    "num_keywords": num_keywords
                },
                "avg_precisions": expanded_precisions,
                "avg_recalls": expanded_recalls,
                "num_evaluated": expanded_num_evaluated
            },
            "raw_data": {
                "baseline_results": baseline_results,
                "expanded_results": expanded_results
            }
        }
        
        # Cache results
        try:
            logger.info(f"Caching expansion results to {self.expansion_cache_path}")
            with open(self.expansion_cache_path, "wb") as f:
                pickle.dump(result_data, f)
        except Exception as e:
            logger.warning(f"Error caching results: {e}")
        
        return result_data
    
    def _evaluate_results(
        self,
        results: List[Dict],
        relevant_docs_by_query: Dict[int, Set[str]],
        highly_relevant_docs_by_query: Dict[int, Set[str]],
        overall_relevant_docs_by_query: Dict[int, Set[str]],
        top_k_p: int = 20,
        top_k_r: int = 1000
    ) -> Tuple[Dict[str, float], Dict[str, float], int]:
        """
        Evaluate search results
        
        Args:
            results: List of result dictionaries
            relevant_docs_by_query: Dictionary of relevant document sets by query ID
            highly_relevant_docs_by_query: Dictionary of highly relevant document sets by query ID
            overall_relevant_docs_by_query: Dictionary of all relevant document sets by query ID
            top_k_p: Number of results to consider for precision
            top_k_r: Number of results to consider for recall
            
        Returns:
            Tuple of (avg_precisions, avg_recalls, num_evaluated)
        """
        all_precisions = {'relevant': [], 'highly_relevant': [], 'overall': []}
        all_recalls = {'relevant': [], 'highly_relevant': [], 'overall': []}
        num_evaluated = 0
        
        for result_item in results:
            query_id = result_item["query_id"]
            search_results = result_item["results"]
            
            # Extract document IDs
            retrieved_docs = [doc_id for doc_id, _ in search_results[:top_k_p]]
            full_retrieved_docs = [doc_id for doc_id, _ in search_results[:top_k_r]]
            
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
        
        return avg_precisions, avg_recalls, num_evaluated
    
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
        
        # Track all results
        all_results = []
        
        # First, load or build indices once
        bm25, corpus_texts, corpus_ids, sbert_model, doc_embeddings = self._load_or_build_indices(
            corpus_dataset,
            force_reindex=force_reindex
        )
        
        # Initialize search engine once
        search_engine = SearchEngine(self.preprocessor, self.cross_encoder_model_name)
        
        # Build qrels dictionaries for evaluation once
        relevant_docs_by_query, highly_relevant_docs_by_query, overall_relevant_docs_by_query = (
            EvaluationUtils.build_qrels_dicts(qrels_dataset)
        )
        
        # Run baseline evaluation once (no expansion)
        logger.info("Running baseline evaluation...")
        baseline_cache_path = os.path.join(self.cache_dir, "baseline_results.pkl")
        
        if not force_regenerate and os.path.exists(baseline_cache_path):
            try:
                logger.info(f"Loading cached baseline results from {baseline_cache_path}")
                with open(baseline_cache_path, "rb") as f:
                    baseline_data = pickle.load(f)
                    baseline_result = baseline_data["baseline"]
            except Exception as e:
                logger.warning(f"Error loading cached baseline results: {e}")
                baseline_result = None
        else:
            baseline_result = None
        
        if baseline_result is None:
            # Baseline search (hybrid simple sum, no MMR, no cross-encoder)
            baseline_results = []
            for query_item in tqdm(queries_dataset, desc="Running baseline searches"):
                query_id = int(query_item["_id"])
                query_text = query_item["text"]
                
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
                
                baseline_results.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "results": baseline_search_results
                })
            
            # Evaluate baseline results
            logger.info("Evaluating baseline search results...")
            baseline_precisions, baseline_recalls, baseline_num_evaluated = self._evaluate_results(
                results=baseline_results,
                relevant_docs_by_query=relevant_docs_by_query,
                highly_relevant_docs_by_query=highly_relevant_docs_by_query,
                overall_relevant_docs_by_query=overall_relevant_docs_by_query,
                top_k_p=20,
                top_k_r=top_k_results
            )
            
            baseline_result = {
                "config": {
                    "name": "Baseline",
                    "method": RetrievalMethod.HYBRID.value,
                    "hybrid_strategy": HybridStrategy.SIMPLE_SUM.value,
                    "use_mmr": False,
                    "use_cross_encoder": False
                },
                "avg_precisions": baseline_precisions,
                "avg_recalls": baseline_recalls,
                "num_evaluated": baseline_num_evaluated,
                "raw_results": baseline_results
            }
            
            # Cache baseline results
            try:
                logger.info(f"Caching baseline results to {baseline_cache_path}")
                with open(baseline_cache_path, "wb") as f:
                    pickle.dump({"baseline": baseline_result}, f)
            except Exception as e:
                logger.warning(f"Error caching baseline results: {e}")
        
        # Add baseline to all results
        all_results.append(baseline_result)
        
        # Extract keywords for all queries using the existing method
        logger.info("Extracting keywords for all queries...")
        all_keywords = self.keyword_extractor.extract_keywords_for_queries(
            queries_dataset=queries_dataset,
            corpus_dataset=corpus_dataset,
            num_keywords=num_keywords,
            diversity=0.7,
            force_regenerate=force_regenerate,
            sbert_model_name=self.sbert_model_name,
            force_reindex=force_reindex
        )
        
        # Now run evaluations for each expansion strategy using the extracted keywords
        for config in expansion_configs:
            logger.info(f"Evaluating {config['name']}...")
            
            # Try to load cached results for this config
            if not force_regenerate and os.path.exists(config["cache_path"]):
                try:
                    logger.info(f"Loading cached results from {config['cache_path']}")
                    with open(config["cache_path"], "rb") as f:
                        result_data = pickle.load(f)
                        all_results.append(result_data["expanded"])
                        continue
                except Exception as e:
                    logger.warning(f"Error loading cached results: {e}")
            
            # Perform search with the specific expansion strategy
            expanded_results = []
            for query_item in tqdm(queries_dataset, desc=f"Running searches with {config['name']}"):
                query_id = str(query_item["_id"])
                query_text = query_item["text"]
                
                # Get keywords for this query
                if query_id in all_keywords:
                    keywords = all_keywords[query_id]
                else:
                    logger.warning(f"No keywords found for query {query_id}, skipping")
                    continue
                
                # Apply the specific expansion strategy
                if config["combination_strategy"] == QueryCombinationStrategy.WEIGHTED_RRF:
                    # First, get baseline results
                    baseline_results = search_engine.search(
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
                    all_rankings = [baseline_results] + keyword_results
                    weights = [0.7] + [(0.3 / len(keywords))] * len(keywords)
                    
                    combined_results = self._reciprocal_rank_fusion(all_rankings, weights)
                    
                    expanded_results.append({
                        "query_id": int(query_id),
                        "query_text": query_text,
                        "expanded_text": ", ".join(keywords),  # Just for logging
                        "strategy": config["combination_strategy"].value,
                        "results": combined_results
                    })
                    
                elif config["combination_strategy"] == QueryCombinationStrategy.CONCATENATED:
                    # Concatenate original query with keywords
                    expanded_query_text = self._deduplicate_query_and_keywords(query_text, keywords)
                    
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
                        "query_id": int(query_id),
                        "query_text": query_text,
                        "expanded_text": expanded_query_text,
                        "strategy": config["combination_strategy"].value,
                        "results": expanded_search_results
                    })
                    
                elif config["combination_strategy"] == QueryCombinationStrategy.CONCATENATED_RERANKED:
                    # Concatenate original query with keywords
                    expanded_query_text = self._deduplicate_query_and_keywords(query_text, keywords)
                    
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
                        "query_id": int(query_id),
                        "query_text": query_text,
                        "expanded_text": expanded_query_text,
                        "strategy": config["combination_strategy"].value,
                        "results": expanded_search_results
                    })
            
            # Evaluate results for this expansion strategy
            logger.info(f"Evaluating {config['name']} results...")
            expanded_precisions, expanded_recalls, expanded_num_evaluated = self._evaluate_results(
                results=expanded_results,
                relevant_docs_by_query=relevant_docs_by_query,
                highly_relevant_docs_by_query=highly_relevant_docs_by_query,
                overall_relevant_docs_by_query=overall_relevant_docs_by_query,
                top_k_p=20,
                top_k_r=top_k_results
            )
            
            expanded_result = {
                "config": {
                    "name": config["name"],
                    "method": RetrievalMethod.HYBRID.value,
                    "hybrid_strategy": HybridStrategy.SIMPLE_SUM.value,
                    "use_mmr": False,
                    "use_cross_encoder": config["combination_strategy"] == QueryCombinationStrategy.CONCATENATED_RERANKED,
                    "expansion_method": config["expansion_method"].value,
                    "combination_strategy": config["combination_strategy"].value,
                    "num_keywords": num_keywords
                },
                "avg_precisions": expanded_precisions,
                "avg_recalls": expanded_recalls,
                "num_evaluated": expanded_num_evaluated,
                "raw_results": expanded_results
            }
            
            # Add to all results
            all_results.append(expanded_result)
            
            # Cache results
            try:
                logger.info(f"Caching results to {config['cache_path']}")
                with open(config["cache_path"], "wb") as f:
                    pickle.dump({"expanded": expanded_result}, f)
            except Exception as e:
                logger.warning(f"Error caching results: {e}")
        
        # Save combined results
        output_file = os.path.join(output_dir, "query_expansion_results.json")
        
        # Convert results to serializable format
        serializable_results = []
        for result in all_results:
            serializable_result = {
                "config": {
                    "name": result["config"]["name"],
                    "method": result["config"]["method"],
                    "use_mmr": result["config"]["use_mmr"],
                    "use_cross_encoder": result["config"]["use_cross_encoder"]
                },
                "avg_precisions": result["avg_precisions"],
                "avg_recalls": result["avg_recalls"],
                "num_evaluated": result["num_evaluated"]
            }
            
            # Add hybrid strategy if present
            if "hybrid_strategy" in result["config"]:
                serializable_result["config"]["hybrid_strategy"] = result["config"]["hybrid_strategy"]
                
            # Add expansion-specific params if present
            for param in ["expansion_method", "combination_strategy", "num_keywords"]:
                if param in result["config"]:
                    serializable_result["config"][param] = result["config"][param]
                
            serializable_results.append(serializable_result)
        
        # Save to file
        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_file}")
        
        # Generate visualization plots
        logger.info("Generating visualization plots...")
        plots_dir = os.path.join(output_dir, "plots")
        plot_paths = visualize_all_results(
            serializable_results,
            top_k_p=20,
            top_k_r=top_k_results,
            output_dir=plots_dir
        )
        logger.info(f"Generated {len(plot_paths)} visualization plots in {plots_dir}")
        
        return {
            "all_results": all_results,
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
    ORIGINAL_WEIGHT = 0.7
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
            "expansion_method": QueryExpansionMethod.KEYBERT,
            "combination_strategy": QueryCombinationStrategy.WEIGHTED_RRF,
            "cache_path": os.path.join(CACHE_DIR, "keybert_rrf_results.pkl")
        },
        {
            "name": "KeyBERT + Concatenated",
            "expansion_method": QueryExpansionMethod.KEYBERT,
            "combination_strategy": QueryCombinationStrategy.CONCATENATED,
            "cache_path": os.path.join(CACHE_DIR, "keybert_concat_results.pkl")
        },
        {
            "name": "KeyBERT + Concatenated + CrossEncoder",
            "expansion_method": QueryExpansionMethod.KEYBERT,
            "combination_strategy": QueryCombinationStrategy.CONCATENATED_RERANKED,
            "cache_path": os.path.join(CACHE_DIR, "keybert_concat_rerank_results.pkl")
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
    
    for result in results["all_results"]:
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
