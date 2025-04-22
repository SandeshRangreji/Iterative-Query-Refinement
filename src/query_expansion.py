# query_expansion.py
import logging
import numpy as np
import re
from typing import List, Dict, Tuple, Set, Union, Optional
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import torch
from keybert import KeyBERT
import math

# Import functionality from search.py
from search import (
    TextPreprocessor, 
    SearchEngine, 
    IndexManager,
    RetrievalMethod,
    HybridStrategy,
    EvaluationUtils
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class QueryExpander:
    """Class for query expansion methods"""
    
    def __init__(
        self, 
        corpus_texts: List[str],
        window_size: int = 10,
        min_count: int = 3,
        keybert_model: str = 'all-mpnet-base-v2'
    ):
        """
        Initialize query expander
        
        Args:
            corpus_texts: List of document texts for co-occurrence analysis
            window_size: Window size for co-occurrence counting
            min_count: Minimum count for terms to be considered
            keybert_model: Model name for KeyBERT
        """
        self.corpus_texts = corpus_texts
        self.window_size = window_size
        self.min_count = min_count
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize KeyBERT
        self.keybert = KeyBERT(model=keybert_model)
        
        # Build co-occurrence stats (lazy initialization)
        self._cooccurrence_matrix = None
        self._word_counts = None
        self._vocab = None
        self._word_to_idx = None
    
    def expand_with_keybert(
        self, 
        query: str, 
        num_keywords: int = 5,
        diversity: float = 0.5
    ) -> List[str]:
        """
        Expand query using KeyBERT
        
        Args:
            query: Original query
            num_keywords: Number of keywords to extract
            diversity: Diversity parameter for MMR (0-1)
            
        Returns:
            List of expanded query terms
        """
        logger.info(f"Expanding query with KeyBERT: {query}")
        
        # Extract keywords using KeyBERT with MMR for diversity
        keywords = self.keybert.extract_keywords(
            query, 
            keyphrase_ngram_range=(1, 2), 
            stop_words='english',
            use_mmr=True,
            diversity=diversity,
            top_n=num_keywords
        )
        
        # Extract just the keywords (not scores)
        expanded_terms = [k for k, _ in keywords]
        
        logger.info(f"KeyBERT expansion: {expanded_terms}")
        return expanded_terms
    
    def _build_cooccurrence_stats(self):
        """Build co-occurrence statistics from corpus"""
        if self._cooccurrence_matrix is not None:
            return
        
        logger.info("Building co-occurrence statistics from corpus...")
        
        # Tokenize and preprocess corpus
        tokenized_docs = []
        for doc in self.corpus_texts:
            tokens = word_tokenize(doc.lower())
            # Filter stopwords and non-alphabetic tokens
            tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words and len(t) > 2]
            tokenized_docs.append(tokens)
        
        # Build vocabulary with frequency filtering
        word_counts = Counter()
        for doc in tokenized_docs:
            word_counts.update(doc)
        
        # Filter by minimum count
        vocab = [word for word, count in word_counts.items() if count >= self.min_count]
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        
        # Initialize co-occurrence matrix
        cooccurrence_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
        
        # Count co-occurrences within window
        for doc in tokenized_docs:
            for i, word in enumerate(doc):
                if word not in word_to_idx:
                    continue
                    
                word_idx = word_to_idx[word]
                
                # Look at words within window
                window_start = max(0, i - self.window_size)
                window_end = min(len(doc), i + self.window_size + 1)
                
                for j in range(window_start, window_end):
                    if i == j:
                        continue
                        
                    context_word = doc[j]
                    if context_word not in word_to_idx:
                        continue
                        
                    context_idx = word_to_idx[context_word]
                    cooccurrence_matrix[word_idx, context_idx] += 1
        
        # Store results
        self._cooccurrence_matrix = cooccurrence_matrix
        self._word_counts = word_counts
        self._vocab = vocab
        self._word_to_idx = word_to_idx
        
        logger.info(f"Built co-occurrence statistics with {len(vocab)} terms")
    
    def expand_with_pmi(
        self, 
        query: str, 
        num_terms: int = 5,
        min_pmi: float = 0.0
    ) -> List[str]:
        """
        Expand query using Pointwise Mutual Information
        
        Args:
            query: Original query
            num_terms: Number of terms to add
            min_pmi: Minimum PMI threshold
            
        Returns:
            List of expanded query terms
        """
        logger.info(f"Expanding query with PMI: {query}")
        
        # Build co-occurrence stats if needed
        self._build_cooccurrence_stats()
        
        # Tokenize query
        query_tokens = word_tokenize(query.lower())
        query_tokens = [t for t in query_tokens if t.isalpha() and t not in self.stop_words]
        
        # Find query terms in vocabulary
        query_indices = []
        for token in query_tokens:
            if token in self._word_to_idx:
                query_indices.append(self._word_to_idx[token])
        
        if not query_indices:
            logger.warning("No query terms found in vocabulary")
            return []
        
        # Calculate PMI for all terms with query terms
        expansion_scores = defaultdict(float)
        corpus_size = sum(self._word_counts.values())
        
        for query_idx in query_indices:
            query_term = self._vocab[query_idx]
            query_count = self._word_counts[query_term]
            
            for candidate_idx, candidate_term in enumerate(self._vocab):
                if candidate_idx in query_indices:
                    continue
                    
                candidate_count = self._word_counts[candidate_term]
                cooccurrence = self._cooccurrence_matrix[query_idx, candidate_idx]
                
                if cooccurrence == 0:
                    continue
                
                # Calculate PMI
                p_xy = cooccurrence / corpus_size
                p_x = query_count / corpus_size
                p_y = candidate_count / corpus_size
                pmi = math.log2(p_xy / (p_x * p_y))
                
                if pmi > min_pmi:
                    expansion_scores[candidate_term] += pmi
        
        # Sort by score and take top-n
        expansion_terms = sorted(
            expansion_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:num_terms]
        
        expanded_query = [term for term, _ in expansion_terms]
        
        logger.info(f"PMI expansion: {expanded_query}")
        return expanded_query
    
    def expand_with_second_order_pmi(
        self, 
        query: str, 
        num_terms: int = 5,
        min_similarity: float = 0.3
    ) -> List[str]:
        """
        Expand query using Second-order PMI
        
        Args:
            query: Original query
            num_terms: Number of terms to add
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of expanded query terms
        """
        logger.info(f"Expanding query with Second-order PMI: {query}")
        
        # Build co-occurrence stats if needed
        self._build_cooccurrence_stats()
        
        # Tokenize query
        query_tokens = word_tokenize(query.lower())
        query_tokens = [t for t in query_tokens if t.isalpha() and t not in self.stop_words]
        
        # Find query terms in vocabulary
        query_indices = []
        for token in query_tokens:
            if token in self._word_to_idx:
                query_indices.append(self._word_to_idx[token])
        
        if not query_indices:
            logger.warning("No query terms found in vocabulary")
            return []
        
        # Calculate PMI vectors for query terms
        query_pmi_vectors = []
        for query_idx in query_indices:
            # Get co-occurrence vector
            cooc_vector = self._cooccurrence_matrix[query_idx]
            
            # Convert to PMI
            corpus_size = sum(self._word_counts.values())
            query_term = self._vocab[query_idx]
            query_count = self._word_counts[query_term]
            p_x = query_count / corpus_size
            
            pmi_vector = np.zeros_like(cooc_vector)
            for i, cooc in enumerate(cooc_vector):
                if cooc > 0:
                    cand_term = self._vocab[i]
                    p_y = self._word_counts[cand_term] / corpus_size
                    p_xy = cooc / corpus_size
                    pmi_vector[i] = max(0, math.log2(p_xy / (p_x * p_y)))
            
            query_pmi_vectors.append(pmi_vector)
        
        # Combine query PMI vectors
        if len(query_pmi_vectors) > 1:
            combined_pmi = np.mean(query_pmi_vectors, axis=0)
        else:
            combined_pmi = query_pmi_vectors[0]
        
        # Calculate cosine similarity between combined PMI and all terms
        expansion_scores = {}
        for candidate_idx, candidate_term in enumerate(self._vocab):
            if candidate_idx in query_indices:
                continue
                
            candidate_pmi = self._cooccurrence_matrix[candidate_idx]
            
            # Convert to PMI
            candidate_count = self._word_counts[candidate_term]
            p_x = candidate_count / corpus_size
            
            for i, cooc in enumerate(candidate_pmi):
                if cooc > 0:
                    cand_term = self._vocab[i]
                    p_y = self._word_counts[cand_term] / corpus_size
                    p_xy = cooc / corpus_size
                    candidate_pmi[i] = max(0, math.log2(p_xy / (p_x * p_y)))
            
            # Calculate cosine similarity
            norm_a = np.linalg.norm(combined_pmi)
            norm_b = np.linalg.norm(candidate_pmi)
            
            if norm_a > 0 and norm_b > 0:
                similarity = np.dot(combined_pmi, candidate_pmi) / (norm_a * norm_b)
                
                if similarity >= min_similarity:
                    expansion_scores[candidate_term] = similarity
        
        # Sort by score and take top-n
        expansion_terms = sorted(
            expansion_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:num_terms]
        
        expanded_query = [term for term, _ in expansion_terms]
        
        logger.info(f"Second-order PMI expansion: {expanded_query}")
        return expanded_query
    
    def expand_query(
        self, 
        query: str,
        methods: List[str] = ["keybert", "pmi", "sopmi"],
        num_terms_per_method: int = 3,
        deduplicate: bool = True
    ) -> str:
        """
        Expand query using multiple methods
        
        Args:
            query: Original query
            methods: List of expansion methods to use
            num_terms_per_method: Number of terms to add per method
            deduplicate: Whether to remove duplicate terms
            
        Returns:
            Expanded query string
        """
        expanded_terms = []
        
        for method in methods:
            if method.lower() == "keybert":
                terms = self.expand_with_keybert(query, num_terms_per_method)
            elif method.lower() == "pmi":
                terms = self.expand_with_pmi(query, num_terms_per_method)
            elif method.lower() == "sopmi":
                terms = self.expand_with_second_order_pmi(query, num_terms_per_method)
            else:
                logger.warning(f"Unknown expansion method: {method}")
                continue
                
            expanded_terms.extend(terms)
        
        # Deduplicate
        if deduplicate:
            expanded_terms = list(dict.fromkeys(expanded_terms))
        
        # Combine with original query
        expanded_query = f"{query} {' '.join(expanded_terms)}"
        
        logger.info(f"Final expanded query: {expanded_query}")
        return expanded_query


def evaluate_expansion_methods(
    corpus_dataset,
    queries_dataset,
    qrels_dataset,
    force_reindex: bool = False,
    top_k_p: int = 20,
    top_k_r: int = 1000,
    sbert_model_name: str = "all-mpnet-base-v2",
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    expansion_methods: List[Dict] = None
):
    """
    Evaluate different query expansion methods
    
    Args:
        corpus_dataset: Dataset containing the corpus
        queries_dataset: Dataset containing the queries
        qrels_dataset: Dataset containing relevance judgments
        force_reindex: Whether to force rebuilding indices
        top_k_p: Number of results to consider for precision
        top_k_r: Number of results to consider for recall
        sbert_model_name: Name of the sentence transformer model
        cross_encoder_model_name: Name of the cross-encoder model
        expansion_methods: List of expansion configurations to evaluate
        
    Returns:
        Dictionary of evaluation results
    """
    # Initialize components
    preprocessor = TextPreprocessor()
    index_manager = IndexManager(preprocessor)
    
    # Build indices
    logger.info("Building BM25 index...")
    bm25, corpus_texts, corpus_ids = index_manager.build_bm25_index(
        corpus_dataset,
        cache_path="bm25_index.pkl",
        force_reindex=force_reindex
    )
    
    logger.info("Building SBERT index...")
    sbert_model, doc_embeddings = index_manager.build_sbert_index(
        corpus_texts,
        model_name=sbert_model_name,
        batch_size=64,
        cache_path="sbert_index.pt",
        force_reindex=force_reindex
    )
    
    # Initialize search engine
    search_engine = SearchEngine(preprocessor, cross_encoder_model_name)
    
    # Initialize query expander
    query_expander = QueryExpander(corpus_texts, keybert_model=sbert_model_name)
    
    # Default expansion methods if none provided
    if expansion_methods is None:
        expansion_methods = [
            {
                "name": "No Expansion (Baseline)",
                "methods": [],
                "search_method": RetrievalMethod.HYBRID,
                "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
                "use_mmr": False,
                "use_cross_encoder": False
            },
            {
                "name": "KeyBERT Expansion",
                "methods": ["keybert"],
                "num_terms": 5,
                "search_method": RetrievalMethod.HYBRID,
                "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
                "use_mmr": False,
                "use_cross_encoder": False
            },
            {
                "name": "PMI Expansion",
                "methods": ["pmi"],
                "num_terms": 5,
                "search_method": RetrievalMethod.HYBRID,
                "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
                "use_mmr": False,
                "use_cross_encoder": False
            },
            {
                "name": "SoPMI Expansion",
                "methods": ["sopmi"],
                "num_terms": 5,
                "search_method": RetrievalMethod.HYBRID,
                "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
                "use_mmr": False,
                "use_cross_encoder": False
            },
            {
                "name": "Combined Expansion",
                "methods": ["keybert", "pmi", "sopmi"],
                "num_terms": 3,
                "search_method": RetrievalMethod.HYBRID,
                "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
                "use_mmr": True,
                "use_cross_encoder": True
            }
        ]
    
    # Build qrels dictionaries
    relevant_docs_by_query, highly_relevant_docs_by_query, overall_relevant_docs_by_query = (
        EvaluationUtils.build_qrels_dicts(qrels_dataset)
    )
    
    # Run evaluations
    results = []
    for config in expansion_methods:
        logger.info(f"Evaluating expansion method: {config['name']}")
        
        # Initialize metrics containers
        all_precisions = {'relevant': [], 'highly_relevant': [], 'overall': []}
        all_recalls = {'relevant': [], 'highly_relevant': [], 'overall': []}
        num_evaluated = 0
        
        # Get search parameters
        search_method = config.get("search_method", RetrievalMethod.HYBRID)
        hybrid_strategy = config.get("hybrid_strategy", HybridStrategy.SIMPLE_SUM)
        hybrid_weight = config.get("hybrid_weight", 0.5)
        use_mmr = config.get("use_mmr", False)
        mmr_lambda = config.get("mmr_lambda", 0.5)
        use_cross_encoder = config.get("use_cross_encoder", False)
        
        # Evaluate each query
        for query_item in queries_dataset:
            query_id = int(query_item["_id"])
            original_query = query_item["text"]
            
            # Expand query if methods are specified
            if config.get("methods", []):
                expanded_query = query_expander.expand_query(
                    original_query,
                    methods=config["methods"],
                    num_terms_per_method=config.get("num_terms", 3),
                    deduplicate=True
                )
            else:
                expanded_query = original_query
            
            # Log for debugging
            logger.debug(f"Query {query_id}: '{original_query}' -> '{expanded_query}'")
            
            # Retrieve documents using expanded query
            search_results = search_engine.search(
                query=expanded_query,
                bm25=bm25,
                corpus_texts=corpus_texts,
                corpus_ids=corpus_ids,
                sbert_model=sbert_model,
                doc_embeddings=doc_embeddings,
                top_k=max(top_k_p, top_k_r),
                method=search_method,
                hybrid_strategy=hybrid_strategy,
                hybrid_weight=hybrid_weight,
                use_mmr=use_mmr,
                mmr_lambda=mmr_lambda,
                use_cross_encoder=use_cross_encoder
            )
            
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
        
        # Log results
        logger.info(f"[{config['name']}] Queries evaluated: {num_evaluated}")
        for level in ['relevant', 'highly_relevant', 'overall']:
            logger.info(f"[{config['name']}] [{level.capitalize()}] Precision@{top_k_p}: {avg_precisions[level]:.4f}, Recall@{top_k_r}: {avg_recalls[level]:.4f}")
        
        # Store results
        results.append({
            "config": config,
            "avg_precisions": avg_precisions,
            "avg_recalls": avg_recalls,
            "num_evaluated": num_evaluated
        })
    
    return results


def main():
    """Main function to evaluate query expansion methods"""
    # Define constants
    FORCE_REINDEX = False
    TOP_K_P = 20
    TOP_K_R = 1000
    SBERT_MODEL = 'all-mpnet-base-v2'
    CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    LOG_LEVEL = 'INFO'
    
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
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")
    logger.info(f"Loaded {len(corpus_dataset)} documents, {len(queries_dataset)} queries, {len(qrels_dataset)} relevance judgments")
    
    # Define expansion methods to evaluate
    expansion_configs = [
        {
            "name": "No Expansion (Baseline)",
            "methods": [],
            "search_method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "use_mmr": False,
            "use_cross_encoder": False
        },
        {
            "name": "KeyBERT Expansion",
            "methods": ["keybert"],
            "num_terms": 5,
            "search_method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "use_mmr": False,
            "use_cross_encoder": False
        },
        {
            "name": "PMI Expansion",
            "methods": ["pmi"],
            "num_terms": 5,
            "search_method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "use_mmr": False,
            "use_cross_encoder": False
        },
        {
            "name": "SoPMI Expansion",
            "methods": ["sopmi"],
            "num_terms": 5,
            "search_method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "use_mmr": False,
            "use_cross_encoder": False
        },
        {
            "name": "Combined Expansion",
            "methods": ["keybert", "pmi", "sopmi"],
            "num_terms": 3,
            "search_method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "use_mmr": True,
            "mmr_lambda": 0.5,
            "use_cross_encoder": True
        },
        {
            "name": "KeyBERT + MMR",
            "methods": ["keybert"],
            "num_terms": 5,
            "search_method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "use_mmr": True,
            "mmr_lambda": 0.5,
            "use_cross_encoder": False
        },
        {
            "name": "KeyBERT + CrossEncoder",
            "methods": ["keybert"],
            "num_terms": 5,
            "search_method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "use_mmr": False,
            "use_cross_encoder": True
        }
    ]
    
    # Run evaluation
    results = evaluate_expansion_methods(
        corpus_dataset=corpus_dataset,
        queries_dataset=queries_dataset,
        qrels_dataset=qrels_dataset,
        force_reindex=FORCE_REINDEX,
        top_k_p=TOP_K_P,
        top_k_r=TOP_K_R,
        sbert_model_name=SBERT_MODEL,
        cross_encoder_model_name=CROSS_ENCODER_MODEL,
        expansion_methods=expansion_configs
    )
    
    # Print results summary
    logger.info("\n===== QUERY EXPANSION EVALUATION RESULTS =====")
    logger.info(f"{'Method':<30} {'P@'+str(TOP_K_P):<10} {'R@'+str(TOP_K_R):<10}")
    logger.info("-" * 50)
    
    for result in results:
        config_name = result["config"]["name"]
        precision = result["avg_precisions"]["overall"]
        recall = result["avg_recalls"]["overall"]
        logger.info(f"{config_name:<30} {precision:.4f}     {recall:.4f}")
    
    return results


# Execute main function if called directly
if __name__ == "__main__":
    main()