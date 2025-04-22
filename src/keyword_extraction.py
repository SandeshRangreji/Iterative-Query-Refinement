# query_expansion.py
import logging
import numpy as np
import re
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import torch
from keybert import KeyBERT
import math

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


# Example usage
def main():
    """Main function to demonstrate query expansion functionality"""
    # Define constants
    QUERY = "covid symptoms"
    EXPANSION_METHODS = ["keybert", "pmi", "sopmi"]
    TERMS_PER_METHOD = 3
    LOG_LEVEL = 'INFO'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load dataset
    logger.info("Loading dataset...")
    from datasets import load_dataset
    
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    corpus_texts = [doc["title"] + "\n\n" + doc["text"] for doc in corpus_dataset]
    
    # Initialize expander
    logger.info("Initializing query expander...")
    expander = QueryExpander(corpus_texts)
    
    # Expand query
    logger.info(f"Expanding query: {QUERY}")
    expanded_query = expander.expand_query(
        QUERY,
        methods=EXPANSION_METHODS,
        num_terms_per_method=TERMS_PER_METHOD
    )
    
    print("\nOriginal query:", QUERY)
    print("Expanded query:", expanded_query)
    
    # Demonstrate individual methods
    print("\nExpansion by individual methods:")
    
    # KeyBERT expansion
    keybert_terms = expander.expand_with_keybert(QUERY, TERMS_PER_METHOD)
    print(f"KeyBERT: {', '.join(keybert_terms)}")
    
    # PMI expansion
    pmi_terms = expander.expand_with_pmi(QUERY, TERMS_PER_METHOD)
    print(f"PMI: {', '.join(pmi_terms)}")
    
    # Second-order PMI expansion
    sopmi_terms = expander.expand_with_second_order_pmi(QUERY, TERMS_PER_METHOD)
    print(f"Second-order PMI: {', '.join(sopmi_terms)}")


# Execute main function if called directly
if __name__ == "__main__":
    main()