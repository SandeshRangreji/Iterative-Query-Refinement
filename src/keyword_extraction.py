# query_expansion.py
import logging
import numpy as np
import re
import os
import json
import pickle
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import torch
from keybert import KeyBERT
import math
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm

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
        keybert_model: str = 'all-mpnet-base-v2',
        cache_dir: str = 'cache'
    ):
        """
        Initialize query expander
        
        Args:
            corpus_texts: List of document texts for co-occurrence analysis
            window_size: Window size for co-occurrence counting
            min_count: Minimum count for terms to be considered
            keybert_model: Model name for KeyBERT or SentenceTransformer model
            cache_dir: Directory to store cache files
        """
        self.corpus_texts = corpus_texts
        self.window_size = window_size
        self.min_count = min_count
        self.stop_words = set(stopwords.words('english'))
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize KeyBERT with sbert model
        if isinstance(keybert_model, str):
            self.sbert_model = SentenceTransformer(keybert_model)
            self.keybert = KeyBERT(model=self.sbert_model)
        else:
            # Assume it's already a SentenceTransformer model
            self.sbert_model = keybert_model
            self.keybert = KeyBERT(model=self.sbert_model)
        
        # Path for cooccurrence cache
        self.cooccurrence_cache_path = os.path.join(
            self.cache_dir, 
            f"cooccurrence_w{window_size}_m{min_count}.pkl"
        )
        
        # Path for expanded queries cache
        self.expanded_queries_cache_path = os.path.join(
            self.cache_dir,
            "expanded_queries.json"
        )
        
        # Load expanded queries cache if it exists
        self.expanded_queries_cache = self._load_expanded_queries_cache()
        
        # Build co-occurrence stats (lazy initialization)
        self._cooccurrence_matrix = None
        self._word_counts = None
        self._vocab = None
        self._word_to_idx = None
    
    def _load_expanded_queries_cache(self) -> Dict:
        """Load expanded queries cache"""
        if os.path.exists(self.expanded_queries_cache_path):
            try:
                with open(self.expanded_queries_cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading expanded queries cache: {e}")
                return {}
        return {}
    
    def _save_expanded_queries_cache(self):
        """Save expanded queries cache"""
        try:
            with open(self.expanded_queries_cache_path, 'w') as f:
                json.dump(self.expanded_queries_cache, f)
        except Exception as e:
            logger.warning(f"Error saving expanded queries cache: {e}")
    
    def _load_cooccurrence_cache(self) -> bool:
        """Load co-occurrence statistics from cache if available"""
        if os.path.exists(self.cooccurrence_cache_path):
            try:
                logger.info(f"Loading co-occurrence stats from cache: {self.cooccurrence_cache_path}")
                with open(self.cooccurrence_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self._cooccurrence_matrix = cache_data['matrix']
                    self._word_counts = cache_data['word_counts']
                    self._vocab = cache_data['vocab']
                    self._word_to_idx = cache_data['word_to_idx']
                return True
            except Exception as e:
                logger.warning(f"Error loading co-occurrence cache: {e}")
                return False
        return False
    
    def _save_cooccurrence_cache(self):
        """Save co-occurrence statistics to cache"""
        try:
            logger.info(f"Saving co-occurrence stats to cache: {self.cooccurrence_cache_path}")
            cache_data = {
                'matrix': self._cooccurrence_matrix,
                'word_counts': self._word_counts,
                'vocab': self._vocab,
                'word_to_idx': self._word_to_idx
            }
            with open(self.cooccurrence_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Error saving co-occurrence cache: {e}")
    
    def remove_stopwords(self, query: str) -> List[str]:
        """Remove stopwords from query"""
        words = word_tokenize(query)
        words = [word for word in words if not all(char in '"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' for char in word)]
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return filtered_words
    
    def compute_idf(self, top_docs_keywords: List[List[str]], total_docs: int) -> Dict[str, float]:
        """Compute IDF scores for keywords"""
        df = defaultdict(int)
        for kws in top_docs_keywords:
            for kw in set(kws):
                df[kw] += 1
        idf = {kw: math.log(total_docs / (1 + freq)) for kw, freq in df.items()}
        return idf
    
    def expand_with_keybert(
        self, 
        query: str, 
        num_keywords: int = 5,
        diversity: float = 0.7,
        query_embedding=None
    ) -> List[str]:
        """
        Expand query using KeyBERT with filtering based on similarity and IDF
        
        Args:
            query: Original query
            num_keywords: Number of keywords to extract
            diversity: Diversity parameter for MMR (0-1)
            query_embedding: Pre-computed query embedding
            
        Returns:
            List of expanded query terms
        """
        logger.info(f"Expanding query with KeyBERT: {query}")
        
        # Use pre-computed query embedding or create one
        if query_embedding is None:
            query_embedding = self.sbert_model.encode(query, convert_to_tensor=True)
        
        # Use query as seed keyword for KeyBERT
        seed_keywords = [query]
        logger.info(f"Using seed keywords: {seed_keywords}")
        
        # For shorter queries, use documents from retrieval to extract keywords
        # Since we don't have documents here, use the query itself but with larger ngram range
        # Extract keywords using KeyBERT with MMR for diversity
        original_keywords = self.keybert.extract_keywords(
            query, 
            keyphrase_ngram_range=(1, 3),  # Use shorter ngrams
            stop_words='english',
            use_mmr=True,
            diversity=diversity,
            top_n=20,  # Get more candidates for filtering
            nr_candidates=30  # Increase candidate pool
        )
        
        # Get just the keywords
        original_keywords = [k for k, _ in original_keywords]
        
        # Filter out keywords that are substrings of the query or vice versa
        filtered_keywords = []
        query_lower = query.lower()
        for kw in original_keywords:
            kw_lower = kw.lower()
            if kw_lower not in query_lower and query_lower not in kw_lower:
                filtered_keywords.append(kw)
        
        # If we have at least some filtered keywords, use them
        if filtered_keywords:
            expanded_terms = filtered_keywords[:num_keywords]
        else:
            # Fall back to original keywords without filtering
            expanded_terms = original_keywords[:num_keywords]
        
        logger.info(f"KeyBERT expansion: {expanded_terms}")
        return expanded_terms
    
    def _build_cooccurrence_stats(self):
        """Build co-occurrence statistics from corpus"""
        # Try to load from cache first
        if self._load_cooccurrence_cache():
            return
        
        logger.info("Building co-occurrence statistics from corpus...")
        
        # Tokenize and preprocess corpus
        tokenized_docs = []
        for doc in tqdm(self.corpus_texts, desc="Tokenizing documents"):
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
        for doc in tqdm(tokenized_docs, desc="Building co-occurrence matrix"):
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
        
        # Save to cache
        self._save_cooccurrence_cache()
        
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
        query_id: Optional[str] = None,
        methods: List[str] = ["keybert", "pmi", "sopmi"],
        num_terms_per_method: int = 3,
        deduplicate: bool = True,
        force_regenerate: bool = False
    ) -> str:
        """
        Expand query using multiple methods with caching
        
        Args:
            query: Original query
            query_id: Query ID for caching (if None, the query text is used)
            methods: List of expansion methods to use
            num_terms_per_method: Number of terms to add per method
            deduplicate: Whether to remove duplicate terms
            force_regenerate: Whether to force regeneration even if cached
            
        Returns:
            Expanded query string
        """
        # Use query as cache key if no ID provided
        cache_key = str(query_id) if query_id is not None else query
        
        # Check cache first if not forcing regeneration
        if not force_regenerate and cache_key in self.expanded_queries_cache:
            logger.info(f"Using cached expanded query for: {query}")
            return self.expanded_queries_cache[cache_key]
        
        # Precompute query embedding for KeyBERT if needed
        query_embedding = None
        if "keybert" in methods:
            query_embedding = self.sbert_model.encode(query, convert_to_tensor=True)
        
        expanded_terms = []
        
        for method in methods:
            if method.lower() == "keybert":
                terms = self.expand_with_keybert(query, num_terms_per_method, query_embedding=query_embedding)
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
        
        # Update cache
        self.expanded_queries_cache[cache_key] = expanded_query
        self._save_expanded_queries_cache()
        
        logger.info(f"Final expanded query: {expanded_query}")
        return expanded_query
    
    def expand_queries_batch(
        self,
        queries_dataset,
        methods: List[str] = ["keybert", "pmi", "sopmi"],
        num_terms_per_method: int = 3,
        deduplicate: bool = True,
        force_regenerate: bool = False
    ) -> Dict[str, str]:
        """
        Expand multiple queries and return a dictionary of expanded queries
        
        Args:
            queries_dataset: Dataset containing queries with _id and text fields
            methods: List of expansion methods to use
            num_terms_per_method: Number of terms to add per method
            deduplicate: Whether to remove duplicate terms
            force_regenerate: Whether to force regeneration even if cached
            
        Returns:
            Dictionary mapping query IDs to expanded queries
        """
        expanded_queries = {}
        
        # Use tqdm for the entire dataset instead of individual operations
        for query_item in tqdm(queries_dataset, desc="Expanding queries", total=len(queries_dataset)):
            query_id = str(query_item["_id"])
            query_text = query_item["text"]
            
            expanded_query = self.expand_query(
                query=query_text,
                query_id=query_id,
                methods=methods,
                num_terms_per_method=num_terms_per_method,
                deduplicate=deduplicate,
                force_regenerate=force_regenerate
            )
            
            expanded_queries[query_id] = expanded_query
        
        return expanded_queries


# Example usage
def main():
    """Main function to demonstrate query expansion functionality"""
    # Define constants
    EXPANSION_METHODS = ["keybert"] # ["keybert", "pmi", "sopmi"]
    TERMS_PER_METHOD = 5
    CACHE_DIR = "cache"
    FORCE_REGENERATE = True
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
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    
    corpus_texts = [doc["title"] + "\n\n" + doc["text"] for doc in corpus_dataset]
    
    # Initialize expander
    logger.info("Initializing query expander...")
    expander = QueryExpander(
        corpus_texts,
        cache_dir=CACHE_DIR
    )
    
    # Expand all queries
    logger.info("Expanding all queries...")
    expanded_queries = expander.expand_queries_batch(
        queries_dataset,
        methods=EXPANSION_METHODS,
        num_terms_per_method=TERMS_PER_METHOD,
        force_regenerate=FORCE_REGENERATE
    )
    
    # Print some examples
    logger.info("Example expansions:")
    for i, (query_id, expanded) in enumerate(list(expanded_queries.items())[:5]):
        original = queries_dataset[i]["text"]
        print(f"[Query {query_id}] Original: {original}")
        print(f"[Query {query_id}] Expanded: {expanded}")
        print()


# Execute main function if called directly
if __name__ == "__main__":
    main()