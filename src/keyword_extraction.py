# keyword_extraction.py
import logging
import os
import json
import pickle
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import from search.py
from search import IndexManager, SearchEngine, TextPreprocessor, RetrievalMethod, HybridStrategy

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
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class KeywordExtractor:
    """Class for extracting keywords from retrieved documents"""
    
    def __init__(
        self, 
        keybert_model: str = 'all-mpnet-base-v2',
        cache_dir: str = 'cache',
        top_k_docs: int = 1000,  # Number of docs to retrieve
        top_n_docs_for_extraction: int = 10  # Number of docs to use for extraction
    ):
        """
        Initialize keyword extractor
        
        Args:
            keybert_model: Model name for KeyBERT or SentenceTransformer model
            cache_dir: Directory to store cache files
            top_k_docs: Number of documents to retrieve
            top_n_docs_for_extraction: Number of top documents to use for extraction
        """
        self.top_k_docs = top_k_docs
        self.top_n_docs_for_extraction = top_n_docs_for_extraction
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
        
        # Path for keywords cache
        self.keywords_cache_path = os.path.join(
            self.cache_dir,
            "extracted_keywords.json"
        )
        
        # Load keywords cache if it exists
        self.keywords_cache = self._load_keywords_cache()
    
    def _load_keywords_cache(self) -> Dict:
        """Load keywords cache"""
        if os.path.exists(self.keywords_cache_path):
            try:
                with open(self.keywords_cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading keywords cache: {e}")
                return {}
        return {}
    
    def _save_keywords_cache(self):
        """Save keywords cache"""
        try:
            with open(self.keywords_cache_path, 'w') as f:
                json.dump(self.keywords_cache, f)
        except Exception as e:
            logger.warning(f"Error saving keywords cache: {e}")
    
    def extract_keywords(
        self, 
        query: str, 
        docs_text: List[str],
        num_keywords: int = 5,
        diversity: float = 0.7,
        keyphrase_ngram_range: Tuple[int, int] = (1, 2)
    ) -> List[str]:
        """
        Extract keywords from documents using KeyBERT with diversity
        
        Args:
            query: Original query
            docs_text: List of document texts
            num_keywords: Number of keywords to extract
            diversity: Diversity parameter for MMR (0-1)
            keyphrase_ngram_range: Range of n-gram lengths for keyphrases
            
        Returns:
            List of extracted keywords
        """
        # Concatenate document texts with double newlines
        combined_text = "\n\n".join(docs_text)
        
        # Extract keywords using KeyBERT with MMR for diversity
        keywords = self.keybert.extract_keywords(
            combined_text, 
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words='english',
            use_mmr=True,
            diversity=diversity,
            top_n=num_keywords,
            nr_candidates=num_keywords * 3  # Increase candidate pool
        )
        
        # Get just the keywords without scores
        extracted_keywords = [k for k, _ in keywords]
        
        return extracted_keywords
    
    def extract_keywords_for_queries(
        self,
        queries_dataset,
        corpus_dataset,
        num_keywords: int = 5,
        diversity: float = 0.7,
        force_regenerate: bool = False,
        sbert_model_name: str = 'all-mpnet-base-v2',
        force_reindex: bool = False
    ) -> Dict[str, List[str]]:
        """
        Extract keywords for multiple queries based on retrieved documents
        
        Args:
            queries_dataset: Dataset containing queries with _id and text fields
            corpus_dataset: Dataset containing corpus documents
            num_keywords: Number of keywords to extract per query
            diversity: Diversity parameter for MMR
            force_regenerate: Whether to force regeneration even if cached
            sbert_model_name: Name of the SBERT model for retrieval
            force_reindex: Whether to force rebuilding indices
            
        Returns:
            Dictionary mapping query IDs to lists of extracted keywords
        """
        # Initialize components for search
        preprocessor = TextPreprocessor()
        index_manager = IndexManager(preprocessor)
        
        # Build indices
        logger.info("Building search indices...")
        bm25, corpus_texts, corpus_ids = index_manager.build_bm25_index(
            corpus_dataset,
            cache_path=os.path.join(self.cache_dir, "bm25_index.pkl"),
            force_reindex=force_reindex
        )
        
        sbert_model, doc_embeddings = index_manager.build_sbert_index(
            corpus_texts,
            model_name=sbert_model_name,
            batch_size=64,
            cache_path=os.path.join(self.cache_dir, "sbert_index.pt"),
            force_reindex=force_reindex
        )
        
        # Initialize search engine
        search_engine = SearchEngine(preprocessor)
        
        # Dictionary to store extracted keywords
        all_keywords = {}
        
        # Process each query with tqdm progress bar
        for query_item in tqdm(queries_dataset, desc="Extracting keywords for queries"):
            query_id = str(query_item["_id"])
            query_text = query_item["text"]
            
            # Check cache first if not forcing regeneration
            if not force_regenerate and query_id in self.keywords_cache:
                all_keywords[query_id] = self.keywords_cache[query_id]
                continue
            
            # Retrieve documents using hybrid search
            results = search_engine.search(
                query=query_text,
                bm25=bm25,
                corpus_texts=corpus_texts,
                corpus_ids=corpus_ids,
                sbert_model=sbert_model,
                doc_embeddings=doc_embeddings,
                top_k=self.top_k_docs,
                method=RetrievalMethod.HYBRID,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=False
            )
            
            # Get top N documents for keyword extraction
            top_docs = []
            for doc_id, _ in results[:self.top_n_docs_for_extraction]:
                idx = corpus_ids.index(doc_id)
                top_docs.append(corpus_texts[idx])
            
            # Extract keywords
            keywords = self.extract_keywords(
                query=query_text,
                docs_text=top_docs,
                num_keywords=num_keywords,
                diversity=diversity
            )
            
            # Store results
            all_keywords[query_id] = keywords
            self.keywords_cache[query_id] = keywords
        
        # Save cache
        self._save_keywords_cache()
        
        return all_keywords


# Example usage
def main():
    """Main function to demonstrate keyword extraction functionality"""
    # Define constants
    NUM_KEYWORDS = 10
    DIVERSITY = 0.7
    CACHE_DIR = "cache"
    FORCE_REGENERATE = True
    FORCE_REINDEX = False
    TOP_K_DOCS = 1000
    TOP_N_DOCS_FOR_EXTRACTION = 10
    LOG_LEVEL = 'INFO'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load dataset
    logger.info("Loading dataset...")
    from datasets import load_dataset
    
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    
    # Initialize extractor
    logger.info("Initializing keyword extractor...")
    extractor = KeywordExtractor(
        cache_dir=CACHE_DIR,
        top_k_docs=TOP_K_DOCS,
        top_n_docs_for_extraction=TOP_N_DOCS_FOR_EXTRACTION
    )
    
    # Extract keywords for all queries
    logger.info("Extracting keywords for all queries...")
    all_keywords = extractor.extract_keywords_for_queries(
        queries_dataset=queries_dataset,
        corpus_dataset=corpus_dataset,
        num_keywords=NUM_KEYWORDS,
        diversity=DIVERSITY,
        force_regenerate=FORCE_REGENERATE,
        force_reindex=FORCE_REINDEX
    )
    
    # Print some examples
    logger.info("Example keyword extractions:")
    for i, (query_id, keywords) in enumerate(list(all_keywords.items())[:5]):
        original = queries_dataset[i]["text"]
        print(f"[Query {query_id}] Original: {original}")
        print(f"[Query {query_id}] Keywords: {keywords}")
        print()


# Execute main function if called directly
if __name__ == "__main__":
    main()