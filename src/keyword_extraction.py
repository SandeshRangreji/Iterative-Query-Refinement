# keyword_extraction.py
import logging
import os
import json
from typing import List, Dict, Tuple, Set, Optional
import nltk
from nltk.corpus import stopwords
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datasets import load_dataset

# Import from other modules
from search import (
    TextPreprocessor, 
    IndexManager,
    SearchEngine,
    RetrievalMethod,
    HybridStrategy
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
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class KeywordExtractor:
    """Class for extracting keywords from retrieved documents"""
    
    def __init__(
        self,
        keybert_model: str = 'all-mpnet-base-v2',
        cache_dir: str = 'cache',
        top_k_docs: int = 1000,  # Number of docs to retrieve
        top_n_docs_for_extraction: int = 10,  # Number of docs to use for extraction
        device: str = None  # Device to use ('cpu', 'cuda', 'mps'). None for auto-detection
    ):
        """
        Initialize keyword extractor

        Args:
            keybert_model: Model name for KeyBERT or SentenceTransformer model
            cache_dir: Directory to store cache files
            top_k_docs: Number of documents to retrieve
            top_n_docs_for_extraction: Number of top documents to use for extraction
            device: Device to use ('cpu', 'cuda', 'mps'). None for auto-detection
        """
        self.top_k_docs = top_k_docs
        self.top_n_docs_for_extraction = top_n_docs_for_extraction
        self.stop_words = set(stopwords.words('english'))
        self.cache_dir = cache_dir
        self.keybert_model_name = keybert_model

        # Select device with auto-detection
        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize KeyBERT with sbert model on specified device
        if isinstance(keybert_model, str):
            self.sbert_model = SentenceTransformer(keybert_model, device=self.device)
            self.keybert = KeyBERT(model=self.sbert_model)
        else:
            # Assume it's already a SentenceTransformer model
            self.sbert_model = keybert_model
            self.keybert = KeyBERT(model=self.sbert_model)
    
    def _generate_keywords_cache_path(
        self, 
        method: str = "keybert",
        num_keywords: int = 5, 
        diversity: float = 0.7,
        top_n_docs: int = 10,
        top_k_docs: int = 1000,
        model_name: str = "all-mpnet-base-v2",
        keyphrase_ngram_range: Tuple[int, int] = (1, 2)
    ) -> str:
        """Generate structured cache path for extracted keywords"""
        # Get base model name (e.g., "mpnet" from "all-mpnet-base-v2")
        if "/" in model_name:
            base_model_name = model_name.split("/")[-1]
        else:
            base_model_name = model_name
        
        # Extract key part of model name
        if "mpnet" in base_model_name.lower():
            clean_model_name = "mpnet"
        elif "minilm" in base_model_name.lower():
            clean_model_name = "minilm"
        elif "bert" in base_model_name.lower():
            clean_model_name = "bert"
        else:
            clean_model_name = base_model_name.replace("-", "").replace("_", "")[:8]
        
        cache_dir = os.path.join("cache", "keywords")
        
        # Create filename with key parameters
        filename = f"{method}_k{num_keywords}_div{diversity}_top{top_n_docs}docs_{clean_model_name}_k{top_k_docs}_ngram{keyphrase_ngram_range[0]}-{keyphrase_ngram_range[1]}.json"
        
        return os.path.join(cache_dir, filename)
    
    def _load_keywords_cache(self, cache_path: str) -> Dict:
        """Load keywords cache from specific path"""
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading keywords cache from {cache_path}: {e}")
                return {}
        return {}
    
    def _save_keywords_cache(self, cache_data: Dict, cache_path: str):
        """Save keywords cache to specific path"""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved keywords cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Error saving keywords cache to {cache_path}: {e}")
    
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
        keyphrase_ngram_range: Tuple[int, int] = (1, 2),
        retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        use_mmr: bool = False,
        use_cross_encoder: bool = False,
        mmr_lambda: float = 0.5,
        hybrid_weight: float = 0.5,
        force_regenerate: bool = False,
        sbert_model_name: str = 'all-mpnet-base-v2',
        force_reindex: bool = False,
        dataset_name: str = 'trec-covid'
    ) -> Dict[str, List[str]]:
        """
        Extract keywords for multiple queries based on retrieved documents
        
        Args:
            queries_dataset: Dataset containing queries with _id and text fields
            corpus_dataset: Dataset containing corpus documents
            num_keywords: Number of keywords to extract per query
            diversity: Diversity parameter for MMR
            keyphrase_ngram_range: Range of n-gram lengths for keyphrases
            retrieval_method: Method for document retrieval
            hybrid_strategy: Strategy for hybrid retrieval
            use_mmr: Whether to use MMR for retrieval
            use_cross_encoder: Whether to use cross-encoder for retrieval
            mmr_lambda: Lambda parameter for MMR
            hybrid_weight: Weight for hybrid strategy
            force_regenerate: Whether to force regeneration even if cached
            sbert_model_name: Name of the SBERT model for retrieval
            force_reindex: Whether to force rebuilding indices
            dataset_name: Name of the dataset for index caching
            
        Returns:
            Dictionary mapping query IDs to lists of extracted keywords
        """
        # Generate cache path based on parameters
        cache_path = self._generate_keywords_cache_path(
            method="keybert",
            num_keywords=num_keywords,
            diversity=diversity,
            top_n_docs=self.top_n_docs_for_extraction,
            top_k_docs=self.top_k_docs,
            model_name=self.keybert_model_name,
            keyphrase_ngram_range=keyphrase_ngram_range
        )
        
        # Load existing cache
        keywords_cache = self._load_keywords_cache(cache_path)
        
        # Check if we can use cached results
        if not force_regenerate and keywords_cache:
            logger.info(f"Loading keywords from cache: {cache_path}")
            # Check if all required queries are in cache
            query_ids_needed = {str(item["_id"]) for item in queries_dataset}
            cached_query_ids = set(keywords_cache.keys())
            
            if query_ids_needed.issubset(cached_query_ids):
                logger.info(f"All {len(query_ids_needed)} queries found in cache")
                return {qid: keywords_cache[qid] for qid in query_ids_needed}
            else:
                missing_queries = query_ids_needed - cached_query_ids
                logger.info(f"Cache miss: {len(missing_queries)} queries not in cache")
        
        # Initialize components for search
        preprocessor = TextPreprocessor()
        index_manager = IndexManager(preprocessor)
        
        # Build indices
        logger.info("Building search indices...")
        
        # Use structured cache paths for indices
        bm25, corpus_texts, corpus_ids = index_manager.build_bm25_index(
            corpus_dataset,
            dataset_name=dataset_name,
            force_reindex=force_reindex
        )
        
        sbert_model, doc_embeddings = index_manager.build_sbert_index(
            corpus_texts,
            model_name=sbert_model_name,
            dataset_name=dataset_name,
            batch_size=64,
            force_reindex=force_reindex
        )
        
        # Initialize search engine with all indices and models
        search_engine = SearchEngine(
            preprocessor=preprocessor,
            bm25=bm25,
            corpus_texts=corpus_texts,
            corpus_ids=corpus_ids,
            sbert_model=sbert_model,
            doc_embeddings=doc_embeddings
        )
        
        # Dictionary to store extracted keywords
        all_keywords = keywords_cache.copy()  # Start with existing cache
        
        # Process each query with tqdm progress bar
        queries_to_process = []
        for query_item in queries_dataset:
            query_id = str(query_item["_id"])
            if force_regenerate or query_id not in all_keywords:
                queries_to_process.append(query_item)
        
        if queries_to_process:
            logger.info(f"Processing {len(queries_to_process)} queries for keyword extraction...")
            
            for query_item in tqdm(queries_to_process, desc="Extracting keywords"):
                query_id = str(query_item["_id"])
                query_text = query_item["text"]
                
                # Retrieve documents using specified parameters
                results = search_engine.search(
                    query=query_text,
                    top_k=self.top_k_docs,
                    method=retrieval_method,
                    hybrid_strategy=hybrid_strategy,
                    hybrid_weight=hybrid_weight,
                    use_mmr=use_mmr,
                    mmr_lambda=mmr_lambda,
                    use_cross_encoder=use_cross_encoder
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
                    diversity=diversity,
                    keyphrase_ngram_range=keyphrase_ngram_range
                )
                
                # Store results
                all_keywords[query_id] = keywords
        
        # Save updated cache
        self._save_keywords_cache(all_keywords, cache_path)
        
        return all_keywords


# Example usage
def main():
    """Main function to demonstrate keyword extraction functionality"""
    
    # ===== CACHING CONTROL FLAGS =====
    FORCE_REINDEX = False               # For search indices
    FORCE_REGENERATE_KEYWORDS = True   # For keyword extraction
    
    # ===== DATASET PARAMETERS =====
    DATASET_NAME = 'trec-covid'
    
    # ===== KEYWORD EXTRACTION PARAMETERS =====
    NUM_KEYWORDS = 10
    DIVERSITY = 0.7
    KEYPHRASE_NGRAM_RANGE = (1, 2)
    TOP_K_DOCS = 1000                   # Documents to retrieve for extraction
    TOP_N_DOCS_FOR_EXTRACTION = 10     # Top documents to use for extraction
    
    # ===== SEARCH PARAMETERS =====
    SBERT_MODEL = 'all-mpnet-base-v2'
    RETRIEVAL_METHOD = RetrievalMethod.HYBRID
    HYBRID_STRATEGY = HybridStrategy.SIMPLE_SUM
    HYBRID_WEIGHT = 0.5
    USE_MMR = False
    USE_CROSS_ENCODER = False
    MMR_LAMBDA = 0.5
    
    # ===== OTHER PARAMETERS =====
    CACHE_DIR = "cache"
    LOG_LEVEL = 'INFO'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load dataset
    logger.info("Loading datasets...")
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    logger.info(f"Loaded {len(corpus_dataset)} documents, {len(queries_dataset)} queries")
    
    # Initialize extractor
    logger.info("Initializing keyword extractor...")
    extractor = KeywordExtractor(
        keybert_model=SBERT_MODEL,
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
        keyphrase_ngram_range=KEYPHRASE_NGRAM_RANGE,
        retrieval_method=RETRIEVAL_METHOD,
        hybrid_strategy=HYBRID_STRATEGY,
        hybrid_weight=HYBRID_WEIGHT,
        use_mmr=USE_MMR,
        use_cross_encoder=USE_CROSS_ENCODER,
        mmr_lambda=MMR_LAMBDA,
        force_regenerate=FORCE_REGENERATE_KEYWORDS,
        sbert_model_name=SBERT_MODEL,
        force_reindex=FORCE_REINDEX,
        dataset_name=DATASET_NAME
    )
    
    # Print some examples
    logger.info("\n===== KEYWORD EXTRACTION RESULTS =====")
    logger.info("Example keyword extractions:")
    
    for i, (query_id, keywords) in enumerate(list(all_keywords.items())[:5]):
        # Find the original query text
        original_query = None
        for query_item in queries_dataset:
            if str(query_item["_id"]) == query_id:
                original_query = query_item["text"]
                break
        
        logger.info(f"\n[Query {query_id}] Original: {original_query}")
        logger.info(f"[Query {query_id}] Keywords: {keywords}")
    
    # Print summary statistics
    total_queries = len(all_keywords)
    avg_keywords_per_query = sum(len(kw) for kw in all_keywords.values()) / total_queries if total_queries > 0 else 0
    
    logger.info(f"\n===== SUMMARY STATISTICS =====")
    logger.info(f"Total queries processed: {total_queries}")
    logger.info(f"Average keywords per query: {avg_keywords_per_query:.2f}")
    logger.info(f"Parameters used:")
    logger.info(f"  - Number of keywords: {NUM_KEYWORDS}")
    logger.info(f"  - Diversity: {DIVERSITY}")
    logger.info(f"  - N-gram range: {KEYPHRASE_NGRAM_RANGE}")
    logger.info(f"  - Retrieval docs: {TOP_K_DOCS}")
    logger.info(f"  - Extraction docs: {TOP_N_DOCS_FOR_EXTRACTION}")
    logger.info(f"  - Model: {SBERT_MODEL}")
    
    return all_keywords


# Execute main function if called directly
if __name__ == "__main__":
    main()