# search.py
import os
import pickle
import re
import logging
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from typing import List, Tuple, Dict, Set, Union, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from enum import Enum
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

# Constants and enums
class RetrievalMethod(str, Enum):
    BM25 = "bm25"
    SBERT = "sbert"
    HYBRID = "hybrid"

class HybridStrategy(str, Enum):
    SIMPLE_SUM = "simple_sum"
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"

class TextPreprocessor:
    """Class for text preprocessing and normalization operations"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by lowercasing, removing punctuation,
        and stemming non-stopwords.
        
        Args:
            text: The input text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        tokens = text.split()
        filtered = [self.stemmer.stem(w) for w in tokens if w not in self.stop_words]
        return filtered

class IndexManager:
    """Class for managing and building search indices"""
    
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
    
    def _generate_bm25_cache_path(self, dataset_name: str, b_param: float = 0.75, k1_param: float = 1.5, epsilon: float = 0.25, stemmer: str = "porter") -> str:
        """Generate structured cache path for BM25 index"""
        cache_dir = os.path.join("cache", "indices", "bm25")
        filename = f"{dataset_name}_b{b_param}_k1-{k1_param}_eps{epsilon}_stem-{stemmer}.pkl"
        return os.path.join(cache_dir, filename)
    
    def _generate_sbert_cache_path(self, model_name: str, dataset_name: str, batch_size: int = 64) -> str:
        """Generate structured cache path for SBERT embeddings"""
        # Clean model name for filesystem
        clean_model_name = model_name.replace("/", "_").replace("-", "_")
        cache_dir = os.path.join("cache", "indices", "embeddings", clean_model_name)
        filename = f"{dataset_name}_batch{batch_size}.pt"
        return os.path.join(cache_dir, filename)
    
    def build_bm25_index(
        self,
        corpus_dataset,
        dataset_name: str,
        b_param: float = 0.75,
        k1_param: float = 1.5,
        epsilon: float = 0.25,
        stemmer: str = "porter",
        force_reindex: bool = False
    ) -> Tuple[BM25Okapi, List[str], List[str]]:
        """
        Build or load a BM25 index for the corpus
        
        Args:
            corpus_dataset: Dataset containing corpus documents
            dataset_name: Name of the dataset for caching
            b_param: BM25 b parameter (document length normalization)
            k1_param: BM25 k1 parameter (term frequency saturation)
            epsilon: BM25 epsilon parameter (score normalization)
            stemmer: Stemmer to use ("porter", "none")
            force_reindex: Whether to force rebuilding the index
            
        Returns:
            Tuple of (BM25 index, corpus texts, corpus ids)
        """
        # Generate cache path
        cache_path = self._generate_bm25_cache_path(dataset_name, b_param, k1_param, epsilon, stemmer)
        
        if (not force_reindex) and os.path.exists(cache_path):
            logger.info(f"Loading BM25 index from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                bm25, corpus_texts, corpus_ids = pickle.load(f)
            return bm25, corpus_texts, corpus_ids

        logger.info("Building BM25 index from scratch...")
        corpus_texts = []
        corpus_ids = []

        for doc in corpus_dataset:
            # Handle documents with or without title field
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title:
                corpus_texts.append(title + "\n\n" + text)
            else:
                corpus_texts.append(text)
            corpus_ids.append(doc["_id"])

        tokenized_corpus = [self.preprocessor.preprocess_text(text) for text in corpus_texts]
        
        # Use exposed parameters with correct defaults
        bm25 = BM25Okapi(tokenized_corpus, b=b_param, k1=k1_param, epsilon=epsilon)

        logger.info(f"Saving BM25 index to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump((bm25, corpus_texts, corpus_ids), f)

        return bm25, corpus_texts, corpus_ids
    
    def build_sbert_index(
        self,
        corpus_texts: List[str],
        dataset_name: str,
        model_name: str = "sentence-transformers/msmarco-MiniLM-L6-cos-v5",
        batch_size: int = 64,
        max_seq_length: Optional[int] = None,
        normalize_embeddings: bool = True,
        force_reindex: bool = False,
        device: str = None
    ) -> Tuple[SentenceTransformer, torch.Tensor]:
        """
        Build or load SBERT embeddings for the corpus
        
        Args:
            corpus_texts: List of document texts
            model_name: Name of the sentence transformer model
            dataset_name: Name of the dataset for caching
            batch_size: Batch size for encoding
            max_seq_length: Maximum sequence length (None for model default)
            normalize_embeddings: Whether to normalize embeddings
            force_reindex: Whether to force recomputing embeddings
            device: Device to use for encoding (None for auto-selection)
            
        Returns:
            Tuple of (SBERT model, document embeddings tensor)
        """
        # Select device based on availability
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Generate cache path
        cache_path = self._generate_sbert_cache_path(model_name, dataset_name, batch_size)
        
        logger.info(f"Initializing SBERT model on {device}...")
        model = SentenceTransformer(model_name, device=device)
        
        # Set max sequence length if provided
        if max_seq_length is not None:
            model.max_seq_length = max_seq_length

        if (not force_reindex) and os.path.exists(cache_path):
            logger.info(f"Loading SBERT embeddings from cache: {cache_path}")
            data = torch.load(cache_path, map_location=device)
            doc_embeddings = data["doc_embeddings"]
            return model, doc_embeddings

        logger.info(f"Encoding corpus from scratch with batch size {batch_size}...")
        doc_embeddings = model.encode(
            corpus_texts, 
            batch_size=batch_size, 
            convert_to_tensor=True,
            normalize_to_unit_sphere=normalize_embeddings,
            show_progress_bar=True
        )
        
        # Ensure embeddings are on CPU for storage
        doc_embeddings = doc_embeddings.cpu().contiguous()

        logger.info(f"Saving SBERT embeddings to cache: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"doc_embeddings": doc_embeddings}, cache_path)

        return model, doc_embeddings

class SearchEngine:
    """Main search engine implementing multiple retrieval methods"""
    
    def __init__(
        self, 
        preprocessor: TextPreprocessor,
        bm25: Optional[BM25Okapi] = None,
        corpus_texts: Optional[List[str]] = None,
        corpus_ids: Optional[List[str]] = None,
        sbert_model: Optional[SentenceTransformer] = None, 
        doc_embeddings: Optional[torch.Tensor] = None,
        cross_encoder_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        device: str = None
    ):
        """
        Initialize search engine with indices and models
        
        Args:
            preprocessor: TextPreprocessor for query processing
            bm25: BM25 index (optional)
            corpus_texts: List of document texts (optional)
            corpus_ids: List of document IDs (optional)
            sbert_model: SentenceTransformer model (optional)
            doc_embeddings: Document embeddings tensor (optional)
            cross_encoder_model_name: Name of cross-encoder model to use
            device: Device to use for models
        """
        self.preprocessor = preprocessor
        self.bm25 = bm25
        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids
        self.sbert_model = sbert_model
        self.doc_embeddings = doc_embeddings
        
        # Select device based on availability if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize cross-encoder if model name provided
        if cross_encoder_model_name:
            logger.info(f"Initializing cross-encoder model on {self.device}...")
            self.cross_encoder = CrossEncoder(cross_encoder_model_name, device=self.device)
        else:
            self.cross_encoder = None
    
    def search(
        self,
        query: str,
        top_k: int = 1000,
        method: RetrievalMethod = RetrievalMethod.BM25,
        hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
        hybrid_weight: float = 0.5,  # Weight for BM25 vs SBERT in weighted hybrid
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        use_cross_encoder: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Main search function supporting multiple retrieval methods
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: Retrieval method (bm25, sbert, or hybrid)
            hybrid_strategy: Strategy for combining BM25 and SBERT results
            hybrid_weight: Weight for BM25 vs SBERT in weighted hybrid (0-1)
            use_mmr: Whether to use Maximal Marginal Relevance
            mmr_lambda: Lambda parameter for MMR (higher values favor relevance)
            use_cross_encoder: Whether to use cross-encoder for reranking
            
        Returns:
            List of tuples (document_id, score)
        """
        # Validate inputs based on method
        if method == RetrievalMethod.BM25 and self.bm25 is None:
            raise ValueError("BM25 index is required for method='bm25'")
        
        if (method == RetrievalMethod.SBERT or method == RetrievalMethod.HYBRID) and (self.sbert_model is None or self.doc_embeddings is None):
            raise ValueError(f"SBERT model and embeddings are required for method='{method}'")
            
        if use_cross_encoder and self.cross_encoder is None:
            raise ValueError("Cross-encoder was not initialized but use_cross_encoder=True")
        
        # Handle different retrieval methods
        if method == RetrievalMethod.BM25:
            return self._bm25_search(
                query, top_k, use_mmr, mmr_lambda
            )
        
        elif method == RetrievalMethod.SBERT:
            return self._sbert_search(
                query, top_k, use_mmr, mmr_lambda
            )
        
        elif method == RetrievalMethod.HYBRID:
            return self._hybrid_search(
                query, top_k, hybrid_strategy, hybrid_weight,
                use_mmr, mmr_lambda, use_cross_encoder
            )
        
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    def _bm25_search(
        self,
        query: str,
        top_k: int,
        use_mmr: bool,
        mmr_lambda: float
    ) -> List[Tuple[str, float]]:
        """
        BM25 search implementation
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_mmr: Whether to use MMR for diversity
            mmr_lambda: MMR lambda parameter
            
        Returns:
            List of tuples (document_id, score)
        """
        tokenized_query = self.preprocessor.preprocess_text(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort by BM25 score (descending)
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.corpus_ids[idx], scores[idx]) for idx in sorted_indices]
        
        # Apply MMR if requested and SBERT is available
        if use_mmr and self.sbert_model is not None and self.doc_embeddings is not None:
            logger.info("Applying MMR reranking to BM25 results...")
            query_embedding = self.sbert_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
            candidate_indices = sorted_indices.tolist()
            candidate_embeddings = self.doc_embeddings[candidate_indices]
            candidate_ids = [self.corpus_ids[i] for i in candidate_indices]
            
            results = self._mmr_rerank(
                query_embedding,
                candidate_embeddings,
                candidate_ids,
                [scores[i] for i in candidate_indices],
                top_k=top_k,
                lambda_param=mmr_lambda
            )
        
        return results
    
    def _sbert_search(
        self,
        query: str,
        top_k: int,
        use_mmr: bool,
        mmr_lambda: float
    ) -> List[Tuple[str, float]]:
        """
        SBERT semantic search implementation
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_mmr: Whether to use MMR for diversity
            mmr_lambda: MMR lambda parameter
            
        Returns:
            List of tuples (document_id, score)
        """
        query_embedding = self.sbert_model.encode(query, convert_to_tensor=True, show_progress_bar=False)

        # Move to same device as query_embedding
        device = query_embedding.device
        doc_embeddings = self.doc_embeddings
        if doc_embeddings.device != device:
            doc_embeddings = doc_embeddings.to(device)

        # Keep similarity computation on GPU
        cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

        # Sort by cosine similarity (descending) using GPU operations
        sorted_indices = torch.argsort(cos_scores, descending=True)[:top_k]

        # Only convert to Python at the final step
        sorted_indices_cpu = sorted_indices.cpu().tolist()
        cos_scores_cpu = cos_scores.cpu().tolist()
        results = [(self.corpus_ids[idx], cos_scores_cpu[idx]) for idx in sorted_indices_cpu]
        
        # Apply MMR if requested
        if use_mmr:
            logger.info("Applying MMR reranking to SBERT results...")
            candidate_indices = sorted_indices.tolist()
            candidate_embeddings = doc_embeddings[candidate_indices]
            candidate_ids = [self.corpus_ids[i] for i in candidate_indices]
            
            results = self._mmr_rerank(
                query_embedding,
                candidate_embeddings,
                candidate_ids,
                [cos_scores[i] for i in candidate_indices],
                top_k=top_k,
                lambda_param=mmr_lambda
            )
        
        return results
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        hybrid_strategy: HybridStrategy,
        hybrid_weight: float,
        use_mmr: bool,
        mmr_lambda: float,
        use_cross_encoder: bool
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search combining BM25 and SBERT results
        
        Args:
            query: Search query
            top_k: Number of results to return
            hybrid_strategy: Strategy for combining results
            hybrid_weight: Weight for BM25 vs SBERT (0-1)
            use_mmr: Whether to use MMR for diversity
            mmr_lambda: MMR lambda parameter
            use_cross_encoder: Whether to use cross-encoder reranking
            
        Returns:
            List of tuples (document_id, score)
        """
        # Get BM25 scores
        tokenized_query = self.preprocessor.preprocess_text(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_sorted_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Get SBERT scores
        query_embedding = self.sbert_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        
        # Move to same device as query_embedding
        device = query_embedding.device
        doc_embeddings = self.doc_embeddings
        if doc_embeddings.device != device:
            doc_embeddings = doc_embeddings.to(device)

        # Keep similarity computation on GPU
        cos_scores_tensor = util.cos_sim(query_embedding, doc_embeddings)[0]

        # Sort on GPU, then convert top-k indices to NumPy
        sbert_sorted_indices = torch.argsort(cos_scores_tensor, descending=True)[:top_k].cpu().numpy()

        # Convert scores to NumPy for hybrid combination (needed for BM25 compatibility)
        cos_scores = cos_scores_tensor.cpu().numpy()
        
        # Apply hybrid strategy
        if hybrid_strategy == HybridStrategy.SIMPLE_SUM:
            # Combine sets of top-k from both methods
            combined_indices = list(set(bm25_sorted_indices) | set(sbert_sorted_indices))
            
            # Normalize scores
            bm25_max = max(bm25_scores[combined_indices]) if combined_indices else 1e-9
            cos_max = max(cos_scores[combined_indices]) if combined_indices else 1e-9
            
            doc_final_scores = {}
            for idx in combined_indices:
                # Normalize scores
                bm25_score = bm25_scores[idx] / (bm25_max + 1e-9)
                cos_score = cos_scores[idx] / (cos_max + 1e-9)
                # Simple sum of normalized scores
                final_score = bm25_score + cos_score
                doc_final_scores[idx] = final_score
            
        elif hybrid_strategy == HybridStrategy.WEIGHTED:
            # Combine sets of top-k from both methods
            combined_indices = list(set(bm25_sorted_indices) | set(sbert_sorted_indices))
            
            # Normalize scores
            bm25_max = max(bm25_scores[combined_indices]) if combined_indices else 1e-9
            cos_max = max(cos_scores[combined_indices]) if combined_indices else 1e-9
            
            doc_final_scores = {}
            for idx in combined_indices:
                # Normalize scores
                bm25_score = bm25_scores[idx] / (bm25_max + 1e-9)
                cos_score = cos_scores[idx] / (cos_max + 1e-9)
                # Weighted combination
                final_score = hybrid_weight * bm25_score + (1 - hybrid_weight) * cos_score
                doc_final_scores[idx] = final_score
        
        elif hybrid_strategy == HybridStrategy.RRF:
            # Reciprocal Rank Fusion
            # Get ranks from both methods
            bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_sorted_indices)}
            sbert_ranks = {idx: rank + 1 for rank, idx in enumerate(sbert_sorted_indices)}
            
            # Combine sets
            combined_indices = list(set(bm25_sorted_indices) | set(sbert_sorted_indices))
            
            # RRF constant (typically 60)
            k = 60
            
            doc_final_scores = {}
            for idx in combined_indices:
                # Get ranks (default to a large value if not in top-k)
                bm25_rank = bm25_ranks.get(idx, len(self.corpus_ids) + 1)
                sbert_rank = sbert_ranks.get(idx, len(self.corpus_ids) + 1)
                
                # RRF score = sum of 1/(rank + k) for each method
                rrf_score = 1/(bm25_rank + k) + 1/(sbert_rank + k)
                doc_final_scores[idx] = rrf_score
        
        else:
            raise ValueError(f"Unknown hybrid strategy: {hybrid_strategy}")
            
        # Sort by final score
        sorted_indices = sorted(
            doc_final_scores.keys(), 
            key=lambda x: doc_final_scores[x], 
            reverse=True
        )[:top_k]
        
        results = [(self.corpus_ids[i], doc_final_scores[i]) for i in sorted_indices]
        
        # Apply MMR if requested
        if use_mmr:
            logger.info("Applying MMR reranking to hybrid results...")
            candidate_indices = list(sorted_indices)
            candidate_embeddings = doc_embeddings[candidate_indices]
            candidate_ids = [self.corpus_ids[i] for i in candidate_indices]
            candidate_scores = [doc_final_scores[i] for i in candidate_indices]
            
            results = self._mmr_rerank(
                query_embedding, 
                candidate_embeddings,
                candidate_ids,
                candidate_scores,
                top_k=top_k,
                lambda_param=mmr_lambda
            )
        
        # Apply Cross-Encoder re-ranking if requested
        if use_cross_encoder and self.cross_encoder:
            doc_ids = [doc_id for doc_id, _ in results]
            
            # Create pairs of (query, document) for cross-encoder
            query_doc_pairs = []
            for doc_id in doc_ids:
                # Find the document text for this ID
                doc_idx = self.corpus_ids.index(doc_id)
                query_doc_pairs.append((query, self.corpus_texts[doc_idx]))
            
            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(query_doc_pairs, show_progress_bar=False)
            
            # Create new results with cross-encoder scores
            results = [(doc_id, score) for doc_id, score in zip(doc_ids, ce_scores)]
            
            # Sort by cross-encoder score
            results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _mmr_rerank(
        self,
        query_embedding: torch.Tensor,
        doc_embeddings: torch.Tensor,
        doc_ids: List[str],
        scores: List[float],
        top_k: int = 1000,
        lambda_param: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Perform Maximal Marginal Relevance (MMR) reranking to diversify results.
        GPU-optimized version that keeps all operations on GPU.

        Args:
            query_embedding: Query embedding tensor
            doc_embeddings: Document embeddings tensor
            doc_ids: List of document IDs corresponding to doc_embeddings
            scores: List of initial relevance scores
            top_k: Number of results to return
            lambda_param: MMR lambda parameter (0-1), higher values favor relevance

        Returns:
            List of tuples (document_id, score)
        """
        # Ensure we have valid data
        if len(doc_ids) == 0:
            return []

        # Move to same device as query_embedding
        device = query_embedding.device
        if doc_embeddings.device != device:
            doc_embeddings = doc_embeddings.to(device)

        # Compute similarity between query and docs (KEEP ON GPU)
        query_sim = util.cos_sim(query_embedding, doc_embeddings)[0]  # Shape: [n_docs]

        # Precompute pairwise similarities between docs (KEEP ON GPU)
        doc_sims = util.cos_sim(doc_embeddings, doc_embeddings)  # Shape: [n_docs, n_docs]

        # MMR selection using GPU tensors
        num_docs = len(doc_ids)
        selected_indices = []
        remaining_mask = torch.ones(num_docs, dtype=torch.bool, device=device)

        for _ in range(min(top_k, num_docs)):
            # Get indices of remaining candidates
            remaining_indices = torch.where(remaining_mask)[0]

            if len(remaining_indices) == 0:
                break

            # If no documents selected yet, pick highest relevance
            if len(selected_indices) == 0:
                best_idx = remaining_indices[torch.argmax(query_sim[remaining_indices])].item()
            else:
                # Compute MMR scores for remaining candidates
                # relevance: query similarity for remaining docs
                relevance = query_sim[remaining_indices]

                # diversity penalty: max similarity to any selected doc
                # doc_sims[remaining_indices][:, selected_indices] -> [n_remaining, n_selected]
                selected_tensor = torch.tensor(selected_indices, device=device, dtype=torch.long)
                similarities_to_selected = doc_sims[remaining_indices][:, selected_tensor]
                diversity_penalty = torch.max(similarities_to_selected, dim=1)[0]

                # Compute MMR scores
                mmr_scores = lambda_param * relevance - (1 - lambda_param) * diversity_penalty

                # Select best document
                best_local_idx = torch.argmax(mmr_scores).item()
                best_idx = remaining_indices[best_local_idx].item()

            # Update selection
            selected_indices.append(best_idx)
            remaining_mask[best_idx] = False

        # Return selected documents with their original scores (only move to CPU at the end)
        return [(doc_ids[idx], scores[idx]) for idx in selected_indices]
        
    def load_indices(
        self,
        bm25: BM25Okapi,
        corpus_texts: List[str],
        corpus_ids: List[str],
        sbert_model: Optional[SentenceTransformer] = None,
        doc_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Load indices and models into the search engine
        
        Args:
            bm25: BM25 index
            corpus_texts: List of document texts
            corpus_ids: List of document IDs
            sbert_model: SentenceTransformer model (optional)
            doc_embeddings: Document embeddings tensor (optional)
        """
        self.bm25 = bm25
        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids
        
        if sbert_model is not None:
            self.sbert_model = sbert_model
        
        if doc_embeddings is not None:
            self.doc_embeddings = doc_embeddings
            
    def get_document_by_id(self, doc_id: str) -> Optional[str]:
        """
        Get document text by ID
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document text if found, None otherwise
        """
        if self.corpus_ids is None or self.corpus_texts is None:
            return None
            
        try:
            idx = self.corpus_ids.index(doc_id)
            return self.corpus_texts[idx]
        except ValueError:
            return None

def run_search_for_multiple_queries(
    search_engine: SearchEngine,
    queries_dataset,
    top_k: int = 1000,
    method: RetrievalMethod = RetrievalMethod.BM25,
    hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
    hybrid_weight: float = 0.5,
    use_mmr: bool = False,
    mmr_lambda: float = 0.5,
    use_cross_encoder: bool = False
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Run search for multiple queries
    
    Args:
        search_engine: Initialized SearchEngine instance
        queries_dataset: Dataset containing queries
        top_k: Number of results to retrieve per query
        method: Retrieval method
        hybrid_strategy: Strategy for hybrid retrieval
        hybrid_weight: Weight for BM25 vs SBERT in weighted hybrid
        use_mmr: Whether to use MMR
        mmr_lambda: Lambda parameter for MMR
        use_cross_encoder: Whether to use cross-encoder reranking
        
    Returns:
        Dictionary mapping query IDs to search results
    """
    query_results = {}
    for query_item in tqdm(queries_dataset, desc="Running searches"):
        query_id = int(query_item["_id"])
        query_text = query_item["text"]
        
        # Perform search
        search_results = search_engine.search(
            query=query_text,
            top_k=top_k,
            method=method,
            hybrid_strategy=hybrid_strategy,
            hybrid_weight=hybrid_weight,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            use_cross_encoder=use_cross_encoder
        )
        
        # Store results for this query
        query_results[query_id] = search_results
    
    return query_results

def search_single_query(
    search_engine: SearchEngine,
    query_text: str,
    top_k: int = 10,
    method: RetrievalMethod = RetrievalMethod.HYBRID,
    hybrid_strategy: HybridStrategy = HybridStrategy.SIMPLE_SUM,
    use_cross_encoder: bool = True
) -> List[Tuple[str, str, float]]:
    """
    Convenience function to search for a single query and return document texts
    
    Args:
        search_engine: Initialized SearchEngine instance
        query_text: Query text
        top_k: Number of results to retrieve
        method: Retrieval method
        hybrid_strategy: Strategy for hybrid retrieval
        use_cross_encoder: Whether to use cross-encoder reranking
        
    Returns:
        List of tuples (doc_id, doc_text, score)
    """
    # Perform search
    results = search_engine.search(
        query=query_text,
        top_k=top_k,
        method=method,
        hybrid_strategy=hybrid_strategy,
        use_cross_encoder=use_cross_encoder
    )
    
    # Get document texts
    search_results_with_text = []
    for doc_id, score in results:
        doc_text = search_engine.get_document_by_id(doc_id)
        search_results_with_text.append((doc_id, doc_text, score))
    
    return search_results_with_text


def main():
    """Main function to demonstrate search functionality"""
    
    # ===== CACHING CONTROL FLAGS =====
    FORCE_REINDEX_BM25 = False
    FORCE_REINDEX_SBERT = False
    FORCE_REGENERATE_SEARCH_RESULTS = False
    FORCE_REGENERATE_EVALUATION = False
    
    # ===== DATASET PARAMETERS =====
    DATASET_NAME = 'trec-covid'
    
    # ===== BM25 PARAMETERS =====
    BM25_B_PARAM = 0.75         # Document length normalization (default)
    BM25_K1_PARAM = 1.5         # Term frequency saturation (default)
    BM25_EPSILON = 0.25         # Score normalization (default)
    BM25_STEMMER = "porter"     # Stemming method
    
    # ===== SBERT PARAMETERS =====
    SBERT_MODEL = 'all-mpnet-base-v2'
    SBERT_BATCH_SIZE = 64
    SBERT_MAX_SEQ_LENGTH = None  # Use model default
    SBERT_NORMALIZE = True
    
    # ===== SEARCH PARAMETERS =====
    CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    TOP_K = 1000
    HYBRID_WEIGHT = 0.5
    MMR_LAMBDA = 0.5
    
    # ===== EVALUATION PARAMETERS =====
    EVALUATION_TOP_K_P = 20     # Precision@k
    EVALUATION_TOP_K_R = 1000   # Recall@k
    
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
    
    # Initialize components
    preprocessor = TextPreprocessor()
    index_manager = IndexManager(preprocessor)
    
    # Build indices with exposed parameters
    logger.info("Building BM25 index...")
    bm25, corpus_texts, corpus_ids = index_manager.build_bm25_index(
        corpus_dataset,
        dataset_name=DATASET_NAME,
        b_param=BM25_B_PARAM,
        k1_param=BM25_K1_PARAM,
        epsilon=BM25_EPSILON,
        stemmer=BM25_STEMMER,
        force_reindex=FORCE_REINDEX_BM25
    )
    
    logger.info("Building SBERT index...")
    sbert_model, doc_embeddings = index_manager.build_sbert_index(
        corpus_texts,
        model_name=SBERT_MODEL,
        dataset_name=DATASET_NAME,
        batch_size=SBERT_BATCH_SIZE,
        max_seq_length=SBERT_MAX_SEQ_LENGTH,
        normalize_embeddings=SBERT_NORMALIZE,
        force_reindex=FORCE_REINDEX_SBERT
    )
    
    # Initialize search engine with all indices and models
    search_engine = SearchEngine(
        preprocessor=preprocessor, 
        bm25=bm25, 
        corpus_texts=corpus_texts, 
        corpus_ids=corpus_ids,
        sbert_model=sbert_model, 
        doc_embeddings=doc_embeddings,
        cross_encoder_model_name=CROSS_ENCODER_MODEL
    )
    
    # Define search configurations to test
    search_configs = [
        {
            "name": "BM25",
            "method": RetrievalMethod.BM25,
            "use_mmr": False,
            "use_cross_encoder": False
        },
        {
            "name": "SBERT",
            "method": RetrievalMethod.SBERT,
            "use_mmr": False,
            "use_cross_encoder": False
        },
        {
            "name": "Hybrid (Simple Sum)",
            "method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "use_mmr": False,
            "use_cross_encoder": False
        },
        {
            "name": "Hybrid (RRF)",
            "method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.RRF,
            "use_mmr": False,
            "use_cross_encoder": False
        },
        {
            "name": "Hybrid + CrossEncoder",
            "method": RetrievalMethod.HYBRID,
            "hybrid_strategy": HybridStrategy.SIMPLE_SUM,
            "use_mmr": False,
            "use_cross_encoder": True
        }
    ]
    
    # Run searches for each configuration
    logger.info("Running searches with different configurations...")
    all_results = {}
    
    for config in search_configs:
        logger.info(f"Running search with {config['name']}...")
        method = config.get("method", RetrievalMethod.BM25)
        hybrid_strategy = config.get("hybrid_strategy", HybridStrategy.SIMPLE_SUM)
        use_mmr = config.get("use_mmr", False)
        use_cross_encoder = config.get("use_cross_encoder", False)
        
        # Run search for all queries with this configuration
        query_results = run_search_for_multiple_queries(
            search_engine=search_engine,
            queries_dataset=queries_dataset,
            top_k=TOP_K,
            method=method,
            hybrid_strategy=hybrid_strategy,
            hybrid_weight=HYBRID_WEIGHT,
            use_mmr=use_mmr,
            mmr_lambda=MMR_LAMBDA,
            use_cross_encoder=use_cross_encoder
        )
        
        all_results[config["name"]] = query_results
    
    # Import the SearchEvaluationUtils for evaluation
    from evaluation import SearchEvaluationUtils
    
    # Evaluate each search configuration
    logger.info("Evaluating search results...")
    evaluation_results = []
    
    for config_name, query_results in all_results.items():
        logger.info(f"Evaluating {config_name}...")
        
        # Calculate metrics
        avg_precisions, avg_recalls, num_evaluated = SearchEvaluationUtils.evaluate_results(
            results_by_query_id=query_results,
            qrels_dataset=qrels_dataset,
            top_k_p=EVALUATION_TOP_K_P,
            top_k_r=EVALUATION_TOP_K_R
        )
        
        # Get the corresponding config
        config = next(config for config in search_configs if config["name"] == config_name)
        
        # Store evaluation results
        evaluation_results.append({
            "config": config,
            "avg_precisions": avg_precisions,
            "avg_recalls": avg_recalls,
            "num_evaluated": num_evaluated
        })
    
    # Calculate F1 scores
    metrics_with_f1 = SearchEvaluationUtils.calculate_f1_scores(evaluation_results)
    
    # Print results summary
    logger.info("\n===== EVALUATION RESULTS SUMMARY =====")
    logger.info(f"{'Method':<30} {'P@20':<10} {'R@1000':<10} {'F1':<10}")
    logger.info("-" * 60)
    
    for metric in metrics_with_f1:
        name = metric["config"]["name"]
        precision = metric["precision"]
        recall = metric["recall"]
        f1 = metric["f1"]
        logger.info(f"{name:<30} {precision:.4f}     {recall:.4f}     {f1:.4f}")
    
    # Save results
    output_dir = "results/search"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, "search_evaluation_results.json")
    SearchEvaluationUtils.save_evaluation_results(evaluation_results, results_file)
    logger.info(f"Results saved to {results_file}")
    
    # Import visualization functions from search_viz.py
    from search_viz import visualize_all_results
    
    # Generate visualization plots
    logger.info("Generating visualization plots...")
    plots_dir = os.path.join(output_dir, "plots")
    plot_paths = visualize_all_results(
        results=evaluation_results,
        top_k_p=EVALUATION_TOP_K_P,
        top_k_r=EVALUATION_TOP_K_R,
        output_dir=plots_dir
    )
    
    # Log visualization outputs
    logger.info(f"Generated {len(plot_paths)} visualization plots:")
    for path in plot_paths:
        logger.info(f"- {path}")
    
    # Example of a single query search
    example_query = queries_dataset[0]["text"]
    logger.info(f"\nRunning example search for query: {example_query}")
    
    example_results = search_single_query(
        search_engine=search_engine,
        query_text=example_query,
        top_k=5,
        method=RetrievalMethod.HYBRID,
        use_cross_encoder=True
    )
    
    logger.info("Top 5 search results:")
    for i, (doc_id, doc_text, score) in enumerate(example_results):
        doc_preview = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
        logger.info(f"{i+1}. [ID: {doc_id}, Score: {score:.4f}] {doc_preview}")
    
    return evaluation_results


# Execute main function if called directly
if __name__ == "__main__":
    main()