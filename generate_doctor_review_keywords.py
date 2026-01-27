#!/usr/bin/env python3
"""
Generate keywords for doctor review queries using KeyBERT.

Uses the same parameters as TREC-COVID keyword extraction:
- 10 keywords per query
- diversity=0.7
- top 10 retrieved docs for extraction
- mpnet embeddings
- ngram range 1-2

Saves to: /home/srangre1/cache/keywords/doctor_reviews_keybert.json
"""

import os
import sys
import json
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datasets import load_from_disk, Dataset
from keyword_extraction import KeywordExtractor
from search import RetrievalMethod, HybridStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Generate keywords for doctor review queries"""

    print('='*80)
    print('GENERATING KEYWORDS FOR DOCTOR REVIEW QUERIES')
    print('='*80)
    print()

    # ===== Configuration (matching TREC-COVID parameters) =====
    CORPUS_PATH = '/home/srangre1/datasets/doctor_reviews_family_med_filtered'
    OUTPUT_PATH = '/home/srangre1/cache/keywords/doctor_reviews_keybert.json'
    DATASET_NAME = 'doctor-reviews'

    # Keyword extraction parameters (same as TREC-COVID)
    NUM_KEYWORDS = 10
    DIVERSITY = 0.7
    KEYPHRASE_NGRAM_RANGE = (1, 2)
    TOP_K_DOCS = 1000
    TOP_N_DOCS_FOR_EXTRACTION = 10

    # Model parameters
    SBERT_MODEL = 'all-mpnet-base-v2'

    # Search parameters
    RETRIEVAL_METHOD = RetrievalMethod.HYBRID
    HYBRID_STRATEGY = HybridStrategy.SIMPLE_SUM
    HYBRID_WEIGHT = 0.5
    USE_MMR = False
    USE_CROSS_ENCODER = False
    MMR_LAMBDA = 0.5

    # Force flags
    FORCE_REINDEX = False
    FORCE_REGENERATE_KEYWORDS = True  # Always regenerate for new dataset

    # ===== Define queries =====
    queries = [
        {"_id": "1", "text": "How do patients find and choose their doctors?"},
        {"_id": "2", "text": "What are patients' experiences with specialist referrals?"},
        {"_id": "3", "text": "What breathing problems do patients report and how are they treated?"},
        {"_id": "4", "text": "How do doctors manage patients with asthma?"},
        {"_id": "5", "text": "What do patients like about their doctors?"},
        {"_id": "6", "text": "What do patients dislike about their doctors?"},
    ]

    # Convert to HuggingFace-like format for compatibility
    queries_dataset = Dataset.from_list(queries)

    # ===== Load corpus =====
    logger.info(f"Loading corpus from {CORPUS_PATH}...")
    corpus_dataset = load_from_disk(CORPUS_PATH)
    logger.info(f"Loaded {len(corpus_dataset):,} documents")

    # ===== Initialize extractor =====
    logger.info("Initializing keyword extractor...")
    extractor = KeywordExtractor(
        keybert_model=SBERT_MODEL,
        cache_dir='cache',
        top_k_docs=TOP_K_DOCS,
        top_n_docs_for_extraction=TOP_N_DOCS_FOR_EXTRACTION
    )

    # ===== Extract keywords =====
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

    # ===== Save to specific output path =====
    logger.info(f"Saving keywords to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(all_keywords, f, indent=2)

    # ===== Print results =====
    print()
    print('='*80)
    print('KEYWORD EXTRACTION RESULTS')
    print('='*80)
    print()

    for query in queries:
        query_id = query["_id"]
        query_text = query["text"]
        keywords = all_keywords.get(query_id, [])

        print(f'Query {query_id}: {query_text}')
        print(f'  Keywords: {keywords}')
        print()

    print('='*80)
    print(f'✅ Keywords saved to: {OUTPUT_PATH}')
    print('='*80)


if __name__ == "__main__":
    main()
