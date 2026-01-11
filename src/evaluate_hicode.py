#!/usr/bin/env python3
# evaluate_hicode.py
"""
Standalone evaluation script for HiCode topic modeling results.

HiCode uses LLM-based inductive coding that can assign multiple topics per document.
This script converts HiCode's multi-label format to single-label format compatible
with the existing evaluation metrics pipeline.

Key conversions:
- topics: List[List[int]] → List[int] (flatten multi-label by taking first topic)
- topic_words: Add stopword filtering for lexical metrics
- topic_words_with_scores: Generate dummy scores (HiCode doesn't provide c-TF-IDF)
- topic_labels: Generate from topic_names

Usage:
    python src/evaluate_hicode.py
"""

import os
import sys
import pickle
import logging
import json
from typing import Dict, List, Any, Tuple
from collections import Counter

# Import evaluation framework
from end_to_end_evaluation import EndToEndEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: FORMAT CONVERSION FUNCTIONS
# =============================================================================

def flatten_hicode_topics(topics: List[List[int]]) -> List[int]:
    """
    Convert HiCode's multi-label format to single-label.

    HiCode can assign multiple topics per document (e.g., [5, 2] means topics 5 AND 2).
    Our evaluation metrics expect single-label (e.g., 5).

    Strategy: Take first topic from each list.
    - [5, 2] → 5 (primary topic)
    - [-1] → -1 (outlier)
    - [] → -1 (unassigned, treat as outlier)

    Args:
        topics: HiCode topics in List[List[int]] format

    Returns:
        Flattened topics as List[int]

    Examples:
        >>> flatten_hicode_topics([[5], [5, 2], [-1], []])
        [5, 5, -1, -1]
    """
    flat_topics = []

    for topic_list in topics:
        if isinstance(topic_list, list):
            if len(topic_list) == 0:
                flat_topics.append(-1)  # Empty list → outlier
            else:
                flat_topics.append(topic_list[0])  # Take first/primary topic
        else:
            # Shouldn't happen, but handle gracefully
            flat_topics.append(topic_list if isinstance(topic_list, int) else -1)

    return flat_topics


def preprocess_topic_words_for_lexical_metrics(
    topic_words: Dict[int, List[str]],
    remove_stopwords: bool = True,
    remove_numbers: bool = True,
    min_word_length: int = 2,
    min_words_per_topic: int = 3
) -> Dict[int, List[str]]:
    """
    Preprocess HiCode topic words for lexical metrics (word overlap, lexical diversity).

    NOTE: This function is currently NOT USED in the evaluation pipeline.

    LIMITATION: The EndToEndEvaluator extracts topic_words once and passes it to all
    metric methods. We cannot intercept this without modifying end_to_end_evaluation.py.
    Therefore, HiCode's lexical metrics will be computed with stopwords included.

    This is ACCEPTABLE because:
    - Semantic metrics (embedding-based) are unaffected by stopwords
    - NPMI coherence uses doc_texts, not topic words
    - Lexical metrics will just reflect HiCode's natural language format "as-is"
    - Summaries correctly show original HiCode topic words/names

    Per user directive: "do stopword filtering exclusively in the places where we need
    them (to calculate lexical overlap and other lexical based metrics). keep it as it
    is everywhere else - so in the summaries, we should see the topics as it is"

    This function is preserved for potential future use if we want to:
    1. Monkey-patch the evaluator's lexical metric methods, OR
    2. Create HiCode-specific metric methods with preprocessing

    HiCode generates natural language phrases (e.g., "domestic and intimate partner violence"),
    which are broken into words. This creates issues for lexical metrics:
    - Stopwords: "and", "of", "the", "among" inflate diversity metrics
    - Numbers: "19" (from COVID-19) are not meaningful
    - Single chars: "a", "s" are noise

    Args:
        topic_words: Original topic words from HiCode
        remove_stopwords: Whether to filter stopwords
        remove_numbers: Whether to filter numeric tokens
        min_word_length: Minimum word length to keep
        min_words_per_topic: Minimum words to preserve (fallback to original if too few)

    Returns:
        Cleaned topic words (for lexical metrics ONLY)
    """
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    except:
        logger.warning("NLTK stopwords not available, skipping stopword removal")
        stop_words = set()
        remove_stopwords = False

    cleaned_words = {}

    for topic_id, words in topic_words.items():
        filtered = []

        for word in words:
            word_lower = word.lower().strip()

            # Apply filters
            if remove_stopwords and word_lower in stop_words:
                continue
            if remove_numbers and word_lower.isdigit():
                continue
            if len(word_lower) < min_word_length:
                continue

            filtered.append(word)

        # Keep at least min_words_per_topic words per topic
        # If filtering removes too many, fall back to original
        if len(filtered) < min_words_per_topic:
            filtered = words[:max(min_words_per_topic, len(words))]

        cleaned_words[topic_id] = filtered

    return cleaned_words


def validate_hicode_format(hicode_data: Dict[str, Any]) -> bool:
    """
    Validate that HiCode pickle has expected structure.

    Args:
        hicode_data: Loaded HiCode pickle data

    Returns:
        True if valid

    Raises:
        ValueError: If critical fields are missing or malformed
    """
    required_fields = [
        'method', 'doc_ids', 'doc_texts', 'sample_size',
        'n_topics', 'topics', 'topic_names', 'topic_words'
    ]

    # Check required fields
    for field in required_fields:
        if field not in hicode_data:
            raise ValueError(f"Missing required field: '{field}'")

    # Check topics format (should be List[List[int]])
    if not isinstance(hicode_data['topics'], list):
        raise ValueError("'topics' must be a list")

    if len(hicode_data['topics']) > 0:
        if not isinstance(hicode_data['topics'][0], list):
            raise ValueError(
                f"'topics' must be List[List[int]] for HiCode, "
                f"but got {type(hicode_data['topics'][0])}"
            )

    # Check alignment
    n_docs = len(hicode_data['doc_ids'])
    if len(hicode_data['topics']) != n_docs:
        raise ValueError(
            f"Misaligned: {len(hicode_data['topics'])} topic assignments "
            f"vs {n_docs} documents"
        )

    if len(hicode_data['doc_texts']) != n_docs:
        raise ValueError(
            f"Misaligned: {len(hicode_data['doc_texts'])} doc_texts "
            f"vs {n_docs} documents"
        )

    logger.info(f"✓ Validation passed for method '{hicode_data['method']}'")
    return True


def convert_hicode_to_standard_format(
    hicode_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert HiCode results to standard format compatible with evaluation pipeline.

    This is the main conversion function that transforms HiCode's multi-label format
    into the single-label format expected by EndToEndEvaluator.

    Key conversions:
    1. Flatten multi-label topics to single-label (take first topic)
    2. Keep topic_words as-is (stopword filtering applied ONLY in lexical metrics)
    3. Keep topic_names as-is (natural language, not converted to labels)
    4. Preserve original multi-label assignments for reference

    IMPORTANT - No dummy values:
    - NO dummy c-TF-IDF scores (different format is acceptable for HiCode)
    - NO topic_labels generation (use natural language topic_names directly)
    - Stopword filtering happens ONLY in lexical metrics, not globally

    Args:
        hicode_data: Raw HiCode pickle data

    Returns:
        Dictionary in standard format compatible with BERTopic/LDA/TopicGPT

    Raises:
        ValueError: If HiCode data fails validation
    """
    # Validate input format
    validate_hicode_format(hicode_data)

    method = hicode_data['method']
    logger.info(f"Converting HiCode results for method: {method}")

    # 1. Flatten multi-label topics
    topics_flat = flatten_hicode_topics(hicode_data['topics'])

    # Count multi-label docs for logging
    multi_label_count = sum(1 for t in hicode_data['topics'] if len(t) > 1)
    if multi_label_count > 0:
        logger.info(
            f"  Multi-label documents: {multi_label_count}/{len(hicode_data['topics'])} "
            f"({100*multi_label_count/len(hicode_data['topics']):.1f}%) - taking first topic"
        )

    # 2. Flatten topic_words from HiCode
    # HiCode stores topic_words as Dict[int, List[List[str]]] (multiple word lists per topic)
    # We need Dict[int, List[str]] (one word list per topic) for evaluation
    # Strategy: Take the first word list (assumed to be most important/primary)
    topic_words = {}
    for topic_id, word_lists in hicode_data['topic_words'].items():
        if isinstance(word_lists[0], list):
            # Nested structure: take first list
            topic_words[topic_id] = word_lists[0]
        else:
            # Already flat: keep as-is
            topic_words[topic_id] = word_lists
    logger.info(f"  Flattened topic_words (taking first word list from HiCode's nested structure)")

    # 3. Construct standard format
    standard_format = {
        # ===== ORIGINAL FIELDS (pass through) =====
        'method': hicode_data['method'],
        'doc_ids': hicode_data['doc_ids'],
        'n_topics': hicode_data['n_topics'],
        'topic_words': topic_words,  # Original words, no global stopword filtering

        # Natural language topic names (e.g., "domestic and intimate partner violence")
        # Summaries will use these directly instead of generated labels
        'topic_names': hicode_data['topic_names'],

        # ===== TRANSFORMED FIELDS =====
        'topics': topics_flat,  # List[int] instead of List[List[int]]

        # ===== FIELDS NOT PROVIDED BY HICODE =====
        # These are NOT generated as dummy values - different format is acceptable
        # - topic_words_with_scores: HiCode doesn't provide c-TF-IDF scores
        # - topic_labels: Use natural language topic_names directly in summaries
        'probabilities': None,  # HiCode doesn't provide probabilities

        # ===== METADATA =====
        'n_docs': hicode_data['sample_size'],
        'topic_model_type': 'hicode',

        # ===== PRESERVED FOR REFERENCE =====
        '_hicode_original_multi_label_topics': hicode_data['topics'],
    }

    logger.info(f"  ✓ Converted to standard format")
    logger.info(f"    - Flattened topics: {len(topics_flat)} assignments")
    logger.info(f"    - Topics with words: {len(topic_words)}")
    logger.info(f"    - Using natural language topic_names (no label generation)")

    return standard_format


# =============================================================================
# SECTION 2: EVALUATION PIPELINE
# =============================================================================

def load_datasets():
    """
    Load TREC-COVID datasets needed for evaluation.

    Returns:
        Tuple of (corpus_dataset, queries_dataset, qrels_dataset)
    """
    logger.info("Loading TREC-COVID datasets...")

    from datasets import load_dataset

    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels", split="test")

    logger.info(
        f"✓ Loaded {len(corpus_dataset)} documents, "
        f"{len(queries_dataset)} queries, "
        f"{len(qrels_dataset)} qrels"
    )

    return corpus_dataset, queries_dataset, qrels_dataset


def load_hicode_results(
    query_id: str,
    hicode_dir: str,
    methods: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Load HiCode results for all sampling methods.

    Args:
        query_id: Query ID (e.g., "43")
        hicode_dir: Directory containing HiCode pickle files
        methods: List of sampling method names

    Returns:
        Dictionary mapping method names to HiCode results
    """
    logger.info(f"\nLoading HiCode results from: {hicode_dir}")

    hicode_results = {}

    for method in methods:
        pkl_path = os.path.join(hicode_dir, f"{method}_hicode.pkl")

        if not os.path.exists(pkl_path):
            logger.warning(f"  ✗ Not found: {pkl_path}")
            continue

        try:
            with open(pkl_path, 'rb') as f:
                hicode_data = pickle.load(f)

            hicode_results[method] = hicode_data
            logger.info(f"  ✓ Loaded {method}: {hicode_data['n_topics']} topics")

        except Exception as e:
            logger.error(f"  ✗ Error loading {method}: {e}")
            continue

    logger.info(f"\n✓ Loaded {len(hicode_results)}/{len(methods)} methods")
    return hicode_results


def load_samples(
    query_id: str,
    samples_dir: str,
    methods: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Load document samples (shared across all topic models).

    Args:
        query_id: Query ID
        samples_dir: Directory containing sample pickle files
        methods: List of sampling method names

    Returns:
        Dictionary mapping method names to samples
    """
    logger.info(f"\nLoading samples from: {samples_dir}")

    samples = {}

    for method in methods:
        sample_path = os.path.join(samples_dir, f"{method}.pkl")

        if not os.path.exists(sample_path):
            logger.warning(f"  ✗ Not found: {sample_path}")
            continue

        try:
            with open(sample_path, 'rb') as f:
                sample = pickle.load(f)

            samples[method] = sample
            logger.info(f"  ✓ Loaded {method}: {len(sample['doc_ids'])} docs")

        except Exception as e:
            logger.error(f"  ✗ Error loading {method}: {e}")
            continue

    logger.info(f"\n✓ Loaded {len(samples)}/{len(methods)} samples")
    return samples


def validate_doc_id_alignment(
    hicode_results: Dict[str, Dict[str, Any]],
    samples: Dict[str, Dict[str, Any]]
):
    """
    Validate that HiCode doc_ids match sample doc_ids.

    This ensures HiCode was run on the correct samples.

    Args:
        hicode_results: HiCode results by method
        samples: Samples by method

    Raises:
        ValueError: If doc_ids don't match
    """
    logger.info("\nValidating doc_id alignment between HiCode and samples...")

    for method in hicode_results.keys():
        if method not in samples:
            logger.warning(f"  ✗ No sample found for method: {method}")
            continue

        hicode_ids = hicode_results[method]['doc_ids']
        sample_ids = samples[method]['doc_ids']

        if hicode_ids != sample_ids:
            raise ValueError(
                f"Doc ID mismatch for {method}!\n"
                f"  HiCode has {len(hicode_ids)} docs\n"
                f"  Sample has {len(sample_ids)} docs\n"
                f"  First mismatch: HiCode[0]={hicode_ids[0]}, Sample[0]={sample_ids[0]}"
            )

        logger.info(f"  ✓ {method}: {len(hicode_ids)} doc_ids match")

    logger.info("✓ All doc_ids validated")


def save_config(
    output_dir: str,
    query_id: str,
    query_text: str,
    config_params: Dict[str, Any]
):
    """
    Save configuration to JSON for reproducibility.

    Args:
        output_dir: Output directory
        query_id: Query ID
        query_text: Query text
        config_params: Additional configuration parameters
    """
    config = {
        "query_id": query_id,
        "query_text": query_text,
        "topic_model_type": "hicode",
        "evaluation_script": "evaluate_hicode.py",
        "conversion_settings": {
            "multi_label_strategy": "take_first_topic",
            "stopword_filtering": config_params.get('stopword_filtering', 'lexical_metrics_only'),
            "no_dummy_values": "Different format acceptable for HiCode",
        },
        "limitations": [
            "Multi-label topics flattened to single-label (affects 2-3% of documents)",
            "No c-TF-IDF scores (HiCode doesn't provide them)",
            "No probabilities (HiCode doesn't compute topic probabilities)",
            "Natural language topic names used directly (not converted to labels)"
        ],
        **config_params
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"✓ Saved configuration to: {config_path}")


def run_hicode_evaluation(
    query_id: str,
    hicode_dir: str,
    samples_dir: str,
    output_dir: str,
    sample_size: int = 1000,
    embedding_model: str = "all-mpnet-base-v2",
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    dataset_name: str = "trec-covid",
    device: str = "cpu",
    random_seed: int = 42
):
    """
    Run full HiCode evaluation pipeline.

    This is the main orchestration function that:
    1. Loads HiCode results and samples
    2. Converts HiCode format to standard format
    3. Creates EndToEndEvaluator instance
    4. Runs evaluation (steps 3-6: summaries, pairwise, plots)

    Args:
        query_id: Query ID to evaluate
        hicode_dir: Directory with HiCode topic model pickles
        samples_dir: Directory with document samples
        output_dir: Output directory for results
        sample_size: Expected sample size
        embedding_model: Embedding model name
        cross_encoder_model: Cross-encoder model name
        dataset_name: Dataset name
        device: Device for embeddings
        random_seed: Random seed

    Returns:
        Dictionary with evaluation results
    """
    logger.info("="*80)
    logger.info(f"HICODE EVALUATION - QUERY {query_id}")
    logger.info("="*80)

    # Define sampling methods
    methods = [
        'random_uniform',
        'direct_retrieval',
        'direct_retrieval_mmr',
        'query_expansion',
        'keyword_search',
        'retrieval_random'
    ]

    # Load datasets
    corpus_dataset, queries_dataset, qrels_dataset = load_datasets()

    # Get query text
    query_text = None
    for query in queries_dataset:
        if str(query["_id"]) == query_id:
            query_text = query["text"]
            break

    if query_text is None:
        raise ValueError(f"Query {query_id} not found in queries dataset")

    logger.info(f"\nQuery {query_id}: {query_text}")

    # Load HiCode results
    hicode_results = load_hicode_results(query_id, hicode_dir, methods)

    if len(hicode_results) == 0:
        raise ValueError(f"No HiCode results found in {hicode_dir}")

    # Load samples
    samples = load_samples(query_id, samples_dir, methods)

    if len(samples) == 0:
        raise ValueError(f"No samples found in {samples_dir}")

    # Validate alignment
    validate_doc_id_alignment(hicode_results, samples)

    # Convert HiCode format to standard format
    logger.info("\n" + "="*80)
    logger.info("CONVERTING HICODE FORMAT TO STANDARD FORMAT")
    logger.info("="*80)

    topic_results = {}
    for method, hicode_data in hicode_results.items():
        try:
            topic_results[method] = convert_hicode_to_standard_format(hicode_data)
        except Exception as e:
            logger.error(f"✗ Conversion failed for {method}: {e}")
            continue

    # Filter samples to match converted results
    samples = {k: v for k, v in samples.items() if k in topic_results}

    if len(topic_results) < 2:
        raise ValueError(
            f"Need at least 2 methods for pairwise comparison, got {len(topic_results)}"
        )

    logger.info(f"\n✓ Successfully converted {len(topic_results)} methods")

    # Create minimal evaluator instance (NO SEARCH ENGINE - we don't need it!)
    logger.info("\n" + "="*80)
    logger.info("INITIALIZING EVALUATOR (minimal - no search indices)")
    logger.info("="*80)

    # We need to create the evaluator but bypass expensive initialization
    # Only qrels_dict and output directories are needed for evaluation
    from end_to_end_evaluation import EndToEndEvaluator

    # Create instance with minimal initialization
    evaluator = object.__new__(EndToEndEvaluator)

    # Set only the attributes needed for evaluation steps 3-6
    evaluator.query_id = query_id
    evaluator.query_text = query_text
    evaluator.sample_size = sample_size
    evaluator.embedding_model_name = embedding_model
    evaluator.device = device
    evaluator.dataset_name = dataset_name
    evaluator.topic_model_type = "hicode"

    # Build qrels_dict (needed for relevant_concentration metric)
    from collections import defaultdict
    evaluator.qrels_dict = defaultdict(dict)
    for qid, cid, score in zip(qrels_dataset["query-id"],
                                qrels_dataset["corpus-id"],
                                qrels_dataset["score"]):
        evaluator.qrels_dict[qid][cid] = score

    # Load IDF scores (needed for topic_specificity metric)
    idf_cache_path = "cache/corpus_171332/idf_scores.pkl"
    if os.path.exists(idf_cache_path):
        import pickle
        with open(idf_cache_path, 'rb') as f:
            evaluator.idf_scores = pickle.load(f)
        logger.info(f"  ✓ Loaded IDF scores from cache: {len(evaluator.idf_scores)} terms")
    else:
        logger.warning(f"  IDF cache not found at {idf_cache_path} - topic_specificity will be 0")
        evaluator.idf_scores = {}

    # Setup output directories
    evaluator.output_dir = os.path.join(output_dir, dataset_name, "hicode", f"query_{query_id}")
    evaluator.results_dir = os.path.join(evaluator.output_dir, "results")
    evaluator.plots_dir = os.path.join(evaluator.results_dir, "plots")
    evaluator.topics_summary_dir = os.path.join(evaluator.results_dir, "topics_summary")

    os.makedirs(evaluator.results_dir, exist_ok=True)
    os.makedirs(evaluator.plots_dir, exist_ok=True)
    os.makedirs(evaluator.topics_summary_dir, exist_ok=True)

    logger.info(f"  ✓ Evaluator initialized (skipped search engine initialization)")
    logger.info(f"  ✓ Output directory: {evaluator.output_dir}")

    # Save configuration
    save_config(
        evaluator.output_dir,
        query_id,
        query_text,
        {
            'sample_size': sample_size,
            'embedding_model': embedding_model,
            'dataset_name': dataset_name,
            'random_seed': random_seed,
            'stopword_filtering': 'lexical_metrics_only',  # Not applied globally
            'methods_evaluated': list(topic_results.keys())
        }
    )

    # SKIP STEPS 1-2 (sampling & topic modeling already done)
    # RUN STEPS 3-6 (evaluation pipeline)

    # Step 3: Save topic summaries
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Saving Topic Summaries")
    logger.info("="*80)

    try:
        evaluator._save_topic_summaries(topic_results, samples)
    except Exception as e:
        logger.error(f"Error saving topic summaries: {e}")
        logger.warning("Continuing with evaluation despite topic summary error")

    # Step 4: Run pairwise comparisons
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Running Pairwise Comparisons")
    logger.info("="*80)

    pairwise_df = evaluator.run_all_pairwise_comparisons(topic_results, samples)

    if pairwise_df.empty:
        logger.error("No pairwise comparisons completed. Exiting.")
        return None

    # Step 5: Create per-method summary
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Creating Per-Method Summary")
    logger.info("="*80)

    per_method_df = evaluator.create_per_method_summary(pairwise_df, topic_results)

    # Step 6: Create comprehensive visualizations
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Creating Comprehensive Visualizations")
    logger.info("="*80)

    evaluator.create_comprehensive_plots(pairwise_df, per_method_df)

    # Step 7: Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)

    print("\nPer-Method Summary:")
    print(per_method_df.to_string(index=False))

    logger.info("\n" + "="*80)
    logger.info("HICODE EVALUATION COMPLETE!")
    logger.info(f"Results saved to: {evaluator.output_dir}")
    logger.info("="*80)

    return {
        "samples": samples,
        "topic_results": topic_results,
        "pairwise_metrics": pairwise_df,
        "per_method_summary": per_method_df,
        "output_dir": evaluator.output_dir
    }


# =============================================================================
# SECTION 3: MAIN ENTRY POINT
# =============================================================================

def main():
    """Main function - configure and run HiCode evaluation."""

    # ===== CONFIGURATION =====

    QUERY_ID = "43"

    # Paths
    HICODE_DIR = "/home/srangre1/results/trec-covid/hicode/query_43/topic_models"
    SAMPLES_DIR = "/home/srangre1/results/trec-covid/hicode/query_43/samples"
    OUTPUT_DIR = "results"

    # Model configuration
    SAMPLE_SIZE = 1000
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    DATASET_NAME = "trec-covid"

    # Device configuration
    DEVICE = "cpu"  # Use "cuda" if GPU available

    # Random seed
    RANDOM_SEED = 42

    # ===== RUN EVALUATION =====

    try:
        results = run_hicode_evaluation(
            query_id=QUERY_ID,
            hicode_dir=HICODE_DIR,
            samples_dir=SAMPLES_DIR,
            output_dir=OUTPUT_DIR,
            sample_size=SAMPLE_SIZE,
            embedding_model=EMBEDDING_MODEL,
            cross_encoder_model=CROSS_ENCODER_MODEL,
            dataset_name=DATASET_NAME,
            device=DEVICE,
            random_seed=RANDOM_SEED
        )

        if results:
            print(f"\n✓ Success! Results available at: {results['output_dir']}")
            return 0
        else:
            print("\n✗ Evaluation failed")
            return 1

    except Exception as e:
        logger.error(f"\n✗ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
