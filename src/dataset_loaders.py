# dataset_loaders.py
"""
Simple dataset loader with if/else ladder.
Each dataset gets its own loading function.
"""

import logging
logger = logging.getLogger(__name__)

def load_dataset(dataset_name: str):
    """
    Load any dataset into standardized format.

    Returns:
        Tuple of (corpus_dataset, queries_dataset, qrels_dataset)
        Any of these can be None if not applicable for the dataset.
    """

    if dataset_name == "trec-covid":
        return _load_trec_covid()

    elif dataset_name == "msmarco":
        return _load_msmarco()

    elif dataset_name == "20newsgroups":
        return _load_20newsgroups()

    elif dataset_name == "doctor-reviews":
        return _load_doctor_reviews()

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_trec_covid():
    """Load TREC-COVID dataset (BeIR format)"""
    from datasets import load_dataset as hf_load_dataset

    logger.info("Loading TREC-COVID dataset...")
    corpus = hf_load_dataset("BeIR/trec-covid", "corpus")["corpus"]
    queries = hf_load_dataset("BeIR/trec-covid", "queries")["queries"]
    qrels = hf_load_dataset("BeIR/trec-covid-qrels", split="test")

    logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} qrels")
    return corpus, queries, qrels


def _load_msmarco():
    """Load MS MARCO dataset (BeIR format)"""
    from datasets import load_dataset as hf_load_dataset

    logger.info("Loading MS MARCO dataset...")
    # TODO: Implement when needed
    raise NotImplementedError("MS MARCO not implemented yet")


def _load_20newsgroups():
    """Load 20 Newsgroups dataset (no queries)"""
    from sklearn.datasets import fetch_20newsgroups

    logger.info("Loading 20 Newsgroups dataset...")
    # TODO: Implement when needed
    raise NotImplementedError("20 Newsgroups not implemented yet")


def _load_doctor_reviews():
    """
    Load pre-filtered Family Medicine doctor reviews dataset.

    The corpus must be created first by running:
        sbatch run_create_filtered_corpus.sh

    Returns:
        corpus: HuggingFace dataset with _id, title, text fields
        queries: List of query dicts with _id and text fields
        qrels: None (no relevance judgments for this dataset)
    """
    from datasets import load_from_disk, Dataset

    CORPUS_PATH = '/home/srangre1/datasets/doctor_reviews_family_med_filtered'

    logger.info("Loading Doctor Reviews dataset...")

    # Load pre-filtered corpus
    corpus = load_from_disk(CORPUS_PATH)

    # Define queries (manual list - no HuggingFace dataset)
    queries_list = [
        {"_id": "1", "text": "How do patients find and choose their doctors?"},
        {"_id": "2", "text": "What are patients' experiences with specialist referrals?"},
        {"_id": "3", "text": "What breathing problems do patients report and how are they treated?"},
        {"_id": "4", "text": "How do doctors manage patients with asthma?"},
        {"_id": "5", "text": "What do patients like about their doctors?"},
        {"_id": "6", "text": "What do patients dislike about their doctors?"},
    ]

    # Convert to HuggingFace Dataset for compatibility with existing code
    queries = Dataset.from_list(queries_list)

    # No relevance judgments for doctor reviews
    qrels = None

    logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries, 0 qrels")
    return corpus, queries, qrels
