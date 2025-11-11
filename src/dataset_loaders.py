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
