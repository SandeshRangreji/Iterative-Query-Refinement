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

    elif dataset_name == "salescontest":
        return _load_oida_salescontest()
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

    CORPUS_PATH = "" #TODO: Add path to pre-filtered corpus

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
        {"_id": "7", "text": "What follow-up care or testing do doctors recommend for people with asthma?"},
        {"_id": "8", "text": "What do patients like about treatment or management recommendations?"},
        {"_id": "9", "text": "What do patients dislike about treatment or management recommendations?"},
        {"_id": "10", "text": "What lifestyle challenges do patients with asthma report?"},
        {"_id": "11", "text": "What symptoms do patients with asthma report?"},

        # generated questions gpt-5.6
        {"_id": "12", "text": "What barriers do patients report when scheduling appointments?"},
        {"_id": "13", "text": "How long do patients report waiting for appointments?"},
        {"_id": "14", "text": "What barriers do patients report when contacting physicians between visits?"},
        {"_id": "15", "text": "What insurance-related barriers do patients report when seeking asthma care?"},
        {"_id": "16", "text": "What out-of-pocket costs for asthma care do patients report?"},
        {"_id": "17", "text": "How do patients describe delays in receiving an asthma diagnosis?"},
        {"_id": "18", "text": "What concerns do patients express about diagnostic uncertainty?"},
        {"_id": "19", "text": "How do patients describe physicians’ explanations of asthma?"},
        {"_id": "20", "text": "How do patients describe physicians’ listening behavior?"},
        {"_id": "21", "text": "How do patients describe their involvement in asthma treatment decisions?"},
        {"_id": "22", "text": "What experiences of disrespect do patients report during asthma care?"},
        {"_id": "23", "text": "What experiences of discrimination do patients report in asthma care?"},
        {"_id": "24", "text": "What language barriers do patients report during asthma care?"},
        {"_id": "25", "text": "What concerns do patients report about interactions with clinic staff?"},
        {"_id": "26", "text": "What problems do patients report with continuity of care?"},
        {"_id": "27", "text": "What medication side effects do patients report?"},
        {"_id": "28", "text": "What difficulties do patients report when using inhalers?"},
        {"_id": "29", "text": "What barriers do patients report when obtaining asthma medications?"},
        {"_id": "30", "text": "What concerns do patients express about long-term asthma medication safety?"},
        {"_id": "31", "text": "What experiences do patients report with written asthma action plans?"},
        {"_id": "32", "text": "What experiences do patients report with emergency care for asthma attacks?"},
        {"_id": "33", "text": "What experiences do patients report with telehealth asthma care?"}
    ]

    # Convert to HuggingFace Dataset for compatibility with existing code
    queries = Dataset.from_list(queries_list)

    # No relevance judgments for doctor reviews
    qrels = None

    logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries, 0 qrels")
    return corpus, queries, qrels


def _load_oida_salescontest():
    from datasets import load_from_disk, Dataset
    
    CORPUS_PATH = "" #TODO: Add path to pre-filtered corpus

    logger.info("Loading OIDA Sales Contest dataset...")

    # Load pre-filtered corpus
    corpus = load_from_disk(CORPUS_PATH)

    # Define queries (manual list - no HuggingFace dataset)
    queries_list = [
        {"_id": "1", "text": "What corporate strategies did opioid manufacturers use to create and expand markets?"},
        {"_id": "2", "text": "How have corporations represented the risks of opioids?"},
        {"_id": "3", "text": "How have corporations represented the benefits of opioids?"},
        {"_id": "4", "text": "How have pharmaceutical sales representatives been trained to interact with prescribers?"},
        {"_id": "5", "text": "What types of prescribers are targeted in sales contests?"},
        {"_id": "6", "text": "What role have health practitioners played in marketing?"},
        {"_id": "7", "text": "What are communication techniques used in marketing opioids during sales contests?"},
        {"_id": "8", "text": "What incentives have corporations provide for sales representatitives in sales contests?"},
        {"_id": "9", "text": "What strategies did manufacturers use to respond to regulatory concerns regarding opioid safety?"},
        {"_id": "10", "text": "How did manufacturers respond to regulatory standards in the United States for their marketing practices?"},
        {"_id": "11", "text": "What are patterns about corporate social responsibilities in these sales contests?"},
        {"_id": "12", "text": "What are marketing challenges during the sales contests?"},
        {"_id": "13", "text": "How have corporations represented the objectives and goals of sales contests?"},
        {"_id": "14", "text": "What penalties, if any, have corporations implemented to their sales representatives in contests?"},
        {"_id": "15", "text": "How do sales representatives discover potential prescribers?"},
        {"_id": "16", "text": "What are symptoms of patients mentioned in these corporation communications?"},

        {"_id": "17", "text": "How were sales contests scheduled in relation to product launches or promotional campaigns?"},
        {"_id": "18", "text": "What performance metrics were used to evaluate sales representatives during contests?"},
        {"_id": "19", "text": "How were sales representatives ranked during contests?"},
        {"_id": "20", "text": "How were sales territories assigned during sales contests?"},
        {"_id": "21", "text": "How were team-based contests structured?"},
        {"_id": "22", "text": "How did contest materials use game-like features to motivate sales representatives?"},
        {"_id": "23", "text": "How did sales managers monitor representative performance during contests?"},
        {"_id": "24", "text": "What sources of sales data were used to determine contest performance?"},
        {"_id": "25", "text": "How were prescribers segmented for contest-related marketing activities?"},
        {"_id": "26", "text": "How were samples or promotional products used during sales contests?"},
        {"_id": "27", "text": "How were opioid products compared with competing treatments in contest communications?"},
        {"_id": "28", "text": "How were different opioid formulations promoted during sales contests?"},
        {"_id": "29", "text": "How were prescribing targets adapted to local markets?"},
        {"_id": "30", "text": "How did contest communications address off-label prescribing?"},
        {"_id": "31", "text": "How were regulatory requirements documented in contest materials?"},
        {"_id": "32", "text": "What types of clinical evidence were cited in contest communications?"},
        {"_id": "33", "text": "How were patient experiences used in contest-related marketing?"},
        {"_id": "34", "text": "How did sales contests influence prescribing behavior?"},
        {"_id": "35", "text": "How did contest communications describe competing pharmaceutical manufacturers?"},
        {"_id": "36", "text": "How were sales contests evaluated after they ended?"}
    ]

    # Convert to HuggingFace Dataset for compatibility with existing code
    queries = Dataset.from_list(queries_list)
    # No relevance judgments for oida sales
    qrels = None
    
    logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries, 0 qrels")
    return corpus, queries, qrels