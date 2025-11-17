# topic_models.py
"""
Topic model wrapper - dispatches to different implementations.
All models MUST return the exact same dictionary structure.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class TopicModelWrapper:
    """
    Simple wrapper that dispatches to different topic modeling implementations.

    ALL implementations must return the EXACT SAME dictionary structure
    to ensure compatibility with evaluation metrics and visualizations.
    """

    def __init__(self, model_type: str = "bertopic", **params):
        """
        Initialize topic model wrapper.

        Args:
            model_type: Type of model ("bertopic", "lda", "topicgpt")
            **params: Model-specific parameters
        """
        self.model_type = model_type
        self.params = params
        self.model = None  # Will be set during fit

    def fit_and_get_results(
        self,
        docs: List[str],
        doc_ids: List[str],
        method_name: str,
        embedding_model_name: str,
        device: str,
        save_model: bool = False,
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fit model and return results in standardized format.

        This is the ONLY public method - it ensures all models return
        the exact same dictionary structure.

        Args:
            docs: List of document texts
            doc_ids: List of document IDs
            method_name: Sampling method name
            embedding_model_name: Name of embedding model
            device: Device to use ('cpu', 'cuda', 'mps')
            save_model: Whether to save the full model
            model_path: Path to save model (if save_model=True)

        Returns:
            Dictionary with EXACT structure expected by evaluation code
        """

        if self.model_type == "bertopic":
            return self._fit_bertopic(
                docs, doc_ids, method_name, embedding_model_name,
                device, save_model, model_path
            )

        elif self.model_type == "lda":
            return self._fit_lda(
                docs, doc_ids, method_name, embedding_model_name,
                device, save_model, model_path
            )

        elif self.model_type == "topicgpt":
            return self._fit_topicgpt(
                docs, doc_ids, method_name, embedding_model_name,
                device, save_model, model_path
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _fit_bertopic(
        self,
        docs: List[str],
        doc_ids: List[str],
        method_name: str,
        embedding_model_name: str,
        device: str,
        save_model: bool,
        model_path: Optional[str]
    ) -> Dict[str, Any]:
        """
        BERTopic implementation - EXTRACTED FROM CURRENT CODE.
        Returns EXACT same format as current run_topic_modeling().
        """
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from sentence_transformers import SentenceTransformer

        # Get params with defaults (EXACT same as current code)
        min_cluster_size = self.params.get("min_cluster_size", 5)
        metric = self.params.get("metric", "euclidean")
        cluster_selection_method = self.params.get("cluster_selection_method", "eom")
        min_df = self.params.get("min_df", 2)
        ngram_range = self.params.get("ngram_range", (1, 2))
        max_features = self.params.get("max_features", 10000)

        # Initialize SentenceTransformer (EXACT same as current code)
        embedding_model = SentenceTransformer(embedding_model_name, device=device)

        # Initialize HDBSCAN (EXACT same as current code)
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
            prediction_data=True
        )

        # Configure CountVectorizer (EXACT same as current code)
        vectorizer_model = CountVectorizer(
            stop_words='english',
            min_df=min_df,
            ngram_range=ngram_range,
            max_features=max_features
        )

        # Initialize BERTopic (EXACT same as current code)
        self.model = BERTopic(
            embedding_model=embedding_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            verbose=True,
            calculate_probabilities=True
        )

        # Fit model (EXACT same as current code)
        logger.info(f"Fitting BERTopic on {len(docs)} documents...")
        topics, probs = self.model.fit_transform(docs)

        # Get topic info (EXACT same as current code)
        topic_info = self.model.get_topic_info()

        # Get topic representations with scores (EXACT same as current code)
        topic_words = {}
        topic_words_with_scores = {}
        topic_labels = {}

        for topic_id in topic_info['Topic']:
            if topic_id != -1:  # Skip outlier topic
                words = self.model.get_topic(topic_id)
                topic_words[topic_id] = [word for word, _ in words]
                topic_words_with_scores[topic_id] = [(word, float(score)) for word, score in words]

                # Generate readable topic label from top words
                top_words = "_".join([word for word, _ in words[:3]])
                topic_labels[topic_id] = top_words

        # Get topic names (EXACT same as current code)
        topic_names = {}
        for topic_id in topic_info['Topic']:
            if topic_id != -1:
                topic_row = topic_info[topic_info['Topic'] == topic_id]
                if not topic_row.empty and 'Name' in topic_row.columns:
                    topic_names[topic_id] = topic_row['Name'].values[0]
                else:
                    topic_names[topic_id] = topic_labels.get(topic_id, f"Topic_{topic_id}")

        # Save model (EXACT same as current code)
        if save_model and model_path:
            self.model.save(model_path, serialization="pickle")
            logger.info(f"Saved topic model to {model_path}")
        else:
            logger.info(f"Skipping model save (save_topic_models=False) - saving ~420 MB")

        # Convert to list (EXACT same as current code)
        topics_list = topics.tolist() if hasattr(topics, 'tolist') else topics
        probs_list = probs.tolist() if probs is not None and hasattr(probs, 'tolist') else probs

        # Return EXACT same structure as current code
        return {
            "method": method_name,
            "doc_ids": doc_ids,
            "topics": topics_list,
            "probabilities": probs_list,
            "topic_info": topic_info.to_dict(),
            "topic_words": topic_words,
            "topic_words_with_scores": topic_words_with_scores,
            "topic_labels": topic_labels,
            "topic_names": topic_names,
            "n_topics": len(topic_words),
            "n_docs": len(docs)
        }

    def _fit_lda(
        self,
        docs: List[str],
        doc_ids: List[str],
        method_name: str,
        embedding_model_name: str,  # Not used by LDA, kept for API consistency
        device: str,                # Not used by LDA, kept for API consistency
        save_model: bool,
        model_path: Optional[str]
    ) -> Dict[str, Any]:
        """
        LDA (Latent Dirichlet Allocation) implementation using Gensim.

        Returns EXACT same format as _fit_bertopic() for metric compatibility.

        Key differences from BERTopic:
        - No embeddings (bag-of-words only)
        - Fixed number of topics (not auto-determined by HDBSCAN)
        - No outliers (all documents assigned to topics)
        - Probabilistic topic assignment (returns full distribution)

        Args:
            docs: List of document texts
            doc_ids: List of document IDs
            method_name: Sampling method name
            embedding_model_name: IGNORED (kept for API compatibility)
            device: IGNORED (LDA runs on CPU only)
            save_model: Whether to save the full model
            model_path: Path to save model (if save_model=True)

        Returns:
            Dictionary with EXACT structure expected by evaluation code
        """
        from gensim.models import LdaMulticore
        from sklearn.feature_extraction.text import CountVectorizer

        # ===== STEP 1: EXTRACT PARAMETERS =====

        n_topics = self.params.get("n_topics", 20)  # Default: 20 topics
        alpha = self.params.get("alpha", "symmetric")  # Document-topic prior
        eta = self.params.get("eta", 0.01)  # Topic-word prior (0.01 = sparse for biomedical text)
        passes = self.params.get("passes", 15)  # Number of passes through corpus
        iterations = self.params.get("iterations", 100)  # Max iterations per pass
        random_state = self.params.get("random_state", 42)
        workers = self.params.get("workers", 20)  # Multi-core training

        # Vocabulary parameters (MATCH BERTopic defaults for fair comparison)
        min_df = self.params.get("min_df", 2)
        ngram_range = self.params.get("ngram_range", (1, 2))
        max_features = self.params.get("max_features", 10000)

        logger.info(f"Fitting LDA with {n_topics} topics on {len(docs)} documents...")
        logger.info(f"  alpha={alpha}, eta={eta}, passes={passes}, iterations={iterations}, workers={workers}")
        logger.info(f"  min_df={min_df}, ngram_range={ngram_range}, max_features={max_features}")

        # ===== STEP 2: PREPROCESSING (MATCH BERTOPIC) =====

        # Use sklearn CountVectorizer to match BERTopic's vocabulary exactly
        vectorizer = CountVectorizer(
            stop_words='english',
            min_df=min_df,
            ngram_range=ngram_range,
            max_features=max_features,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens, 2+ chars
        )

        # Fit vectorizer and transform docs
        doc_term_matrix = vectorizer.fit_transform(docs)
        vocabulary = vectorizer.get_feature_names_out()

        logger.info(f"  Vocabulary size: {len(vocabulary)} terms")

        # Convert sklearn sparse matrix to Gensim format
        # Gensim expects: list of (word_id, count) tuples per document
        corpus_gensim = []
        for doc_idx in range(doc_term_matrix.shape[0]):
            doc_bow = []
            row = doc_term_matrix.getrow(doc_idx)
            for term_idx, count in zip(row.indices, row.data):
                doc_bow.append((int(term_idx), int(count)))
            corpus_gensim.append(doc_bow)

        # Create Gensim dictionary from vocabulary
        id2word = {i: word for i, word in enumerate(vocabulary)}

        # ===== STEP 3: TRAIN LDA MODEL =====

        logger.info(f"  Training LDA (this may take 2-5 minutes)...")

        self.model = LdaMulticore(
            corpus=corpus_gensim,
            id2word=id2word,
            num_topics=n_topics,
            alpha=alpha,
            eta=eta,
            passes=passes,
            iterations=iterations,
            random_state=random_state,
            workers=workers,
            per_word_topics=False  # We don't need per-word topic distributions
        )

        logger.info(f"  LDA training complete!")

        # ===== STEP 4: GET TOPIC ASSIGNMENTS =====

        # For each document, get topic distribution and assign to most probable topic
        topics = []  # Hard assignments (argmax of distribution)
        probs = []   # Full probability distributions

        for doc_bow in corpus_gensim:
            # Get topic distribution for this document
            topic_dist = self.model.get_document_topics(doc_bow, minimum_probability=0.0)

            # Sort by topic ID to ensure consistent ordering
            topic_dist = sorted(topic_dist, key=lambda x: x[0])

            # Extract probabilities (fill missing topics with 0.0)
            doc_probs = [0.0] * n_topics
            for topic_id, prob in topic_dist:
                doc_probs[topic_id] = float(prob)

            # Assign to most probable topic
            assigned_topic = int(np.argmax(doc_probs))

            topics.append(assigned_topic)
            probs.append(doc_probs)

        # ===== STEP 5: EXTRACT TOPIC REPRESENTATIONS =====

        topic_words = {}
        topic_words_with_scores = {}
        topic_labels = {}
        topic_names = {}

        for topic_id in range(n_topics):
            # Get top words for this topic (returns list of (word, probability) tuples)
            top_words = self.model.show_topic(topic_id, topn=20)

            # Extract word list and word-score pairs
            words = [word for word, _ in top_words]
            words_with_scores = [(word, float(score)) for word, score in top_words]

            topic_words[topic_id] = words
            topic_words_with_scores[topic_id] = words_with_scores

            # Generate readable labels (match BERTopic format)
            top_3_words = "_".join(words[:3])
            topic_labels[topic_id] = top_3_words
            topic_names[topic_id] = f"Topic_{topic_id}_{top_3_words}"

        # ===== STEP 6: CREATE TOPIC_INFO (MATCH BERTOPIC FORMAT) =====

        # Count documents per topic
        topic_counts = {}
        for t in topics:
            topic_counts[t] = topic_counts.get(t, 0) + 1

        topic_info_data = {
            "Topic": [],
            "Count": [],
            "Name": [],
            "Representation": []  # Top words as list
        }

        for topic_id in range(n_topics):
            topic_info_data["Topic"].append(topic_id)
            topic_info_data["Count"].append(topic_counts.get(topic_id, 0))
            topic_info_data["Name"].append(topic_names[topic_id])
            topic_info_data["Representation"].append(topic_words[topic_id][:10])

        # ===== STEP 7: SAVE MODEL (OPTIONAL) =====

        if save_model and model_path:
            self.model.save(model_path)
            logger.info(f"Saved LDA model to {model_path}")
        else:
            logger.info(f"Skipping model save (save_topic_models=False)")

        # ===== STEP 8: RETURN STANDARDIZED STRUCTURE =====

        return {
            "method": method_name,
            "doc_ids": doc_ids,
            "topics": topics,  # List[int], length = n_docs, NO -1 outliers
            "probabilities": probs,  # List[List[float]], shape = (n_docs, n_topics)
            "topic_info": topic_info_data,  # Dict (convertible to DataFrame)
            "topic_words": topic_words,  # Dict[int, List[str]]
            "topic_words_with_scores": topic_words_with_scores,  # Dict[int, List[Tuple[str, float]]]
            "topic_labels": topic_labels,  # Dict[int, str]
            "topic_names": topic_names,  # Dict[int, str]
            "n_topics": n_topics,
            "n_docs": len(docs)
        }

    def _fit_topicgpt(self, *args, **kwargs) -> Dict[str, Any]:
        """TopicGPT implementation - placeholder for future"""
        raise NotImplementedError("TopicGPT not implemented yet")
