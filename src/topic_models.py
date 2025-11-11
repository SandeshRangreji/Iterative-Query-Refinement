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

    def _fit_lda(self, *args, **kwargs) -> Dict[str, Any]:
        """LDA implementation - placeholder for future"""
        raise NotImplementedError("LDA not implemented yet")

    def _fit_topicgpt(self, *args, **kwargs) -> Dict[str, Any]:
        """TopicGPT implementation - placeholder for future"""
        raise NotImplementedError("TopicGPT not implemented yet")
