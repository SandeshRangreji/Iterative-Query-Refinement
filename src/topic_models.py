# topic_models.py
"""
Topic model wrapper - dispatches to different implementations.
All models MUST return the exact same dictionary structure.
"""

import logging
import os
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter

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

    def _fit_topicgpt(
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
        TopicGPT implementation using LLM-based topic modeling.

        Returns EXACT same format as _fit_bertopic() and _fit_lda() for metric compatibility.

        Requires OPENAI_API_KEY environment variable to be set.

        Args:
            docs: List of document texts
            doc_ids: List of document IDs
            method_name: Sampling method name
            embedding_model_name: Name of embedding model (used for word embeddings)
            device: Device to use for embeddings
            save_model: Whether to save the model artifacts
            model_path: Path to save model (if save_model=True)

        Returns:
            Dictionary with EXACT structure expected by evaluation code
        """
        # Import TopicGPT functions
        try:
            from topicgpt_python import generate_topic_lvl1, assign_topics
        except ImportError:
            raise ImportError(
                "topicgpt_python not installed. Install with: pip install topicgpt_python"
            )

        # Check for API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "TopicGPT requires an OpenAI API key."
            )

        # ===== STEP 1: EXTRACT PARAMETERS =====

        # Model configuration
        generation_model = self.params.get("generation_model", "gpt-4o-mini")
        assignment_model = self.params.get("assignment_model", "gpt-4o-mini")

        # Sampling for topic generation (use subset for efficiency)
        generation_sample_size = self.params.get("generation_sample_size", min(500, len(docs)))

        # Verbose output
        verbose = self.params.get("verbose", True)

        logger.info(f"Fitting TopicGPT on {len(docs)} documents...")
        logger.info(f"  Generation model: {generation_model}")
        logger.info(f"  Assignment model: {assignment_model}")
        logger.info(f"  Generation sample size: {generation_sample_size}")

        # ===== STEP 2: CREATE TEMPORARY FILES FOR TOPICGPT API =====

        import tempfile
        import json
        import pandas as pd

        # Create temp directory for TopicGPT files
        temp_dir = tempfile.mkdtemp(prefix="topicgpt_")
        logger.info(f"  Temp directory: {temp_dir}")

        try:
            # --- Create data file (JSONL format) ---
            # Use top-K documents for topic generation (no sampling)
            if len(docs) > generation_sample_size:
                # Take first generation_sample_size documents (top-ranked by relevance)
                generation_docs = docs[:generation_sample_size]
                generation_ids = doc_ids[:generation_sample_size]
                logger.info(f"  Using top {len(generation_docs)} documents for topic generation")
            else:
                generation_docs = docs
                generation_ids = doc_ids

            # Write generation data
            gen_data_file = os.path.join(temp_dir, "generation_data.jsonl")
            with open(gen_data_file, "w") as f:
                for doc_id, text in zip(generation_ids, generation_docs):
                    json.dump({"id": str(doc_id), "text": text}, f)
                    f.write("\n")

            # Write full assignment data
            assign_data_file = os.path.join(temp_dir, "assignment_data.jsonl")
            with open(assign_data_file, "w") as f:
                for doc_id, text in zip(doc_ids, docs):
                    json.dump({"id": str(doc_id), "text": text}, f)
                    f.write("\n")

            # --- Create prompt files ---
            # Using original TopicGPT prompts from https://github.com/chtmp223/topicGPT
            generation_prompt = """You will receive a document and a set of top-level topics from a topic hierarchy. Your task is to identify generalizable topics within the document that can act as top-level topics in the hierarchy. If any relevant topics are missing from the provided set, please add them. Otherwise, output the existing top-level topics as identified in the document.

[Top-level topics]
{Topics}

[Examples]
Example 1: Adding "[1] Infectious Disease Research"
Document:
A study investigating the transmission dynamics of SARS-CoV-2 in healthcare settings, examining viral load correlations with disease severity and outcomes in hospitalized COVID-19 patients.

Your response:
[1] Infectious Disease Research: Studies on pathogen transmission, disease mechanisms, and clinical outcomes.

Example 2: Duplicate "[1] Public Health Interventions", returning the existing topic
Document:
Analysis of social distancing measures' effectiveness in reducing COVID-19 transmission rates across different population densities and demographic groups.

Your response:
[1] Public Health Interventions: Measures and policies implemented to control disease spread.

[Instructions]
Step 1: Determine topics mentioned in the document.
- The topic labels must be as GENERALIZABLE as possible. They must not be document-specific.
- The topics must reflect a SINGLE topic instead of a combination of topics.
- The new topics must have a level number, a short general label, and a topic description.
- The topics must be broad enough to accommodate future subtopics.
Step 2: Perform ONE of the following operations:
1. If there are already duplicates or relevant topics in the hierarchy, output those topics and stop here.
2. If the document contains no topic, return "None".
3. Otherwise, add your topic as a top-level topic. Stop here and output the added topic(s). DO NOT add any additional levels.

[Document]
{Document}

Please ONLY return the relevant or modified topics at the top level in the hierarchy. Your response should be in the following format:
[Topic Level] Topic Label: Topic Description

Your response:"""

            assignment_prompt = """You will receive a document and a topic hierarchy. Assign the document to the most relevant topics the hierarchy. Then, output the topic labels, assignment reasoning and supporting quotes from the document. DO NOT make up new topics or quotes.

[Topic Hierarchy]
{tree}

[Examples]
Example 1: Assign "[1] Infectious Disease Research" to the document
Document:
A study investigating the transmission dynamics of SARS-CoV-2 in healthcare settings, examining viral load correlations with disease severity and outcomes in hospitalized COVID-19 patients.

Assignment:
[1] Infectious Disease Research: Examines pathogen transmission and clinical outcomes ("...examining viral load correlations with disease severity...")

Example 2: Assigned "[1] Public Health Interventions" to the document
Document:
Analysis of social distancing measures' effectiveness in reducing COVID-19 transmission rates across different population densities and demographic groups.

Assignment:
[1] Public Health Interventions: Evaluates disease control measures ("...social distancing measures' effectiveness in reducing COVID-19 transmission...")

[Instructions]
1. Topic labels must be present in the provided topic hierarchy. You MUST NOT make up new topics.
2. The quote must be taken from the document. You MUST NOT make up quotes.

[Document]
{Document}

Double check that your assignment exists in the hierarchy!
Your response should be in the following format:
[Topic Level] Topic Label: Assignment reasoning (Supporting quote)

Your response:"""

            gen_prompt_file = os.path.join(temp_dir, "generation_prompt.txt")
            with open(gen_prompt_file, "w") as f:
                f.write(generation_prompt)

            assign_prompt_file = os.path.join(temp_dir, "assignment_prompt.txt")
            with open(assign_prompt_file, "w") as f:
                f.write(assignment_prompt)

            # --- Create seed file (empty for fresh topic discovery) ---
            seed_file = os.path.join(temp_dir, "seed.md")
            with open(seed_file, "w") as f:
                f.write("")  # Empty seed for fresh discovery

            # --- Create output file paths ---
            gen_out_file = os.path.join(temp_dir, "generation_output.jsonl")
            topic_file = os.path.join(temp_dir, "topics.md")
            assign_out_file = os.path.join(temp_dir, "assignment_output.jsonl")

            # ===== STEP 3: GENERATE TOPICS (STAGE 1) =====

            logger.info("Stage 1: Generating topics...")

            # Call TopicGPT generation API
            # generate_topic_lvl1(api, model, data, prompt_file, seed_file, out_file, topic_file, verbose)
            topics_root = generate_topic_lvl1(
                api="openai",
                model=generation_model,
                data=gen_data_file,
                prompt_file=gen_prompt_file,
                seed_file=seed_file,
                out_file=gen_out_file,
                topic_file=topic_file,
                verbose=verbose
            )

            # Extract topics from TopicTree
            # topics_root.to_topic_list() returns list like "[1] Label (Count: N): Description"
            topic_list_with_counts = topics_root.to_topic_list(desc=True, count=True)
            topic_list_no_counts = topics_root.to_topic_list(desc=True, count=False)

            # Parse topic names and descriptions
            import re
            refined_topics = []  # Store as "Label: Description"
            topic_names = []  # Just the labels

            for topic_str in topic_list_no_counts:
                # Parse format: "[1] Label: Description"
                match = re.match(r"^\[(\d+)\]\s*(.+?):\s*(.+)$", topic_str.strip())
                if match:
                    level, label, desc = match.groups()
                    refined_topics.append(f"{label.strip()}: {desc.strip()}")
                    topic_names.append(label.strip())
                else:
                    # Fallback
                    refined_topics.append(topic_str)
                    topic_names.append(topic_str.split(":")[0].strip() if ":" in topic_str else topic_str)

            logger.info(f"  Generated {len(refined_topics)} topics")

            # Ensure we have at least one topic
            if len(refined_topics) == 0:
                refined_topics = ["General Topic: Documents that don't fit other categories"]
                topic_names = ["General Topic"]
                # Create a minimal topic file for assignment
                with open(topic_file, "w") as f:
                    f.write("[1] General Topic (Count: 1): Documents that don't fit other categories\n")
                logger.warning("No topics generated, using fallback 'General Topic'")

            # ===== STEP 4: ASSIGN TOPICS TO ALL DOCUMENTS (STAGE 2) =====

            logger.info(f"Stage 2: Assigning topics to {len(docs)} documents...")

            # Call TopicGPT assignment API
            # assign_topics(api, model, data, prompt_file, out_file, topic_file, verbose)
            assign_topics(
                api="openai",
                model=assignment_model,
                data=assign_data_file,
                prompt_file=assign_prompt_file,
                out_file=assign_out_file,
                topic_file=topic_file,
                verbose=verbose
            )

            # Parse assignment results
            topic_assignments = []
            assignment_quotes = []

            # Read assignment output
            assign_df = pd.read_json(assign_out_file, lines=True)

            # Log first few responses for debugging
            logger.info(f"  Sample assignment responses (first 3):")
            for idx in range(min(3, len(assign_df))):
                resp = assign_df.iloc[idx].get("responses", "")
                logger.info(f"    Doc {idx}: {resp[:150]}...")

            for i in range(len(docs)):
                if i < len(assign_df):
                    response = assign_df.iloc[i].get("responses", "")
                    response_str = str(response).strip()

                    # Parse multi-line response format:
                    # [1] Topic Label: <topic_name>
                    # Reasoning: <reasoning>
                    # Supporting quote from document: "<quote>"

                    # Extract topic label - try different patterns
                    label_match = re.search(r'^\[(\d+)\]\s*Topic Label:\s*(.+?)(?:\n|$)', response_str, re.MULTILINE | re.IGNORECASE)
                    if not label_match:
                        # Fallback: try simpler pattern [1] Label: ...
                        label_match = re.match(r'^\[(\d+)\]\s*(.+?):\s', response_str)

                    if label_match:
                        level = label_match.group(1)
                        assigned_label = label_match.group(2).strip()

                        # Extract quote - try multiple patterns
                        quote = ""
                        quote_patterns = [
                            r'Supporting quote from document:\s*["\']([^"\']+)["\']',  # Supporting quote from document: "..."
                            r'["\']([^"\']{20,})["\']',  # Any quoted text with 20+ chars
                            r'\(["\'"]([^"\']+)["\']\)',  # ("...")
                        ]
                        for pattern in quote_patterns:
                            quote_match = re.search(pattern, response_str, re.IGNORECASE)
                            if quote_match:
                                quote = quote_match.group(1).strip()
                                break

                        # Find matching topic index - exact match only, then first partial match
                        topic_idx = 0
                        found = False

                        # Try exact match first
                        for j, name in enumerate(topic_names):
                            if name.lower().strip() == assigned_label.lower().strip():
                                topic_idx = j
                                found = True
                                break

                        # If no exact match, try partial match (take first only)
                        if not found:
                            for j, name in enumerate(topic_names):
                                if assigned_label.lower() in name.lower() or name.lower() in assigned_label.lower():
                                    topic_idx = j
                                    break

                        topic_assignments.append(topic_idx)
                        assignment_quotes.append(quote)
                    else:
                        # Parsing failed - default to topic 0
                        logger.warning(f"  Failed to parse response for doc {i}: {response_str[:100]}...")
                        topic_assignments.append(0)
                        assignment_quotes.append("")
                else:
                    topic_assignments.append(0)
                    assignment_quotes.append("")

            logger.info(f"  Assigned {len(topic_assignments)} documents to topics")

        finally:
            # Clean up temp directory
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"  Cleaned up temp directory")
            except Exception as e:
                logger.warning(f"  Failed to clean up temp dir: {str(e)}")

        # ===== STEP 6: EXTRACT TOPIC WORDS =====

        # For TopicGPT, we extract keywords from topic descriptions
        # and from assigned documents

        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

        # Get vocabulary parameters (match BERTopic/LDA for fair comparison)
        min_df = self.params.get("min_df", 2)
        ngram_range = self.params.get("ngram_range", (1, 2))
        max_features = self.params.get("max_features", 10000)

        # Build TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=min_df,
            ngram_range=ngram_range,
            max_features=max_features,
            token_pattern=r'\b[a-zA-Z]{2,}\b'
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(docs)
            feature_names = vectorizer.get_feature_names_out()
        except Exception as e:
            logger.warning(f"TF-IDF vectorization failed: {str(e)}")
            # Fallback to simple word extraction from topic names
            feature_names = []
            tfidf_matrix = None

        topic_words = {}
        topic_words_with_scores = {}
        topic_labels = {}
        topic_names_dict = {}

        n_topics = len(refined_topics)

        for topic_id in range(n_topics):
            # Get documents assigned to this topic
            doc_indices = [i for i, t in enumerate(topic_assignments) if t == topic_id]

            if len(doc_indices) > 0 and tfidf_matrix is not None:
                # Get average TF-IDF scores for this topic's documents
                topic_tfidf = tfidf_matrix[doc_indices].mean(axis=0).A1

                # Get top words by TF-IDF score
                top_indices = topic_tfidf.argsort()[::-1][:20]
                words = [feature_names[i] for i in top_indices]
                scores = [float(topic_tfidf[i]) for i in top_indices]

                topic_words[topic_id] = words
                topic_words_with_scores[topic_id] = list(zip(words, scores))
            else:
                # Fallback: extract words from topic description
                topic_desc = refined_topics[topic_id]
                words = [w.lower() for w in topic_desc.split() if len(w) > 2][:20]
                if not words:
                    words = [f"topic_{topic_id}_word_{i}" for i in range(10)]

                topic_words[topic_id] = words
                topic_words_with_scores[topic_id] = [(w, 1.0) for w in words]

            # Generate labels from topic description or top words
            topic_desc = refined_topics[topic_id]
            if len(topic_desc) > 50:
                # Use top words for label
                label = "_".join(topic_words[topic_id][:3])
            else:
                # Use topic description
                label = topic_desc.replace(" ", "_")[:50]

            topic_labels[topic_id] = label
            topic_names_dict[topic_id] = f"Topic_{topic_id}_{label}"

        # ===== STEP 7: CREATE TOPIC_INFO (MATCH BERTOPIC FORMAT) =====

        # Count documents per topic
        topic_counts = Counter(topic_assignments)

        topic_info_data = {
            "Topic": [],
            "Count": [],
            "Name": [],
            "Representation": [],
            "Description": []  # Extra field for TopicGPT's natural language descriptions
        }

        for topic_id in range(n_topics):
            topic_info_data["Topic"].append(topic_id)
            topic_info_data["Count"].append(topic_counts.get(topic_id, 0))
            topic_info_data["Name"].append(topic_names_dict[topic_id])
            topic_info_data["Representation"].append(topic_words[topic_id][:10])
            topic_info_data["Description"].append(refined_topics[topic_id])

        # ===== STEP 8: COMPUTE PROBABILITIES (OPTIONAL) =====

        # TopicGPT doesn't provide probabilities by default
        # We can estimate them based on document-topic similarity
        # For now, use one-hot encoding
        probabilities = []
        for topic_idx in topic_assignments:
            prob_dist = [0.0] * n_topics
            prob_dist[topic_idx] = 1.0
            probabilities.append(prob_dist)

        # ===== STEP 9: SAVE MODEL ARTIFACTS (OPTIONAL) =====

        if save_model and model_path:
            import pickle

            model_artifacts = {
                "topics": refined_topics,
                "topic_assignments": topic_assignments,
                "assignment_quotes": assignment_quotes,
                "params": self.params,
                "generation_model": generation_model,
                "assignment_model": assignment_model
            }

            with open(model_path, "wb") as f:
                pickle.dump(model_artifacts, f)

            logger.info(f"Saved TopicGPT model artifacts to {model_path}")
        else:
            logger.info(f"Skipping model save (save_topic_models=False)")

        # ===== STEP 10: RETURN STANDARDIZED STRUCTURE =====

        return {
            "method": method_name,
            "doc_ids": doc_ids,
            "topics": topic_assignments,  # List[int], length = n_docs
            "probabilities": probabilities,  # List[List[float]], shape = (n_docs, n_topics)
            "topic_info": topic_info_data,  # Dict (convertible to DataFrame)
            "topic_words": topic_words,  # Dict[int, List[str]]
            "topic_words_with_scores": topic_words_with_scores,  # Dict[int, List[Tuple[str, float]]]
            "topic_labels": topic_labels,  # Dict[int, str]
            "topic_names": topic_names_dict,  # Dict[int, str]
            "n_topics": n_topics,
            "n_docs": len(docs),
            # TopicGPT-specific extras
            "topic_descriptions": refined_topics,  # Original LLM-generated descriptions
            "assignment_quotes": assignment_quotes  # Supporting quotes from documents
        }
