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

# HiCode lives in a git submodule (external/hicode) as flat modules, not a package
HICODE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "hicode")

# Dataset context prepended to HiCode's label-generation prompt.
# Override per run with the "background" topic model param.
HICODE_BACKGROUNDS = {
    "trec-covid": """
      """, #TODO
    "doctor-reviews": """
    The online physician review dataset contains millions of public patient reviews about physicians from different healthcare providers.
    We focus on reviews related to Family Medicine physicians containing 171,216 reviews about 5,986 physicians.
    Public health researchers are interested in understanding exploratory research questions about patient experiences, pathways to providers, relationship with physicians, and healthcare quality from these reviews.
    """,

    "salescontest": """
    The sales contest dataset contains a collection of internal corporation email communications related to potentially different sales contests in a US pharmaceutical company. 
    Public health researchers are interested in inductively understanding the key themes and topics for their research questions.
    """,
}


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
            model_type: Type of model ("bertopic", "lda", "topicgpt", "hicode")
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

        elif self.model_type == "hicode":
            return self._fit_hicode(
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
        except ImportError as e:
            raise ImportError(
                f"topicgpt_python not installed. Install with: pip install topicgpt_python. Original error: {e}"
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
        import shutil
        import pandas as pd

        # Use project-local temp directory instead of system /tmp
        # (/tmp on compute nodes may have limited space or be cleaned mid-job)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_tmp_base = os.path.join(project_root, ".tmp", "topicgpt")
        os.makedirs(local_tmp_base, exist_ok=True)

        # Pre-flight check: verify temp directory is writable and log disk space
        disk_usage = shutil.disk_usage(local_tmp_base)
        logger.info(f"  Temp base: {local_tmp_base} "
                     f"(free: {disk_usage.free / (1024**3):.1f} GB)")
        if disk_usage.free < 1 * 1024**3:  # Warn if less than 1 GB free
            logger.warning(f"  Low disk space on temp directory: "
                           f"{disk_usage.free / (1024**3):.1f} GB free")

        # Ensure data/output/ exists (topicgpt_python library writes backup
        # files to hardcoded "data/output/" relative to the working directory)
        os.makedirs("data/output", exist_ok=True)

        # Create temp directory for TopicGPT files
        temp_dir = tempfile.mkdtemp(prefix="topicgpt_", dir=local_tmp_base)
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

    def _fit_hicode(
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
        HiCode implementation (hierarchical inductive coding with LLMs).
        See https://github.com/mianzg/HICode, vendored as the external/hicode submodule.

        Pipeline: split each document into segments -> LLM labels each segment
        relevant to the query -> LLM iteratively clusters labels into themes ->
        each document is assigned the themes of its labeled segments.

        Returns the same format evaluate_hicode.py produces from offline HiCode
        pickles (topic_model_type="hicode"), which the evaluator special-cases:
        - topics: List[int], each doc's most frequent theme (-1 = no relevant segment)
        - topic_words: Dict[int, List[str]] of generation labels in each theme
        - _hicode_original_multi_label_topics: List[List[int]], all themes per doc

        Required params (injected by EndToEndEvaluator.run_topic_modeling):
            query_text: Query used as the coding goal
            work_dir: Directory for HiCode's intermediate generation/clustering files
            dataset_name: Selects the prompt background from HICODE_BACKGROUNDS
                (or pass "background" directly)

        Optional params:
            model_name: Label generation model (default "gpt-4o-mini")
            cluster_model_name: Label clustering model (default "gpt-4o-mini")
            max_n_iter: Max clustering iterations (default 3)
            segment_delimiter: Splits documents into segments (default "\\n\\n")

        Requires OPENAI_API_KEY environment variable to be set.
        """
        import sys
        import json
        import shutil
        import pickle
        from sklearn.feature_extraction.text import TfidfVectorizer

        if HICODE_DIR not in sys.path:
            sys.path.insert(0, HICODE_DIR)
        try:
            from label_generation import generate_labels, save_generation_output
            from label_clustering import cluster_labels_gpt, make_clustering_prompt
            from metrics import create_mapping
        except ImportError as e:
            raise ImportError(
                f"HiCode not found at {HICODE_DIR}. Run: git submodule update --init external/hicode. "
                f"Original error: {e}"
            )
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "HiCode requires an OpenAI API key."
            )

        # ===== STEP 1: EXTRACT PARAMETERS =====

        query_text = self.params.get("query_text")
        work_dir = self.params.get("work_dir")
        if not query_text or not work_dir:
            raise ValueError("HiCode requires 'query_text' and 'work_dir' params")

        background = self.params.get("background")
        if background is None:
            dataset_name = self.params.get("dataset_name")
            if dataset_name not in HICODE_BACKGROUNDS:
                raise ValueError(
                    f"No HiCode background for dataset '{dataset_name}'. "
                    f"Add one to HICODE_BACKGROUNDS or pass a 'background' param."
                )
            background = HICODE_BACKGROUNDS[dataset_name]

        segment_delimiter = self.params.get("segment_delimiter", "\n\n")
        config = {
            "model_name": self.params.get("model_name", "gpt-4o-mini"),
            "cluster_model_name": self.params.get("cluster_model_name", "gpt-4o-mini"),
            "max_n_iter": self.params.get("max_n_iter", 3),
            "cluster_output_dir": os.path.join(work_dir, "clustering"),
        }
        generation_dir = os.path.join(work_dir, "generation")

        logger.info(f"Fitting HiCode on {len(docs)} documents...")
        logger.info(f"  Generation model: {config['model_name']}")
        logger.info(f"  Clustering model: {config['cluster_model_name']}")
        logger.info(f"  Work directory: {work_dir}")

        # ===== STEP 2: SPLIT DOCUMENTS INTO SEGMENTS =====

        # HiCode keys segments as "<doc_id>_<segment_idx>" and recovers the doc id
        # by stripping the last "_" part, so doc ids may themselves contain "_"
        segments = {}
        for doc_id, text in zip(doc_ids, docs):
            for j, seg in enumerate(text.split(segment_delimiter)):
                if len(seg.strip()) > 0:
                    segments[f"{doc_id}_{j}"] = seg
        logger.info(f"  Split into {len(segments)} segments")

        # ===== STEP 3: LABEL GENERATION =====

        coding_goal = "understand " + query_text
        system_prompt = f"""
    {background}

    We are using the queries to this bot to conduct INDUCTIVE Coding. The labeling aims to {coding_goal}

    Instruction:
    - Label the input only when it is HIGHLY RELEVANT and USEFUL for {coding_goal}.
    - Then, define the phrase of the label. The label description should be observational, concise and clear.
    - ONLY output the label and DO NOT output any explanation.

    Format:
    - Define the label using the format \"LABEL: [The phrase of the label]\".
    - If there are multiple labels, each label is a new line.
    - If the input is irrelevant, use \"LABEL: [Irrelevant]\".
    """

        # Reuse saved generation output so a crashed run doesn't repeat the LLM calls
        generation_file = os.path.join(generation_dir, f"generation_{method_name}.json")
        if os.path.exists(generation_file):
            with open(generation_file, "r") as f:
                gen_result = json.load(f)
            logger.info(f"  Loaded existing generation output from {generation_file}")
        else:
            logger.info("Stage 1: Generating labels...")
            gen_result = generate_labels(segments, system_prompt, config)
            save_generation_output(gen_result, generation_dir, output_id=method_name)
        logger.info(f"  {len(gen_result)} documents have relevant segments")

        # ===== STEP 4: HIERARCHICAL LABEL CLUSTERING =====

        clustering_dir = os.path.join(config["cluster_output_dir"], f"clustering_{method_name}")
        if not os.path.exists(clustering_dir):
            logger.info("Stage 2: Clustering labels...")
            try:
                cluster_labels_gpt(
                    gen_result, make_clustering_prompt(goal=coding_goal), config,
                    gen_result_id=method_name
                )
            except BaseException:
                # create_mapping reads every cluster_iter_*.json, so never leave a partial run behind
                shutil.rmtree(clustering_dir, ignore_errors=True)
                raise
        else:
            logger.info(f"  Loaded existing clustering output from {clustering_dir}")

        # Maps lowercased generation label -> final theme
        label_to_theme = create_mapping(clustering_dir)

        # ===== STEP 5: ASSIGN THEMES TO DOCUMENTS =====

        themes = sorted(set(label_to_theme.values()))
        theme_to_id = {theme: i for i, theme in enumerate(themes)}
        n_topics = len(themes)
        logger.info(f"  Discovered {n_topics} themes")
        if n_topics == 0:
            logger.warning("HiCode produced no themes - all documents are outliers")

        multi_label_topics = []
        topic_assignments = []
        for doc_id in doc_ids:
            theme_counts = Counter()
            for annotation in gen_result.get(str(doc_id), {}).get("LLM_Annotation", []):
                seg_themes = {label_to_theme.get(label.lower()) for label in annotation["label"]}
                theme_counts.update(theme_to_id[t] for t in seg_themes if t is not None)

            if theme_counts:
                # Primary topic = theme on the most segments, ties broken by lowest id
                ranked = sorted(theme_counts, key=lambda t: (-theme_counts[t], t))
                multi_label_topics.append(ranked)
                topic_assignments.append(ranked[0])
            else:
                multi_label_topics.append([-1])
                topic_assignments.append(-1)

        n_outliers = topic_assignments.count(-1)
        n_multi = sum(1 for t in multi_label_topics if len(t) > 1)
        logger.info(f"  Outlier documents: {n_outliers}/{len(docs)}")
        logger.info(f"  Multi-label documents: {n_multi}/{len(docs)} - using most frequent theme")

        # ===== STEP 6: TOPIC WORDS =====

        # Generation labels per theme are HiCode's topic "words"; the evaluator
        # embeds the joined labels for semantic matching
        topic_words = {i: [] for i in range(n_topics)}
        for label, theme in label_to_theme.items():
            topic_words[theme_to_id[theme]].append(label)

        # Individual high TF-IDF words from each theme's labels (stored as fallback)
        topic_words_processed = {}
        for topic_id, labels in topic_words.items():
            try:
                vectorizer = TfidfVectorizer(stop_words="english")
                tfidf = vectorizer.fit_transform(labels)
                mean_tfidf = tfidf.mean(axis=0).A1
                feature_names = vectorizer.get_feature_names_out()
                top_indices = mean_tfidf.argsort()[-20:][::-1]
                topic_words_processed[topic_id] = [feature_names[i] for i in top_indices]
            except ValueError:
                # Labels made only of stop words
                topic_words_processed[topic_id] = []

        topic_names = {i: theme for i, theme in enumerate(themes)}
        topic_names[-1] = "irrelevant"

        # ===== STEP 7: SAVE MODEL ARTIFACTS (OPTIONAL) =====

        if save_model and model_path:
            model_artifacts = {
                "generation_result": gen_result,
                "label_to_theme": label_to_theme,
                "params": self.params,
            }
            with open(model_path, "wb") as f:
                pickle.dump(model_artifacts, f)
            logger.info(f"Saved HiCode model artifacts to {model_path}")
        else:
            logger.info(f"Skipping model save (save_topic_models=False)")

        # ===== STEP 8: RETURN STANDARDIZED STRUCTURE =====

        return {
            "method": method_name,
            "doc_ids": doc_ids,
            "topics": topic_assignments,  # List[int], length = n_docs
            "probabilities": None,  # HiCode doesn't provide probabilities
            "topic_words": topic_words,  # Dict[int, List[str]] of generation labels
            "topic_names": topic_names,  # Dict[int, str], natural language themes
            "n_topics": n_topics,  # Excludes -1, matching BERTopic
            "n_docs": len(docs),
            "topic_model_type": "hicode",
            # HiCode-specific extras
            "_hicode_original_multi_label_topics": multi_label_topics,
            "_topic_words_processed": topic_words_processed,
        }
