# end_to_end_evaluation.py
"""
End-to-end evaluation of topic modeling (BERTopic or LDA) using different sampling strategies.

Supports multiple topic modeling methods:
- BERTopic: Embedding-based with HDBSCAN clustering (auto-determines topic count)
- LDA: Bag-of-words Latent Dirichlet Allocation (requires fixed topic count)

Compares 6 sampling methods:
1. Random Uniform Sampling
2. Direct Retrieval (Hybrid BM25+SBERT)
3. Direct Retrieval + MMR (High Diversity)
4. Query Expansion + Retrieval
5. Simple Keyword Search (BM25 only)
6. Retrieval Pool + Random Sampling (Hybrid retrieval of 5000 docs, then random sample 1000)

METRICS IMPLEMENTED:

Intrinsic Quality Metrics (Per-Method):
- NPMI Coherence: Semantic coherence of topic words based on co-occurrence
- Embedding Coherence: Average pairwise cosine similarity of word embeddings within topics
- Semantic Diversity: Average pairwise distance between topics (embedding-based)
- Lexical Diversity: Unique word fraction across all topics (vocabulary redundancy)
- Document Coverage: Fraction of documents assigned to topics (1 - outlier_ratio)
- Number of Topics: Total topics discovered
- Average Topic Size: Mean documents per topic

Query Alignment Metrics (Per-Method):
- Topic-Query Similarity (Avg): Average similarity between all topics and the query
- Max Query Similarity: Highest topic-query similarity score
- Query-Relevant Ratio: Fraction of topics above similarity threshold (0.5)
- Top-3 Avg Similarity: Average similarity of top-3 most query-relevant topics

Pairwise Comparison Metrics:
- Topic Word Overlap (Jaccard): Lexical overlap between matched topics
- Topic Semantic Similarity: Embedding-based similarity between matched topics
- Precision @ thresholds (0.5, 0.6, 0.7): What fraction of B's topics match A's topics
- Recall @ thresholds (0.5, 0.6, 0.7): What fraction of A's topics match B's topics
- F1 @ thresholds (0.5, 0.6, 0.7): Topic matching quality at different thresholds
- NPMI Coherence Difference: Difference in coherence between methods
- Embedding Coherence Difference: Difference in embedding coherence
- Semantic Diversity Difference: Difference in semantic diversity
- Lexical Diversity Difference: Difference in lexical diversity
- ARI (Adjusted Rand Index): Agreement on document clustering (overlap only)
- NMI (Normalized Mutual Information): Shared clustering information (overlap only)

Visualizations Generated:
1. intrinsic_quality_metrics.png: 4x2 grid of NPMI, Embedding Coherence, Semantic/Lexical Diversity, Coverage, Topic Specificity, Topic Count, Relevant Concentration
2. query_alignment_metrics.png: 1x3 grid of Avg/Max/Ratio query similarity metrics
3. diversity_scatter.png: Semantic vs Lexical diversity scatter plot
4. Pairwise heatmaps (14 plots): Similarity, Overlap, F1 @0.5/0.6/0.7, Precision @0.5/0.6/0.7, Recall @0.5/0.6/0.7, NPMI diff, Embedding diff, ARI
"""

import os
import sys
import json
import pickle
import logging
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# BERTopic imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Import from existing modules
from search import (
    TextPreprocessor,
    IndexManager,
    SearchEngine,
    RetrievalMethod,
    HybridStrategy
)
from evaluation import SearchEvaluationUtils
from topic_models import TopicModelWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class TimingTracker:
    """
    Track timing for pipeline steps and write to JSONL file.

    Each timing event is written as a JSON object with:
    - event: Name of the event (e.g., "sample_generation", "topic_modeling")
    - timestamp: ISO format timestamp when the event completed
    - duration_seconds: Time taken in seconds
    - Additional fields depending on event type (e.g., method, query_id)
    """

    def __init__(self, output_path: str):
        """
        Initialize timing tracker.

        Args:
            output_path: Path to write timing.jsonl file
        """
        self.output_path = output_path
        self.events = []
        self._start_times = {}

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def start(self, event_name: str, **metadata):
        """Start timing an event."""
        import time
        self._start_times[event_name] = {
            'start_time': time.time(),
            'metadata': metadata
        }

    def stop(self, event_name: str, **extra_metadata):
        """Stop timing an event and record it."""
        import time
        from datetime import datetime

        if event_name not in self._start_times:
            logger.warning(f"Timing event '{event_name}' was never started")
            return

        start_info = self._start_times.pop(event_name)
        duration = time.time() - start_info['start_time']

        # Merge metadata
        metadata = {**start_info['metadata'], **extra_metadata}

        event = {
            'event': event_name,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': round(duration, 2),
            **metadata
        }

        self.events.append(event)

        # Append to file immediately (so we don't lose data on crash)
        with open(self.output_path, 'a') as f:
            f.write(json.dumps(event) + '\n')

    def record(self, event_name: str, duration_seconds: float, **metadata):
        """Record a timing event directly (without start/stop)."""
        from datetime import datetime

        event = {
            'event': event_name,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': round(duration_seconds, 2),
            **metadata
        }

        self.events.append(event)

        with open(self.output_path, 'a') as f:
            f.write(json.dumps(event) + '\n')

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all recorded events."""
        return {
            'total_events': len(self.events),
            'events': self.events
        }


# ============================================================================
# Statistical Testing Functions
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for two groups.

    Args:
        group1: Array of values for group 1
        group2: Array of values for group 2

    Returns:
        Cohen's d effect size (positive if group1 > group2)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        String interpretation of effect size
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def paired_ttest(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """
    Perform paired t-test between two groups.

    Args:
        group1: Array of values for group 1
        group2: Array of values for group 2

    Returns:
        Tuple of (t-statistic, p-value)
    """
    from scipy import stats

    # Handle edge cases
    if len(group1) != len(group2):
        raise ValueError("Groups must have same length for paired t-test")

    if len(group1) < 2:
        return (np.nan, np.nan)

    # Check if all differences are zero (no variance)
    differences = group1 - group2
    if np.all(differences == 0):
        return (0.0, 1.0)

    t_stat, p_value = stats.ttest_rel(group1, group2)
    return (t_stat, p_value)


def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Apply Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
        p_values: List of raw p-values
        alpha: Significance threshold (default 0.05)

    Returns:
        Tuple of (adjusted p-values, list of booleans indicating significance)
    """
    n = len(p_values)
    if n == 0:
        return [], []

    # Handle NaN values
    p_array = np.array(p_values)
    valid_mask = ~np.isnan(p_array)

    if not np.any(valid_mask):
        return list(p_array), [False] * n

    # Sort p-values and get ranks
    sorted_indices = np.argsort(p_array[valid_mask])
    sorted_p = p_array[valid_mask][sorted_indices]

    # Calculate adjusted p-values
    m = len(sorted_p)
    ranks = np.arange(1, m + 1)
    adjusted_p_sorted = sorted_p * m / ranks

    # Ensure monotonicity (each adjusted p-value >= previous)
    adjusted_p_sorted = np.minimum.accumulate(adjusted_p_sorted[::-1])[::-1]
    adjusted_p_sorted = np.clip(adjusted_p_sorted, 0, 1)

    # Map back to original order
    adjusted_p = np.full(n, np.nan)
    valid_indices = np.where(valid_mask)[0]
    adjusted_p[valid_indices[sorted_indices]] = adjusted_p_sorted

    # Determine significance
    significant = adjusted_p < alpha

    return list(adjusted_p), list(significant)


def run_pairwise_statistical_tests(
    data: pd.DataFrame,
    metric: str,
    method_col: str = 'method',
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run pairwise statistical tests (paired t-test + Cohen's d) for a metric across methods.

    Args:
        data: DataFrame with columns [method_col, metric, 'query_id']
        metric: Name of the metric column to test
        method_col: Name of the method column
        alpha: Significance threshold

    Returns:
        DataFrame with columns:
        - method_a, method_b: Method pair
        - mean_a, mean_b: Mean values
        - diff: Difference in means (a - b)
        - t_stat: t-statistic
        - p_value: Raw p-value
        - p_adjusted: BH-adjusted p-value
        - cohens_d: Effect size
        - effect_size: Interpretation of Cohen's d
        - significant: Whether difference is significant after correction
    """
    methods = sorted(data[method_col].unique())
    results = []

    for i, method_a in enumerate(methods):
        for method_b in methods[i+1:]:
            # Get paired data (same queries for both methods)
            data_a = data[data[method_col] == method_a].set_index('query_id')[metric]
            data_b = data[data[method_col] == method_b].set_index('query_id')[metric]

            # Align by query_id
            common_queries = data_a.index.intersection(data_b.index)
            values_a = data_a.loc[common_queries].values
            values_b = data_b.loc[common_queries].values

            if len(common_queries) < 2:
                continue

            # Compute statistics
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            diff = mean_a - mean_b

            t_stat, p_value = paired_ttest(values_a, values_b)
            d = compute_cohens_d(values_a, values_b)

            results.append({
                'method_a': method_a,
                'method_b': method_b,
                'n_queries': len(common_queries),
                'mean_a': round(mean_a, 4),
                'mean_b': round(mean_b, 4),
                'diff': round(diff, 4),
                't_stat': round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
                'p_value': round(p_value, 6) if not np.isnan(p_value) else np.nan,
                'cohens_d': round(d, 4),
                'effect_size': interpret_cohens_d(d)
            })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Apply BH correction
    p_values = results_df['p_value'].tolist()
    adjusted_p, significant = benjamini_hochberg_correction(p_values, alpha)

    results_df['p_adjusted'] = [round(p, 6) if not np.isnan(p) else np.nan for p in adjusted_p]
    results_df['significant'] = significant

    return results_df


def plot_statistical_heatmaps(
    stats_df: pd.DataFrame,
    metric_name: str,
    output_dir: str,
    filename_prefix: str
) -> None:
    """
    Create heatmap visualizations for statistical test results.

    Generates two heatmaps:
    1. P-value heatmap with significance markers
    2. Cohen's d heatmap with effect size coloring

    Args:
        stats_df: DataFrame from run_pairwise_statistical_tests()
        metric_name: Human-readable name of the metric
        output_dir: Directory to save plots
        filename_prefix: Prefix for output filenames
    """
    if stats_df.empty:
        logger.warning(f"No statistical results to plot for {metric_name}")
        return

    methods = sorted(set(stats_df['method_a'].tolist() + stats_df['method_b'].tolist()))
    n_methods = len(methods)
    method_to_idx = {m: i for i, m in enumerate(methods)}

    # Create matrices
    p_matrix = np.ones((n_methods, n_methods))  # Default p=1 (no difference)
    d_matrix = np.zeros((n_methods, n_methods))  # Default d=0 (no effect)
    sig_matrix = np.zeros((n_methods, n_methods), dtype=bool)

    for _, row in stats_df.iterrows():
        i = method_to_idx[row['method_a']]
        j = method_to_idx[row['method_b']]

        p_val = row['p_adjusted'] if not np.isnan(row['p_adjusted']) else 1.0
        d_val = row['cohens_d']
        sig = row['significant']

        # Fill both triangles (symmetric)
        p_matrix[i, j] = p_val
        p_matrix[j, i] = p_val
        d_matrix[i, j] = d_val
        d_matrix[j, i] = -d_val  # Reverse sign for other direction
        sig_matrix[i, j] = sig
        sig_matrix[j, i] = sig

    # Set diagonal to NaN for display
    np.fill_diagonal(p_matrix, np.nan)
    np.fill_diagonal(d_matrix, np.nan)

    # ===== Plot 1: P-value Heatmap =====
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use log scale for p-values for better visualization
    p_display = p_matrix.copy()

    # Create annotation matrix with significance markers
    annot_matrix = np.empty((n_methods, n_methods), dtype=object)
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                annot_matrix[i, j] = ""
            elif np.isnan(p_matrix[i, j]):
                annot_matrix[i, j] = ""
            else:
                p_val = p_matrix[i, j]
                if p_val < 0.001:
                    annot_matrix[i, j] = f"{p_val:.0e}***"
                elif p_val < 0.01:
                    annot_matrix[i, j] = f"{p_val:.3f}**"
                elif p_val < 0.05:
                    annot_matrix[i, j] = f"{p_val:.3f}*"
                else:
                    annot_matrix[i, j] = f"{p_val:.3f}"

    # Mask diagonal
    mask = np.eye(n_methods, dtype=bool)

    sns.heatmap(
        p_display,
        annot=annot_matrix,
        fmt="",
        cmap="RdYlGn_r",  # Red = low p (significant), Green = high p
        vmin=0,
        vmax=0.1,
        mask=mask,
        xticklabels=methods,
        yticklabels=methods,
        ax=ax,
        cbar_kws={'label': 'Adjusted p-value'}
    )

    ax.set_title(f'{metric_name}\nPairwise Significance (Paired t-test, BH-corrected)\n* p<0.05, ** p<0.01, *** p<0.001',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Method', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_pvalues.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved {filename_prefix}_pvalues.png")

    # ===== Plot 2: Cohen's d Heatmap =====
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create annotation with effect size labels
    annot_d = np.empty((n_methods, n_methods), dtype=object)
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                annot_d[i, j] = ""
            elif np.isnan(d_matrix[i, j]):
                annot_d[i, j] = ""
            else:
                d_val = d_matrix[i, j]
                effect = interpret_cohens_d(d_val)
                annot_d[i, j] = f"{d_val:.2f}\n({effect[0].upper()})"  # First letter of effect size

    # Diverging colormap centered at 0
    vmax = max(abs(np.nanmin(d_matrix)), abs(np.nanmax(d_matrix)), 0.8)

    sns.heatmap(
        d_matrix,
        annot=annot_d,
        fmt="",
        cmap="RdBu_r",  # Red = positive (row > col), Blue = negative
        vmin=-vmax,
        vmax=vmax,
        center=0,
        mask=mask,
        xticklabels=methods,
        yticklabels=methods,
        ax=ax,
        cbar_kws={'label': "Cohen's d (row - column)"}
    )

    ax.set_title(f"{metric_name}\nEffect Size (Cohen's d)\nN=negligible, S=small, M=medium, L=large",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Method', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_cohens_d.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved {filename_prefix}_cohens_d.png")


def run_aggregate_statistical_analysis(
    all_per_method: pd.DataFrame,
    output_dir: str,
    metrics_to_test: List[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run statistical analysis for multiple metrics and save results.

    Args:
        all_per_method: DataFrame with per-method data across all queries
        output_dir: Directory to save results
        metrics_to_test: List of dicts with 'column' and 'name' keys
                        If None, uses default metrics

    Returns:
        Dictionary mapping metric names to results DataFrames
    """
    if metrics_to_test is None:
        metrics_to_test = [
            {'column': 'topic_query_similarity', 'name': 'Avg Topic-Query Similarity'},
            {'column': 'diversity_semantic', 'name': 'Semantic Diversity'},
            {'column': 'topic_specificity', 'name': 'Topic Specificity'}
        ]

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    logger.info("Running aggregate statistical analysis...")

    for metric_info in metrics_to_test:
        column = metric_info['column']
        name = metric_info['name']

        if column not in all_per_method.columns:
            logger.warning(f"Column '{column}' not found, skipping statistical tests")
            continue

        logger.info(f"  Testing: {name}")

        # Run pairwise tests
        stats_df = run_pairwise_statistical_tests(
            data=all_per_method,
            metric=column,
            method_col='method',
            alpha=0.05
        )

        if stats_df.empty:
            logger.warning(f"  No results for {name}")
            continue

        # Save CSV
        csv_path = os.path.join(output_dir, f"statistical_tests_{column}.csv")
        stats_df.to_csv(csv_path, index=False)
        logger.info(f"  ✓ Saved {csv_path}")

        # Create heatmaps
        plot_statistical_heatmaps(
            stats_df=stats_df,
            metric_name=name,
            output_dir=output_dir,
            filename_prefix=f"statistical_{column}"
        )

        all_results[column] = stats_df

        # Log summary
        n_significant = stats_df['significant'].sum()
        n_total = len(stats_df)
        logger.info(f"  Summary: {n_significant}/{n_total} pairs significantly different")

    logger.info("✓ Aggregate statistical analysis complete")

    return all_results


class EndToEndEvaluator:
    """End-to-end evaluator for topic modeling with pairwise method comparisons"""

    def __init__(
        self,
        corpus_dataset,
        queries_dataset,
        qrels_dataset,
        query_id: str,
        sample_size: Optional[int] = None,
        embedding_model_name: str = "all-mpnet-base-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        dataset_name: str = "trec-covid",
        topic_model_type: str = "bertopic",
        topic_model_params: Optional[Dict] = None,
        output_dir: str = "results",
        cache_dir: str = "cache",
        random_seed: int = 42,
        device: str = "cpu",
        save_topic_models: bool = False,
        force_regenerate_samples: bool = False,
        force_regenerate_topics: bool = False,
        force_regenerate_evaluation: bool = False,
        keyword_cache_path: Optional[str] = None
    ):
        """
        Initialize the end-to-end evaluator

        Args:
            corpus_dataset: Full corpus dataset
            queries_dataset: Queries dataset
            qrels_dataset: Relevance judgments dataset
            query_id: Query ID to evaluate
            sample_size: Sample size (if None, determined by qrels count)
            embedding_model_name: Name of embedding model
            cross_encoder_model_name: Name of cross-encoder model
            dataset_name: Dataset name for caching
            topic_model_type: Type of topic model ('bertopic', 'lda', 'topicgpt')
            topic_model_params: Model-specific parameters dict
            output_dir: Output directory for results
            cache_dir: Cache directory
            random_seed: Random seed for reproducibility
            device: Device to use ('cpu', 'cuda', 'mps')
            save_topic_models: Whether to save full BERTopic models (420 MB each, only needed for interactive exploration)
            force_regenerate_samples: Force regenerate samples
            force_regenerate_topics: Force regenerate topic models
            force_regenerate_evaluation: Force regenerate evaluation
            keyword_cache_path: Path to keyword cache for query expansion (dataset-specific)
        """
        self.corpus_dataset = corpus_dataset
        self.queries_dataset = queries_dataset
        self.qrels_dataset = qrels_dataset
        self.query_id = str(query_id)
        self.embedding_model_name = embedding_model_name
        self.cross_encoder_model_name = cross_encoder_model_name
        self.dataset_name = dataset_name
        self.topic_model_type = topic_model_type
        self.topic_model_params = topic_model_params or {}
        self.cache_dir = cache_dir
        self.random_seed = random_seed
        self.device = device
        self.save_topic_models = save_topic_models
        self.force_regenerate_samples = force_regenerate_samples
        self.force_regenerate_topics = force_regenerate_topics
        self.force_regenerate_evaluation = force_regenerate_evaluation
        self.keyword_cache_path = keyword_cache_path

        # Set random seeds
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Get query text
        self.query_text = self._get_query_text()
        logger.info(f"Query {query_id}: {self.query_text}")

        # Set sample size
        self.sample_size = sample_size
        logger.info(f"Sample size: {self.sample_size}")

        # Create output directories (new unified structure)
        self.output_dir = os.path.join(
            output_dir,
            self.dataset_name,
            self.topic_model_type,
            f"query_{query_id}"
        )
        self.samples_dir = os.path.join(self.output_dir, "samples")
        self.topic_models_dir = os.path.join(self.output_dir, "topic_models")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.topics_summary_dir = os.path.join(self.results_dir, "topics_summary")

        for directory in [self.samples_dir, self.topic_models_dir, self.results_dir,
                          self.plots_dir, self.topics_summary_dir]:
            os.makedirs(directory, exist_ok=True)

        # Save configuration
        self._save_config()

        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.index_manager = IndexManager(self.preprocessor)

        # Build or load indices
        self._initialize_indices()

        # Initialize IDF scores cache (corpus-level, shared across all queries)
        self.idf_cache_dir = os.path.join(self.cache_dir, f"corpus_{len(self.corpus_dataset)}")
        os.makedirs(self.idf_cache_dir, exist_ok=True)
        self.idf_scores = self._load_or_compute_idf()

        # Build qrels dictionary for fast relevance lookup
        self.qrels_dict = self._build_qrels_dict()
        logger.info(f"QRELs loaded for {len(self.qrels_dict)} queries")

        # Initialize shared embedding model for metric computation (reused across all metrics)
        # This prevents loading the model multiple times and saves GPU memory
        logger.info(f"Loading shared embedding model for metrics: {self.embedding_model_name}")
        self.metrics_embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
        logger.info("Shared embedding model loaded")

    def _get_query_text(self) -> str:
        """Get query text for given query ID"""
        for query in self.queries_dataset:
            if str(query["_id"]) == self.query_id:
                return query["text"]
        raise ValueError(f"Query ID {self.query_id} not found in queries dataset")


    def _save_config(self):
        """Save configuration to JSON"""
        config = {
            "query_id": self.query_id,
            "query_text": self.query_text if hasattr(self, 'query_text') else None,
            "sample_size": self.sample_size,
            "embedding_model": self.embedding_model_name,
            "cross_encoder_model": self.cross_encoder_model_name,
            "dataset_name": self.dataset_name,
            "topic_model_type": self.topic_model_type,
            "topic_model_params": self.topic_model_params,
            "random_seed": self.random_seed,
            "force_flags": {
                "samples": self.force_regenerate_samples,
                "topics": self.force_regenerate_topics,
                "evaluation": self.force_regenerate_evaluation
            }
        }

        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")

    def _initialize_indices(self):
        """Initialize or load search indices"""
        logger.info(f"Initializing search indices on device: {self.device}...")

        # Build BM25 index
        self.bm25, self.corpus_texts, self.corpus_ids = self.index_manager.build_bm25_index(
            self.corpus_dataset,
            dataset_name=self.dataset_name,
            force_reindex=False
        )

        # Build SBERT index
        self.sbert_model, self.doc_embeddings = self.index_manager.build_sbert_index(
            self.corpus_texts,
            model_name=self.embedding_model_name,
            dataset_name=self.dataset_name,
            batch_size=64,
            force_reindex=False,
            device=self.device
        )

        # Initialize search engine
        self.search_engine = SearchEngine(
            preprocessor=self.preprocessor,
            bm25=self.bm25,
            corpus_texts=self.corpus_texts,
            corpus_ids=self.corpus_ids,
            sbert_model=self.sbert_model,
            doc_embeddings=self.doc_embeddings,
            cross_encoder_model_name=self.cross_encoder_model_name,
            device=self.device
        )

        logger.info("Search indices initialized")

    def _load_or_compute(self, cache_path: str, compute_fn, force: bool = False):
        """Universal caching pattern"""
        if not force and os.path.exists(cache_path):
            logger.info(f"Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        logger.info(f"Computing (cache miss or force=True)")
        result = compute_fn()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Saved to cache: {cache_path}")

        return result

    def _load_or_compute_sample(self, sample_name: str, compute_fn, force: bool = False):
        """
        Load sample with cross-model sharing support.

        Samples are identical across topic model types (BERTopic, LDA, TopicGPT),
        so we check other model directories if the sample doesn't exist locally.

        Priority order: current model dir -> bertopic -> lda -> topicgpt -> compute
        """
        cache_path = os.path.join(self.samples_dir, f"{sample_name}.pkl")

        # If not forcing and file exists locally, use it
        if not force and os.path.exists(cache_path):
            logger.info(f"Loading sample from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Check other model type directories for shared samples
        if not force:
            other_model_types = ["bertopic", "lda", "topicgpt"]
            for other_type in other_model_types:
                if other_type == self.topic_model_type:
                    continue

                other_path = os.path.join(
                    os.path.dirname(os.path.dirname(self.output_dir)),  # Go up to dataset level
                    other_type,
                    f"query_{self.query_id}",
                    "samples",
                    f"{sample_name}.pkl"
                )

                if os.path.exists(other_path):
                    logger.info(f"Loading shared sample from {other_type}: {other_path}")
                    with open(other_path, 'rb') as f:
                        sample = pickle.load(f)

                    # Copy to local cache for future use
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(sample, f)
                    logger.info(f"Copied sample to local cache: {cache_path}")

                    return sample

        # Compute if no cache found
        logger.info(f"Computing sample (cache miss or force=True)")
        result = compute_fn()

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Saved sample to cache: {cache_path}")

        return result

    def _build_qrels_dict(self) -> Dict[int, Dict[str, int]]:
        """Build qrels dictionary for fast relevance lookup

        Returns:
            Dict mapping query_id -> {doc_id: relevance_score}
        """
        # Handle datasets without QRELs (e.g., doctor-reviews)
        if self.qrels_dataset is None:
            logger.info("No QRELs provided - relevant_concentration metric will return 0.0")
            return {}

        qrels_dict = defaultdict(dict)
        for item in self.qrels_dataset:
            query_id = int(item['query-id'])
            corpus_id = str(item['corpus-id'])
            score = int(item['score'])
            qrels_dict[query_id][corpus_id] = score
        return dict(qrels_dict)

    def _compute_idf_scores(self) -> Dict[str, float]:
        """Compute IDF scores from full corpus using TfidfVectorizer

        Returns:
            Dict mapping words to IDF scores
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("Computing IDF scores on full corpus (this may take 2-5 minutes)...")

        # Extract corpus texts
        corpus_texts = []
        for doc in tqdm(self.corpus_dataset, desc="Loading corpus for IDF"):
            text = doc.get('text', '') or doc.get('title', '')
            corpus_texts.append(text)

        # Fit vectorizer on full corpus
        vectorizer = TfidfVectorizer(
            max_features=50000,  # Top 50K vocabulary
            stop_words='english',
            ngram_range=(1, 2),   # Unigrams + bigrams
            min_df=2,             # Must appear in at least 2 docs
            lowercase=True
        )

        vectorizer.fit(corpus_texts)

        # Extract IDF scores
        feature_names = vectorizer.get_feature_names_out()
        idf_values = vectorizer.idf_

        idf_dict = dict(zip(feature_names, idf_values))

        logger.info(f"✓ Computed IDF for {len(idf_dict)} terms from {len(corpus_texts)} documents")
        return idf_dict

    def _load_or_compute_idf(self) -> Dict[str, float]:
        """Load cached IDF scores or compute if not exists"""
        cache_path = os.path.join(self.idf_cache_dir, "idf_scores.pkl")

        return self._load_or_compute(
            cache_path=cache_path,
            compute_fn=self._compute_idf_scores,
            force=False  # Never force - IDF computation is expensive
        )

    def sample_random_uniform(self) -> Dict[str, Any]:
        """Method 1: Random uniform sampling"""
        logger.info(f"Method 1: Random uniform sampling ({self.sample_size} docs)")

        def compute():
            # Pure random sampling
            all_indices = list(range(len(self.corpus_dataset)))
            sample_indices = random.sample(all_indices, self.sample_size)

            docs = []
            doc_ids = []
            for idx in tqdm(sample_indices, desc="Extracting random docs"):
                doc = self.corpus_dataset[idx]
                # Handle documents with or without title field
                title = doc.get("title", "")
                text = doc.get("text", "")
                if title:
                    docs.append(title + "\n\n" + text)
                else:
                    docs.append(text)
                doc_ids.append(doc["_id"])

            return {
                "method": "random_uniform",
                "doc_ids": doc_ids,
                "doc_texts": docs,
                "sample_size": len(docs)
            }

        return self._load_or_compute_sample("random_uniform", compute, self.force_regenerate_samples)

    def sample_direct_retrieval(self) -> Dict[str, Any]:
        """Method 2: Direct retrieval (Hybrid BM25+SBERT)"""
        logger.info(f"Method 2: Direct retrieval ({self.sample_size} docs)")

        def compute():
            # Perform hybrid search
            results = self.search_engine.search(
                query=self.query_text,
                top_k=self.sample_size,
                method=RetrievalMethod.HYBRID,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=False
            )

            doc_ids = [doc_id for doc_id, _ in results]
            doc_texts = []

            for doc_id in tqdm(doc_ids, desc="Extracting retrieved docs"):
                doc_text = self.search_engine.get_document_by_id(doc_id)
                doc_texts.append(doc_text)

            return {
                "method": "direct_retrieval",
                "doc_ids": doc_ids,
                "doc_texts": doc_texts,
                "sample_size": len(doc_ids)
            }

        return self._load_or_compute_sample("direct_retrieval", compute, self.force_regenerate_samples)

    def sample_query_expansion(self) -> Dict[str, Any]:
        """Method 3: Query expansion + retrieval"""
        logger.info(f"Method 3: Query expansion + retrieval ({self.sample_size} docs)")

        def compute():
            # Load cached keywords (use configured path or default)
            keyword_cache_path = self.keyword_cache_path
            if keyword_cache_path is None:
                # Default to TREC-COVID keywords for backward compatibility
                keyword_cache_path = "/home/srangre1/cache/keywords/keybert_k10_div0.7_top10docs_mpnet_k1000_ngram1-2.json"

            if not os.path.exists(keyword_cache_path):
                raise FileNotFoundError(f"Keyword cache not found: {keyword_cache_path}")

            with open(keyword_cache_path, 'r') as f:
                all_keywords = json.load(f)

            query_keywords = all_keywords.get(self.query_id, [])

            if not query_keywords:
                logger.warning(f"No keywords found for query {self.query_id}, using direct retrieval")
                return self.sample_direct_retrieval()

            logger.info(f"Using keywords: {query_keywords}")

            # Perform baseline search
            baseline_results = self.search_engine.search(
                query=self.query_text,
                top_k=self.sample_size * 2,  # Retrieve more for fusion
                method=RetrievalMethod.HYBRID,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=False
            )

            # Search with each keyword
            keyword_results = []
            for keyword in tqdm(query_keywords, desc="Searching with keywords"):
                kw_results = self.search_engine.search(
                    query=keyword,
                    top_k=self.sample_size * 2,
                    method=RetrievalMethod.HYBRID,
                    hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                    use_mmr=False,
                    use_cross_encoder=False
                )
                keyword_results.append(kw_results)

            # Weighted RRF fusion
            all_rankings = [baseline_results] + keyword_results
            weights = [0.7] + [0.3 / len(query_keywords)] * len(query_keywords)

            combined = self._reciprocal_rank_fusion(all_rankings, weights)

            # Take top sample_size
            doc_ids = [doc_id for doc_id, _ in combined[:self.sample_size]]
            doc_texts = []

            for doc_id in tqdm(doc_ids, desc="Extracting expanded docs"):
                doc_text = self.search_engine.get_document_by_id(doc_id)
                doc_texts.append(doc_text)

            return {
                "method": "query_expansion",
                "doc_ids": doc_ids,
                "doc_texts": doc_texts,
                "sample_size": len(doc_ids),
                "keywords": query_keywords
            }

        return self._load_or_compute_sample("query_expansion", compute, self.force_regenerate_samples)

    def sample_keyword_search(self) -> Dict[str, Any]:
        """Method 4: Simple keyword search (BM25 only)"""
        logger.info(f"Method 4: Simple keyword search ({self.sample_size} docs)")

        def compute():
            # Perform BM25-only search
            results = self.search_engine.search(
                query=self.query_text,
                top_k=self.sample_size,
                method=RetrievalMethod.BM25,
                use_mmr=False,
                use_cross_encoder=False
            )

            doc_ids = [doc_id for doc_id, _ in results]
            doc_texts = []

            for doc_id in tqdm(doc_ids, desc="Extracting keyword search docs"):
                doc_text = self.search_engine.get_document_by_id(doc_id)
                doc_texts.append(doc_text)

            return {
                "method": "keyword_search",
                "doc_ids": doc_ids,
                "doc_texts": doc_texts,
                "sample_size": len(doc_ids)
            }

        return self._load_or_compute_sample("keyword_search", compute, self.force_regenerate_samples)

    def sample_direct_retrieval_mmr(self) -> Dict[str, Any]:
        """Method 5: Direct Retrieval + MMR (High Diversity)

        Retrieves 5000 candidates, applies MMR reranking with lambda=0.3 (70% diversity),
        then selects top 1000 documents.
        """
        logger.info(f"Method 5: Direct Retrieval + MMR (High Diversity, lambda=0.3)")

        def compute():
            # STEP 1: Retrieve large candidate pool (5000 docs) WITHOUT MMR
            logger.info("Retrieving 5000 candidate documents...")
            candidate_results = self.search_engine.search(
                query=self.query_text,
                top_k=5000,  # Large candidate pool for MMR to select from
                method=RetrievalMethod.HYBRID,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,  # Don't apply MMR yet
                use_cross_encoder=False
            )

            # STEP 2: Apply MMR reranking with high diversity (lambda=0.3)
            logger.info("Applying MMR reranking with lambda=0.3 (30% relevance, 70% diversity)...")

            # Get embeddings for MMR
            candidate_ids = [r[0] for r in candidate_results]
            candidate_scores = [r[1] for r in candidate_results]

            # Get document indices
            candidate_indices = [self.search_engine.corpus_ids.index(doc_id)
                                for doc_id in candidate_ids]
            candidate_embeddings = self.search_engine.doc_embeddings[candidate_indices]

            # Compute query embedding
            query_embedding = self.search_engine.sbert_model.encode(
                self.query_text,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            # STEP 3: Apply MMR to select top 1000 from 5000 candidates
            mmr_results = self.search_engine._mmr_rerank(
                query_embedding=query_embedding,
                doc_embeddings=candidate_embeddings,
                doc_ids=candidate_ids,
                scores=candidate_scores,
                top_k=self.sample_size,  # 1000
                lambda_param=0.3  # 30% relevance, 70% diversity
            )

            # Extract final sample
            doc_ids = [r[0] for r in mmr_results]
            doc_texts = []

            for doc_id in tqdm(doc_ids, desc="Extracting MMR docs"):
                doc_text = self.search_engine.get_document_by_id(doc_id)
                doc_texts.append(doc_text)

            logger.info(f"✓ Selected {len(doc_ids)} documents after MMR (from 5000 candidates)")

            return {
                'method': 'direct_retrieval_mmr',
                'doc_ids': doc_ids,
                'doc_texts': doc_texts,
                'sample_size': len(doc_ids),
                'mmr_lambda': 0.3,
                'candidate_pool_size': 5000
            }

        return self._load_or_compute_sample("direct_retrieval_mmr", compute, self.force_regenerate_samples)

    def sample_retrieval_random(self, pool_size: int = 5000) -> Dict[str, Any]:
        """Method 6: Retrieval Pool + Random Sampling

        Retrieves large pool of relevant documents using hybrid search,
        then randomly samples target size from that pool.

        Tests whether relevance filtering alone (without ranking bias) affects
        topic modeling outcomes.

        Args:
            pool_size: Size of retrieval pool to sample from (default: 5000)

        Returns:
            Dictionary with sampled documents and metadata
        """
        logger.info(f"Method 6: Retrieval Pool + Random Sampling ({self.sample_size} from {pool_size} pool)")

        def compute():
            # STEP 1: Retrieve large pool of relevant documents
            logger.info(f"  Retrieving pool of {pool_size} documents using hybrid search...")

            pool_results = self.search_engine.search(
                query=self.query_text,
                top_k=pool_size,
                method=RetrievalMethod.HYBRID,
                hybrid_strategy=HybridStrategy.SIMPLE_SUM,
                use_mmr=False,
                use_cross_encoder=False
            )

            # STEP 2: Random sample from pool
            logger.info(f"  Randomly sampling {self.sample_size} documents from pool...")
            pool_doc_ids = [doc_id for doc_id, _ in pool_results]

            # Pure random sampling (uniform) - respects random seed
            sampled_indices = random.sample(range(len(pool_doc_ids)), self.sample_size)
            doc_ids = [pool_doc_ids[i] for i in sampled_indices]

            # STEP 3: Extract document texts
            doc_texts = []
            for doc_id in tqdm(doc_ids, desc="Extracting retrieval-random docs"):
                doc_text = self.search_engine.get_document_by_id(doc_id)
                doc_texts.append(doc_text)

            logger.info(f"  ✓ Selected {len(doc_ids)} documents (random sample from {len(pool_doc_ids)} retrieved)")

            return {
                'method': 'retrieval_random',
                'doc_ids': doc_ids,
                'doc_texts': doc_texts,
                'sample_size': len(doc_ids),
                'pool_size': len(pool_doc_ids),  # Actual pool size (may be < requested)
                'requested_pool_size': pool_size,
                'sampling_strategy': 'random_from_pool'
            }

        return self._load_or_compute_sample("retrieval_random", compute, self.force_regenerate_samples)


    def _reciprocal_rank_fusion(
        self,
        rankings: List[List[Tuple[str, float]]],
        weights: List[float],
        k: int = 60
    ) -> List[Tuple[str, float]]:
        """Reciprocal rank fusion for combining rankings"""
        scores = defaultdict(float)

        for i, rank_list in enumerate(rankings):
            weight = weights[i]
            for rank, (doc_id, _) in enumerate(rank_list):
                scores[doc_id] += weight * (1.0 / (k + rank + 1))

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _get_bertopic_n_topics(self, method: str) -> int:
        """
        Read n_topics from BERTopic results for this query/method.
        Falls back to default if not found.

        This enables fair comparison: LDA uses same topic count as BERTopic discovered.

        Args:
            method: Sampling method name (e.g., "random_uniform")

        Returns:
            Number of topics found by BERTopic for this query/method, or 20 if not found
        """
        bertopic_csv = f"/home/srangre1/results/trec-covid/bertopic/query_{self.query_id}/results/per_method_summary.csv"

        if os.path.exists(bertopic_csv):
            try:
                import pandas as pd
                df = pd.read_csv(bertopic_csv)
                row = df[df['method'] == method]
                if not row.empty and 'n_topics' in row.columns:
                    n_topics = int(row['n_topics'].values[0])
                    logger.info(f"  Using n_topics={n_topics} from BERTopic results for {method}")
                    return n_topics
            except Exception as e:
                logger.warning(f"  Could not read BERTopic results: {e}")

        logger.info(f"  BERTopic results not found, using default n_topics=20")
        return 20  # Default fallback

    def run_topic_modeling(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run topic modeling on a document sample"""
        method = sample["method"]
        logger.info(f"Running {self.topic_model_type} on {method} ({sample['sample_size']} docs)")

        model_cache_path = os.path.join(self.topic_models_dir, f"{method}_model.pkl")
        results_cache_path = os.path.join(self.topic_models_dir, f"{method}_results.pkl")

        def compute():
            docs = sample["doc_texts"]
            doc_ids = sample["doc_ids"]

            # For LDA with "auto" n_topics, match BERTopic's discovered count
            topic_model_params = self.topic_model_params.copy()
            if self.topic_model_type == "lda":
                if topic_model_params.get("n_topics") == "auto":
                    topic_model_params["n_topics"] = self._get_bertopic_n_topics(method)
                    logger.info(f"  LDA will use n_topics={topic_model_params['n_topics']} (matched to BERTopic)")

            # Use wrapper to dispatch to appropriate topic model
            topic_model = TopicModelWrapper(
                model_type=self.topic_model_type,
                **topic_model_params
            )

            # Call wrapper - returns EXACT same dict structure as before
            results = topic_model.fit_and_get_results(
                docs=docs,
                doc_ids=doc_ids,
                method_name=method,
                embedding_model_name=self.embedding_model_name,
                device=self.device,
                save_model=self.save_topic_models,
                model_path=model_cache_path if self.save_topic_models else None
            )

            return results

        return self._load_or_compute(results_cache_path, compute, self.force_regenerate_topics)

    def compute_pairwise_metrics(
        self,
        method_a: str,
        method_b: str,
        results_a: Dict[str, Any],
        results_b: Dict[str, Any],
        sample_a: Dict[str, Any],
        sample_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute all metrics for a pair of methods

        Args:
            method_a: First method name
            method_b: Second method name
            results_a: Topic results for method A
            results_b: Topic results for method B
            sample_a: Sample for method A
            sample_b: Sample for method B

        Returns:
            Dictionary of pairwise metrics
        """
        metrics = {
            "method_a": method_a,
            "method_b": method_b,
        }

        # Extract topic information
        topics_a = results_a["topic_words"]
        topics_b = results_b["topic_words"]

        # Topic coverage comparison
        metrics["n_topics_a"] = len(topics_a)
        metrics["n_topics_b"] = len(topics_b)
        metrics["topic_count_ratio"] = len(topics_a) / len(topics_b) if len(topics_b) > 0 else 0.0

        # Topic diversity comparison (semantic and lexical)
        diversity_semantic_a = self._compute_topic_diversity_semantic(topics_a)
        diversity_semantic_b = self._compute_topic_diversity_semantic(topics_b)
        metrics["diversity_semantic_a"] = diversity_semantic_a
        metrics["diversity_semantic_b"] = diversity_semantic_b
        metrics["diversity_semantic_diff"] = diversity_semantic_a - diversity_semantic_b

        diversity_lexical_a = self._compute_topic_diversity_lexical(topics_a)
        diversity_lexical_b = self._compute_topic_diversity_lexical(topics_b)
        metrics["diversity_lexical_a"] = diversity_lexical_a
        metrics["diversity_lexical_b"] = diversity_lexical_b
        metrics["diversity_lexical_diff"] = diversity_lexical_a - diversity_lexical_b

        # Intrinsic quality metrics
        # NPMI Coherence
        npmi_a = self._compute_npmi_coherence(topics_a, sample_a["doc_texts"])
        npmi_b = self._compute_npmi_coherence(topics_b, sample_b["doc_texts"])
        metrics["npmi_coherence_a"] = npmi_a
        metrics["npmi_coherence_b"] = npmi_b
        metrics["npmi_coherence_diff"] = npmi_a - npmi_b

        # Embedding Coherence
        emb_coh_a = self._compute_embedding_coherence(topics_a)
        emb_coh_b = self._compute_embedding_coherence(topics_b)
        metrics["embedding_coherence_a"] = emb_coh_a
        metrics["embedding_coherence_b"] = emb_coh_b
        metrics["embedding_coherence_diff"] = emb_coh_a - emb_coh_b

        # Relevant Document Concentration (NEW)
        relevant_conc_a = self._compute_relevant_concentration(sample_a)
        relevant_conc_b = self._compute_relevant_concentration(sample_b)
        metrics["relevant_concentration_a"] = relevant_conc_a
        metrics["relevant_concentration_b"] = relevant_conc_b
        metrics["relevant_concentration_diff"] = relevant_conc_a - relevant_conc_b

        # Topic Specificity (IDF-based) (NEW)
        topic_spec_a = self._compute_topic_specificity(topics_a)
        topic_spec_b = self._compute_topic_specificity(topics_b)
        metrics["topic_specificity_a"] = topic_spec_a
        metrics["topic_specificity_b"] = topic_spec_b
        metrics["topic_specificity_diff"] = topic_spec_a - topic_spec_b

        # Per-topic query alignment metrics
        alignment_a = self._compute_per_topic_query_alignment(results_a)
        alignment_b = self._compute_per_topic_query_alignment(results_b)
        metrics["max_query_similarity_a"] = alignment_a["max_query_similarity"]
        metrics["max_query_similarity_b"] = alignment_b["max_query_similarity"]
        metrics["query_relevant_ratio_a"] = alignment_a["query_relevant_ratio"]
        metrics["query_relevant_ratio_b"] = alignment_b["query_relevant_ratio"]
        metrics["top3_avg_similarity_a"] = alignment_a["top3_avg_similarity"]

        # Relevant topic diversity metrics (diversity among query-relevant topics only)
        rel_div_a = self._compute_relevant_topic_diversity(results_a)
        rel_div_b = self._compute_relevant_topic_diversity(results_b)
        metrics["relevant_topic_diversity_a"] = rel_div_a["relevant_topic_diversity"]
        metrics["relevant_topic_diversity_b"] = rel_div_b["relevant_topic_diversity"]
        metrics["relevance_weighted_diversity_a"] = rel_div_a["relevance_weighted_diversity"]
        metrics["relevance_weighted_diversity_b"] = rel_div_b["relevance_weighted_diversity"]
        metrics["topk_relevant_diversity_a"] = rel_div_a["topk_relevant_diversity"]
        metrics["topk_relevant_diversity_b"] = rel_div_b["topk_relevant_diversity"]
        metrics["n_relevant_topics_a"] = rel_div_a["n_relevant_topics"]
        metrics["n_relevant_topics_b"] = rel_div_b["n_relevant_topics"]
        metrics["relevant_diversity_ratio_a"] = rel_div_a["relevant_diversity_ratio"]
        metrics["relevant_diversity_ratio_b"] = rel_div_b["relevant_diversity_ratio"]
        metrics["top3_avg_similarity_b"] = alignment_b["top3_avg_similarity"]

        # Topic matching using Hungarian algorithm
        word_overlaps, semantic_sims = self._match_topics(topics_a, topics_b)

        if word_overlaps and semantic_sims:
            metrics["topic_word_overlap_mean"] = float(np.mean(word_overlaps))
            metrics["topic_word_overlap_std"] = float(np.std(word_overlaps))
            metrics["topic_semantic_similarity_mean"] = float(np.mean(semantic_sims))
            metrics["topic_semantic_similarity_std"] = float(np.std(semantic_sims))

            # Topic matching at different thresholds
            for threshold in [0.5, 0.6, 0.7]:
                matched = sum(1 for s in semantic_sims if s >= threshold)

                # Precision: how many of B's topics match A's topics
                precision_b = matched / len(topics_b) if len(topics_b) > 0 else 0.0
                # Recall: how many of A's topics match B's topics
                recall_a = matched / len(topics_a) if len(topics_a) > 0 else 0.0
                # F1
                f1 = 2 * precision_b * recall_a / (precision_b + recall_a) if (precision_b + recall_a) > 0 else 0.0

                threshold_str = str(threshold).replace('.', '')
                metrics[f"precision_b_@{threshold_str}"] = precision_b
                metrics[f"recall_a_@{threshold_str}"] = recall_a
                metrics[f"f1_@{threshold_str}"] = f1
        else:
            # No topics to compare
            metrics["topic_word_overlap_mean"] = 0.0
            metrics["topic_word_overlap_std"] = 0.0
            metrics["topic_semantic_similarity_mean"] = 0.0
            metrics["topic_semantic_similarity_std"] = 0.0
            for threshold in [0.5, 0.6, 0.7]:
                threshold_str = str(threshold).replace('.', '')
                metrics[f"precision_b_@{threshold_str}"] = 0.0
                metrics[f"recall_a_@{threshold_str}"] = 0.0
                metrics[f"f1_@{threshold_str}"] = 0.0

        # Topic-Query similarity for both methods
        metrics["topic_query_similarity_a"] = self._compute_topic_query_similarity(results_a)
        metrics["topic_query_similarity_b"] = self._compute_topic_query_similarity(results_b)
        metrics["topic_query_similarity_diff"] = metrics["topic_query_similarity_a"] - metrics["topic_query_similarity_b"]

        # Document distribution comparison
        outlier_a = np.sum(np.array(results_a["topics"]) == -1) / len(results_a["topics"])
        outlier_b = np.sum(np.array(results_b["topics"]) == -1) / len(results_b["topics"])
        metrics["outlier_ratio_a"] = float(outlier_a)
        metrics["outlier_ratio_b"] = float(outlier_b)
        metrics["outlier_ratio_diff"] = float(outlier_a - outlier_b)

        # Average topic size comparison
        topics_arr_a = np.array(results_a["topics"])
        topics_arr_b = np.array(results_b["topics"])

        unique_a, counts_a = np.unique(topics_arr_a[topics_arr_a != -1], return_counts=True)
        unique_b, counts_b = np.unique(topics_arr_b[topics_arr_b != -1], return_counts=True)

        avg_size_a = float(np.mean(counts_a)) if len(counts_a) > 0 else 0.0
        avg_size_b = float(np.mean(counts_b)) if len(counts_b) > 0 else 0.0

        metrics["avg_topic_size_a"] = avg_size_a
        metrics["avg_topic_size_b"] = avg_size_b
        metrics["avg_topic_size_diff"] = avg_size_a - avg_size_b

        # Document-level clustering metrics (if overlap exists)
        doc_ids_a = set(sample_a["doc_ids"])
        doc_ids_b = set(sample_b["doc_ids"])
        overlap = doc_ids_a & doc_ids_b

        metrics["overlap_count"] = len(overlap)
        metrics["overlap_percent_a"] = len(overlap) / len(doc_ids_a) * 100 if len(doc_ids_a) > 0 else 0.0
        metrics["overlap_percent_b"] = len(overlap) / len(doc_ids_b) * 100 if len(doc_ids_b) > 0 else 0.0

        # Document Overlap (Jaccard similarity)
        union_count = len(doc_ids_a | doc_ids_b)
        metrics["document_overlap_jaccard"] = len(overlap) / union_count if union_count > 0 else 0.0

        if len(overlap) >= 2:
            doc_to_idx_a = {doc_id: i for i, doc_id in enumerate(sample_a["doc_ids"])}
            doc_to_idx_b = {doc_id: i for i, doc_id in enumerate(sample_b["doc_ids"])}

            topics_overlap_a = [results_a["topics"][doc_to_idx_a[doc_id]] for doc_id in overlap]
            topics_overlap_b = [results_b["topics"][doc_to_idx_b[doc_id]] for doc_id in overlap]

            try:
                ari = adjusted_rand_score(topics_overlap_a, topics_overlap_b)
                nmi = normalized_mutual_info_score(topics_overlap_a, topics_overlap_b)
                metrics["ari"] = float(ari)
                metrics["nmi"] = float(nmi)
            except:
                metrics["ari"] = None
                metrics["nmi"] = None
        else:
            metrics["ari"] = None
            metrics["nmi"] = None

        return metrics

    def _compute_topic_diversity_semantic(self, topic_words: Dict[int, List[str]]) -> float:
        """
        Compute semantic diversity: average pairwise distance between topics in embedding space

        Higher values indicate topics are more semantically distinct from each other.

        Args:
            topic_words: Dictionary mapping topic IDs to lists of words

        Returns:
            Average pairwise semantic distance (0-1, higher = more diverse)
        """
        if len(topic_words) < 2:
            return 0.0

        # Use shared embedding model (loaded once in __init__)
        model = self.metrics_embedding_model

        # Get embeddings for topics
        topic_embeddings = []
        for topic_id, words in topic_words.items():
            top_words = " ".join(words[:5])
            embedding = model.encode(top_words, convert_to_tensor=False, show_progress_bar=False)
            topic_embeddings.append(embedding)

        topic_embeddings = np.array(topic_embeddings)

        # Compute pairwise distances
        similarities = cosine_similarity(topic_embeddings)
        n = len(topic_embeddings)
        distances = 1 - similarities
        upper_triangle = distances[np.triu_indices(n, k=1)]

        return float(np.mean(upper_triangle))

    def _compute_topic_diversity_lexical(self, topic_words: Dict[int, List[str]], top_k: int = 10) -> float:
        """
        Compute lexical diversity: unique word fraction across all topics

        Measures vocabulary redundancy. Higher values indicate less word overlap across topics.

        Args:
            topic_words: Dictionary mapping topic IDs to lists of words
            top_k: Number of top words to consider per topic

        Returns:
            Ratio of unique words to total words (0-1, higher = less redundancy)
        """
        if not topic_words:
            return 0.0

        unique_words = set()
        total_words = 0

        for topic_id, words in topic_words.items():
            unique_words.update(words[:top_k])
            total_words += min(len(words), top_k)

        return len(unique_words) / total_words if total_words > 0 else 0.0

    def _match_topics(
        self,
        topics_a: Dict[int, List[str]],
        topics_b: Dict[int, List[str]],
        top_n: int = 10
    ) -> Tuple[List[float], List[float]]:
        """Match topics using Hungarian algorithm"""
        if not topics_a or not topics_b:
            return [], []

        # Use shared embedding model (loaded once in __init__)
        model = self.metrics_embedding_model

        # Get topic IDs
        topic_ids_a = list(topics_a.keys())
        topic_ids_b = list(topics_b.keys())

        # Compute embeddings
        embeddings_a = []
        for topic_id in topic_ids_a:
            words = " ".join(topics_a[topic_id][:top_n])
            emb = model.encode(words, convert_to_tensor=False, show_progress_bar=False)
            embeddings_a.append(emb)

        embeddings_b = []
        for topic_id in topic_ids_b:
            words = " ".join(topics_b[topic_id][:top_n])
            emb = model.encode(words, convert_to_tensor=False, show_progress_bar=False)
            embeddings_b.append(emb)

        embeddings_a = np.array(embeddings_a)
        embeddings_b = np.array(embeddings_b)

        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings_a, embeddings_b)

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

        # Get matched similarities and word overlaps
        semantic_sims = []
        word_overlaps = []

        for i, j in zip(row_ind, col_ind):
            topic_id_a = topic_ids_a[i]
            topic_id_b = topic_ids_b[j]

            # Semantic similarity
            semantic_sims.append(similarity_matrix[i, j])

            # Word overlap (Jaccard)
            words_a = set(topics_a[topic_id_a][:top_n])
            words_b = set(topics_b[topic_id_b][:top_n])

            intersection = len(words_a & words_b)
            union = len(words_a | words_b)
            jaccard = intersection / union if union > 0 else 0.0

            word_overlaps.append(jaccard)

        return word_overlaps, semantic_sims

    def _compute_topic_query_similarity(self, results: Dict[str, Any]) -> float:
        """
        Compute average similarity between topics and the query

        Args:
            results: Topic modeling results

        Returns:
            Average cosine similarity between all topics and the query
        """
        topic_words = results["topic_words"]

        if not topic_words:
            return 0.0

        # Use shared embedding model (loaded once in __init__)
        model = self.metrics_embedding_model

        # Encode query
        query_embedding = model.encode(self.query_text, convert_to_tensor=False, show_progress_bar=False)

        # Encode all topics (top 10 words each)
        topic_embeddings = []
        for topic_id, words in topic_words.items():
            top_words = " ".join(words[:10])
            emb = model.encode(top_words, convert_to_tensor=False, show_progress_bar=False)
            topic_embeddings.append(emb)

        if not topic_embeddings:
            return 0.0

        topic_embeddings = np.array(topic_embeddings)
        query_embedding = query_embedding.reshape(1, -1)

        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, topic_embeddings)[0]

        # Return average similarity
        return float(np.mean(similarities))

    def _compute_per_topic_query_alignment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute per-topic query alignment metrics

        Returns detailed metrics about how individual topics align with the query.

        Args:
            results: Topic modeling results

        Returns:
            Dictionary with:
            - max_query_similarity: Highest topic-query similarity
            - query_relevant_ratio: Fraction of topics above similarity threshold (0.5)
            - top3_avg_similarity: Average similarity of top-3 most query-relevant topics
            - per_topic_similarities: List of (topic_id, similarity) tuples
        """
        topic_words = results["topic_words"]

        if not topic_words:
            return {
                "max_query_similarity": 0.0,
                "query_relevant_ratio": 0.0,
                "top3_avg_similarity": 0.0,
                "per_topic_similarities": []
            }

        # Use shared embedding model (loaded once in __init__)
        model = self.metrics_embedding_model

        # Encode query
        query_embedding = model.encode(self.query_text, convert_to_tensor=False, show_progress_bar=False)

        # Encode all topics and compute similarities
        similarities = []
        for topic_id, words in topic_words.items():
            top_words = " ".join(words[:10])
            topic_emb = model.encode(top_words, convert_to_tensor=False, show_progress_bar=False)
            sim = float(cosine_similarity([query_embedding], [topic_emb])[0][0])
            similarities.append((topic_id, sim))

        # Sort by similarity (descending)
        similarities_sorted = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Compute metrics
        max_sim = similarities_sorted[0][1] if similarities_sorted else 0.0

        # Count topics above threshold
        threshold = 0.5
        relevant_count = sum(1 for _, sim in similarities if sim >= threshold)
        relevant_ratio = relevant_count / len(similarities) if similarities else 0.0

        # Top-3 average
        top3_sims = [sim for _, sim in similarities_sorted[:3]]
        top3_avg = float(np.mean(top3_sims)) if top3_sims else 0.0

        return {
            "max_query_similarity": max_sim,
            "query_relevant_ratio": relevant_ratio,
            "top3_avg_similarity": top3_avg,
            "per_topic_similarities": similarities_sorted
        }

    def _compute_relevant_topic_diversity(
        self,
        results: Dict[str, Any],
        relevance_threshold: float = 0.5,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Compute semantic diversity among query-relevant topics only.

        This metric addresses the insight that overall diversity can be inflated by
        "noise" topics unrelated to the query. By focusing on relevant topics, we
        measure whether the method discovers distinct facets of the query.

        Three approaches are computed:
        1. Hard Threshold: Diversity among topics with query similarity >= threshold
        2. Top-K: Diversity among top-K most query-relevant topics
        3. Relevance-Weighted: Diversity weighted by query relevance (no threshold)

        Args:
            results: Topic modeling results containing topic_words
            relevance_threshold: Threshold for "relevant" topics (default 0.5)
            top_k: Number of top topics for top-k diversity (default 5)

        Returns:
            Dictionary with:
            - relevant_topic_diversity: Diversity among relevant topics (hard threshold)
            - relevance_weighted_diversity: Diversity weighted by query relevance
            - topk_relevant_diversity: Diversity among top-K topics
            - n_relevant_topics: Count of topics above threshold
            - relevant_diversity_ratio: relevant_div / overall_div (>1 = good facet coverage)
        """
        topic_words = results.get("topic_words", {})

        # Filter out outlier topic
        topic_words = {k: v for k, v in topic_words.items() if k != -1}

        if len(topic_words) < 2:
            return {
                "relevant_topic_diversity": None,
                "relevance_weighted_diversity": None,
                "topk_relevant_diversity": None,
                "n_relevant_topics": 0,
                "relevant_diversity_ratio": None
            }

        model = self.metrics_embedding_model

        # Encode query
        query_embedding = model.encode(
            self.query_text,
            convert_to_tensor=False,
            show_progress_bar=False
        )

        # Compute embeddings and query similarities for all topics
        topic_data = []
        for topic_id, words in topic_words.items():
            topic_text = " ".join(words[:10])
            topic_embedding = model.encode(
                topic_text,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            similarity = float(cosine_similarity([query_embedding], [topic_embedding])[0][0])
            topic_data.append({
                "topic_id": topic_id,
                "embedding": topic_embedding,
                "query_similarity": similarity
            })

        # Helper to compute diversity from embeddings
        def compute_diversity(embeddings):
            if len(embeddings) < 2:
                return None
            sim_matrix = cosine_similarity(embeddings)
            n = len(embeddings)
            distances = 1 - sim_matrix
            upper_triangle = distances[np.triu_indices(n, k=1)]
            return float(np.mean(upper_triangle))

        # ===== 1. Hard Threshold Diversity =====
        relevant_topics = [t for t in topic_data if t["query_similarity"] >= relevance_threshold]
        n_relevant = len(relevant_topics)

        if n_relevant >= 2:
            relevant_embeddings = np.array([t["embedding"] for t in relevant_topics])
            relevant_diversity = compute_diversity(relevant_embeddings)
        else:
            relevant_diversity = None

        # ===== 2. Top-K Diversity =====
        topic_data_sorted = sorted(topic_data, key=lambda x: x["query_similarity"], reverse=True)
        top_k_topics = topic_data_sorted[:min(top_k, len(topic_data_sorted))]

        if len(top_k_topics) >= 2:
            topk_embeddings = np.array([t["embedding"] for t in top_k_topics])
            topk_diversity = compute_diversity(topk_embeddings)
        else:
            topk_diversity = None

        # ===== 3. Relevance-Weighted Diversity =====
        n = len(topic_data)
        embeddings = np.array([t["embedding"] for t in topic_data])
        query_sims = np.array([t["query_similarity"] for t in topic_data])

        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = 1 - sim_matrix
        weight_matrix = np.outer(query_sims, query_sims)

        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairwise_distances = dist_matrix[mask]
        pairwise_weights = weight_matrix[mask]
        total_weight = pairwise_weights.sum()

        if total_weight > 0:
            weighted_diversity = float((pairwise_distances * pairwise_weights).sum() / total_weight)
        else:
            weighted_diversity = None

        # ===== 4. Overall Diversity (for ratio) =====
        overall_diversity = compute_diversity(embeddings)

        # ===== 5. Compute Ratio =====
        if overall_diversity and overall_diversity > 0 and relevant_diversity is not None:
            relevant_ratio = relevant_diversity / overall_diversity
        else:
            relevant_ratio = None

        return {
            "relevant_topic_diversity": relevant_diversity,
            "relevance_weighted_diversity": weighted_diversity,
            "topk_relevant_diversity": topk_diversity,
            "n_relevant_topics": n_relevant,
            "relevant_diversity_ratio": relevant_ratio
        }

    def _compute_npmi_coherence(self, topic_words: Dict[int, List[str]], doc_texts: List[str], top_k: int = 10) -> float:
        """
        Compute average NPMI (Normalized Pointwise Mutual Information) coherence across topics

        NPMI measures semantic coherence of topic words based on co-occurrence in documents.
        Higher values indicate more interpretable/coherent topics.

        Args:
            topic_words: Dictionary mapping topic IDs to lists of words
            doc_texts: List of document texts for computing co-occurrences
            top_k: Number of top words to consider per topic

        Returns:
            Average NPMI coherence across all topics (range: -1 to 1, higher = more coherent)
        """
        if not topic_words or not doc_texts:
            return 0.0

        from collections import Counter
        import math

        # Tokenize documents (simple whitespace + lowercase)
        tokenized_docs = []
        for doc in doc_texts:
            tokens = set(doc.lower().split())
            tokenized_docs.append(tokens)

        n_docs = len(tokenized_docs)

        # Compute document frequencies for all words
        word_doc_freq = Counter()
        for tokens in tokenized_docs:
            for word in tokens:
                word_doc_freq[word] += 1

        # Compute NPMI for each topic
        topic_npmis = []

        for topic_id, words in topic_words.items():
            top_words = words[:top_k]

            # Compute NPMI for all word pairs
            pair_npmis = []
            for i in range(len(top_words)):
                for j in range(i + 1, len(top_words)):
                    word_i = top_words[i].lower()
                    word_j = top_words[j].lower()

                    # Count co-occurrences
                    co_occur = sum(1 for tokens in tokenized_docs if word_i in tokens and word_j in tokens)

                    # Compute PMI
                    if co_occur > 0:
                        p_i = word_doc_freq.get(word_i, 0) / n_docs
                        p_j = word_doc_freq.get(word_j, 0) / n_docs
                        p_ij = co_occur / n_docs

                        if p_i > 0 and p_j > 0 and p_ij > 0:
                            pmi = math.log(p_ij / (p_i * p_j))
                            npmi = pmi / (-math.log(p_ij))  # Normalize by -log(p(i,j))
                            pair_npmis.append(npmi)

            # Average NPMI for this topic
            if pair_npmis:
                topic_npmi = float(np.mean(pair_npmis))
                topic_npmis.append(topic_npmi)

        # Average across all topics
        return float(np.mean(topic_npmis)) if topic_npmis else 0.0

    def _compute_embedding_coherence(self, topic_words: Dict[int, List[str]], top_k: int = 10) -> float:
        """
        Compute average embedding coherence across topics

        Measures how semantically tight topic words are by computing average pairwise
        cosine similarity of word embeddings within each topic.

        Args:
            topic_words: Dictionary mapping topic IDs to lists of words
            top_k: Number of top words to consider per topic

        Returns:
            Average embedding coherence across all topics (0-1, higher = tighter topics)
        """
        if not topic_words:
            return 0.0

        # Use shared embedding model (loaded once in __init__)
        model = self.metrics_embedding_model

        topic_coherences = []

        for topic_id, words in topic_words.items():
            top_words = words[:top_k]

            if len(top_words) < 2:
                continue

            # Get embeddings for words
            word_embeddings = model.encode(top_words, convert_to_tensor=False, show_progress_bar=False)

            # Compute pairwise cosine similarities
            similarities = cosine_similarity(word_embeddings)

            # Get upper triangle (exclude diagonal)
            n = len(word_embeddings)
            upper_triangle = similarities[np.triu_indices(n, k=1)]

            # Average similarity for this topic
            topic_coherence = float(np.mean(upper_triangle))
            topic_coherences.append(topic_coherence)

        # Average across all topics
        return float(np.mean(topic_coherences)) if topic_coherences else 0.0

    def _compute_relevant_concentration(self, sample: Dict[str, Any]) -> float:
        """
        Compute fraction of sample that is relevant according to QRELs

        Validates that retrieval actually increases sample relevance using ground-truth.

        Args:
            sample: Sample dictionary with doc_ids

        Returns:
            Fraction of documents in sample that are relevant (0-1)
        """
        query_id_int = int(self.query_id)

        if query_id_int not in self.qrels_dict:
            logger.warning(f"No QRELs found for query {query_id_int}")
            return 0.0

        relevant_doc_ids = set(self.qrels_dict[query_id_int].keys())
        sample_doc_ids = set(sample['doc_ids'])

        relevant_in_sample = sample_doc_ids & relevant_doc_ids
        concentration = len(relevant_in_sample) / len(sample_doc_ids) if sample_doc_ids else 0.0

        logger.debug(f"Relevant concentration: {concentration:.3f} ({len(relevant_in_sample)}/{len(sample_doc_ids)})")
        return concentration

    def _compute_topic_specificity(self, topic_words: Dict[int, List[str]]) -> float:
        """
        Compute average IDF (specificity) of topic words

        Operationalizes "less generic" observation - higher IDF means more specific/technical terms.

        Args:
            topic_words: Dictionary mapping topic IDs to lists of words

        Returns:
            Average IDF across all topic words (higher = more specific)
        """
        if not topic_words or not self.idf_scores:
            return 0.0

        topic_specificities = []

        for topic_id, words in topic_words.items():
            if topic_id == -1:  # Skip outlier topic
                continue

            # Get IDF scores for topic words (top 10)
            topic_idf_scores = []
            for word in words[:10]:
                # Normalize word (lowercase, remove special chars)
                normalized_word = word.lower().strip()
                if normalized_word in self.idf_scores:
                    topic_idf_scores.append(self.idf_scores[normalized_word])

            if topic_idf_scores:
                topic_specificities.append(np.mean(topic_idf_scores))

        avg_specificity = np.mean(topic_specificities) if topic_specificities else 0.0
        logger.debug(f"Topic specificity (avg IDF): {avg_specificity:.3f}")
        return float(avg_specificity)

    def _save_topic_summaries(self, topic_results: Dict[str, Dict[str, Any]], samples: Dict[str, Dict[str, Any]]):
        """Save human-readable topic summaries for qualitative analysis"""
        logger.info("Saving human-readable topic summaries...")

        for method_name, results in topic_results.items():
            if results is None:
                continue

            # Create text summary
            summary_path = os.path.join(self.topics_summary_dir, f"{method_name}_topics.txt")
            json_path = os.path.join(self.topics_summary_dir, f"{method_name}_topics.json")

            # Text summary
            with open(summary_path, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"TOPIC SUMMARY: {method_name.upper()}\n")
                f.write(f"Query ID: {self.query_id}\n")
                f.write(f"Query: {self.query_text}\n")
                f.write(f"Number of Topics: {results['n_topics']}\n")
                f.write(f"Number of Documents: {results['n_docs']}\n")
                f.write("="*80 + "\n\n")

                # Get topic info
                topic_labels = results.get('topic_labels', {})
                topic_names = results.get('topic_names', {})
                topic_words_with_scores = results.get('topic_words_with_scores', {})

                # Count documents per topic
                topics_array = np.array(results['topics'])
                unique_topics, counts = np.unique(topics_array[topics_array != -1], return_counts=True)
                topic_counts = dict(zip(unique_topics, counts))

                # Sort topics by size (largest first)
                sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

                for topic_id, count in sorted_topics:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"TOPIC {topic_id}: {topic_labels.get(topic_id, 'N/A')}\n")
                    f.write(f"Name: {topic_names.get(topic_id, 'N/A')}\n")
                    f.write(f"Document Count: {count}\n")
                    f.write(f"-"*80 + "\n")

                    # Top words with scores
                    if topic_id in topic_words_with_scores:
                        f.write("Top Words (with c-TF-IDF scores):\n")
                        for i, (word, score) in enumerate(topic_words_with_scores[topic_id][:10], 1):
                            f.write(f"  {i:2d}. {word:20s} {score:.4f}\n")
                    f.write("\n")

                # Outliers
                n_outliers = np.sum(topics_array == -1)
                if n_outliers > 0:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"OUTLIERS (Topic -1)\n")
                    f.write(f"Document Count: {n_outliers}\n")
                    f.write(f"Percentage: {100 * n_outliers / len(topics_array):.2f}%\n")

            # JSON summary for programmatic access
            json_summary = {
                "method": method_name,
                "query_id": self.query_id,
                "query_text": self.query_text,
                "n_topics": results['n_topics'],
                "n_docs": results['n_docs'],
                "topics": []
            }

            # Get topic info
            topic_labels = results.get('topic_labels', {})
            topic_names = results.get('topic_names', {})
            topic_words_with_scores = results.get('topic_words_with_scores', {})

            # Count documents per topic
            topics_array = np.array(results['topics'])
            unique_topics, counts = np.unique(topics_array[topics_array != -1], return_counts=True)
            topic_counts = dict(zip(unique_topics, counts))
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

            for topic_id, count in sorted_topics:
                topic_data = {
                    "topic_id": int(topic_id),
                    "label": topic_labels.get(topic_id, f"Topic_{topic_id}"),
                    "name": topic_names.get(topic_id, f"Topic_{topic_id}"),
                    "doc_count": int(count),
                    "top_words": topic_words_with_scores.get(topic_id, [])[:10]
                }
                json_summary["topics"].append(topic_data)

            # Add outlier info
            n_outliers = np.sum(topics_array == -1)
            if n_outliers > 0:
                json_summary["outliers"] = {
                    "count": int(n_outliers),
                    "percentage": float(100 * n_outliers / len(topics_array))
                }

            with open(json_path, 'w') as f:
                json.dump(json_summary, f, indent=2)

            logger.info(f"Saved topic summary for {method_name}: {summary_path}")

        logger.info(f"All topic summaries saved to {self.topics_summary_dir}")

    def run_all_pairwise_comparisons(
        self,
        topic_results: Dict[str, Dict[str, Any]],
        samples: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Run pairwise comparisons for all method pairs

        Returns:
            DataFrame with all pairwise metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Running pairwise comparisons...")
        logger.info("="*80)

        methods = list(topic_results.keys())

        if len(methods) < 2:
            logger.error(f"Need at least 2 methods, found {len(methods)}")
            return pd.DataFrame()

        all_metrics = []

        # Compare each pair
        pairs = list(combinations(methods, 2))
        logger.info(f"Computing {len(pairs)} pairwise comparisons...\n")

        for method_a, method_b in tqdm(pairs, desc="Pairwise comparisons"):
            try:
                metrics = self.compute_pairwise_metrics(
                    method_a=method_a,
                    method_b=method_b,
                    results_a=topic_results[method_a],
                    results_b=topic_results[method_b],
                    sample_a=samples[method_a],
                    sample_b=samples[method_b]
                )
                all_metrics.append(metrics)

                logger.debug(f"Compared {method_a} vs {method_b}")
            except Exception as e:
                logger.error(f"Error comparing {method_a} vs {method_b}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)

        # Save results
        csv_path = os.path.join(self.results_dir, "pairwise_metrics.csv")
        json_path = os.path.join(self.results_dir, "pairwise_metrics.json")

        df.to_csv(csv_path, index=False)
        logger.info(f"\n✓ Saved pairwise metrics to {csv_path}")

        with open(json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"✓ Saved pairwise metrics to {json_path}")

        return df

    def create_per_method_summary(
        self,
        pairwise_df: pd.DataFrame,
        topic_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create per-method summary statistics

        Args:
            pairwise_df: DataFrame with pairwise metrics
            topic_results: Dict of topic modeling results

        Returns:
            DataFrame with per-method statistics
        """
        logger.info("Creating per-method summary...")

        methods = list(topic_results.keys())
        summary_data = []

        for method in methods:
            # Get all comparisons involving this method
            method_comparisons = pairwise_df[
                (pairwise_df['method_a'] == method) | (pairwise_df['method_b'] == method)
            ]

            # Extract metrics when this method is method_a or method_b
            topic_counts = []
            diversity_semantic = []
            diversity_lexical = []
            npmi_coherence = []
            embedding_coherence = []
            relevant_concentrations = []
            topic_specificities = []
            outlier_ratios = []
            avg_topic_sizes = []
            topic_query_sims = []
            max_query_sims = []
            query_relevant_ratios = []
            top3_avg_sims = []
            # New: Relevant topic diversity metrics
            relevant_topic_diversities = []
            relevance_weighted_diversities = []
            topk_relevant_diversities = []
            n_relevant_topics_list = []
            relevant_diversity_ratios = []

            for _, row in method_comparisons.iterrows():
                if row['method_a'] == method:
                    topic_counts.append(row['n_topics_a'])
                    diversity_semantic.append(row['diversity_semantic_a'])
                    diversity_lexical.append(row['diversity_lexical_a'])
                    npmi_coherence.append(row['npmi_coherence_a'])
                    embedding_coherence.append(row['embedding_coherence_a'])
                    relevant_concentrations.append(row['relevant_concentration_a'])
                    topic_specificities.append(row['topic_specificity_a'])
                    outlier_ratios.append(row['outlier_ratio_a'])
                    avg_topic_sizes.append(row['avg_topic_size_a'])
                    topic_query_sims.append(row['topic_query_similarity_a'])
                    max_query_sims.append(row['max_query_similarity_a'])
                    query_relevant_ratios.append(row['query_relevant_ratio_a'])
                    top3_avg_sims.append(row['top3_avg_similarity_a'])
                    # New metrics
                    relevant_topic_diversities.append(row.get('relevant_topic_diversity_a'))
                    relevance_weighted_diversities.append(row.get('relevance_weighted_diversity_a'))
                    topk_relevant_diversities.append(row.get('topk_relevant_diversity_a'))
                    n_relevant_topics_list.append(row.get('n_relevant_topics_a'))
                    relevant_diversity_ratios.append(row.get('relevant_diversity_ratio_a'))
                else:  # method_b == method
                    topic_counts.append(row['n_topics_b'])
                    diversity_semantic.append(row['diversity_semantic_b'])
                    diversity_lexical.append(row['diversity_lexical_b'])
                    npmi_coherence.append(row['npmi_coherence_b'])
                    embedding_coherence.append(row['embedding_coherence_b'])
                    relevant_concentrations.append(row['relevant_concentration_b'])
                    topic_specificities.append(row['topic_specificity_b'])
                    outlier_ratios.append(row['outlier_ratio_b'])
                    avg_topic_sizes.append(row['avg_topic_size_b'])
                    topic_query_sims.append(row['topic_query_similarity_b'])
                    max_query_sims.append(row['max_query_similarity_b'])
                    query_relevant_ratios.append(row['query_relevant_ratio_b'])
                    top3_avg_sims.append(row['top3_avg_similarity_b'])
                    # New metrics
                    relevant_topic_diversities.append(row.get('relevant_topic_diversity_b'))
                    relevance_weighted_diversities.append(row.get('relevance_weighted_diversity_b'))
                    topk_relevant_diversities.append(row.get('topk_relevant_diversity_b'))
                    n_relevant_topics_list.append(row.get('n_relevant_topics_b'))
                    relevant_diversity_ratios.append(row.get('relevant_diversity_ratio_b'))

            summary_data.append({
                "method": method,
                "n_topics": topic_counts[0] if topic_counts else 0,
                "diversity_semantic": diversity_semantic[0] if diversity_semantic else 0,
                "diversity_lexical": diversity_lexical[0] if diversity_lexical else 0,
                "npmi_coherence": npmi_coherence[0] if npmi_coherence else 0,
                "embedding_coherence": embedding_coherence[0] if embedding_coherence else 0,
                "relevant_concentration": relevant_concentrations[0] if relevant_concentrations else 0,
                "topic_specificity": topic_specificities[0] if topic_specificities else 0,
                "outlier_ratio": outlier_ratios[0] if outlier_ratios else 0,
                "document_coverage": (1 - outlier_ratios[0]) if outlier_ratios else 1.0,
                "avg_topic_size": avg_topic_sizes[0] if avg_topic_sizes else 0,
                "topic_query_similarity": topic_query_sims[0] if topic_query_sims else 0,
                "max_query_similarity": max_query_sims[0] if max_query_sims else 0,
                "query_relevant_ratio": query_relevant_ratios[0] if query_relevant_ratios else 0,
                "top3_avg_similarity": top3_avg_sims[0] if top3_avg_sims else 0,
                "n_docs": topic_results[method]['n_docs'],
                # New: Relevant topic diversity metrics
                "relevant_topic_diversity": relevant_topic_diversities[0] if relevant_topic_diversities else None,
                "relevance_weighted_diversity": relevance_weighted_diversities[0] if relevance_weighted_diversities else None,
                "topk_relevant_diversity": topk_relevant_diversities[0] if topk_relevant_diversities else None,
                "n_relevant_topics": n_relevant_topics_list[0] if n_relevant_topics_list else 0,
                "relevant_diversity_ratio": relevant_diversity_ratios[0] if relevant_diversity_ratios else None,
            })

        summary_df = pd.DataFrame(summary_data)

        # Save summary
        summary_path = os.path.join(self.results_dir, "per_method_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"✓ Saved per-method summary to {summary_path}")

        return summary_df

    def create_comprehensive_plots(
        self,
        pairwise_df: pd.DataFrame,
        per_method_df: pd.DataFrame
    ):
        """
        Create comprehensive visualization plots for all metrics

        Args:
            pairwise_df: DataFrame with pairwise metrics
            per_method_df: DataFrame with per-method summary
        """
        logger.info("\n" + "="*80)
        logger.info("Creating comprehensive visualizations...")
        logger.info("="*80)

        # Set style
        sns.set_style("whitegrid")
        methods = per_method_df['method'].tolist()
        n_methods = len(methods)

        # Plot 1: Intrinsic Quality Metrics (4x2 grid)
        fig, axes = plt.subplots(4, 2, figsize=(14, 20))
        fig.suptitle(f'Intrinsic Topic Quality Metrics - Query {self.query_id}', fontsize=16, fontweight='bold')

        quality_metrics = [
            ('npmi_coherence', 'NPMI Coherence', 'Blues_d'),
            ('embedding_coherence', 'Embedding Coherence', 'Greens_d'),
            ('diversity_semantic', 'Semantic Diversity', 'Purples_d'),
            ('diversity_lexical', 'Lexical Diversity', 'Oranges_d'),
            ('document_coverage', 'Document Coverage', 'YlGn'),
            ('topic_specificity', 'Topic Specificity (IDF)', 'Reds_d'),
            ('n_topics', 'Number of Topics', 'Greys_d'),
            ('relevant_concentration', 'Relevant Doc Concentration', 'plasma')
        ]

        for idx, (metric, title, color) in enumerate(quality_metrics):
            ax = axes[idx // 2, idx % 2]
            values = per_method_df[metric].values
            colors_list = sns.color_palette(color, n_methods)

            bars = ax.bar(methods, values, color=colors_list, edgecolor='black', linewidth=1.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.set_xlabel('Method', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}' if isinstance(value, float) and value < 10 else f'{value:.0f}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "intrinsic_quality_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved intrinsic quality metrics plot")

        # Plot 2: Query Alignment Metrics (1x3 grid)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Query Alignment Metrics - Query {self.query_id}', fontsize=16, fontweight='bold')

        query_metrics = [
            ('topic_query_similarity', 'Avg Topic-Query Similarity', 'viridis'),
            ('max_query_similarity', 'Max Topic-Query Similarity', 'plasma'),
            ('query_relevant_ratio', 'Query-Relevant Topic Ratio', 'cividis')
        ]

        for idx, (metric, title, cmap) in enumerate(query_metrics):
            ax = axes[idx]
            values = per_method_df[metric].values
            colors_list = sns.color_palette(cmap, n_methods)

            bars = ax.bar(methods, values, color=colors_list, edgecolor='black', linewidth=1.5)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.set_xlabel('Method', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.0)

            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "query_alignment_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved query alignment metrics plot")

        # Plot 3: Diversity Scatter (Semantic vs Lexical)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors_list = sns.color_palette("husl", n_methods)

        for idx, method in enumerate(methods):
            semantic = per_method_df[per_method_df['method'] == method]['diversity_semantic'].values[0]
            lexical = per_method_df[per_method_df['method'] == method]['diversity_lexical'].values[0]
            ax.scatter(semantic, lexical, s=200, color=colors_list[idx],
                      edgecolor='black', linewidth=2, label=method, alpha=0.7)
            ax.annotate(method, (semantic, lexical),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)

        ax.set_xlabel('Semantic Diversity (higher = more distinct concepts)', fontsize=12)
        ax.set_ylabel('Lexical Diversity (higher = less word overlap)', fontsize=12)
        ax.set_title(f'Topic Diversity: Semantic vs Lexical - Query {self.query_id}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        # Add quadrant lines (optional interpretation guides)
        ax.axhline(0.8, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(0.6, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, "diversity_scatter.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved diversity scatter plot")

        # Plot 3b: Relevancy vs Diversity Trade-off
        fig, ax = plt.subplots(figsize=(12, 10))

        relevancy = per_method_df['topic_query_similarity'].values
        diversity = per_method_df['diversity_semantic'].values

        # Color map for methods
        colors_list = sns.color_palette("tab10", n_methods)

        for i, method in enumerate(methods):
            ax.scatter(relevancy[i], diversity[i], s=300, color=colors_list[i],
                      label=method, alpha=0.7, edgecolors='black', linewidths=2)
            ax.annotate(method, (relevancy[i], diversity[i]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=10, fontweight='bold')

        # Add median lines for quadrants
        if len(relevancy) > 0 and len(diversity) > 0:
            relevancy_median = np.median(relevancy)
            diversity_median = np.median(diversity)

            ax.axvline(x=relevancy_median, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axhline(y=diversity_median, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

            # Quadrant labels
            ax.text(relevancy_median + 0.01, diversity_median + 0.01,
                   "High Rel.\nHigh Div.", fontsize=9, alpha=0.6)
            ax.text(relevancy_median - 0.15, diversity_median + 0.01,
                   "Low Rel.\nHigh Div.", fontsize=9, alpha=0.6)
            ax.text(relevancy_median + 0.01, diversity_median - 0.05,
                   "High Rel.\nLow Div.", fontsize=9, alpha=0.6)
            ax.text(relevancy_median - 0.15, diversity_median - 0.05,
                   "Low Rel.\nLow Div.", fontsize=9, alpha=0.6)

        ax.set_xlabel('Query Alignment (Avg Topic-Query Similarity)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Semantic Diversity', fontsize=13, fontweight='bold')
        ax.set_title(f'Relevancy vs. Diversity Trade-off - Query {self.query_id}',
                    fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, 'relevancy_vs_diversity_tradeoff.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved relevancy vs diversity trade-off plot")

        # Plot 4: Pairwise Similarity Heatmaps
        pairwise_metrics_to_plot = [
            ("topic_semantic_similarity_mean", "Topic Semantic Similarity", "Blues", (0, 1)),
            ("topic_word_overlap_mean", "Topic Word Overlap (Jaccard)", "Greens", (0, 1)),
            ("f1_@05", "Topic Matching F1 @0.5", "RdPu", (0, 1)),
            ("f1_@06", "Topic Matching F1 @0.6", "RdPu", (0, 1)),
            ("f1_@07", "Topic Matching F1 @0.7", "RdPu", (0, 1)),
            ("precision_b_@05", "Topic Precision (B) @0.5", "Purples", (0, 1)),
            ("precision_b_@06", "Topic Precision (B) @0.6", "Purples", (0, 1)),
            ("precision_b_@07", "Topic Precision (B) @0.7", "Purples", (0, 1)),
            ("recall_a_@05", "Topic Recall (A) @0.5", "Oranges", (0, 1)),
            ("recall_a_@06", "Topic Recall (A) @0.6", "Oranges", (0, 1)),
            ("recall_a_@07", "Topic Recall (A) @0.7", "Oranges", (0, 1)),
            ("npmi_coherence_diff", "NPMI Coherence Difference (A-B)", "RdBu_r", (-0.5, 0.5)),
            ("embedding_coherence_diff", "Embedding Coherence Difference (A-B)", "RdBu_r", (-0.2, 0.2)),
            ("ari", "Adjusted Rand Index (ARI)", "YlGn", (-0.5, 1)),
        ]

        for metric_key, title, cmap, vrange in pairwise_metrics_to_plot:
            # Create matrix
            matrix = np.zeros((n_methods, n_methods))

            for i, method_a in enumerate(methods):
                for j, method_b in enumerate(methods):
                    if i == j:
                        # Diagonal: self-comparison (perfect score for similarity metrics)
                        if "ari" in metric_key.lower():
                            matrix[i, j] = 1.0
                        elif "diff" in metric_key.lower():
                            matrix[i, j] = 0.0
                        else:
                            matrix[i, j] = 1.0
                    else:
                        # Find the metric value
                        row = pairwise_df[(pairwise_df['method_a'] == method_a) & (pairwise_df['method_b'] == method_b)]
                        if row.empty:
                            # Try reverse order - but use the COMPLEMENTARY metric for precision/recall
                            row = pairwise_df[(pairwise_df['method_a'] == method_b) & (pairwise_df['method_b'] == method_a)]
                            if not row.empty:
                                # Determine which metric to use from the reverse row
                                # For precision_b metrics, use recall_a from reverse (and vice versa)
                                # because precision_b(A,B) = recall_a(B,A)
                                reverse_metric_key = metric_key
                                if 'precision_b' in metric_key:
                                    # precision_b(A=i,B=j) should use recall_a(A=j,B=i)
                                    reverse_metric_key = metric_key.replace('precision_b', 'recall_a')
                                elif 'recall_a' in metric_key:
                                    # recall_a(A=i,B=j) should use precision_b(A=j,B=i)
                                    reverse_metric_key = metric_key.replace('recall_a', 'precision_b')

                                if reverse_metric_key in row.columns:
                                    value = row[reverse_metric_key].values[0]
                                    matrix[i, j] = value if value is not None and not np.isnan(value) else 0.0
                                elif metric_key in row.columns:
                                    # Fallback for symmetric metrics (F1, similarity, etc.)
                                    value = row[metric_key].values[0]
                                    matrix[i, j] = value if value is not None and not np.isnan(value) else 0.0
                        elif metric_key in row.columns:
                            value = row[metric_key].values[0]
                            matrix[i, j] = value if value is not None and not np.isnan(value) else 0.0

            # Create heatmap
            plt.figure(figsize=(10, 8))

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                xticklabels=methods,
                yticklabels=methods,
                vmin=vrange[0],
                vmax=vrange[1],
                cbar_kws={'label': title},
                square=True,
                linewidths=0.5
            )

            plt.title(f"{title}\n(Method A vs Method B) - Query {self.query_id}", fontsize=14, fontweight='bold')
            plt.xlabel("Method B", fontsize=12)
            plt.ylabel("Method A", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Save plot
            filename = metric_key.replace('@', '_at_').replace('_', '-') + "_heatmap.png"
            plot_path = os.path.join(self.plots_dir, filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Saved {title} heatmap")

        logger.info(f"\n✓ All plots saved to {self.plots_dir}")

    def run_full_evaluation(self):
        """Run full end-to-end evaluation pipeline with pairwise comparisons"""

        logger.info("="*80)
        logger.info(f"Starting End-to-End Evaluation for Query {self.query_id}")
        logger.info(f"Sample size: {self.sample_size}")
        logger.info("="*80)

        # Step 1: Generate samples
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Generating Document Samples")
        logger.info("="*80)

        samples = {}
        samples["random_uniform"] = self.sample_random_uniform()
        samples["direct_retrieval"] = self.sample_direct_retrieval()
        samples["direct_retrieval_mmr"] = self.sample_direct_retrieval_mmr()
        samples["query_expansion"] = self.sample_query_expansion()
        samples["keyword_search"] = self.sample_keyword_search()
        samples["retrieval_random"] = self.sample_retrieval_random()

        # Step 2: Run topic modeling
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Running Topic Modeling")
        logger.info("="*80)

        topic_results = {}
        for method_name, sample in samples.items():
            try:
                topic_results[method_name] = self.run_topic_modeling(sample)
            except Exception as e:
                logger.error(f"Error in topic modeling for {method_name}: {e}")
                topic_results[method_name] = None

        # Remove failed methods
        topic_results = {k: v for k, v in topic_results.items() if v is not None}
        samples = {k: v for k, v in samples.items() if k in topic_results}

        if len(topic_results) < 2:
            logger.error("Need at least 2 successful topic models for pairwise comparison")
            return None

        # Step 3: Save topic summaries
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Saving Topic Summaries")
        logger.info("="*80)

        try:
            self._save_topic_summaries(topic_results, samples)
        except Exception as e:
            logger.error(f"Error saving topic summaries: {e}")
            logger.warning("Continuing with evaluation despite topic summary error")

        # Step 4: Run pairwise comparisons
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Running Pairwise Comparisons")
        logger.info("="*80)

        pairwise_df = self.run_all_pairwise_comparisons(topic_results, samples)

        if pairwise_df.empty:
            logger.error("No pairwise comparisons completed. Exiting.")
            return None

        # Step 5: Create per-method summary
        logger.info("\n" + "="*80)
        logger.info("STEP 5: Creating Per-Method Summary")
        logger.info("="*80)

        per_method_df = self.create_per_method_summary(pairwise_df, topic_results)

        # Step 6: Create comprehensive visualizations
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Creating Comprehensive Visualizations")
        logger.info("="*80)

        self.create_comprehensive_plots(pairwise_df, per_method_df)

        # Step 7: Print summary
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)

        print("\nPer-Method Summary:")
        print(per_method_df.to_string(index=False))

        logger.info("\n" + "="*80)
        logger.info("Evaluation Complete!")
        logger.info(f"All results saved to: {self.output_dir}")
        logger.info("="*80)

        return {
            "samples": samples,
            "topic_results": topic_results,
            "pairwise_metrics": pairwise_df,
            "per_method_summary": per_method_df
        }


def create_cross_query_intrinsic_plots(
    all_per_method: pd.DataFrame,
    output_dir: str
):
    """
    Create cross-query intrinsic metric visualizations.
    Consolidates functionality from visualize_intrinsic_metrics.py

    Args:
        all_per_method: DataFrame with per-method metrics across all queries
        output_dir: Directory to save plots
    """
    logger.info("Creating cross-query intrinsic metric visualizations...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define method colors for consistency
    method_colors = {
        "random_uniform": "#e74c3c",      # Red
        "direct_retrieval": "#3498db",    # Blue
        "query_expansion": "#2ecc71",     # Green
        "keyword_search": "#f39c12"       # Orange
    }

    # Get available methods
    methods = sorted(all_per_method['method'].unique())
    queries = sorted(all_per_method['query_id'].unique())

    # 1. DIVERSITY ANALYSIS
    if 'diversity_semantic' in all_per_method.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Topic Diversity Across Methods and Queries', fontsize=16, fontweight='bold')

        # Plot 1: Bar chart by query
        ax = axes[0, 0]
        pivot = all_per_method.pivot(index='query_id', columns='method', values='diversity_semantic')
        pivot.plot(kind='bar', ax=ax, color=[method_colors.get(m, '#95a5a6') for m in pivot.columns], width=0.8)
        ax.set_title('Semantic Diversity by Query', fontsize=14, fontweight='bold')
        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel('Diversity (higher = more diverse)', fontsize=12)
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Method comparison (averaged across queries)
        ax = axes[0, 1]
        method_avg = all_per_method.groupby('method')['diversity_semantic'].agg(['mean', 'std']).reset_index()
        method_avg = method_avg.sort_values('mean', ascending=False)
        x_pos = np.arange(len(method_avg))
        bars = ax.bar(x_pos, method_avg['mean'], yerr=method_avg['std'],
                      color=[method_colors.get(m, '#95a5a6') for m in method_avg['method']],
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_avg['method'], fontsize=11)
        ax.set_title('Average Diversity by Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('Diversity (mean ± std)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        for i, (bar, mean_val) in enumerate(zip(bars, method_avg['mean'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Plot 3: Heatmap
        ax = axes[1, 0]
        heatmap_data = all_per_method.pivot(index='query_id', columns='method', values='diversity_semantic')
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': 'Diversity'})
        ax.set_title('Diversity Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Query', fontsize=12)

        # Plot 4: Line plot (trend across queries)
        ax = axes[1, 1]
        for method in methods:
            method_data = all_per_method[all_per_method['method'] == method].sort_values('query_id')
            ax.plot(method_data['query_id'], method_data['diversity_semantic'],
                    marker='o', linewidth=2, markersize=8, label=method,
                    color=method_colors.get(method, '#95a5a6'))
        ax.set_title('Diversity Trend Across Queries', fontsize=14, fontweight='bold')
        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel('Diversity', fontsize=12)
        ax.legend(title='Method', loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_query_diversity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved diversity analysis")

    # 2. DOCUMENT COVERAGE (1 - outlier_ratio)
    if 'document_coverage' in all_per_method.columns:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Document Coverage Across Methods and Queries', fontsize=16, fontweight='bold')

        # Plot 1: Bar chart by query
        ax = axes[0, 0]
        pivot = all_per_method.pivot(index='query_id', columns='method', values='document_coverage')
        pivot.plot(kind='bar', ax=ax, color=[method_colors.get(m, '#95a5a6') for m in pivot.columns], width=0.8)
        ax.set_title('Document Coverage by Query', fontsize=14, fontweight='bold')
        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel('Coverage (higher = better)', fontsize=12)
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Method comparison
        ax = axes[0, 1]
        method_avg = all_per_method.groupby('method')['document_coverage'].agg(['mean', 'std']).reset_index()
        method_avg = method_avg.sort_values('mean', ascending=False)
        x_pos = np.arange(len(method_avg))
        bars = ax.bar(x_pos, method_avg['mean'], yerr=method_avg['std'],
                      color=[method_colors.get(m, '#95a5a6') for m in method_avg['method']],
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_avg['method'], fontsize=11)
        ax.set_title('Average Coverage by Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coverage (mean ± std)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        for i, (bar, mean_val) in enumerate(zip(bars, method_avg['mean'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Plot 3: Heatmap
        ax = axes[1, 0]
        heatmap_data = all_per_method.pivot(index='query_id', columns='method', values='document_coverage')
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGn', ax=ax,
                    cbar_kws={'label': 'Coverage'})
        ax.set_title('Coverage Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Query', fontsize=12)

        # Plot 4: Line plot
        ax = axes[1, 1]
        for method in methods:
            method_data = all_per_method[all_per_method['method'] == method].sort_values('query_id')
            ax.plot(method_data['query_id'], method_data['document_coverage'],
                    marker='o', linewidth=2, markersize=8, label=method,
                    color=method_colors.get(method, '#95a5a6'))
        ax.set_title('Coverage Trend Across Queries', fontsize=14, fontweight='bold')
        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel('Coverage', fontsize=12)
        ax.legend(title='Method', loc='best')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_query_coverage_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved coverage analysis")

    # 3. TOPIC COUNT ANALYSIS
    if 'n_topics' in all_per_method.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Number of Topics Discovered', fontsize=16, fontweight='bold')

        # Plot 1: Bar chart by query
        ax = axes[0]
        pivot = all_per_method.pivot(index='query_id', columns='method', values='n_topics')
        pivot.plot(kind='bar', ax=ax, color=[method_colors.get(m, '#95a5a6') for m in pivot.columns], width=0.8)
        ax.set_title('Topic Count by Query', fontsize=14, fontweight='bold')
        ax.set_xlabel('Query', fontsize=12)
        ax.set_ylabel('Number of Topics', fontsize=12)
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Plot 2: Heatmap
        ax = axes[1]
        heatmap_data = all_per_method.pivot(index='query_id', columns='method', values='n_topics')
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                    cbar_kws={'label': 'Number of Topics'})
        ax.set_title('Topic Count Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Query', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_query_topic_count.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved topic count analysis")

    # 4. COMBINED SUMMARY
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Intrinsic Metrics Summary: All Methods & Queries', fontsize=18, fontweight='bold')

    # Plot 1: Diversity comparison
    if 'diversity_semantic' in all_per_method.columns:
        ax = axes[0, 0]
        pivot = all_per_method.pivot(index='method', columns='query_id', values='diversity_semantic')
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Semantic Diversity by Method', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Diversity', fontsize=12)
        ax.legend(title='Query', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Coverage comparison
    if 'document_coverage' in all_per_method.columns:
        ax = axes[0, 1]
        pivot = all_per_method.pivot(index='method', columns='query_id', values='document_coverage')
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Document Coverage by Method', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Coverage', fontsize=12)
        ax.legend(title='Query', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Topic count comparison
    if 'n_topics' in all_per_method.columns:
        ax = axes[1, 0]
        pivot = all_per_method.pivot(index='method', columns='query_id', values='n_topics')
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Number of Topics by Method', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Number of Topics', fontsize=12)
        ax.legend(title='Query', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: Coherence comparison
    if 'npmi_coherence' in all_per_method.columns:
        ax = axes[1, 1]
        pivot = all_per_method.pivot(index='method', columns='query_id', values='npmi_coherence')
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('NPMI Coherence by Method', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('NPMI Coherence', fontsize=12)
        ax.legend(title='Query', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_query_intrinsic_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved intrinsic metrics summary")

    logger.info(f"✓ All cross-query intrinsic plots saved to {output_dir}")


def aggregate_cross_query_results(
    results_base_dir: str,
    query_ids: List[str],
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Aggregate results across all queries to understand overall method performance.

    Args:
        results_base_dir: Base directory containing query result folders
        query_ids: List of query IDs to aggregate
        output_dir: Directory to save aggregated results (default: results_base_dir/aggregate_results)

    Returns:
        Dictionary containing aggregated metrics
    """
    logger.info("="*80)
    logger.info("Aggregating Results Across Queries")
    logger.info("="*80)

    if output_dir is None:
        output_dir = os.path.join(results_base_dir, "aggregate_results")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # Collect all per-method summaries
    per_method_data = []
    pairwise_data = []

    logger.info(f"Loading results from {len(query_ids)} queries...")

    for query_id in query_ids:
        query_dir = os.path.join(results_base_dir, f"query_{query_id}", "results")

        # Load per-method summary
        per_method_csv = os.path.join(query_dir, "per_method_summary.csv")
        if os.path.exists(per_method_csv):
            df = pd.read_csv(per_method_csv)
            df['query_id'] = query_id
            per_method_data.append(df)
        else:
            logger.warning(f"Missing per_method_summary.csv for query {query_id}")

        # Load pairwise metrics
        pairwise_json = os.path.join(query_dir, "pairwise_metrics.json")
        if os.path.exists(pairwise_json):
            with open(pairwise_json, 'r') as f:
                pairwise = json.load(f)
                for item in pairwise:
                    item['query_id'] = query_id
                pairwise_data.extend(pairwise)
        else:
            logger.warning(f"Missing pairwise_metrics.json for query {query_id}")

    if not per_method_data:
        logger.error("No per-method data found!")
        return None

    # Combine all data
    all_per_method = pd.concat(per_method_data, ignore_index=True)
    all_pairwise = pd.DataFrame(pairwise_data)

    logger.info(f"Loaded data from {len(per_method_data)} queries")
    logger.info(f"Total per-method records: {len(all_per_method)}")
    logger.info(f"Total pairwise records: {len(all_pairwise)}")

    # ===== PER-METHOD AGGREGATION =====
    logger.info("\nAggregating per-method metrics...")

    # Metrics to aggregate with mean/std/median/min/max
    aggregatable_metrics = [
        'diversity_semantic',
        'diversity_lexical',
        'npmi_coherence',
        'embedding_coherence',
        'relevant_concentration',
        'topic_specificity',
        'outlier_ratio',
        'document_coverage',
        'topic_query_similarity',
        'max_query_similarity',
        'query_relevant_ratio',
        'top3_avg_similarity',
        # Relevant topic diversity metrics
        'relevant_topic_diversity',
        'relevance_weighted_diversity',
        'topk_relevant_diversity',
        'relevant_diversity_ratio'
    ]

    # Metrics to report but not aggregate (query-dependent)
    descriptive_metrics = ['n_topics', 'avg_topic_size', 'n_relevant_topics']

    per_method_aggregates = []

    for method in all_per_method['method'].unique():
        method_data = all_per_method[all_per_method['method'] == method]

        agg_record = {'method': method, 'n_queries': len(method_data)}

        # Aggregate main metrics
        for metric in aggregatable_metrics:
            if metric in method_data.columns:
                values = method_data[metric].dropna()
                if len(values) > 0:
                    agg_record[f'{metric}_mean'] = values.mean()
                    agg_record[f'{metric}_std'] = values.std()
                    agg_record[f'{metric}_median'] = values.median()
                    agg_record[f'{metric}_min'] = values.min()
                    agg_record[f'{metric}_max'] = values.max()

        # Report descriptive metrics
        for metric in descriptive_metrics:
            if metric in method_data.columns:
                values = method_data[metric].dropna()
                if len(values) > 0:
                    agg_record[f'{metric}_mean'] = values.mean()
                    agg_record[f'{metric}_std'] = values.std()

        per_method_aggregates.append(agg_record)

    per_method_agg_df = pd.DataFrame(per_method_aggregates)

    # Save per-method aggregates
    per_method_csv_path = os.path.join(output_dir, "per_method_aggregates.csv")
    per_method_json_path = os.path.join(output_dir, "per_method_aggregates.json")

    per_method_agg_df.to_csv(per_method_csv_path, index=False)
    per_method_agg_df.to_json(per_method_json_path, orient='records', indent=2)

    logger.info(f"Saved per-method aggregates to {per_method_csv_path}")

    # ===== PAIRWISE AGGREGATION =====
    logger.info("\nAggregating pairwise metrics...")

    # Metrics to aggregate
    pairwise_aggregatable = [
        'diversity_semantic_diff',
        'diversity_lexical_diff',
        'npmi_coherence_diff',
        'embedding_coherence_diff',
        'topic_query_similarity_diff',
        'outlier_ratio_diff',
        'topic_word_overlap_mean',
        'topic_semantic_similarity_mean',
        'precision_b_@05',
        'recall_a_@05',
        'f1_@05',
        'precision_b_@06',
        'recall_a_@06',
        'f1_@06',
        'precision_b_@07',
        'recall_a_@07',
        'f1_@07',
        'ari',
        'nmi'
    ]

    pairwise_aggregates = []

    # Group by method pair
    if 'method_a' in all_pairwise.columns and 'method_b' in all_pairwise.columns:
        for (method_a, method_b), group in all_pairwise.groupby(['method_a', 'method_b']):
            agg_record = {
                'method_a': method_a,
                'method_b': method_b,
                'n_queries': len(group)
            }

            for metric in pairwise_aggregatable:
                if metric in group.columns:
                    values = group[metric].dropna()
                    if len(values) > 0:
                        agg_record[f'{metric}_mean'] = values.mean()
                        agg_record[f'{metric}_std'] = values.std()
                        agg_record[f'{metric}_median'] = values.median()
                        agg_record[f'{metric}_min'] = values.min()
                        agg_record[f'{metric}_max'] = values.max()

            pairwise_aggregates.append(agg_record)

    pairwise_agg_df = pd.DataFrame(pairwise_aggregates)

    # Save pairwise aggregates
    pairwise_csv_path = os.path.join(output_dir, "pairwise_aggregates.csv")
    pairwise_json_path = os.path.join(output_dir, "pairwise_aggregates.json")

    pairwise_agg_df.to_csv(pairwise_csv_path, index=False)
    pairwise_agg_df.to_json(pairwise_json_path, orient='records', indent=2)

    logger.info(f"Saved pairwise aggregates to {pairwise_csv_path}")

    # ===== VISUALIZATIONS =====
    logger.info("\nGenerating aggregate visualizations...")

    _generate_aggregate_visualizations(
        all_per_method,
        all_pairwise,
        per_method_agg_df,
        pairwise_agg_df,
        output_dir
    )

    logger.info("="*80)
    logger.info("Cross-Query Aggregation Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)

    return {
        'per_method_aggregates': per_method_agg_df,
        'pairwise_aggregates': pairwise_agg_df,
        'raw_per_method': all_per_method,
        'raw_pairwise': all_pairwise
    }


def _generate_aggregate_visualizations(
    all_per_method: pd.DataFrame,
    all_pairwise: pd.DataFrame,
    per_method_agg: pd.DataFrame,
    pairwise_agg: pd.DataFrame,
    output_dir: str
):
    """Generate visualizations for aggregated results"""

    plots_dir = os.path.join(output_dir, "plots")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # 1. Method Quality Comparison (Boxplots)
    logger.info("Creating method quality comparison plots...")

    quality_metrics = [
        ('npmi_coherence', 'NPMI Coherence'),
        ('embedding_coherence', 'Embedding Coherence'),
        ('diversity_semantic', 'Semantic Diversity'),
        ('diversity_lexical', 'Lexical Diversity'),
        ('document_coverage', 'Document Coverage'),
        ('outlier_ratio', 'Outlier Ratio')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(quality_metrics):
        if metric in all_per_method.columns:
            sns.boxplot(data=all_per_method, x='method', y=metric, ax=axes[idx])
            axes[idx].set_title(f'{title} Across Queries', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Method', fontsize=10)
            axes[idx].set_ylabel(title, fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "method_quality_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Method Alignment Comparison (Boxplots)
    logger.info("Creating method alignment comparison plots...")

    alignment_metrics = [
        ('topic_query_similarity', 'Avg Topic-Query Similarity'),
        ('max_query_similarity', 'Max Topic-Query Similarity'),
        ('query_relevant_ratio', 'Query-Relevant Ratio'),
        ('top3_avg_similarity', 'Top-3 Avg Similarity')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, title) in enumerate(alignment_metrics):
        if metric in all_per_method.columns:
            sns.boxplot(data=all_per_method, x='method', y=metric, ax=axes[idx])
            axes[idx].set_title(f'{title} Across Queries', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Method', fontsize=10)
            axes[idx].set_ylabel(title, fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "method_alignment_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Pairwise Coherence Difference Heatmap
    logger.info("Creating pairwise difference heatmaps...")

    pairwise_heatmap_metrics = [
        # Difference metrics (asymmetric - negate for lower triangle)
        ('npmi_coherence_diff_mean', 'Avg NPMI Coherence Difference', 'RdYlGn'),
        ('embedding_coherence_diff_mean', 'Avg Embedding Coherence Difference', 'RdYlGn'),
        ('diversity_semantic_diff_mean', 'Avg Semantic Diversity Difference', 'RdYlGn'),
        ('topic_query_similarity_diff_mean', 'Avg Topic-Query Similarity Difference', 'RdYlGn'),
        # Similarity/overlap metrics (symmetric - mirror directly)
        ('topic_word_overlap_mean_mean', 'Avg Topic Word Overlap', 'YlOrRd'),
        ('topic_semantic_similarity_mean_mean', 'Avg Topic Semantic Similarity', 'YlOrRd'),
        # Precision metrics (% of B's topics matched) - use reverse row for lower triangle
        ('precision_b_@05_mean', 'Topic Match Rate (% of B\'s topics matched) @ 0.5', 'YlGnBu'),
        ('precision_b_@06_mean', 'Topic Match Rate (% of B\'s topics matched) @ 0.6', 'YlGnBu'),
        ('precision_b_@07_mean', 'Topic Match Rate (% of B\'s topics matched) @ 0.7', 'YlGnBu'),
        # Recall metrics (% of A's topics matched) - use reverse row for lower triangle
        ('recall_a_@05_mean', 'Topic Match Rate (% of A\'s topics matched) @ 0.5', 'YlGnBu'),
        ('recall_a_@06_mean', 'Topic Match Rate (% of A\'s topics matched) @ 0.6', 'YlGnBu'),
        ('recall_a_@07_mean', 'Topic Match Rate (% of A\'s topics matched) @ 0.7', 'YlGnBu'),
        # F1 metrics (harmonic mean - symmetric)
        ('f1_@05_mean', 'Topic Match F1 @ 0.5', 'YlGnBu'),
        ('f1_@06_mean', 'Topic Match F1 @ 0.6', 'YlGnBu'),
        ('f1_@07_mean', 'Topic Match F1 @ 0.7', 'YlGnBu'),
    ]

    # Build a lookup dict for quick access to pairwise rows: (method_a, method_b) -> row
    pairwise_lookup = {}
    for _, row in pairwise_agg.iterrows():
        key = (row['method_a'], row['method_b'])
        pairwise_lookup[key] = row

    for metric, title, cmap in pairwise_heatmap_metrics:
        if metric in pairwise_agg.columns:
            # Create pivot table
            methods = sorted(set(pairwise_agg['method_a'].tolist() + pairwise_agg['method_b'].tolist()))
            pivot_data = np.zeros((len(methods), len(methods)))
            pivot_data[:] = np.nan

            method_to_idx = {m: i for i, m in enumerate(methods)}

            # Determine metric type for proper handling
            is_diff_metric = 'diff' in metric
            is_precision_metric = 'precision_b' in metric
            is_recall_metric = 'recall_a' in metric
            is_f1_metric = 'f1_@' in metric
            is_symmetric = not is_diff_metric and not is_precision_metric and not is_recall_metric

            # Fill the matrix
            for _, row in pairwise_agg.iterrows():
                i = method_to_idx[row['method_a']]
                j = method_to_idx[row['method_b']]
                value = row[metric]
                if pd.notna(value):
                    pivot_data[i, j] = value

                    # Handle lower triangle based on metric type
                    if is_diff_metric:
                        # Difference metrics: negate for reverse direction
                        pivot_data[j, i] = -value
                    elif is_symmetric or is_f1_metric:
                        # Symmetric metrics (overlap, similarity, F1): mirror directly
                        pivot_data[j, i] = value
                    elif is_precision_metric:
                        # For precision_b(A,B), the reverse precision_b(B,A) = recall_a(A,B)
                        # So lower triangle [j,i] gets recall_a from the same row
                        complementary_metric = metric.replace('precision_b', 'recall_a')
                        if complementary_metric in row:
                            complementary_value = row[complementary_metric]
                            if pd.notna(complementary_value):
                                pivot_data[j, i] = complementary_value
                    elif is_recall_metric:
                        # For recall_a(A,B), the reverse recall_a(B,A) = precision_b(A,B)
                        # So lower triangle [j,i] gets precision_b from the same row
                        complementary_metric = metric.replace('recall_a', 'precision_b')
                        if complementary_metric in row:
                            complementary_value = row[complementary_metric]
                            if pd.notna(complementary_value):
                                pivot_data[j, i] = complementary_value

            # Set diagonal to 1.0 for non-diff metrics (self-comparison is perfect)
            if not is_diff_metric:
                for i in range(len(methods)):
                    pivot_data[i, i] = 1.0

            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))

            # Determine color scale
            if is_diff_metric:
                vmax = np.nanmax(np.abs(pivot_data))
                vmin = -vmax
                center = 0
            else:
                vmin = 0.0  # These metrics range from 0 to 1
                vmax = 1.0
                center = None

            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                xticklabels=methods,
                yticklabels=methods,
                ax=ax,
                cbar_kws={'label': title}
            )

            ax.set_title(f'{title}\n(Averaged Across Queries)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Method B', fontsize=11)
            ax.set_ylabel('Method A', fontsize=11)

            plt.tight_layout()

            # Create safe filename
            safe_metric = metric.replace('_mean', '').replace('@', '-at-').replace('_', '-')
            plt.savefig(os.path.join(plots_dir, f"pairwise_{safe_metric}_heatmap.png"), dpi=300, bbox_inches='tight')
            plt.close()

    logger.info(f"Saved all aggregate pairwise heatmaps to {plots_dir}")

    # Create cross-query intrinsic metric visualizations (consolidates visualize_intrinsic_metrics.py)
    logger.info("\nCreating cross-query intrinsic metric visualizations...")
    create_cross_query_intrinsic_plots(all_per_method, plots_dir)

    # Create aggregate scatter plots for relevancy vs diversity trade-off
    logger.info("\nCreating aggregate scatter plots...")
    plot_aggregate_scatter_plots(all_per_method, plots_dir)

    # Create standalone aggregate bar charts for key metrics
    logger.info("\nCreating standalone aggregate bar charts...")
    plot_aggregate_bar_charts(all_per_method, plots_dir)

    # Run statistical significance tests
    logger.info("\nRunning statistical significance tests...")
    stats_dir = os.path.join(output_dir, "statistical_tests")
    run_aggregate_statistical_analysis(all_per_method, stats_dir)

    # Return aggregated data
    return {
        "per_method": all_per_method,
        "pairwise": pairwise_agg
    }


def draw_error_ellipse(ax, x_mean, x_std, y_mean, y_std, color, alpha=0.2, **kwargs):
    """
    Draw a 2D error ellipse representing ±1 standard deviation in both X and Y.

    Args:
        ax: Matplotlib axes object
        x_mean: Mean X value
        x_std: Standard deviation in X
        y_mean: Mean Y value
        y_std: Standard deviation in Y
        color: Color of the ellipse
        alpha: Transparency
        **kwargs: Additional arguments for Ellipse patch
    """
    from matplotlib.patches import Ellipse

    ellipse = Ellipse(
        xy=(x_mean, y_mean),
        width=2 * x_std,  # ±1 std
        height=2 * y_std,  # ±1 std
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=1.5,
        **kwargs
    )
    ax.add_patch(ellipse)


def draw_convex_hull(ax, points, color, alpha=0.15, linewidth=2):
    """
    Draw a convex hull around a set of 2D points.

    Args:
        ax: Matplotlib axes object
        points: Nx2 numpy array of (x, y) coordinates
        color: Color of the hull outline
        alpha: Fill transparency
        linewidth: Line width for hull boundary
    """
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon

    if len(points) < 3:
        # Need at least 3 points for a hull
        return

    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        polygon = Polygon(
            hull_points,
            facecolor=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=linewidth
        )
        ax.add_patch(polygon)
    except Exception as e:
        logger.warning(f"Could not compute convex hull: {e}")


def plot_aggregate_scatter_plots(all_per_method: pd.DataFrame, plots_dir: str):
    """
    Create aggregate scatter plots across all queries.

    Generates 4 plots:
    1. Relevancy vs Diversity - Mean with error ellipses
    2. Relevancy vs Diversity - All points with convex hulls
    3. Relevancy vs Diversity - Combined (all points + means)
    4. Diversity Scatter (Semantic vs Lexical) - Mean with error ellipses

    Args:
        all_per_method: DataFrame with all per-method data across queries
        plots_dir: Directory to save plots
    """
    logger.info("Generating aggregate scatter plots...")

    # Compute aggregated statistics per method
    methods = sorted(all_per_method['method'].unique())
    n_methods = len(methods)

    # Aggregate stats
    agg_stats = all_per_method.groupby('method').agg({
        'topic_query_similarity': ['mean', 'std'],
        'diversity_semantic': ['mean', 'std'],
        'diversity_lexical': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    agg_stats.columns = ['_'.join(col).strip('_') for col in agg_stats.columns.values]

    # Color palette
    colors_list = sns.color_palette("tab10", n_methods)
    method_colors = {method: colors_list[i] for i, method in enumerate(methods)}

    # ===== Plot 1: Relevancy vs Diversity - Mean with Error Ellipses =====
    logger.info("Creating relevancy vs diversity (mean + ellipses)...")

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, method in enumerate(methods):
        stats = agg_stats[agg_stats['method'] == method].iloc[0]

        x_mean = stats['topic_query_similarity_mean']
        x_std = stats['topic_query_similarity_std']
        y_mean = stats['diversity_semantic_mean']
        y_std = stats['diversity_semantic_std']

        color = method_colors[method]

        # Draw error ellipse first (background)
        draw_error_ellipse(ax, x_mean, x_std, y_mean, y_std, color, alpha=0.2)

        # Draw mean point on top
        ax.scatter(x_mean, y_mean, s=400, color=color, label=method,
                  alpha=0.8, edgecolors='black', linewidths=2, zorder=10)

        # Annotate
        ax.annotate(method, (x_mean, y_mean),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold', zorder=11)

    # Add median lines for quadrants
    all_x_means = agg_stats['topic_query_similarity_mean'].values
    all_y_means = agg_stats['diversity_semantic_mean'].values

    x_median = np.median(all_x_means)
    y_median = np.median(all_y_means)

    ax.axvline(x=x_median, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=y_median, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # Quadrant labels
    ax.text(x_median + 0.01, y_median + 0.01,
           "High Rel.\nHigh Div.", fontsize=9, alpha=0.6)
    ax.text(x_median - 0.08, y_median + 0.01,
           "Low Rel.\nHigh Div.", fontsize=9, alpha=0.6)
    ax.text(x_median + 0.01, y_median - 0.02,
           "High Rel.\nLow Div.", fontsize=9, alpha=0.6)
    ax.text(x_median - 0.08, y_median - 0.02,
           "Low Rel.\nLow Div.", fontsize=9, alpha=0.6)

    ax.set_xlabel('Query Alignment (Avg Topic-Query Similarity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Semantic Diversity', fontsize=13, fontweight='bold')
    ax.set_title('Relevancy vs. Diversity Trade-off - Aggregated Across Queries\n(Mean ± 1 Std)',
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'relevancy_vs_diversity_mean.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved {plot_path}")

    # ===== Plot 2: Relevancy vs Diversity - All Points with Convex Hulls =====
    logger.info("Creating relevancy vs diversity (all points + convex hulls)...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot convex hulls first (background)
    for i, method in enumerate(methods):
        method_data = all_per_method[all_per_method['method'] == method]
        points = method_data[['topic_query_similarity', 'diversity_semantic']].values

        color = method_colors[method]
        draw_convex_hull(ax, points, color, alpha=0.15, linewidth=2)

    # Plot all individual points
    for i, method in enumerate(methods):
        method_data = all_per_method[all_per_method['method'] == method]

        x_vals = method_data['topic_query_similarity'].values
        y_vals = method_data['diversity_semantic'].values

        color = method_colors[method]

        ax.scatter(x_vals, y_vals, s=100, color=color, label=method,
                  alpha=0.5, edgecolors='white', linewidths=0.5)

    # Add median lines
    all_x = all_per_method['topic_query_similarity'].values
    all_y = all_per_method['diversity_semantic'].values

    x_median_all = np.median(all_x)
    y_median_all = np.median(all_y)

    ax.axvline(x=x_median_all, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=y_median_all, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Query Alignment (Avg Topic-Query Similarity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Semantic Diversity', fontsize=13, fontweight='bold')
    ax.set_title('Relevancy vs. Diversity Trade-off - All Queries\n(Individual Points + Convex Hulls)',
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'relevancy_vs_diversity_all_points.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved {plot_path}")

    # ===== Plot 3: Relevancy vs Diversity - Combined (All + Means) =====
    logger.info("Creating relevancy vs diversity (combined)...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # 1. Plot all individual points (semi-transparent, small)
    for i, method in enumerate(methods):
        method_data = all_per_method[all_per_method['method'] == method]

        x_vals = method_data['topic_query_similarity'].values
        y_vals = method_data['diversity_semantic'].values

        color = method_colors[method]

        ax.scatter(x_vals, y_vals, s=60, color=color,
                  alpha=0.25, edgecolors='none', zorder=1)

    # 2. Plot convex hulls
    for i, method in enumerate(methods):
        method_data = all_per_method[all_per_method['method'] == method]
        points = method_data[['topic_query_similarity', 'diversity_semantic']].values

        color = method_colors[method]
        draw_convex_hull(ax, points, color, alpha=0.1, linewidth=1.5)

    # 3. Plot error ellipses
    for i, method in enumerate(methods):
        stats = agg_stats[agg_stats['method'] == method].iloc[0]

        x_mean = stats['topic_query_similarity_mean']
        x_std = stats['topic_query_similarity_std']
        y_mean = stats['diversity_semantic_mean']
        y_std = stats['diversity_semantic_std']

        color = method_colors[method]
        draw_error_ellipse(ax, x_mean, x_std, y_mean, y_std, color, alpha=0.25)

    # 4. Plot mean points on top
    for i, method in enumerate(methods):
        stats = agg_stats[agg_stats['method'] == method].iloc[0]

        x_mean = stats['topic_query_similarity_mean']
        y_mean = stats['diversity_semantic_mean']

        color = method_colors[method]

        ax.scatter(x_mean, y_mean, s=400, color=color, label=method,
                  alpha=0.9, edgecolors='black', linewidths=2.5, zorder=10)

        ax.annotate(method, (x_mean, y_mean),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold', zorder=11)

    # Add median lines
    ax.axvline(x=x_median, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=y_median, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Query Alignment (Avg Topic-Query Similarity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Semantic Diversity', fontsize=13, fontweight='bold')
    ax.set_title('Relevancy vs. Diversity Trade-off - Combined View\n(All Points + Mean ± 1 Std + Convex Hulls)',
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'relevancy_vs_diversity_combined.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved {plot_path}")

    # ===== Plot 4: Diversity Scatter (Semantic vs Lexical) - Mean with Ellipses =====
    logger.info("Creating diversity scatter (semantic vs lexical)...")

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, method in enumerate(methods):
        stats = agg_stats[agg_stats['method'] == method].iloc[0]

        x_mean = stats['diversity_lexical_mean']
        x_std = stats['diversity_lexical_std']
        y_mean = stats['diversity_semantic_mean']
        y_std = stats['diversity_semantic_std']

        color = method_colors[method]

        # Draw error ellipse
        draw_error_ellipse(ax, x_mean, x_std, y_mean, y_std, color, alpha=0.2)

        # Draw mean point
        ax.scatter(x_mean, y_mean, s=400, color=color, label=method,
                  alpha=0.8, edgecolors='black', linewidths=2, zorder=10)

        # Annotate
        ax.annotate(method, (x_mean, y_mean),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold', zorder=11)

    # Add median lines
    all_x_lex = agg_stats['diversity_lexical_mean'].values
    all_y_sem = agg_stats['diversity_semantic_mean'].values

    x_median_lex = np.median(all_x_lex)
    y_median_sem = np.median(all_y_sem)

    ax.axvline(x=x_median_lex, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=y_median_sem, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Lexical Diversity', fontsize=13, fontweight='bold')
    ax.set_ylabel('Semantic Diversity', fontsize=13, fontweight='bold')
    ax.set_title('Diversity Balance - Aggregated Across Queries\n(Mean ± 1 Std)',
                fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'diversity_scatter_mean.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved {plot_path}")

    logger.info("✓ All aggregate scatter plots created successfully")


def plot_aggregate_bar_charts(all_per_method: pd.DataFrame, plots_dir: str):
    """
    Create standalone aggregate bar charts for key metrics.

    Generates 6 plots (3 metrics × 2 error types):
    - Topic Specificity (std and 95% CI versions)
    - Average Topic-Query Similarity (std and 95% CI versions)
    - Semantic Diversity (std and 95% CI versions)

    Args:
        all_per_method: DataFrame with all per-method data across queries
        plots_dir: Directory to save plots
    """
    logger.info("Generating standalone aggregate bar charts...")

    # Compute aggregated statistics per method
    methods = sorted(all_per_method['method'].unique())
    n_methods = len(methods)

    # Define metrics to plot
    metrics_config = [
        {
            'column': 'topic_specificity',
            'title': 'Topic Specificity (IDF-based)',
            'ylabel': 'Specificity (higher = more specific terms)',
            'filename': 'aggregate_topic_specificity'
        },
        {
            'column': 'topic_query_similarity',
            'title': 'Average Topic-Query Similarity',
            'ylabel': 'Similarity (0-1)',
            'filename': 'aggregate_topic_query_similarity'
        },
        {
            'column': 'diversity_semantic',
            'title': 'Semantic Diversity',
            'ylabel': 'Diversity (higher = more distinct topics)',
            'filename': 'aggregate_semantic_diversity'
        },
        {
            'column': 'relevant_topic_diversity',
            'title': 'Relevant Topic Diversity',
            'ylabel': 'Diversity among query-relevant topics (similarity ≥ 0.5)',
            'filename': 'aggregate_relevant_topic_diversity'
        },
        {
            'column': 'n_relevant_topics',
            'title': 'Number of Relevant Topics',
            'ylabel': 'Count of topics with query similarity ≥ 0.5',
            'filename': 'aggregate_n_relevant_topics'
        }
    ]

    # Color palette
    colors_list = sns.color_palette("tab10", n_methods)
    method_colors = {method: colors_list[i] for i, method in enumerate(methods)}

    for metric_info in metrics_config:
        column = metric_info['column']
        title_base = metric_info['title']
        ylabel = metric_info['ylabel']
        filename_base = metric_info['filename']

        if column not in all_per_method.columns:
            logger.warning(f"Column '{column}' not found in data, skipping...")
            continue

        # Compute statistics per method
        stats = all_per_method.groupby('method')[column].agg(['mean', 'std', 'count']).reset_index()
        stats.columns = ['method', 'mean', 'std', 'count']

        # Calculate 95% CI (using SEM-based approach)
        stats['sem'] = stats['std'] / np.sqrt(stats['count'])
        stats['ci_95'] = 1.96 * stats['sem']

        # Sort by method name for consistency
        stats = stats.sort_values('method').reset_index(drop=True)

        # ===== Plot 1: Bar chart with ±1 Std Error Bars =====
        fig, ax = plt.subplots(figsize=(12, 7))

        x_pos = np.arange(len(stats))
        bars = ax.bar(
            x_pos,
            stats['mean'],
            yerr=stats['std'],
            color=[method_colors[m] for m in stats['method']],
            capsize=6,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5,
            error_kw={'elinewidth': 2, 'capthick': 2}
        )

        # Add value labels on top of bars
        for i, (bar, mean_val, std_val) in enumerate(zip(bars, stats['mean'], stats['std'])):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std_val + 0.01,
                f'{mean_val:.3f}',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats['method'], fontsize=11)
        ax.set_xlabel('Sampling Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{title_base}\nAggregated Across Queries (Mean ± 1 Std)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Set y-axis to start from 0 if all values are positive
        if stats['mean'].min() >= 0:
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'{filename_base}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved {plot_path}")

        # ===== Plot 2: Bar chart with 95% CI Error Bars =====
        fig, ax = plt.subplots(figsize=(12, 7))

        bars = ax.bar(
            x_pos,
            stats['mean'],
            yerr=stats['ci_95'],
            color=[method_colors[m] for m in stats['method']],
            capsize=6,
            alpha=0.8,
            edgecolor='black',
            linewidth=1.5,
            error_kw={'elinewidth': 2, 'capthick': 2}
        )

        # Add value labels on top of bars
        for i, (bar, mean_val, ci_val) in enumerate(zip(bars, stats['mean'], stats['ci_95'])):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci_val + 0.01,
                f'{mean_val:.3f}',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold'
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(stats['method'], fontsize=11)
        ax.set_xlabel('Sampling Method', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(f'{title_base}\nAggregated Across Queries (Mean ± 95% CI)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Set y-axis to start from 0 if all values are positive
        if stats['mean'].min() >= 0:
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'{filename_base}_95ci.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved {plot_path}")

    logger.info("✓ All standalone aggregate bar charts created successfully")


def main():
    """Main function"""

    # ===== DATASET SELECTION =====
    # Options: "trec-covid", "doctor-reviews"
    DATASET_NAME = "trec-covid"

    # ===== DATASET-SPECIFIC CONFIGURATION =====
    DATASET_CONFIGS = {
        "trec-covid": {
            "query_ids": ["2", "9", "10", "13", "18", "21", "23", "24", "26", "27", "34", "43", "45", "47", "48"],
            "keyword_cache_path": "/home/srangre1/cache/keywords/keybert_k10_div0.7_top10docs_mpnet_k1000_ngram1-2.json",
        },
        "doctor-reviews": {
            "query_ids": ["1", "2", "3", "4", "5", "6"],
            "keyword_cache_path": "/home/srangre1/cache/keywords/doctor_reviews_keybert.json",
        }
    }

    # Get config for selected dataset
    dataset_config = DATASET_CONFIGS[DATASET_NAME]

    # ===== CONFIGURATION PARAMETERS =====

    # Query configuration - use dataset-specific query IDs or override here
    QUERY_IDS = dataset_config["query_ids"]  # All queries for selected dataset
    # QUERY_IDS = [dataset_config["query_ids"][0]]  # Single test query

    # Keyword cache path for query expansion (dataset-specific)
    KEYWORD_CACHE_PATH = dataset_config["keyword_cache_path"]

    # Sample size (fixed at 1000 documents for all methods)
    SAMPLE_SIZE = 1000

    # Model configuration
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Topic modeling configuration
    TOPIC_MODEL_TYPE = "topicgpt"  # Options: "bertopic", "lda", "topicgpt"

    # TopicGPT parameters - ACTIVE CONFIGURATION
    # Requires OPENAI_API_KEY environment variable
    TOPIC_MODEL_PARAMS = {
        # Model selection (use gpt-4o-mini for cost efficiency)
        "generation_model": "gpt-4o-mini",    # Model for topic generation
        "assignment_model": "gpt-4o-mini",    # Model for topic assignment

        # Sampling for topic generation (use subset of documents)
        "generation_sample_size": 500,        # Use 500 docs for topic discovery (out of 1000 sampled)

        # Vocabulary params (match BERTopic/LDA for fair comparison)
        "min_df": 2,
        "ngram_range": (1, 2),
        "max_features": 10000,

        # Output
        "verbose": True
    }

    # BERTopic parameters (currently commented out)
    # TOPIC_MODEL_PARAMS = {}

    # TopicGPT parameters (currently commented out)
    # Requires OPENAI_API_KEY environment variable
    # TOPIC_MODEL_PARAMS = {
    #     # Model selection (use gpt-4o-mini for cost efficiency)
    #     "generation_model": "gpt-4o-mini",    # Model for topic generation
    #     "assignment_model": "gpt-4o-mini",    # Model for topic assignment
    #
    #     # Sampling for topic generation (use subset of documents)
    #     "generation_sample_size": 500,        # Use 500 docs for topic discovery (out of 1000 sampled)
    #
    #     # Vocabulary params (match BERTopic/LDA for fair comparison)
    #     "min_df": 2,
    #     "ngram_range": (1, 2),
    #     "max_features": 10000,
    #
    #     # Output
    #     "verbose": True
    # }

    # BERTopic parameters (when TOPIC_MODEL_TYPE = "bertopic") - COMMENTED OUT
    # TOPIC_MODEL_PARAMS = {
    #     # Using BERTopic defaults that match current behavior
    #     # Can override: "min_cluster_size": 5, "metric": "euclidean", etc.
    # }

    # TopicGPT parameters (when TOPIC_MODEL_TYPE = "topicgpt")
    # TOPIC_MODEL_PARAMS = {
    #     # Model selection (cheapest available)
    #     "generation_model": "gpt-4o-mini",    # $0.15/1M input, $0.60/1M output
    #     "assignment_model": "gpt-4o-mini",    # Same pricing
    #
    #     # Topic generation parameters
    #     "max_topics": 50,
    #     "min_topics": 5,
    #     "generation_sample_size": 500,        # Use 500 docs for topic discovery
    #     "topic_temperature": 0.0,
    #     "topic_max_tokens": 500,
    #
    #     # Topic assignment parameters
    #     "assignment_temperature": 0.0,
    #     "assignment_max_tokens": 300,
    #
    #     # Refinement
    #     "do_refine": True,
    #
    #     # Vocabulary params (match BERTopic/LDA)
    #     "min_df": 2,
    #     "ngram_range": (1, 2),
    #     "max_features": 10000,
    #
    #     # Output
    #     "verbose": True
    # }

    # BERTopic parameters (when TOPIC_MODEL_TYPE = "bertopic")
    # TOPIC_MODEL_PARAMS = {
    #     # Using defaults that match current behavior
    #     # Can override: "min_cluster_size": 5, "metric": "euclidean", etc.
    # }

    # LDA parameters (when TOPIC_MODEL_TYPE = "lda")
    # TOPIC_MODEL_PARAMS = {
    #     "n_topics": "auto",       # "auto" matches BERTopic results, or specify int (e.g., 20)
    #     "alpha": "symmetric",     # Document-topic prior (standard default)
    #     "eta": 0.01,              # Topic-word prior (0.01 = sparse, focused topics for biomedical text)
    #     "passes": 15,             # Training passes (higher for technical corpus)
    #     "iterations": 100,        # Iterations per pass (ensure convergence)
    #     "random_state": 42,
    #     "workers": 12,            # Multi-core training (use ~80% of available cores)
    #     # Vocabulary params (match BERTopic for fair comparison)
    #     "min_df": 2,
    #     "ngram_range": (1, 2),
    #     "max_features": 10000
    # }

    # TopicGPT parameters (when TOPIC_MODEL_TYPE = "topicgpt")
    # Requires OPENAI_API_KEY environment variable
    # TOPIC_MODEL_PARAMS = {
    #     # Model selection (use cheapest models for cost efficiency)
    #     "generation_model": "gpt-4o-mini",    # Model for topic generation (needs good reasoning)
    #     "assignment_model": "gpt-4o-mini",    # Model for topic assignment (simpler task)
    #     # Alternative cheap models: "gpt-4o-mini", "gpt-3.5-turbo"
    #     # Higher quality: "gpt-4o", "gpt-4-turbo"
    #
    #     # Topic generation parameters
    #     "max_topics": 50,                     # Maximum topics to generate
    #     "min_topics": 5,                      # Minimum topics
    #     "generation_sample_size": 500,        # Documents to use for topic generation (subset for cost)
    #     "topic_temperature": 0.0,             # Temperature for generation (0 = deterministic)
    #     "topic_max_tokens": 500,              # Max tokens for topic generation response
    #
    #     # Topic assignment parameters
    #     "assignment_temperature": 0.0,        # Temperature for assignment (0 = deterministic)
    #     "assignment_max_tokens": 300,         # Max tokens for assignment response
    #
    #     # Refinement
    #     "do_refine": True,                    # Whether to refine topics (merge similar, remove irrelevant)
    #
    #     # Vocabulary params (match BERTopic/LDA for fair comparison)
    #     "min_df": 2,
    #     "ngram_range": (1, 2),
    #     "max_features": 10000,
    #
    #     # Output
    #     "verbose": True
    # }

    # Directory configuration
    OUTPUT_DIR = "results"
    CACHE_DIR = "cache"

    # Device configuration - Auto-detect GPU with CPU fallback
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Topic model saving configuration
    # Set to True to save full BERTopic models (420 MB each, only needed for interactive exploration)
    # Set to False to save only results (30-500 KB each, sufficient for evaluation)
    # Default: False (saves ~25.2 GB for 15 queries × 4 methods)
    SAVE_TOPIC_MODELS = False

    # Force flags
    # Set to True to force regeneration (ignoring cache), False to use cache
    #
    # EVALUATION-ONLY MODE (for regenerating plots from cached topic models):
    #   FORCE_REGENERATE_SAMPLES = False
    #   FORCE_REGENERATE_TOPICS = False
    #   FORCE_REGENERATE_EVALUATION = True
    # This regenerates metrics/plots in ~2-5 min per query (vs. ~8-12 min full pipeline).
    # Useful for: (1) Fixing visualization bugs, (2) Adding new aggregate scatter plots
    #
    # CURRENT MODE: Evaluation-only (recompute metrics from cached topic models)
    FORCE_REGENERATE_SAMPLES = False      # Use cached samples
    FORCE_REGENERATE_TOPICS = False       # Use cached topic models
    FORCE_REGENERATE_EVALUATION = False   # Use cached metrics, just regenerate plots

    # Random seed
    RANDOM_SEED = 42

    # ===== INITIALIZE TIMING TRACKER =====
    import time
    from datetime import datetime
    pipeline_start_time = time.time()

    # Create timing filename with dataset, model, and timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_filename = f"timing_{DATASET_NAME}_{TOPIC_MODEL_TYPE}_{timestamp_str}.jsonl"
    timing_file_path = os.path.join(OUTPUT_DIR, DATASET_NAME, TOPIC_MODEL_TYPE, timing_filename)
    os.makedirs(os.path.dirname(timing_file_path), exist_ok=True)
    timing = TimingTracker(timing_file_path)
    logger.info(f"Timing will be written to: {timing_file_path}")

    # ===== LOAD DATASETS =====

    timing.start("dataset_loading", dataset=DATASET_NAME)
    logger.info(f"Loading dataset: {DATASET_NAME}...")
    from dataset_loaders import load_dataset as load_project_dataset

    corpus_dataset, queries_dataset, qrels_dataset = load_project_dataset(DATASET_NAME)

    # Log dataset info
    qrels_count = len(qrels_dataset) if qrels_dataset is not None else 0
    logger.info(f"Loaded {len(corpus_dataset)} documents, {len(queries_dataset)} queries, {qrels_count} qrels")
    timing.stop("dataset_loading", n_documents=len(corpus_dataset), n_queries=len(queries_dataset))

    # ===== HANDLE SINGLE QUERY VS MULTIPLE QUERIES =====

    # Convert single query ID to list for uniform processing
    if isinstance(QUERY_IDS, str):
        query_ids_to_run = [QUERY_IDS]
    else:
        query_ids_to_run = QUERY_IDS

    # ===== RUN EVALUATION FOR EACH QUERY =====

    all_results = {}
    first_query = True  # Track if this is the first query (index initialization happens here)

    for query_id in query_ids_to_run:
        logger.info("\n" + "="*80)
        logger.info(f"PROCESSING QUERY {query_id} ({query_ids_to_run.index(query_id) + 1}/{len(query_ids_to_run)})")
        logger.info("="*80 + "\n")

        # Time index initialization only for first query (indices are reused)
        if first_query:
            timing.start("index_initialization", dataset=DATASET_NAME)

        evaluator = EndToEndEvaluator(
            corpus_dataset=corpus_dataset,
            queries_dataset=queries_dataset,
            qrels_dataset=qrels_dataset,
            query_id=query_id,
            sample_size=SAMPLE_SIZE,
            embedding_model_name=EMBEDDING_MODEL,
            cross_encoder_model_name=CROSS_ENCODER_MODEL,
            dataset_name=DATASET_NAME,
            topic_model_type=TOPIC_MODEL_TYPE,
            topic_model_params=TOPIC_MODEL_PARAMS,
            output_dir=OUTPUT_DIR,
            cache_dir=CACHE_DIR,
            random_seed=RANDOM_SEED,
            device=DEVICE,
            save_topic_models=SAVE_TOPIC_MODELS,
            force_regenerate_samples=FORCE_REGENERATE_SAMPLES,
            force_regenerate_topics=FORCE_REGENERATE_TOPICS,
            force_regenerate_evaluation=FORCE_REGENERATE_EVALUATION,
            keyword_cache_path=KEYWORD_CACHE_PATH
        )

        if first_query:
            timing.stop("index_initialization", embedding_model=EMBEDDING_MODEL)
            first_query = False

        # Time the full evaluation for this query
        timing.start("query_evaluation", query_id=query_id)
        results = evaluator.run_full_evaluation()
        timing.stop("query_evaluation", query_id=query_id)

        all_results[query_id] = results

        # Monitor GPU memory usage if using CUDA
        if DEVICE == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            torch.cuda.empty_cache()  # Clear unused cached memory

    # ===== FINAL SUMMARY =====

    logger.info("\n" + "="*80)
    logger.info("ALL QUERIES COMPLETED!")
    logger.info("="*80)
    logger.info(f"Processed {len(query_ids_to_run)} queries: {', '.join(query_ids_to_run)}")
    logger.info(f"Results saved to: {os.path.join(OUTPUT_DIR, DATASET_NAME, TOPIC_MODEL_TYPE)}")
    logger.info("="*80 + "\n")

    # ===== AGGREGATE CROSS-QUERY RESULTS =====

    # Only run aggregation if we processed multiple queries
    if len(query_ids_to_run) > 1:
        logger.info("\n" + "="*80)
        logger.info("AGGREGATING RESULTS ACROSS ALL QUERIES")
        logger.info("="*80 + "\n")

        timing.start("aggregation", n_queries=len(query_ids_to_run))
        results_base_dir = os.path.join(OUTPUT_DIR, DATASET_NAME, TOPIC_MODEL_TYPE)
        aggregate_results = aggregate_cross_query_results(
            results_base_dir=results_base_dir,
            query_ids=query_ids_to_run
        )
        timing.stop("aggregation")

        if aggregate_results:
            logger.info("\n" + "="*80)
            logger.info("AGGREGATION COMPLETE!")
            logger.info("="*80)
            logger.info(f"Aggregate results saved to: {results_base_dir}/aggregate_results")
            logger.info("="*80 + "\n")
    else:
        logger.info("\nSkipping cross-query aggregation (only 1 query processed)")

    # Record total pipeline time
    total_duration = time.time() - pipeline_start_time
    timing.record("total_pipeline", total_duration,
                  dataset=DATASET_NAME,
                  topic_model=TOPIC_MODEL_TYPE,
                  n_queries=len(query_ids_to_run))

    logger.info(f"\nTotal pipeline time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logger.info(f"Timing data saved to: {timing_file_path}")

    return all_results


if __name__ == "__main__":
    main()
