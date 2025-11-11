# pairwise_evaluation.py
"""
Pairwise comparison evaluation for different sampling methods.

Instead of comparing everything against a single baseline (random sampling),
this evaluates each method against every other method to understand:
1. Which method preserves topics best overall
2. Which pairs of methods are most similar/different
3. Relative rankings of methods across different metrics

Usage:
    python pairwise_evaluation.py --query-id 27
    python pairwise_evaluation.py --query-id 43 --include-full-corpus
    python pairwise_evaluation.py --results-dir /custom/path --query-id 2
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Set, Optional, Any
from itertools import combinations
from scipy.stats import wilcoxon, mannwhitneyu
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PairwiseEvaluator:
    """Pairwise comparison evaluator for topic modeling methods"""

    def __init__(
        self,
        results_dir: str,
        query_id: str,
        include_full_corpus: bool = False,
        embedding_model: str = "all-mpnet-base-v2"
    ):
        """
        Initialize pairwise evaluator

        Args:
            results_dir: Base results directory (e.g., "/home/srangre1/results/end_to_end_evaluation")
            query_id: Query ID to evaluate
            include_full_corpus: Whether to include full_corpus in comparisons (default: False)
            embedding_model: Sentence transformer model for embeddings
        """
        self.results_dir = results_dir
        self.query_id = str(query_id)
        self.query_dir = os.path.join(results_dir, f"query_{query_id}")
        self.include_full_corpus = include_full_corpus
        self.embedding_model = embedding_model

        # Check if query directory exists
        if not os.path.exists(self.query_dir):
            raise ValueError(f"Query directory not found: {self.query_dir}")

        logger.info(f"Query directory: {self.query_dir}")

        # Output directory for pairwise evaluation
        self.pairwise_dir = os.path.join(self.query_dir, "pairwise_evaluation")
        os.makedirs(self.pairwise_dir, exist_ok=True)

        # Load all topic results and samples
        self.topic_results = {}
        self.samples = {}
        self._load_all_results()

    def _load_all_results(self):
        """Load all topic modeling results and samples"""
        logger.info("Loading topic modeling results...")

        topic_models_dir = os.path.join(self.query_dir, "topic_models")
        samples_dir = os.path.join(self.query_dir, "samples")

        # Define method names (conditionally include full_corpus)
        if self.include_full_corpus:
            methods = ["full_corpus", "random_uniform", "direct_retrieval", "query_expansion", "qrels_labeled"]
        else:
            methods = ["random_uniform", "direct_retrieval", "query_expansion", "qrels_labeled"]
            logger.info("Excluding full_corpus from comparisons (use --include-full-corpus to include)")

        for method in methods:
            # Load topic results
            results_path = os.path.join(topic_models_dir, f"{method}_results.pkl")
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    self.topic_results[method] = pickle.load(f)
                logger.info(f"✓ Loaded {method} topic results")
            else:
                logger.warning(f"✗ Topic results not found for {method}: {results_path}")

            # Load samples (try different naming conventions)
            sample_paths = [
                os.path.join(samples_dir, f"method1_{method}.pkl"),  # method1_random_uniform.pkl
                os.path.join(samples_dir, f"method2_{method}.pkl"),  # method2_direct_retrieval.pkl
                os.path.join(samples_dir, f"method3_{method}.pkl"),  # method3_query_expansion.pkl
                os.path.join(samples_dir, f"method4_{method}.pkl"),  # method4_qrels_labeled.pkl
                os.path.join(samples_dir, f"{method}.pkl"),          # full_corpus.pkl
            ]

            loaded = False
            for sample_path in sample_paths:
                if os.path.exists(sample_path):
                    with open(sample_path, 'rb') as f:
                        self.samples[method] = pickle.load(f)
                    logger.info(f"✓ Loaded {method} sample from {os.path.basename(sample_path)}")
                    loaded = True
                    break

            if not loaded:
                logger.warning(f"✗ Sample not found for {method}")

        logger.info(f"\nSuccessfully loaded {len(self.topic_results)} topic results and {len(self.samples)} samples")

        if len(self.topic_results) < 2:
            raise ValueError("Need at least 2 methods with results to perform pairwise comparison")

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
        from sentence_transformers import SentenceTransformer
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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

        # Topic diversity comparison
        diversity_a = self._compute_topic_diversity(topics_a)
        diversity_b = self._compute_topic_diversity(topics_b)
        metrics["diversity_a"] = diversity_a
        metrics["diversity_b"] = diversity_b
        metrics["diversity_diff"] = diversity_a - diversity_b

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

    def _compute_topic_diversity(self, topic_words: Dict[int, List[str]]) -> float:
        """Compute average pairwise distance between topics"""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        if len(topic_words) < 2:
            return 0.0

        # Load embedding model (force CPU to avoid CUDA compatibility issues)
        model = SentenceTransformer(self.embedding_model, device='cpu')

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

    def _match_topics(
        self,
        topics_a: Dict[int, List[str]],
        topics_b: Dict[int, List[str]],
        top_n: int = 10
    ) -> Tuple[List[float], List[float]]:
        """Match topics using Hungarian algorithm"""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.optimize import linear_sum_assignment

        if not topics_a or not topics_b:
            return [], []

        # Load embedding model (force CPU to avoid CUDA compatibility issues)
        model = SentenceTransformer(self.embedding_model, device='cpu')

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

    def run_all_pairwise_comparisons(self) -> pd.DataFrame:
        """
        Run pairwise comparisons for all method pairs

        Returns:
            DataFrame with all pairwise metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Running pairwise comparisons...")
        logger.info("="*80)

        methods = list(self.topic_results.keys())

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
                    results_a=self.topic_results[method_a],
                    results_b=self.topic_results[method_b],
                    sample_a=self.samples[method_a],
                    sample_b=self.samples[method_b]
                )
                all_metrics.append(metrics)

                logger.debug(f"Compared {method_a} vs {method_b}")
            except Exception as e:
                logger.error(f"Error comparing {method_a} vs {method_b}: {e}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)

        # Save results
        csv_path = os.path.join(self.pairwise_dir, "pairwise_metrics.csv")
        json_path = os.path.join(self.pairwise_dir, "pairwise_metrics.json")

        df.to_csv(csv_path, index=False)
        logger.info(f"\n✓ Saved pairwise metrics to {csv_path}")

        with open(json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"✓ Saved pairwise metrics to {json_path}")

        return df

    def create_comparison_matrices(self, df: pd.DataFrame):
        """
        Create comparison matrices for key metrics

        Args:
            df: DataFrame with pairwise metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Creating comparison matrices...")
        logger.info("="*80)

        methods = list(self.topic_results.keys())
        n = len(methods)

        # Define metrics to visualize
        metrics_to_plot = [
            ("topic_semantic_similarity_mean", "Topic Semantic Similarity", "Blues", "higher_better"),
            ("topic_word_overlap_mean", "Topic Word Overlap (Jaccard)", "Greens", "higher_better"),
            ("precision_b_@05", "Topic Matching Precision @0.5", "Purples", "higher_better"),
            ("recall_a_@05", "Topic Matching Recall @0.5", "Oranges", "higher_better"),
            ("f1_@05", "Topic Matching F1 @0.5", "RdPu", "higher_better"),
            ("precision_b_@06", "Topic Matching Precision @0.6", "Purples", "higher_better"),
            ("recall_a_@06", "Topic Matching Recall @0.6", "Oranges", "higher_better"),
            ("f1_@06", "Topic Matching F1 @0.6", "RdPu", "higher_better"),
            ("precision_b_@07", "Topic Matching Precision @0.7", "Purples", "higher_better"),
            ("recall_a_@07", "Topic Matching Recall @0.7", "Oranges", "higher_better"),
            ("f1_@07", "Topic Matching F1 @0.7", "RdPu", "higher_better"),
            ("diversity_diff", "Topic Diversity Difference", "RdBu_r", "centered"),
            ("outlier_ratio_diff", "Outlier Ratio Difference", "RdYlGn_r", "centered"),
            ("ari", "Adjusted Rand Index", "YlGn", "higher_better"),
        ]

        for metric_key, title, cmap, scale_type in metrics_to_plot:
            # Create matrix
            matrix = np.zeros((n, n))

            for i, method_a in enumerate(methods):
                for j, method_b in enumerate(methods):
                    if i == j:
                        # Diagonal: self-comparison (perfect score)
                        if scale_type == "higher_better":
                            matrix[i, j] = 1.0
                        else:  # centered
                            matrix[i, j] = 0.0
                    else:
                        # Find the metric value
                        row = df[(df['method_a'] == method_a) & (df['method_b'] == method_b)]
                        if row.empty:
                            # Try reverse order
                            row = df[(df['method_a'] == method_b) & (df['method_b'] == method_a)]
                            if not row.empty and metric_key in row.columns:
                                value = row[metric_key].values[0]
                                # For difference metrics, reverse sign
                                if "diff" in metric_key:
                                    value = -value
                                matrix[i, j] = value if value is not None and not np.isnan(value) else 0.0
                        elif metric_key in row.columns:
                            value = row[metric_key].values[0]
                            matrix[i, j] = value if value is not None and not np.isnan(value) else 0.0

            # Create heatmap
            plt.figure(figsize=(10, 8))

            # Determine color scale
            if scale_type == "higher_better":
                vmin, vmax = 0, 1
            elif scale_type == "centered":
                abs_max = max(abs(matrix.min()), abs(matrix.max())) if matrix.size > 0 else 1
                vmin, vmax = -abs_max, abs_max
            else:
                vmin, vmax = matrix.min(), matrix.max()

            sns.heatmap(
                matrix,
                annot=True,
                fmt='.3f',
                cmap=cmap,
                xticklabels=methods,
                yticklabels=methods,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': title},
                square=True
            )

            plt.title(f"{title}\n(Method A vs Method B)", fontsize=14, fontweight='bold')
            plt.xlabel("Method B", fontsize=12)
            plt.ylabel("Method A", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Save plot
            filename = metric_key.replace('@', '_at_').replace('_', '-') + "_matrix.png"
            plot_path = os.path.join(self.pairwise_dir, filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Saved {title} matrix")

        logger.info(f"\n✓ All matrices saved to {self.pairwise_dir}")

    def create_ranking_summary(self, df: pd.DataFrame):
        """
        Create a ranking summary showing which methods perform best

        Args:
            df: DataFrame with pairwise metrics
        """
        logger.info("\n" + "="*80)
        logger.info("Creating ranking summary...")
        logger.info("="*80)

        methods = list(self.topic_results.keys())

        # Define metrics where higher is better
        higher_better_metrics = [
            "topic_semantic_similarity_mean",
            "topic_word_overlap_mean",
            "f1_@07",
            "ari",
            "nmi"
        ]

        # For each method, count how many pairwise comparisons it "wins"
        wins = {method: 0 for method in methods}
        total_comparisons = {method: 0 for method in methods}

        for metric in higher_better_metrics:
            if metric not in df.columns:
                continue

            for _, row in df.iterrows():
                method_a = row['method_a']
                method_b = row['method_b']
                value = row[metric]

                if value is None or np.isnan(value):
                    continue

                # A has higher score than B
                if value > 0.5:  # Assuming normalized 0-1 scale
                    wins[method_a] += 1
                    total_comparisons[method_a] += 1
                    total_comparisons[method_b] += 1
                elif value < 0.5:
                    wins[method_b] += 1
                    total_comparisons[method_a] += 1
                    total_comparisons[method_b] += 1

        # Calculate win rate
        ranking = []
        for method in methods:
            win_rate = wins[method] / total_comparisons[method] if total_comparisons[method] > 0 else 0.0
            ranking.append({
                "method": method,
                "wins": wins[method],
                "total_comparisons": total_comparisons[method],
                "win_rate": win_rate
            })

        # Sort by win rate
        ranking = sorted(ranking, key=lambda x: x['win_rate'], reverse=True)

        # Create ranking DataFrame
        ranking_df = pd.DataFrame(ranking)

        # Save ranking
        ranking_path = os.path.join(self.pairwise_dir, "method_ranking.csv")
        ranking_df.to_csv(ranking_path, index=False)
        logger.info(f"✓ Saved method ranking to {ranking_path}")

        # Print ranking
        logger.info("\n" + "="*60)
        logger.info("METHOD RANKING (by pairwise win rate)")
        logger.info("="*60)
        for i, row in ranking_df.iterrows():
            logger.info(f"{i+1}. {row['method']:<25} Win Rate: {row['win_rate']:.3f} ({row['wins']}/{row['total_comparisons']})")
        logger.info("="*60)

        return ranking_df

    def create_summary_report(self, df: pd.DataFrame, ranking_df: pd.DataFrame):
        """
        Create a comprehensive summary report

        Args:
            df: DataFrame with pairwise metrics
            ranking_df: DataFrame with method rankings
        """
        logger.info("\n" + "="*80)
        logger.info("Creating summary report...")
        logger.info("="*80)

        report_path = os.path.join(self.pairwise_dir, "pairwise_summary_report.txt")

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PAIRWISE COMPARISON EVALUATION REPORT\n")
            f.write(f"Query ID: {self.query_id}\n")
            f.write("="*80 + "\n\n")

            # Method ranking
            f.write("METHOD RANKING (by pairwise win rate)\n")
            f.write("-"*80 + "\n")
            for i, row in ranking_df.iterrows():
                f.write(f"{i+1}. {row['method']:<30} Win Rate: {row['win_rate']:.3f} ({row['wins']}/{row['total_comparisons']})\n")
            f.write("\n")

            # Key findings
            f.write("KEY FINDINGS\n")
            f.write("-"*80 + "\n")

            # Most similar pair
            if 'topic_semantic_similarity_mean' in df.columns and len(df) > 0:
                most_similar = df.loc[df['topic_semantic_similarity_mean'].idxmax()]
                f.write(f"\nMost Similar Methods:\n")
                f.write(f"  {most_similar['method_a']} vs {most_similar['method_b']}\n")
                f.write(f"  Semantic Similarity: {most_similar['topic_semantic_similarity_mean']:.3f}\n")

                # Most different pair
                most_different = df.loc[df['topic_semantic_similarity_mean'].idxmin()]
                f.write(f"\nMost Different Methods:\n")
                f.write(f"  {most_different['method_a']} vs {most_different['method_b']}\n")
                f.write(f"  Semantic Similarity: {most_different['topic_semantic_similarity_mean']:.3f}\n")

            # Average metrics across all pairs
            f.write("\n\nAVERAGE METRICS ACROSS ALL PAIRS\n")
            f.write("-"*80 + "\n")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['method_a', 'method_b']:
                    mean_val = df[col].mean()
                    if not np.isnan(mean_val):
                        f.write(f"  {col:<45} {mean_val:.4f}\n")

            f.write("\n" + "="*80 + "\n")

        logger.info(f"✓ Saved summary report to {report_path}")

    def run_full_evaluation(self):
        """Run complete pairwise evaluation pipeline"""
        logger.info("\n" + "="*80)
        logger.info(f"Starting Pairwise Evaluation for Query {self.query_id}")
        logger.info("="*80)

        # Run pairwise comparisons
        df = self.run_all_pairwise_comparisons()

        if df.empty:
            logger.error("No pairwise comparisons completed. Exiting.")
            return None

        # Create comparison matrices
        self.create_comparison_matrices(df)

        # Create ranking summary
        ranking_df = self.create_ranking_summary(df)

        # Create summary report
        self.create_summary_report(df, ranking_df)

        logger.info("\n" + "="*80)
        logger.info("Pairwise evaluation complete!")
        logger.info(f"Results saved to: {self.pairwise_dir}")
        logger.info("="*80 + "\n")

        return {
            "pairwise_metrics": df,
            "ranking": ranking_df,
            "output_dir": self.pairwise_dir
        }


def main():
    """Main function"""

    # ===== CONFIGURATION PARAMETERS =====
    # Modify these parameters as needed

    RESULTS_DIR = "/home/srangre1/results/end_to_end_evaluation"
    QUERY_ID = "9"
    INCLUDE_FULL_CORPUS = False  # Set to True if you have full_corpus results
    EMBEDDING_MODEL = "all-mpnet-base-v2"

    # ===== END CONFIGURATION =====
    # NOTE: Device is hardcoded to 'cpu' to avoid CUDA compatibility issues with GTX 1080 Ti

    logger.info("\n" + "="*80)
    logger.info("PAIRWISE EVALUATION CONFIGURATION")
    logger.info("="*80)
    logger.info(f"  Results Directory: {RESULTS_DIR}")
    logger.info(f"  Query ID: {QUERY_ID}")
    logger.info(f"  Include Full Corpus: {INCLUDE_FULL_CORPUS}")
    logger.info(f"  Embedding Model: {EMBEDDING_MODEL}")
    logger.info("="*80)

    # Initialize evaluator
    try:
        evaluator = PairwiseEvaluator(
            results_dir=RESULTS_DIR,
            query_id=QUERY_ID,
            include_full_corpus=INCLUDE_FULL_CORPUS,
            embedding_model=EMBEDDING_MODEL
        )
    except ValueError as e:
        logger.error(f"\nError initializing evaluator: {e}")
        sys.exit(1)

    # Run evaluation
    results = evaluator.run_full_evaluation()

    if results is None:
        logger.error("\nEvaluation failed!")
        sys.exit(1)

    logger.info("\nSuccess! Check the output directory for results.")
    return results


if __name__ == "__main__":
    main()
