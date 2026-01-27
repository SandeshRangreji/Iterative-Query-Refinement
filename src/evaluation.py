# evaluation.py
import os
import logging
import json
from typing import List, Dict, Tuple, Set, Union, Optional
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SearchEvaluationUtils:
    """Utility class for search evaluation metrics and operations"""
    
    @staticmethod
    def build_qrels_dicts(qrels_dataset):
        """
        Convert qrels dataset into dictionaries for different relevance levels
        
        Args:
            qrels_dataset: Dataset containing relevance judgments
            
        Returns:
            Tuple of dictionaries (relevant, highly_relevant, overall_relevant)
        """
        relevant_docs_by_query = defaultdict(set)
        highly_relevant_docs_by_query = defaultdict(set)
        overall_relevant_docs_by_query = defaultdict(set)

        for qid, cid, score in zip(qrels_dataset["query-id"],
                                qrels_dataset["corpus-id"],
                                qrels_dataset["score"]):
            if score == 1:
                relevant_docs_by_query[qid].add(cid)
            if score == 2:
                highly_relevant_docs_by_query[qid].add(cid)
            if score > 0:
                overall_relevant_docs_by_query[qid].add(cid)

        return relevant_docs_by_query, highly_relevant_docs_by_query, overall_relevant_docs_by_query
    
    @staticmethod
    def calculate_precision_recall(retrieved_docs: List[str], full_retrieved_docs: List[str], relevant_set: Set[str]):
        """
        Calculate precision and recall for a set of retrieved documents
        
        Args:
            retrieved_docs: List of document IDs for precision calculation
            full_retrieved_docs: List of document IDs for recall calculation (typically larger set)
            relevant_set: Set of relevant document IDs
            
        Returns:
            Tuple of (precision, recall) or (None, None) if relevant_set is empty
        """
        if not relevant_set:
            return None, None

        # Precision@k
        retrieved_set = set(retrieved_docs)
        num_relevant_retrieved = len(relevant_set.intersection(retrieved_set))
        precision = num_relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0

        # Recall@k
        full_retrieved_set = set(full_retrieved_docs)
        num_relevant_in_full = len(relevant_set.intersection(full_retrieved_set))
        recall = num_relevant_in_full / len(relevant_set) if relevant_set else 0

        return precision, recall
    
    @staticmethod
    def evaluate_results(
        results_by_query_id: Dict[int, List[Tuple[str, float]]],
        qrels_dataset,
        top_k_p: int = 20,
        top_k_r: int = 1000
    ) -> Tuple[Dict[str, float], Dict[str, float], int]:
        """
        Evaluate retrieval results
        
        Args:
            results_by_query_id: Dictionary mapping query IDs to a list of (doc_id, score) tuples
            qrels_dataset: Dataset containing relevance judgments
            top_k_p: Number of results to consider for precision
            top_k_r: Number of results to consider for recall
            
        Returns:
            Tuple of (avg_precisions, avg_recalls, num_evaluated)
        """
        # Build qrels dictionaries
        relevant_docs_by_query, highly_relevant_docs_by_query, overall_relevant_docs_by_query = (
            SearchEvaluationUtils.build_qrels_dicts(qrels_dataset)
        )

        # Initialize metrics containers
        all_precisions = {'relevant': [], 'highly_relevant': [], 'overall': []}
        all_recalls = {'relevant': [], 'highly_relevant': [], 'overall': []}
        num_evaluated = 0

        for query_id, results in results_by_query_id.items():
            # Extract document IDs
            retrieved_docs = [doc_id for doc_id, _ in results[:top_k_p]]
            full_retrieved_docs = [doc_id for doc_id, _ in results[:top_k_r]]

            # Calculate metrics for different relevance levels
            if query_id in relevant_docs_by_query:
                precision, recall = SearchEvaluationUtils.calculate_precision_recall(
                    retrieved_docs, full_retrieved_docs, relevant_docs_by_query[query_id]
                )
                if precision is not None:
                    all_precisions['relevant'].append(precision)
                    all_recalls['relevant'].append(recall)

            if query_id in highly_relevant_docs_by_query:
                precision, recall = SearchEvaluationUtils.calculate_precision_recall(
                    retrieved_docs, full_retrieved_docs, highly_relevant_docs_by_query[query_id]
                )
                if precision is not None:
                    all_precisions['highly_relevant'].append(precision)
                    all_recalls['highly_relevant'].append(recall)

            if query_id in overall_relevant_docs_by_query:
                precision, recall = SearchEvaluationUtils.calculate_precision_recall(
                    retrieved_docs, full_retrieved_docs, overall_relevant_docs_by_query[query_id]
                )
                if precision is not None:
                    all_precisions['overall'].append(precision)
                    all_recalls['overall'].append(recall)

            num_evaluated += 1

        # Compute averages
        avg_precisions = {
            level: (sum(precisions) / len(precisions)) if precisions else 0.0
            for level, precisions in all_precisions.items()
        }
        
        avg_recalls = {
            level: (sum(recalls) / len(recalls)) if recalls else 0.0
            for level, recalls in all_recalls.items()
        }

        return avg_precisions, avg_recalls, num_evaluated
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall
        
        Args:
            precision: Precision value
            recall: Recall value
            
        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_f1_scores(results, relevance_level="overall"):
        """
        Calculate F1 scores from precision and recall
        
        Args:
            results: List of result dictionaries from evaluation
            relevance_level: Which relevance level to use
            
        Returns:
            List of dictionaries with added F1 scores
        """
        metrics_results = []
        
        for result in results:
            precision = result["avg_precisions"][relevance_level]
            recall = result["avg_recalls"][relevance_level]
            
            # Calculate F1 score
            f1 = SearchEvaluationUtils.calculate_f1_score(precision, recall)
            
            # Create metrics dict
            metrics = {
                "config": result["config"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "num_evaluated": result["num_evaluated"]
            }
            
            metrics_results.append(metrics)
        
        return metrics_results
    
    @staticmethod
    def save_evaluation_results(results, output_file="search_evaluation_results.json"):
        """
        Save evaluation results to a JSON file
        
        Args:
            results: List of result dictionaries from evaluation
            output_file: Path to save results
            
        Returns:
            Path to the saved file
        """
        # Convert results to a serializable format
        serializable_results = []
        for result in results:
            # Handle method field - could be string, enum, or missing
            method_value = result["config"].get("method")
            if method_value is not None:
                if not isinstance(method_value, str):
                    method_value = method_value.value

            serializable_result = {
                "config": {
                    "name": result["config"]["name"],
                    "method": method_value,
                    "use_mmr": result["config"]["use_mmr"],
                    "use_cross_encoder": result["config"]["use_cross_encoder"]
                },
                "avg_precisions": result["avg_precisions"],
                "avg_recalls": result["avg_recalls"],
                "num_evaluated": result["num_evaluated"]
            }
            
            # Add hybrid strategy if present
            if "hybrid_strategy" in result["config"] and result["config"]["hybrid_strategy"] is not None:
                serializable_result["config"]["hybrid_strategy"] = (
                    result["config"]["hybrid_strategy"]
                    if isinstance(result["config"]["hybrid_strategy"], str)
                    else result["config"]["hybrid_strategy"].value
                )
                
            # Add other optional parameters if present
            for param in ["mmr_lambda", "hybrid_weight", "expansion_method", "combination_strategy"]:
                if param in result["config"]:
                    serializable_result["config"][param] = result["config"][param]

            serializable_results.append(serializable_result)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to file
        with open(output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {output_file}")
        return output_file
    
    @staticmethod
    def compare_evaluation_results(baseline_results, comparison_results, relevance_level="overall"):
        """
        Compare two sets of evaluation results to measure improvements
        
        Args:
            baseline_results: Baseline evaluation results
            comparison_results: Results to compare against baseline
            relevance_level: Which relevance level to use
            
        Returns:
            Dictionary with improvement percentages
        """
        baseline_precision = baseline_results["avg_precisions"][relevance_level]
        baseline_recall = baseline_results["avg_recalls"][relevance_level]
        
        comparison_precision = comparison_results["avg_precisions"][relevance_level]
        comparison_recall = comparison_results["avg_recalls"][relevance_level]
        
        # Calculate improvements
        precision_change = ((comparison_precision - baseline_precision) / baseline_precision) * 100 if baseline_precision > 0 else float('inf')
        recall_change = ((comparison_recall - baseline_recall) / baseline_recall) * 100 if baseline_recall > 0 else float('inf')
        
        # Calculate F1 scores
        baseline_f1 = SearchEvaluationUtils.calculate_f1_score(baseline_precision, baseline_recall)
        comparison_f1 = SearchEvaluationUtils.calculate_f1_score(comparison_precision, comparison_recall)
        
        f1_change = ((comparison_f1 - baseline_f1) / baseline_f1) * 100 if baseline_f1 > 0 else float('inf')
        
        return {
            "baseline_name": baseline_results["config"]["name"],
            "comparison_name": comparison_results["config"]["name"],
            "precision_baseline": baseline_precision,
            "precision_comparison": comparison_precision,
            "precision_change_percent": precision_change,
            "recall_baseline": baseline_recall,
            "recall_comparison": comparison_recall,
            "recall_change_percent": recall_change,
            "f1_baseline": baseline_f1,
            "f1_comparison": comparison_f1,
            "f1_change_percent": f1_change
        }
    
    @staticmethod
    def calculate_per_query_metrics(
        results_by_query_id: Dict[int, List[Tuple[str, float]]],
        qrels_dataset,
        top_k_p: int = 20,
        top_k_r: int = 1000
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Calculate metrics for each individual query
        
        Args:
            results_by_query_id: Dictionary mapping query IDs to a list of (doc_id, score) tuples
            qrels_dataset: Dataset containing relevance judgments
            top_k_p: Number of results to consider for precision
            top_k_r: Number of results to consider for recall
            
        Returns:
            Dictionary mapping query IDs to metrics
        """
        # Build qrels dictionaries
        relevant_docs_by_query, highly_relevant_docs_by_query, overall_relevant_docs_by_query = (
            SearchEvaluationUtils.build_qrels_dicts(qrels_dataset)
        )
        
        # Dictionary to store per-query metrics
        per_query_metrics = {}
        
        for query_id, results in results_by_query_id.items():
            # Extract document IDs
            retrieved_docs = [doc_id for doc_id, _ in results[:top_k_p]]
            full_retrieved_docs = [doc_id for doc_id, _ in results[:top_k_r]]
            
            query_metrics = {"precision": {}, "recall": {}, "f1": {}}
            
            # Calculate metrics for different relevance levels
            for level, relevant_set_dict in [
                ("relevant", relevant_docs_by_query),
                ("highly_relevant", highly_relevant_docs_by_query),
                ("overall", overall_relevant_docs_by_query)
            ]:
                if query_id not in relevant_set_dict:
                    continue
                
                relevant_set = relevant_set_dict[query_id]
                
                precision, recall = SearchEvaluationUtils.calculate_precision_recall(
                    retrieved_docs, full_retrieved_docs, relevant_set
                )
                
                if precision is None:
                    continue
                
                # Calculate F1 score
                f1 = SearchEvaluationUtils.calculate_f1_score(precision, recall)
                
                # Store metrics
                query_metrics["precision"][level] = precision
                query_metrics["recall"][level] = recall
                query_metrics["f1"][level] = f1
            
            per_query_metrics[query_id] = query_metrics
        
        return per_query_metrics
    