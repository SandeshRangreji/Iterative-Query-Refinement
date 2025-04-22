# visualization.py
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_dir(output_dir: str = "plots") -> str:
    """Create output directory for plots if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_precision_recall_comparison(
    results: List[Dict[str, Any]],
    relevance_level: str = "overall",
    top_k_p: int = 20,
    top_k_r: int = 1000,
    output_dir: str = "plots",
    filename: str = "precision_recall_comparison.png"
) -> str:
    """
    Create a precision-recall comparison bar chart

    Args:
        results: List of result dictionaries from evaluation
        relevance_level: Which relevance level to use ('relevant', 'highly_relevant', 'overall')
        top_k_p: k value used for precision
        top_k_r: k value used for recall
        output_dir: Directory to save the plot
        filename: Filename for the saved plot

    Returns:
        Path to the saved plot
    """
    # Create output directory if needed
    create_output_dir(output_dir)
    
    # Extract data for plotting
    method_names = [result["config"]["name"] for result in results]
    precisions = [result["avg_precisions"][relevance_level] for result in results]
    recalls = [result["avg_recalls"][relevance_level] for result in results]
    
    # Set up plot
    plt.figure(figsize=(12, 6))
    
    # Set width of bars
    bar_width = 0.35
    indices = np.arange(len(method_names))
    
    # Create bars
    plt.bar(indices - bar_width/2, precisions, bar_width, label=f'Precision@{top_k_p}', color='skyblue')
    plt.bar(indices + bar_width/2, recalls, bar_width, label=f'Recall@{top_k_r}', color='salmon')
    
    # Add labels and title
    plt.xlabel('Retrieval Method')
    plt.ylabel('Score')
    plt.title(f'Precision@{top_k_p} and Recall@{top_k_r} Comparison ({relevance_level.capitalize()} Relevance)')
    plt.xticks(indices, method_names, rotation=45, ha='right')
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(precisions):
        plt.text(i - bar_width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    for i, v in enumerate(recalls):
        plt.text(i + bar_width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved precision-recall comparison plot to {output_path}")
    
    # Close figure to free memory
    plt.close()
    
    return output_path

def plot_f1_scores(
    results: List[Dict[str, Any]],
    relevance_level: str = "overall",
    top_k_values: List[int] = [10, 20, 50, 100],
    output_dir: str = "plots",
    filename: str = "f1_scores_comparison.png"
) -> str:
    """
    Create an F1 score comparison chart for different k values
    
    Args:
        results: List of result dictionaries from evaluation
        relevance_level: Which relevance level to use ('relevant', 'highly_relevant', 'overall')
        top_k_values: List of k values to compare
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if needed
    create_output_dir(output_dir)
    
    # Calculate F1 scores for each method and each k value
    method_names = [result["config"]["name"] for result in results]
    f1_scores = []
    
    for result in results:
        precision = result["avg_precisions"][relevance_level]
        recall = result["avg_recalls"][relevance_level]
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    # Set up plot
    plt.figure(figsize=(10, 6))
    
    # Create bars
    plt.bar(method_names, f1_scores, color='mediumseagreen')
    
    # Add labels and title
    plt.xlabel('Retrieval Method')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score Comparison ({relevance_level.capitalize()} Relevance)')
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved F1 score comparison plot to {output_path}")
    
    # Close figure to free memory
    plt.close()
    
    return output_path

def plot_method_radar_chart(
    results: List[Dict[str, Any]],
    metrics: List[str] = ["precision", "recall", "f1"],
    relevance_level: str = "overall",
    output_dir: str = "plots",
    filename: str = "method_radar_comparison.png"
) -> str:
    """
    Create a radar chart comparing different retrieval methods
    
    Args:
        results: List of result dictionaries from evaluation
        metrics: List of metrics to include in the radar chart
        relevance_level: Which relevance level to use
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if needed
    create_output_dir(output_dir)
    
    # Extract data for plotting
    method_names = [result["config"]["name"] for result in results]
    
    # Calculate metrics for each method
    radar_data = []
    for result in results:
        precision = result["avg_precisions"][relevance_level]
        recall = result["avg_recalls"][relevance_level]
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        method_metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        radar_data.append([method_metrics[m] for m in metrics])
    
    # Set up radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics])
    
    # Plot each method
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, (method_name, method_values) in enumerate(zip(method_names, radar_data)):
        values = method_values + method_values[:1]  # Close the loop
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, label=method_name)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(f'Method Comparison Radar Chart ({relevance_level.capitalize()} Relevance)')
    
    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved radar chart to {output_path}")
    
    # Close figure to free memory
    plt.close()
    
    return output_path

def plot_method_heatmap(
    results: List[Dict[str, Any]],
    relevance_levels: List[str] = ["relevant", "highly_relevant", "overall"],
    metric: str = "precision",  # or "recall"
    output_dir: str = "plots",
    filename: str = "method_relevance_heatmap.png"
) -> str:
    """
    Create a heatmap showing method performance across different relevance levels
    
    Args:
        results: List of result dictionaries from evaluation
        relevance_levels: List of relevance levels to include
        metric: Which metric to visualize ('precision' or 'recall')
        output_dir: Directory to save the plot
        filename: Filename for the saved plot
        
    Returns:
        Path to the saved plot
    """
    # Create output directory if needed
    create_output_dir(output_dir)
    
    # Extract data for plotting
    method_names = [result["config"]["name"] for result in results]
    
    # Create data matrix for heatmap
    data_matrix = []
    for result in results:
        if metric == "precision":
            values = [result["avg_precisions"][level] for level in relevance_levels]
        else:  # metric == "recall"
            values = [result["avg_recalls"][level] for level in relevance_levels]
        data_matrix.append(values)
    
    # Convert to numpy array for heatmap
    data_matrix = np.array(data_matrix)
    
    # Set up plot
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=[level.capitalize() for level in relevance_levels],
        yticklabels=method_names
    )
    
    # Add labels and title
    plt.xlabel('Relevance Level')
    plt.ylabel('Retrieval Method')
    plt.title(f'{metric.capitalize()} by Method and Relevance Level')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved heatmap to {output_path}")
    
    # Close figure to free memory
    plt.close()
    
    return output_path

def visualize_all_results(
    results: List[Dict[str, Any]],
    top_k_p: int = 20,
    top_k_r: int = 1000,
    output_dir: str = "plots"
) -> List[str]:
    """
    Create all visualizations for the evaluation results
    
    Args:
        results: List of result dictionaries from evaluation
        top_k_p: k value used for precision
        top_k_r: k value used for recall
        output_dir: Directory to save the plots
        
    Returns:
        List of paths to all generated plots
    """
    output_paths = []
    
    # Create precision-recall comparison for each relevance level
    for relevance_level in ["relevant", "highly_relevant", "overall"]:
        filename = f"precision_recall_{relevance_level}.png"
        path = plot_precision_recall_comparison(
            results,
            relevance_level=relevance_level,
            top_k_p=top_k_p,
            top_k_r=top_k_r,
            output_dir=output_dir,
            filename=filename
        )
        output_paths.append(path)
    
    # Create F1 score comparison
    path = plot_f1_scores(
        results,
        relevance_level="overall",
        output_dir=output_dir
    )
    output_paths.append(path)
    
    # Create radar chart
    path = plot_method_radar_chart(
        results,
        output_dir=output_dir
    )
    output_paths.append(path)
    
    # Create heatmaps for precision and recall
    for metric in ["precision", "recall"]:
        filename = f"method_{metric}_heatmap.png"
        path = plot_method_heatmap(
            results,
            metric=metric,
            output_dir=output_dir,
            filename=filename
        )
        output_paths.append(path)
    
    return output_paths

def main():
    """Example usage of visualization functions"""
    import json
    
    # Load example results from a file (if available)
    try:
        with open("evaluation_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        # Create dummy results for testing
        logger.info("No results file found, creating dummy data for visualization")
        results = [
            {
                "config": {
                    "name": "BM25"
                },
                "avg_precisions": {
                    "relevant": 0.45,
                    "highly_relevant": 0.35,
                    "overall": 0.42
                },
                "avg_recalls": {
                    "relevant": 0.62,
                    "highly_relevant": 0.58,
                    "overall": 0.60
                }
            },
            {
                "config": {
                    "name": "SBERT"
                },
                "avg_precisions": {
                    "relevant": 0.51,
                    "highly_relevant": 0.48,
                    "overall": 0.50
                },
                "avg_recalls": {
                    "relevant": 0.68,
                    "highly_relevant": 0.65,
                    "overall": 0.67
                }
            },
            {
                "config": {
                    "name": "Hybrid (Simple Sum)"
                },
                "avg_precisions": {
                    "relevant": 0.58,
                    "highly_relevant": 0.54,
                    "overall": 0.56
                },
                "avg_recalls": {
                    "relevant": 0.74,
                    "highly_relevant": 0.70,
                    "overall": 0.72
                }
            },
            {
                "config": {
                    "name": "Hybrid (RRF)"
                },
                "avg_precisions": {
                    "relevant": 0.62,
                    "highly_relevant": 0.57,
                    "overall": 0.60
                },
                "avg_recalls": {
                    "relevant": 0.77,
                    "highly_relevant": 0.72,
                    "overall": 0.75
                }
            }
        ]
    
    # Generate all visualizations
    output_paths = visualize_all_results(results)
    
    # Log outputs
    logger.info(f"Generated {len(output_paths)} visualization plots:")
    for path in output_paths:
        logger.info(f"- {path}")

if __name__ == "__main__":
    main()