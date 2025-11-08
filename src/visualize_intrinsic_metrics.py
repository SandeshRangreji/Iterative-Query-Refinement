#!/usr/bin/env python3
"""
Visualize intrinsic metrics (not pairwise comparisons) across methods and queries.

Intrinsic metrics are computed independently for each method:
- Topic Diversity
- Outlier Ratio
- Number of Topics
- Average Topic Size
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def extract_intrinsic_metrics(results_dir="/home/srangre1/results/end_to_end_evaluation"):
    """Extract intrinsic metrics from all queries"""

    queries = ["2", "9", "27", "43"]
    methods = ["random_uniform", "direct_retrieval", "query_expansion", "qrels_labeled"]

    data = []

    for query_id in queries:
        csv_path = os.path.join(results_dir, f"query_{query_id}", "pairwise_evaluation", "pairwise_metrics.csv")

        if not os.path.exists(csv_path):
            print(f"Skipping query {query_id} - no pairwise metrics found")
            continue

        df = pd.read_csv(csv_path)

        # Extract intrinsic metrics for each method
        # Since each method appears in multiple rows, we need to extract unique values
        method_metrics = {}

        for method in methods:
            # Get rows where this method is method_a
            rows_a = df[df['method_a'] == method]
            if not rows_a.empty:
                # Take first row's method_a metrics
                row = rows_a.iloc[0]
                method_metrics[method] = {
                    'diversity': row['diversity_a'],
                    'outlier_ratio': row['outlier_ratio_a'],
                    'n_topics': row['n_topics_a'],
                    'avg_topic_size': row['avg_topic_size_a']
                }
            else:
                # Try method_b
                rows_b = df[df['method_b'] == method]
                if not rows_b.empty:
                    row = rows_b.iloc[0]
                    method_metrics[method] = {
                        'diversity': row['diversity_b'],
                        'outlier_ratio': row['outlier_ratio_b'],
                        'n_topics': row['n_topics_b'],
                        'avg_topic_size': row['avg_topic_size_b']
                    }

        # Convert to records
        for method, metrics in method_metrics.items():
            data.append({
                'query_id': query_id,
                'method': method,
                **metrics
            })

    return pd.DataFrame(data)

def create_intrinsic_visualizations(df, output_dir="/home/srangre1/results/end_to_end_evaluation/intrinsic_metrics"):
    """Create visualizations for intrinsic metrics"""

    os.makedirs(output_dir, exist_ok=True)

    # Get query labels
    query_labels = {
        "2": "Q2: Weather",
        "9": "Q9: Canada",
        "27": "Q27: Asymptomatic",
        "43": "Q43: Violence"
    }
    df['query_label'] = df['query_id'].map(query_labels)

    # Method labels
    method_labels = {
        "random_uniform": "Random",
        "direct_retrieval": "Direct",
        "query_expansion": "QueryExp",
        "qrels_labeled": "QRELs"
    }
    df['method_label'] = df['method'].map(method_labels)

    # Color palette
    method_colors = {
        "Random": "#e74c3c",      # Red
        "Direct": "#3498db",      # Blue
        "QueryExp": "#2ecc71",    # Green
        "QRELs": "#f39c12"        # Orange
    }

    # ===== 1. TOPIC DIVERSITY =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Topic Diversity Across Methods and Queries', fontsize=16, fontweight='bold')

    # Plot 1: Bar chart by query
    ax = axes[0, 0]
    pivot = df.pivot(index='query_label', columns='method_label', values='diversity')
    pivot.plot(kind='bar', ax=ax, color=[method_colors[m] for m in pivot.columns], width=0.8)
    ax.set_title('Topic Diversity by Query', fontsize=14, fontweight='bold')
    ax.set_xlabel('Query', fontsize=12)
    ax.set_ylabel('Diversity (higher = more diverse)', fontsize=12)
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Method comparison (averaged across queries)
    ax = axes[0, 1]
    method_avg = df.groupby('method_label')['diversity'].agg(['mean', 'std']).reset_index()
    method_avg = method_avg.sort_values('mean', ascending=False)
    x_pos = np.arange(len(method_avg))
    bars = ax.bar(x_pos, method_avg['mean'], yerr=method_avg['std'],
                  color=[method_colors[m] for m in method_avg['method_label']],
                  capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_avg['method_label'], fontsize=11)
    ax.set_title('Average Diversity by Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Diversity (mean ± std)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, method_avg['mean'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Heatmap
    ax = axes[1, 0]
    heatmap_data = df.pivot(index='query_label', columns='method_label', values='diversity')
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Diversity'}, vmin=0.3, vmax=0.8)
    ax.set_title('Diversity Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Query', fontsize=12)

    # Plot 4: Line plot (trend across queries)
    ax = axes[1, 1]
    for method in method_labels.values():
        method_data = df[df['method_label'] == method].sort_values('query_id')
        ax.plot(method_data['query_label'], method_data['diversity'],
                marker='o', linewidth=2, markersize=8, label=method, color=method_colors[method])
    ax.set_title('Diversity Trend Across Queries', fontsize=14, fontweight='bold')
    ax.set_xlabel('Query', fontsize=12)
    ax.set_ylabel('Diversity', fontsize=12)
    ax.legend(title='Method', loc='best')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diversity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved diversity analysis")

    # ===== 2. OUTLIER RATIO =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Outlier Ratio Across Methods and Queries', fontsize=16, fontweight='bold')

    # Plot 1: Bar chart by query
    ax = axes[0, 0]
    pivot = df.pivot(index='query_label', columns='method_label', values='outlier_ratio')
    pivot.plot(kind='bar', ax=ax, color=[method_colors[m] for m in pivot.columns], width=0.8)
    ax.set_title('Outlier Ratio by Query', fontsize=14, fontweight='bold')
    ax.set_xlabel('Query', fontsize=12)
    ax.set_ylabel('Outlier Ratio (lower = better)', fontsize=12)
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Method comparison
    ax = axes[0, 1]
    method_avg = df.groupby('method_label')['outlier_ratio'].agg(['mean', 'std']).reset_index()
    method_avg = method_avg.sort_values('mean', ascending=True)  # Lower is better
    x_pos = np.arange(len(method_avg))
    bars = ax.bar(x_pos, method_avg['mean'], yerr=method_avg['std'],
                  color=[method_colors[m] for m in method_avg['method_label']],
                  capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_avg['method_label'], fontsize=11)
    ax.set_title('Average Outlier Ratio by Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Outlier Ratio (mean ± std)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, mean_val) in enumerate(zip(bars, method_avg['mean'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Heatmap
    ax = axes[1, 0]
    heatmap_data = df.pivot(index='query_label', columns='method_label', values='outlier_ratio')
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax,
                cbar_kws={'label': 'Outlier Ratio'}, vmin=0, vmax=0.5)
    ax.set_title('Outlier Ratio Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Query', fontsize=12)

    # Plot 4: Line plot
    ax = axes[1, 1]
    for method in method_labels.values():
        method_data = df[df['method_label'] == method].sort_values('query_id')
        ax.plot(method_data['query_label'], method_data['outlier_ratio'],
                marker='o', linewidth=2, markersize=8, label=method, color=method_colors[method])
    ax.set_title('Outlier Ratio Trend Across Queries', fontsize=14, fontweight='bold')
    ax.set_xlabel('Query', fontsize=12)
    ax.set_ylabel('Outlier Ratio', fontsize=12)
    ax.legend(title='Method', loc='best')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_ratio_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved outlier ratio analysis")

    # ===== 3. NUMBER OF TOPICS =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Number of Topics Discovered', fontsize=16, fontweight='bold')

    # Plot 1: Bar chart by query
    ax = axes[0]
    pivot = df.pivot(index='query_label', columns='method_label', values='n_topics')
    pivot.plot(kind='bar', ax=ax, color=[method_colors[m] for m in pivot.columns], width=0.8)
    ax.set_title('Topic Count by Query', fontsize=14, fontweight='bold')
    ax.set_xlabel('Query', fontsize=12)
    ax.set_ylabel('Number of Topics', fontsize=12)
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Heatmap
    ax = axes[1]
    heatmap_data = df.pivot(index='query_label', columns='method_label', values='n_topics')
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                cbar_kws={'label': 'Number of Topics'})
    ax.set_title('Topic Count Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Query', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_count_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved topic count analysis")

    # ===== 4. COMBINED SUMMARY =====
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Intrinsic Metrics Summary: All Methods & Queries', fontsize=18, fontweight='bold')

    # Plot 1: Diversity comparison
    ax = axes[0, 0]
    pivot = df.pivot(index='method_label', columns='query_label', values='diversity')
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Diversity by Method', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Diversity', fontsize=12)
    ax.legend(title='Query', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Outlier ratio comparison
    ax = axes[0, 1]
    pivot = df.pivot(index='method_label', columns='query_label', values='outlier_ratio')
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Outlier Ratio by Method', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Outlier Ratio', fontsize=12)
    ax.legend(title='Query', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Topic count comparison
    ax = axes[1, 0]
    pivot = df.pivot(index='method_label', columns='query_label', values='n_topics')
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Number of Topics by Method', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Number of Topics', fontsize=12)
    ax.legend(title='Query', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: Scatter - Diversity vs Outlier Ratio
    ax = axes[1, 1]
    for method in method_labels.values():
        method_data = df[df['method_label'] == method]
        ax.scatter(method_data['diversity'], method_data['outlier_ratio'],
                  s=150, alpha=0.7, label=method, color=method_colors[method])
    ax.set_title('Diversity vs Outlier Ratio', fontsize=14, fontweight='bold')
    ax.set_xlabel('Diversity (higher = more diverse)', fontsize=12)
    ax.set_ylabel('Outlier Ratio (lower = better)', fontsize=12)
    ax.legend(title='Method', loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intrinsic_metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved intrinsic metrics summary")

    # ===== 5. CREATE SUMMARY TABLE =====
    summary = df.groupby('method_label').agg({
        'diversity': ['mean', 'std', 'min', 'max'],
        'outlier_ratio': ['mean', 'std', 'min', 'max'],
        'n_topics': ['mean', 'std', 'min', 'max']
    }).round(3)

    summary.to_csv(os.path.join(output_dir, 'intrinsic_metrics_summary.csv'))
    print(f"✓ Saved summary table")

    return summary

def main():
    print("="*80)
    print("INTRINSIC METRICS VISUALIZATION")
    print("="*80)

    # Extract data
    print("\nExtracting intrinsic metrics from all queries...")
    df = extract_intrinsic_metrics()

    print(f"\nLoaded data for {len(df)} method-query combinations")
    print(f"Queries: {sorted(df['query_id'].unique())}")
    print(f"Methods: {sorted(df['method'].unique())}")

    # Create visualizations
    print("\nCreating visualizations...")
    summary = create_intrinsic_visualizations(df)

    print("\n" + "="*80)
    print("INTRINSIC METRICS SUMMARY")
    print("="*80)
    print(summary)

    print("\n" + "="*80)
    print("✓ All visualizations saved to:")
    print("  /home/srangre1/results/end_to_end_evaluation/intrinsic_metrics/")
    print("="*80)

if __name__ == "__main__":
    main()
