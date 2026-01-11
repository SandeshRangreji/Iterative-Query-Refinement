#!/usr/bin/env python3
# analyze_hicode_relevance.py
"""
Analyze how well HiCode identifies highly relevant documents.

For each sampling method, computes:
- Precision: % of HiCode-labeled docs that are actually highly relevant
- Recall: % of highly relevant docs that HiCode found
- F1 Score: Harmonic mean of precision and recall

Relevance definition: TREC qrels score == 2 (highly relevant)
"""

import os
import sys
import pickle
import csv
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


def load_hicode_results(hicode_dir: str, methods: List[str]) -> Dict[str, Dict]:
    """
    Load HiCode topic modeling results for all methods.

    Args:
        hicode_dir: Directory containing HiCode pickle files
        methods: List of sampling method names

    Returns:
        Dictionary mapping method names to HiCode data
    """
    results = {}

    for method in methods:
        pkl_path = os.path.join(hicode_dir, f"{method}_hicode.pkl")

        if not os.path.exists(pkl_path):
            print(f"Warning: {pkl_path} not found, skipping...")
            continue

        with open(pkl_path, 'rb') as f:
            results[method] = pickle.load(f)

    return results


def get_highly_relevant_docs(query_id: str = '43') -> Set[str]:
    """
    Get set of highly relevant document IDs from TREC qrels.

    Args:
        query_id: Query ID to filter

    Returns:
        Set of doc_ids with score == 2 (highly relevant)
    """
    from datasets import load_dataset

    qrels = load_dataset('BeIR/trec-covid-qrels', split='test')

    highly_relevant = set()
    for entry in qrels:
        if str(entry['query-id']) == query_id and entry['score'] == 2:
            highly_relevant.add(entry['corpus-id'])

    return highly_relevant


def flatten_topic_assignments(topics: List) -> List[int]:
    """
    Flatten HiCode's multi-label topic assignments to single-label.

    Args:
        topics: List of topic assignments (may be List[List[int]] or List[int])

    Returns:
        Flattened list of topic IDs
    """
    flattened = []
    for t in topics:
        if isinstance(t, list):
            flattened.append(t[0] if len(t) > 0 else -1)
        else:
            flattened.append(t)
    return flattened


def compute_relevance_metrics(
    hicode_labeled_docs: Set[str],
    highly_relevant_in_sample: Set[str]
) -> Tuple[int, float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        hicode_labeled_docs: Set of doc_ids labeled as relevant by HiCode (topic != -1)
        highly_relevant_in_sample: Set of doc_ids that are highly relevant (score == 2)

    Returns:
        Tuple of (overlap_count, precision, recall, f1)
    """
    overlap = hicode_labeled_docs & highly_relevant_in_sample
    overlap_count = len(overlap)

    # Precision: of docs HiCode says are relevant, what % are actually highly relevant?
    precision = (overlap_count / len(hicode_labeled_docs) * 100) if len(hicode_labeled_docs) > 0 else 0.0

    # Recall: of all highly relevant docs in sample, what % did HiCode find?
    recall = (overlap_count / len(highly_relevant_in_sample) * 100) if len(highly_relevant_in_sample) > 0 else 0.0

    # F1 Score
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return overlap_count, precision, recall, f1


def analyze_per_topic_contribution(
    doc_ids: List[str],
    topic_assignments: List[int],
    highly_relevant_docs: Set[str],
    topic_names: Dict[int, str]
) -> Dict[int, int]:
    """
    Analyze which topics contributed to correctly identifying relevant docs.

    Args:
        doc_ids: List of document IDs
        topic_assignments: List of topic assignments (flattened)
        highly_relevant_docs: Set of highly relevant doc_ids
        topic_names: Mapping of topic IDs to names

    Returns:
        Dictionary mapping topic_id to count of correctly identified docs
    """
    topic_contributions = defaultdict(int)

    for doc_id, topic_id in zip(doc_ids, topic_assignments):
        if topic_id != -1 and doc_id in highly_relevant_docs:
            topic_contributions[topic_id] += 1

    return dict(topic_contributions)


def count_missed_relevant_docs(
    doc_ids: List[str],
    topic_assignments: List[int],
    highly_relevant_docs: Set[str]
) -> int:
    """
    Count how many highly relevant docs were marked as irrelevant (-1) by HiCode.

    Args:
        doc_ids: List of document IDs
        topic_assignments: List of topic assignments (flattened)
        highly_relevant_docs: Set of highly relevant doc_ids

    Returns:
        Count of missed relevant documents
    """
    missed = 0
    for doc_id, topic_id in zip(doc_ids, topic_assignments):
        if topic_id == -1 and doc_id in highly_relevant_docs:
            missed += 1
    return missed


def generate_text_report(results: List[Dict], output_path: str, query_text: str):
    """
    Generate human-readable text report.

    Args:
        results: List of result dictionaries for each method
        output_path: Path to save the report
        query_text: Query text for header
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HiCode Relevance Analysis - Query 43\n")
        f.write("=" * 80 + "\n")
        f.write(f'Query: "{query_text}"\n')
        f.write("\n")
        f.write("Metric: Highly Relevant Documents (TREC score == 2)\n")
        f.write("Sample Size: 1000 docs per method\n")
        f.write("\n")

        # Per-method detailed reports
        for result in results:
            f.write("-" * 80 + "\n")
            f.write(f"Method: {result['method']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Total docs in sample:                {result['total_docs']:4d}\n")
            f.write(f"  HiCode labeled as relevant:          {result['hicode_labeled']:4d}\n")
            f.write(f"  Highly relevant in sample:           {result['highly_relevant_in_sample']:4d}\n")
            f.write(f"  Correctly identified (overlap):      {result['overlap']:4d}\n")
            f.write(f"  Missed (relevant but labeled -1):    {result['missed']:4d}\n")
            f.write("\n")
            f.write(f"  Precision: {result['precision']:5.1f}% ({result['overlap']}/{result['hicode_labeled']})\n")
            f.write(f"  Recall:    {result['recall']:5.1f}% ({result['overlap']}/{result['highly_relevant_in_sample']})\n")
            f.write(f"  F1 Score:  {result['f1']:5.1f}%\n")
            f.write("\n")

            # Per-topic breakdown
            if result['topic_contributions']:
                f.write("  Correctly identified docs by topic:\n")
                for topic_id in sorted(result['topic_contributions'].keys()):
                    count = result['topic_contributions'][topic_id]
                    topic_name = result['topic_names'].get(topic_id, 'Unknown')
                    f.write(f"    Topic {topic_id:2d} ({topic_name:40s}): {count:2d} docs\n")
            else:
                f.write("  No correctly identified docs (precision = 0)\n")
            f.write("\n")

        # Summary comparison table
        f.write("=" * 80 + "\n")
        f.write("Summary Comparison\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Method':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Labeled':>10}\n")
        f.write("-" * 80 + "\n")

        for result in results:
            f.write(f"{result['method']:<25} "
                   f"{result['precision']:>9.1f}% "
                   f"{result['recall']:>9.1f}% "
                   f"{result['f1']:>9.1f}% "
                   f"{result['hicode_labeled']:>10d}\n")

        f.write("=" * 80 + "\n")

    print(f"✓ Text report saved to: {output_path}")


def generate_csv_report(results: List[Dict], output_path: str):
    """
    Generate CSV report for easier analysis.

    Args:
        results: List of result dictionaries for each method
        output_path: Path to save the CSV
    """
    fieldnames = [
        'method',
        'total_docs',
        'hicode_labeled',
        'highly_relevant_in_sample',
        'correctly_identified',
        'missed_relevant',
        'precision',
        'recall',
        'f1'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                'method': result['method'],
                'total_docs': result['total_docs'],
                'hicode_labeled': result['hicode_labeled'],
                'highly_relevant_in_sample': result['highly_relevant_in_sample'],
                'correctly_identified': result['overlap'],
                'missed_relevant': result['missed'],
                'precision': round(result['precision'], 1),
                'recall': round(result['recall'], 1),
                'f1': round(result['f1'], 1)
            })

    print(f"✓ CSV report saved to: {output_path}")


def main():
    """Main analysis function."""

    # ===== CONFIGURATION =====
    QUERY_ID = '43'
    HICODE_DIR = '/home/srangre1/results/trec-covid/hicode/query_43/topic_models'
    OUTPUT_DIR = '/home/srangre1/results/trec-covid/hicode/query_43/results/topics_summary'

    METHODS = [
        'random_uniform',
        'direct_retrieval',
        'direct_retrieval_mmr',
        'query_expansion',
        'keyword_search',
        'retrieval_random'
    ]

    QUERY_TEXT = "How has the COVID-19 pandemic impacted violence in society, including violent crimes?"

    # ===== LOAD DATA =====
    print("=" * 80)
    print("HiCode Relevance Analysis")
    print("=" * 80)
    print(f"\nQuery {QUERY_ID}: {QUERY_TEXT}\n")

    print("Loading HiCode results...")
    hicode_results = load_hicode_results(HICODE_DIR, METHODS)
    print(f"  ✓ Loaded {len(hicode_results)} methods\n")

    print("Loading TREC qrels (highly relevant docs, score == 2)...")
    all_highly_relevant = get_highly_relevant_docs(QUERY_ID)
    print(f"  ✓ Found {len(all_highly_relevant)} highly relevant docs for query {QUERY_ID}\n")

    # ===== COMPUTE METRICS =====
    print("Computing relevance metrics for each method...\n")

    analysis_results = []

    for method in METHODS:
        if method not in hicode_results:
            print(f"  ✗ Skipping {method} (not loaded)")
            continue

        hicode_data = hicode_results[method]

        # Get doc_ids and flatten topic assignments
        doc_ids = hicode_data['doc_ids']
        topic_assignments = flatten_topic_assignments(hicode_data['topics'])
        topic_names = hicode_data['topic_names']

        # Build sets
        hicode_labeled = set(
            doc_id for doc_id, topic_id in zip(doc_ids, topic_assignments)
            if topic_id != -1
        )

        highly_relevant_in_sample = all_highly_relevant & set(doc_ids)

        # Compute metrics
        overlap, precision, recall, f1 = compute_relevance_metrics(
            hicode_labeled,
            highly_relevant_in_sample
        )

        # Per-topic analysis
        topic_contributions = analyze_per_topic_contribution(
            doc_ids,
            topic_assignments,
            highly_relevant_in_sample,
            topic_names
        )

        # Missed relevant docs
        missed = count_missed_relevant_docs(
            doc_ids,
            topic_assignments,
            highly_relevant_in_sample
        )

        # Store results
        result = {
            'method': method,
            'total_docs': len(doc_ids),
            'hicode_labeled': len(hicode_labeled),
            'highly_relevant_in_sample': len(highly_relevant_in_sample),
            'overlap': overlap,
            'missed': missed,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'topic_contributions': topic_contributions,
            'topic_names': topic_names
        }

        analysis_results.append(result)

        print(f"  ✓ {method:25s} - P: {precision:5.1f}%  R: {recall:5.1f}%  F1: {f1:5.1f}%")

    # ===== GENERATE REPORTS =====
    print(f"\nGenerating reports...\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txt_path = os.path.join(OUTPUT_DIR, 'relevance_analysis.txt')
    csv_path = os.path.join(OUTPUT_DIR, 'relevance_metrics.csv')

    generate_text_report(analysis_results, txt_path, QUERY_TEXT)
    generate_csv_report(analysis_results, csv_path)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nReports saved to:")
    print(f"  - {txt_path}")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
