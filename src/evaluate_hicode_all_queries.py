#!/usr/bin/env python3
# evaluate_hicode_all_queries.py
"""
Multi-query evaluation script for HiCode topic modeling results.

Evaluates HiCode results across all queries and aggregates results.

Usage:
    python src/evaluate_hicode_all_queries.py

    # Or with specific queries:
    python src/evaluate_hicode_all_queries.py --queries 2 10 13 43

    # Skip aggregation (just run individual queries):
    python src/evaluate_hicode_all_queries.py --skip-aggregation
"""

import os
import sys
import argparse
import logging
import torch
from typing import List, Optional

# Add src to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_hicode import run_hicode_evaluation
from end_to_end_evaluation import aggregate_cross_query_results, create_cross_query_intrinsic_plots
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_available_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("MPS (Apple Silicon) available")
    else:
        device = "cpu"
        logger.info("No GPU available, using CPU")
    return device


def discover_available_queries(hicode_base_dir: str) -> List[str]:
    """
    Discover all available query directories in the HiCode results.

    Args:
        hicode_base_dir: Base directory containing query_X folders

    Returns:
        List of query IDs (sorted numerically)
    """
    query_ids = []

    if not os.path.exists(hicode_base_dir):
        logger.error(f"HiCode base directory not found: {hicode_base_dir}")
        return []

    for item in os.listdir(hicode_base_dir):
        if item.startswith("query_"):
            query_id = item.replace("query_", "")
            query_dir = os.path.join(hicode_base_dir, item)

            # Verify it has the expected pickle files
            expected_files = [
                "direct_retrieval_hicode.pkl",
                "random_uniform_hicode.pkl"
            ]

            has_files = all(
                os.path.exists(os.path.join(query_dir, f))
                for f in expected_files
            )

            if has_files:
                query_ids.append(query_id)
            else:
                logger.warning(f"Query {query_id} missing expected files, skipping")

    # Sort numerically
    query_ids.sort(key=int)
    return query_ids


def run_all_queries(
    query_ids: List[str],
    hicode_base_dir: str,
    samples_base_dir: str,
    output_dir: str,
    device: str,
    sample_size: int = 1000,
    embedding_model: str = "all-mpnet-base-v2",
    dataset_name: str = "trec-covid"
) -> List[str]:
    """
    Run HiCode evaluation for all queries.

    Args:
        query_ids: List of query IDs to evaluate
        hicode_base_dir: Base directory with HiCode results (query_X subfolders)
        samples_base_dir: Base directory with samples (query_X/samples subfolders)
        output_dir: Output directory for results
        device: Device to use for computation
        sample_size: Expected sample size
        embedding_model: Embedding model name
        dataset_name: Dataset name

    Returns:
        List of successfully processed query IDs
    """
    successful_queries = []
    failed_queries = []

    total = len(query_ids)

    for idx, query_id in enumerate(query_ids, 1):
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"PROCESSING QUERY {query_id} ({idx}/{total})")
        logger.info("=" * 80)

        # Construct paths for this query
        hicode_dir = os.path.join(hicode_base_dir, f"query_{query_id}")
        samples_dir = os.path.join(samples_base_dir, f"query_{query_id}", "samples")

        # Verify paths exist
        if not os.path.exists(hicode_dir):
            logger.error(f"HiCode directory not found: {hicode_dir}")
            failed_queries.append((query_id, "HiCode directory not found"))
            continue

        if not os.path.exists(samples_dir):
            logger.error(f"Samples directory not found: {samples_dir}")
            failed_queries.append((query_id, "Samples directory not found"))
            continue

        try:
            results = run_hicode_evaluation(
                query_id=query_id,
                hicode_dir=hicode_dir,
                samples_dir=samples_dir,
                output_dir=output_dir,
                sample_size=sample_size,
                embedding_model=embedding_model,
                dataset_name=dataset_name,
                device=device,
                random_seed=42
            )

            if results:
                successful_queries.append(query_id)
                logger.info(f"Query {query_id} completed successfully")
            else:
                failed_queries.append((query_id, "Evaluation returned None"))
                logger.error(f"Query {query_id} failed (returned None)")

        except Exception as e:
            failed_queries.append((query_id, str(e)))
            logger.error(f"Query {query_id} failed with error: {e}", exc_info=True)
            continue

        # Log GPU memory if using CUDA
        if device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("QUERY PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total queries: {total}")
    logger.info(f"Successful: {len(successful_queries)}")
    logger.info(f"Failed: {len(failed_queries)}")

    if failed_queries:
        logger.info("\nFailed queries:")
        for qid, reason in failed_queries:
            logger.info(f"  - Query {qid}: {reason}")

    return successful_queries


def main():
    """Main function - configure and run multi-query HiCode evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate HiCode topic modeling results across multiple queries"
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        type=str,
        help="Specific query IDs to evaluate (default: auto-discover all)"
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Skip cross-query aggregation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'mps', or 'cpu' (default: auto-detect)"
    )
    parser.add_argument(
        "--hicode-dir",
        type=str,
        default="/export/fs06/mzhong8/hicode/results/assignment",
        help="Base directory containing HiCode results"
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default="/home/srangre1/results/trec-covid/bertopic",
        help="Base directory containing samples"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/srangre1/results",
        help="Output directory for evaluation results"
    )

    args = parser.parse_args()

    # ===== CONFIGURATION =====

    # Paths
    HICODE_BASE_DIR = args.hicode_dir
    SAMPLES_BASE_DIR = args.samples_dir
    OUTPUT_DIR = args.output_dir

    # Model configuration
    SAMPLE_SIZE = 1000
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    DATASET_NAME = "trec-covid"

    # Device configuration - auto-detect if not specified
    DEVICE = args.device if args.device else get_available_device()
    logger.info(f"Using device: {DEVICE}")

    # ===== DISCOVER OR USE SPECIFIED QUERIES =====

    if args.queries:
        query_ids = args.queries
        logger.info(f"Using specified queries: {query_ids}")
    else:
        logger.info(f"Auto-discovering queries from: {HICODE_BASE_DIR}")
        query_ids = discover_available_queries(HICODE_BASE_DIR)
        logger.info(f"Discovered {len(query_ids)} queries: {query_ids}")

    if not query_ids:
        logger.error("No queries to process!")
        return 1

    # ===== RUN EVALUATION FOR ALL QUERIES =====

    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING MULTI-QUERY HICODE EVALUATION")
    logger.info("=" * 80)
    logger.info(f"HiCode results: {HICODE_BASE_DIR}")
    logger.info(f"Samples: {SAMPLES_BASE_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Queries: {query_ids}")
    logger.info(f"Device: {DEVICE}")

    successful_queries = run_all_queries(
        query_ids=query_ids,
        hicode_base_dir=HICODE_BASE_DIR,
        samples_base_dir=SAMPLES_BASE_DIR,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        sample_size=SAMPLE_SIZE,
        embedding_model=EMBEDDING_MODEL,
        dataset_name=DATASET_NAME
    )

    # ===== AGGREGATE CROSS-QUERY RESULTS =====

    if not args.skip_aggregation and len(successful_queries) > 1:
        logger.info("")
        logger.info("=" * 80)
        logger.info("AGGREGATING RESULTS ACROSS ALL QUERIES")
        logger.info("=" * 80)

        # Results are stored in: OUTPUT_DIR/DATASET_NAME/hicode/query_X/results/
        results_base_dir = os.path.join(OUTPUT_DIR, DATASET_NAME, "hicode")
        aggregate_output_dir = os.path.join(results_base_dir, "aggregate_results")

        try:
            aggregate_results = aggregate_cross_query_results(
                results_base_dir=results_base_dir,
                query_ids=successful_queries,
                output_dir=aggregate_output_dir
            )

            if aggregate_results:
                # Also create cross-query intrinsic plots
                logger.info("\nCreating cross-query intrinsic metric visualizations...")
                plots_dir = os.path.join(aggregate_output_dir, "plots")
                create_cross_query_intrinsic_plots(
                    aggregate_results['raw_per_method'],
                    plots_dir
                )

                logger.info("")
                logger.info("=" * 80)
                logger.info("AGGREGATION COMPLETE!")
                logger.info("=" * 80)
                logger.info(f"Aggregate results saved to: {aggregate_output_dir}")
            else:
                logger.error("Aggregation failed!")

        except Exception as e:
            logger.error(f"Aggregation failed with error: {e}", exc_info=True)

    elif len(successful_queries) <= 1:
        logger.info("\nSkipping cross-query aggregation (need at least 2 successful queries)")

    # ===== FINAL SUMMARY =====

    logger.info("")
    logger.info("=" * 80)
    logger.info("ALL QUERIES COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Processed {len(successful_queries)} queries: {successful_queries}")
    logger.info(f"Results saved to: {os.path.join(OUTPUT_DIR, DATASET_NAME, 'hicode')}")
    logger.info("=" * 80)

    return 0 if successful_queries else 1


if __name__ == "__main__":
    sys.exit(main())
