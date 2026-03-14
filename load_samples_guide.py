"""
Reference Code: How to Load and Use Sampled Documents

This script shows how to load and work with the sampled documents from:
/path/to/results/bertopic/query_43/samples/

Each .pkl file contains a dictionary with:
- method: Sampling method name (str)
- doc_ids: List of document IDs (List[str])
- doc_texts: List of full document texts (List[str])
- sample_size: Number of documents (int)

Author: Sample Loading Guide
Date: 2025-12-04
"""

import pickle
import os
import pandas as pd
from typing import Dict, List, Any


# ============================================================================
# BASIC LOADING
# ============================================================================

def load_sample(sample_path: str) -> Dict[str, Any]:
    """
    Load a single sample file.

    Args:
        sample_path: Path to .pkl file

    Returns:
        Dictionary with keys: ['method', 'doc_ids', 'doc_texts', 'sample_size']
    """
    with open(sample_path, 'rb') as f:
        sample = pickle.load(f)
    return sample


def load_all_samples(samples_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all sampling methods from a directory.

    Args:
        samples_dir: Path to samples directory

    Returns:
        Dictionary mapping method_name -> sample_data
    """
    samples = {}

    # Standard sampling method names
    method_files = [
        'random_uniform.pkl',
        'keyword_search.pkl',
        'direct_retrieval.pkl',
        'direct_retrieval_mmr.pkl',
        'query_expansion.pkl',
        'retrieval_random.pkl'
    ]

    for filename in method_files:
        filepath = os.path.join(samples_dir, filename)
        if os.path.exists(filepath):
            method_name = filename.replace('.pkl', '')
            samples[method_name] = load_sample(filepath)
            print(f"✓ Loaded {method_name}: {samples[method_name]['sample_size']} documents")
        else:
            print(f"✗ Not found: {filename}")

    return samples


# ============================================================================
# EXAMPLE 1: BASIC USAGE
# ============================================================================

def example_basic_usage(samples_dir: str):
    """Basic example: Load and inspect a sample"""

    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)

    # Load one sample
    sample = load_sample(os.path.join(samples_dir, "direct_retrieval.pkl"))

    print(f"\nMethod: {sample['method']}")
    print(f"Sample size: {sample['sample_size']}")
    print(f"Number of documents: {len(sample['doc_ids'])}")

    print("\n" + "-"*80)
    print("First document:")
    print(f"ID: {sample['doc_ids'][0]}")
    print(f"Text (first 300 chars):\n{sample['doc_texts'][0][:300]}...")

    print("\n" + "-"*80)
    print("Access specific document by index:")
    doc_index = 5
    print(f"Document #{doc_index}:")
    print(f"  ID: {sample['doc_ids'][doc_index]}")
    print(f"  Text: {sample['doc_texts'][doc_index][:200]}...")

    return sample


# ============================================================================
# EXAMPLE 2: CONVERT TO PANDAS DATAFRAME
# ============================================================================

def sample_to_dataframe(sample: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert sample to pandas DataFrame for easier analysis.

    Args:
        sample: Sample dictionary

    Returns:
        DataFrame with columns: ['doc_id', 'doc_text', 'method']
    """
    df = pd.DataFrame({
        'doc_id': sample['doc_ids'],
        'doc_text': sample['doc_texts'],
        'method': sample['method']
    })
    return df


def example_pandas_conversion(samples_dir: str):
    """Example: Convert to DataFrame"""

    print("\n" + "="*80)
    print("EXAMPLE 2: Convert to Pandas DataFrame")
    print("="*80)

    sample = load_sample(os.path.join(samples_dir, "direct_retrieval.pkl"))

    # Convert to DataFrame
    df = sample_to_dataframe(sample)

    print("\nDataFrame shape:", df.shape)
    print("\nFirst 3 rows:")
    print(df[['doc_id', 'method']].head(3))
    print("\nFirst document text:")
    print(df.iloc[0]['doc_text'][:300] + "...")

    return df


# ============================================================================
# EXAMPLE 3: LOAD ALL METHODS
# ============================================================================

def example_load_all_methods(samples_dir: str):
    """Example: Load all sampling methods"""

    print("\n" + "="*80)
    print("EXAMPLE 3: Load All Sampling Methods")
    print("="*80)

    # Load all samples
    all_samples = load_all_samples(samples_dir)

    print(f"\n✓ Loaded {len(all_samples)} sampling methods")

    # Access each method
    for method_name, sample in all_samples.items():
        print(f"\n{method_name}:")
        print(f"  Documents: {len(sample['doc_ids'])}")
        print(f"  First doc ID: {sample['doc_ids'][0]}")

    return all_samples


# ============================================================================
# EXAMPLE 4: EXPORT TO CSV/JSON
# ============================================================================

def export_sample(sample: Dict[str, Any], output_dir: str, format: str = 'csv'):
    """
    Export sample to CSV or JSON.

    Args:
        sample: Sample dictionary
        output_dir: Output directory
        format: 'csv' or 'json'
    """
    os.makedirs(output_dir, exist_ok=True)

    method_name = sample['method']
    df = sample_to_dataframe(sample)

    if format == 'csv':
        output_path = os.path.join(output_dir, f"{method_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"✓ Saved to CSV: {output_path}")

    elif format == 'json':
        output_path = os.path.join(output_dir, f"{method_name}.json")
        df.to_json(output_path, orient='records', indent=2)
        print(f"✓ Saved to JSON: {output_path}")


def example_export(samples_dir: str):
    """Example: Export samples to CSV/JSON"""

    print("\n" + "="*80)
    print("EXAMPLE 4: Export to CSV/JSON")
    print("="*80)

    sample = load_sample(os.path.join(samples_dir, "direct_retrieval.pkl"))

    output_dir = "exported_samples"

    print("\nExporting to CSV and JSON...")
    export_sample(sample, output_dir, format='csv')
    export_sample(sample, output_dir, format='json')

    print(f"\n✓ Exports saved to: {output_dir}/")


# ============================================================================
# EXAMPLE 5: ITERATE OVER DOCUMENTS
# ============================================================================

def example_iterate_documents(samples_dir: str):
    """Example: Iterate over all documents"""

    print("\n" + "="*80)
    print("EXAMPLE 5: Iterate Over Documents")
    print("="*80)

    sample = load_sample(os.path.join(samples_dir, "direct_retrieval.pkl"))

    print(f"\nIterating over {len(sample['doc_ids'])} documents...\n")

    # Iterate over first 5 documents
    for i, (doc_id, doc_text) in enumerate(zip(sample['doc_ids'][:5], sample['doc_texts'][:5])):
        print(f"Document {i+1}:")
        print(f"  ID: {doc_id}")
        print(f"  Length: {len(doc_text)} characters")
        print(f"  Preview: {doc_text[:100]}...")
        print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run all examples.

    To use this script:
    1. Update the SAMPLES_DIR path to your location
    2. Run: python load_samples_guide.py
    """

    # ===== CONFIGURE THIS PATH =====
    SAMPLES_DIR = "results/topic_eval_bertopic_6methods_2025-11-28/bertopic/query_43/samples"

    # Check if directory exists
    if not os.path.exists(SAMPLES_DIR):
        print(f"ERROR: Samples directory not found: {SAMPLES_DIR}")
        print("\nPlease update the SAMPLES_DIR variable to point to your samples directory.")
        exit(1)

    # Run all examples
    print("\n" + "="*80)
    print("SAMPLE LOADING REFERENCE GUIDE")
    print("="*80)

    # Example 1: Basic loading
    sample = example_basic_usage(SAMPLES_DIR)

    # Example 2: Pandas DataFrame
    df = example_pandas_conversion(SAMPLES_DIR)

    # Example 3: Load all methods
    all_samples = example_load_all_methods(SAMPLES_DIR)

    # Example 4: Export (commented out to avoid creating files)
    # example_export(SAMPLES_DIR)

    # Example 5: Iterate over documents
    example_iterate_documents(SAMPLES_DIR)

    print("\n" + "="*80)
    print("✓ All examples completed!")
    print("="*80)
