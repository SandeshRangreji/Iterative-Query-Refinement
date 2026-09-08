#!/bin/bash
# grid/run_generate_keywords.sh
#
# Keyword-generation stage: runs generate_doctor_review_keywords.py, which
# uses KeyBERT to extract query-expansion keywords for the doctor-reviews
# dataset and caches them to a JSON file consumed later by
# end_to_end_evaluation.py's query_expansion sampling method.
#
# NOTE: generate_doctor_review_keywords.py itself has CORPUS_PATH/OUTPUT_PATH
# hardcoded inside it (not exposed as flags here) - see CONFIGURATION section
# in USAGE.md for the exact lines to edit for your own account before running.
# It currently only covers a subset of doctor-reviews queries - see CACHING.md
# for why that's an intentional fallback, not a bug.

#SBATCH --job-name=generate_keywords
#SBATCH --output=generate_keywords_%j.out
#SBATCH --error=generate_keywords_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE

# ============================================================================
# CONFIGURATION - EDIT THESE FOR YOUR OWN ACCOUNT BEFORE RUNNING
# ============================================================================
PROJECT_DIR="$HOME/Iterative-Query-Refinement"   # Your clone of this repo
CONDA_ENV="coreset_proj"                          # Your conda env name
# ============================================================================

source /etc/profile.d/modules.sh
module purge
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Acquire GPU using the shared CLSP grid script (not account-specific).
if [ -f /home/gqin2/scripts/acquire-gpu ]; then
    source /home/gqin2/scripts/acquire-gpu
    echo "✅ GPU acquired via acquire-gpu script"
fi

echo "===== Generating Keywords for Doctor Review Queries ====="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

cd "$PROJECT_DIR"
python generate_doctor_review_keywords.py

echo ""
echo "===== Job Complete ====="
