#!/bin/bash
# grid/run_create_filtered_corpus.sh
#
# One-time corpus-preparation stage for the doctor-reviews dataset: runs
# create_filtered_doctor_corpus.py, which applies the 9 documented filters
# (see DOCTOR_REVIEW_FILTERING_GUIDE.md) and saves a HuggingFace-disk-format
# corpus consumed by dataset_loaders.py's _load_doctor_reviews(). Only needs
# to be re-run if the filtering logic or source data changes.
#
# NOTE: create_filtered_doctor_corpus.py has its input/output paths hardcoded
# inside it - see CONFIGURATION section in USAGE.md before running under a
# different account.

#SBATCH --job-name=create_filtered_corpus
#SBATCH --output=create_filtered_corpus_%j.out
#SBATCH --error=create_filtered_corpus_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "===== Creating Filtered Doctor Reviews Corpus ====="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

python "$PROJECT_DIR/create_filtered_doctor_corpus.py"

echo ""
echo "===== Job Complete ====="
