#!/bin/bash
# grid/run_hicode_eval.sh
#
# Runs the HiCode multi-query evaluation (src/evaluate_hicode_all_queries.py)
# on the CLSP SLURM grid. HiCode is a separate track from the main
# end_to_end_evaluation.py pipeline - it evaluates topic assignments produced
# by an external HiCode run rather than running BERTopic/LDA/TopicGPT itself.
# See PIPELINE.md for how this fits alongside the other topic models.

#SBATCH --job-name=hicode_doctor         # Job name
#SBATCH --output=hicode_doctor_%j.out    # Standard output log
#SBATCH --error=hicode_doctor_%j.err     # Standard error log
#SBATCH --partition=gpu                  # GPU partition for accelerated embeddings
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                # CPU cores
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --mem=64G                        # Total memory (for 11 queries)
#SBATCH --time=12:00:00                  # Time limit hrs:min:sec (12 hours should be sufficient)
#SBATCH --mail-type=NONE                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --exclude=c05,octopod            # Exclude problematic GPU nodes

# ============================================================================
# CONFIGURATION - EDIT THESE FOR YOUR OWN ACCOUNT BEFORE RUNNING
# ============================================================================
PROJECT_DIR="$HOME/Iterative-Query-Refinement"    # Your clone of this repo
CONDA_ENV="coreset_proj"                           # Your conda env name
OUTPUT_DIR="$HOME/results"                         # Where evaluation results go
DATASET_NAME="trec-covid"                          # "trec-covid" or "doctor-reviews"

# HiCode's own topic-assignment output is produced by a *separate* HiCode run
# this pipeline does not run HiCode itself, only evaluates its output. The
# path below (/export/fs06/mzhong8/...) points at a specific lab member's
# shared-filesystem directory - confirm you have read access, or point this
# at your own HiCode output directory instead.
HICODE_EXTERNAL_DIR="/export/fs06/mzhong8/hicode/results/assignment"
# ============================================================================

# Initialize module system
source /etc/profile.d/modules.sh

# Load the Conda module
module purge

# Set CUDA library path (CRITICAL for CLSP grid)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Initialize Conda for the shell
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate "$CONDA_ENV"

# Acquire GPU using the shared CLSP grid script (prevents race conditions).
# This is a shared, grid-wide utility (not account-specific) - leave as-is.
if [ -f /home/gqin2/scripts/acquire-gpu ]; then
    source /home/gqin2/scripts/acquire-gpu
    echo "GPU acquired via acquire-gpu script"
else
    echo "acquire-gpu script not found, using SLURM allocation"
fi

echo "===== Environment Info ====="
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "PyTorch CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "============================"

# GPU verification
echo "===== GPU Verification ====="
nvidia-smi || echo "nvidia-smi not available"
echo ""
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "============================"

# CUDA compatibility check
python << 'EOFPYTHON'
import torch
import sys

if not torch.cuda.is_available():
    print("")
    print("ERROR: CUDA not available to PyTorch")
    print("Possible causes:")
    print("  1. LD_LIBRARY_PATH missing /usr/local/cuda/lib64")
    print("  2. CUDA_VISIBLE_DEVICES not set correctly")
    print("  3. GPU not allocated by SLURM")
    sys.exit(1)

print(f"CUDA verified: {torch.cuda.get_device_name(0)}")
EOFPYTHON

if [ $? -ne 0 ]; then
    echo "CUDA verification failed. Exiting."
    exit 1
fi

# =============================================================================
# Run HiCode Evaluation
# =============================================================================

echo ""
echo "===== Starting HiCode Evaluation ====="
echo "HiCode results: $HICODE_EXTERNAL_DIR"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "======================================="
echo ""

cd "$PROJECT_DIR"

# All queries found under HICODE_EXTERNAL_DIR are auto-discovered and
# processed; cross-query aggregation runs automatically at the end.
python src/evaluate_hicode_all_queries.py \
    --hicode-dir "$HICODE_EXTERNAL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-name "$DATASET_NAME"

echo ""
echo "===== Job Complete ====="
echo "Results saved to: $OUTPUT_DIR/$DATASET_NAME/hicode/"
echo "Aggregate results: $OUTPUT_DIR/$DATASET_NAME/hicode/aggregate_results/"
