#!/bin/bash
# grid/run_search.sh
#
# Runs the full sampling -> topic-modeling -> metrics -> statistics pipeline
# (src/end_to_end_evaluation.py) on the CLSP SLURM grid.
#
# NOTE ON THE JOB NAME: this is historically named "search_evaluation" from
# when it may have targeted a different script. It currently runs the full
# end_to_end_evaluation.py pipeline, not the standalone retrieval-quality
# evaluator in src/search.py. This is intentional/known — see PIPELINE.md
# for what src/search.py's own standalone evaluation looks like.
#
# Dataset / topic-model selection is controlled by env vars read inside
# end_to_end_evaluation.py's main(), not by flags to this script. Submit with:
#   sbatch --export=ALL,DATASET=trec-covid,MODEL=bertopic grid/run_search.sh
# See USAGE.md / CONFIGURATION section for all valid DATASET/MODEL values.

#SBATCH --job-name=search_evaluation  # Job name
#SBATCH --output=search_evaluation_%j.out  # Standard output log
#SBATCH --error=search_evaluation_%j.err   # Standard error log
#SBATCH --partition=gpu                # GPU partition for accelerated embeddings
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=8              # CPU cores (reduced - GPU does heavy lifting)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=16G                      # Total memory (peak usage ~5GB, 24GB provides headroom)
#SBATCH --time=14-00:00:00             # Time limit: 14 days (2 weeks)
#SBATCH --mail-type=NONE               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --exclude=c05                  # Exclude node c05 (GPU driver issues)

# ============================================================================
# CONFIGURATION - EDIT THESE FOR YOUR OWN ACCOUNT BEFORE RUNNING
# ============================================================================
PROJECT_DIR="$HOME/Iterative-Query-Refinement"   # Your clone of this repo
CONDA_ENV="coreset_proj"                          # Your conda env name

# NOT set here: OUTPUT_DIR ("results/") is hardcoded inside main() in
# src/end_to_end_evaluation.py (currently /home/srangre1/results is used as
# the working assumption throughout the docs) - edit OUTPUT_DIR there, or run
# from a working directory where a relative "results/" is where you want it.
# See CONFIGURATION section in USAGE.md for the full list of paths baked into
# end_to_end_evaluation.py that are specific to the original author's account.
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

# Fix SSL cert path for compute nodes: system /usr/lib/ssl/cert.pem may not
# exist on all nodes, so use certifi's bundle from the conda env (on NFS)
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

# Acquire GPU using the shared CLSP grid script (prevents race conditions).
# This is a shared, grid-wide utility (not account-specific) - leave as-is.
if [ -f /home/gqin2/scripts/acquire-gpu ]; then
    source /home/gqin2/scripts/acquire-gpu
    echo "✅ GPU acquired via acquire-gpu script"
else
    echo "⚠️  acquire-gpu script not found, using SLURM allocation"
fi

echo "===== Environment Info ====="
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "PyTorch CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "DATASET: ${DATASET:-trec-covid (default)}"
echo "MODEL: ${MODEL:-bertopic (default)}"
echo "============================"

# GPU verification
echo "===== GPU Verification ====="
nvidia-smi || echo "⚠️  nvidia-smi not available"
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
    print("❌ ERROR: CUDA not available to PyTorch")
    print("Possible causes:")
    print("  1. LD_LIBRARY_PATH missing /usr/local/cuda/lib64")
    print("  2. CUDA_VISIBLE_DEVICES not set correctly")
    print("  3. GPU not allocated by SLURM")
    sys.exit(1)

print(f"✅ CUDA verified: {torch.cuda.get_device_name(0)}")
EOFPYTHON

if [ $? -ne 0 ]; then
    echo "CUDA verification failed. Exiting."
    exit 1
fi

# Run end-to-end topic modeling evaluation
python "$PROJECT_DIR/src/end_to_end_evaluation.py"

echo "===== Job Complete ====="
echo "Results saved to: see OUTPUT_DIR in src/end_to_end_evaluation.py's main()"
