#!/bin/bash
# grid/run_regenerate_diversity_plots.sh
#
# LEGACY ONE-OFF UTILITY - not part of the core pipeline stages in PIPELINE.md.
# Runs regenerate_diversity_plots.py, a standalone script written to reformat
# aggregate diversity plots with condensed bars for a specific past report.
# Its target paths (results/trec-covid/{bertopic,topicgpt}/aggregate_results/
# per_method_aggregates.json) predate the dual-embedding-model rework and do
# NOT reflect the current results/{dataset}/{model}/aggregate_results/{tag}/
# layout - see CACHING.md. Kept for reference; expect it to need path updates
# before it will run against current results.

#SBATCH --job-name=regen_diversity_plots  # Job name
#SBATCH --output=regen_diversity_plots_%j.out  # Standard output log
#SBATCH --error=regen_diversity_plots_%j.err   # Standard error log
#SBATCH --partition=cpu                # CPU partition (no GPU needed for plotting)
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=2              # CPU cores (plotting is light)
#SBATCH --mem=4G                       # Total memory (4GB sufficient for plotting)
#SBATCH --time=00:10:00                # Time limit: 10 minutes (should take < 1 min)
#SBATCH --mail-type=NONE               # Mail events

# ============================================================================
# CONFIGURATION - EDIT THESE FOR YOUR OWN ACCOUNT BEFORE RUNNING
# ============================================================================
PROJECT_DIR="$HOME/Iterative-Query-Refinement"   # Your clone of this repo
CONDA_ENV="coreset_proj"                          # Your conda env name
# ============================================================================

# Initialize module system
source /etc/profile.d/modules.sh

# Load the Conda module
module purge

# Initialize Conda for the shell
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate "$CONDA_ENV"

echo "===== Environment Info ====="
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "============================"

cd "$PROJECT_DIR"

echo ""
echo "===== Running Diversity Plot Regeneration ====="
echo "Script: regenerate_diversity_plots.py"
echo "================================================"
echo ""

python regenerate_diversity_plots.py

if [ $? -eq 0 ]; then
    echo ""
    echo "===== Job Complete ====="
    echo "✅ Plots regenerated successfully!"
else
    echo ""
    echo "===== Job Failed ====="
    echo "❌ Error occurred during plot regeneration"
    exit 1
fi
