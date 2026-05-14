#!/bin/bash
#SBATCH --job-name=ai_researcher           # Job name
#SBATCH --output=logs/ai_researcher_%j.out # Standard output log
#SBATCH --error=logs/ai_researcher_%j.err  # Error log
#SBATCH --time=02:00:00                    # Time limit (2 hours with buffer for 1 paper)
#SBATCH --partition=p_48G                  # Use RTX 3090/L40S partition
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --mem=32G                          # Memory allocation
#SBATCH --cpus-per-task=8                  # CPU cores (adjust based on needs)

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate ai_researcher_v2

# Change to project directory
cd projects/AI-Researcher

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo "=========================================="

# Check Singularity image exists
if [ ! -f "/nfs/home/<Username>/Projects/phd/airesearcher.sif" ]; then
    echo "ERROR: Singularity image not found!"
    exit 1
fi
export CATEGORY="vq"
export INSTANCE_ID="bayesian_quadrature"
export TASK_LEVEL="2" 
export MAX_ITER_TIMES="1"  


echo "Configuration:"
echo "  CATEGORY: $CATEGORY"
echo "  INSTANCE_ID: $INSTANCE_ID"
echo "  TASK_LEVEL: $TASK_LEVEL"
echo "  MAX_ITER_TIMES: $MAX_ITER_TIMES"
echo "=========================================="

# Run AI-Researcher
echo "Starting AI-Researcher..."
python run_ai_researcher.py 2>&1 | tee logs/run_${SLURM_JOB_ID}.log

# Capture exit code
EXIT_CODE=$?

# Print completion information
echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Clean up Singularity instances (in case they weren't stopped properly)
singularity instance list
echo "Stopping any remaining instances..."
singularity instance list | grep -v "INSTANCE NAME" | awk '{print $1}' | xargs -I {} singularity instance stop {} 2>/dev/null || true

exit $EXIT_CODE