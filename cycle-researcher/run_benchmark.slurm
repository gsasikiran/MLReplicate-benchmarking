#!/bin/bash
#SBATCH --job-name=cycle_researcher_benchmark       # Job name
#SBATCH --output=logs/%x_%j.out        # Standard output and error log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00                # Time limit hrs:min:sec
#SBATCH --partition=gpu                # Partition name (use 'cpu' if no GPU)
#SBATCH --gres=gpu:4                   # Number of GPUs (remove if CPU-only)


# Load required modules
module purge
module load python/3.11
# module load cuda/12.2   # Optional, only if using GPU

# Activate your virtual environment (optional)


# Create logs directory if it doesn't exist
mkdir -p logs

# Run your benchmark
uv run run_benchmark.py
