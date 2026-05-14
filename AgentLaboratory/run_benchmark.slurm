#!/bin/bash
#SBATCH --job-name=agent_laboratory_benchmark       # Job name
#SBATCH --output=logs/%x_%j.out        # Standard output and error log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=p_80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# module load python/3.9

source .venv/bin/activate

uv run run_benchmark.py