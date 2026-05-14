#!/bin/bash
#SBATCH --job-name=v2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:t2080ti:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --partition=p_12G  # Specify the partition explicitly
#SBATCH --output=/nfs/home/<Username>/AI_Scientist_v2/AI-Scientist-v2/v2.log

# Load your Conda environment (replace 'myenv' with your environment name)
source ~/miniconda3/bin/activate

conda activate ai_scientist

export PDFLATEX=/nfs/home/<Username>/miniconda3/envs/ai_scientist/bin/pdflatex

python -u /nfs/home/<Username>/research/MLReplicate/AI-Scientist-v2/process_papers.py