#!/bin/bash
#SBATCH --job-name=tiny-scientist-batch
#SBATCH --output=slurm_output_%j.log
#SBATCH --error=slurm_error_%j.log
#SBATCH --time=16:00:00                    # Time limit (2 hours with buffer for 1 paper)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --partition=p_48G                  # Use RTX 3090/L40S partition




INTENTS_FILE="intents.txt"
TRACKER_SCRIPT="run_tiny_scientist.py"
MODEL="gpt-4o"
BUDGET=10.0
MAX_RETRIES=2


 eval "$(conda shell.bash hook)"
 conda activate tiny-scientist

echo "========================================="
echo "SLURM Job Information"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""

# Check if files exist
if [ ! -f "$INTENTS_FILE" ]; then
    echo "Error: $INTENTS_FILE not found!"
    exit 1
fi

if [ ! -f "$TRACKER_SCRIPT" ]; then
    echo "Error: $TRACKER_SCRIPT not found!"
    exit 1
fi

# Count total intents
TOTAL=$(wc -l < "$INTENTS_FILE")
echo "Total intents: $TOTAL"
echo "Model: $MODEL"
echo "Budget per run: \$$BUDGET"
echo "========================================="
echo ""

# Function to check if LaTeX was generated (success criteria)
check_latex_success() {
    local latest_dir=$(ls -td tracked_experiments/*/ 2>/dev/null | head -1)
    if [ -z "$latest_dir" ]; then
        return 1
    fi
    
    # Check for LaTeX file in tiny_scientist_output
    local latex_file="${latest_dir}tiny_scientist_output/latex/acl_latex.tex"
    if [ -f "$latex_file" ]; then
        echo "  ✓ Found LaTeX: $latex_file"
        return 0
    fi
    
    return 1
}

# Read intents and run with retry logic
CURRENT=1
SUCCESS_COUNT=0
FAIL_COUNT=0

while IFS= read -r intent || [ -n "$intent" ]; do

    if [ -z "$intent" ] || [[ "$intent" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    echo ""
    echo "========================================="
    echo "[$CURRENT/$TOTAL] Processing intent"
    echo "========================================="
    echo "Intent: $intent"
    echo ""
    
    SUCCESS=0
    for ATTEMPT in $(seq 1 $MAX_RETRIES); do
        if [ $ATTEMPT -gt 1 ]; then
            echo ""
            echo "  Retry attempt $ATTEMPT/$MAX_RETRIES..."
            echo ""
        fi
        
        python "$TRACKER_SCRIPT" "$intent"
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            if check_latex_success; then
                echo ""
                echo "[$CURRENT/$TOTAL] Attempt $ATTEMPT SUCCEEDED"
                SUCCESS=1
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                break
            else
                echo ""
                echo "[$CURRENT/$TOTAL] Attempt $ATTEMPT completed but no LaTeX found"
            fi
        else
            echo ""
            echo "[$CURRENT/$TOTAL] Attempt $ATTEMPT FAILED (exit code: $EXIT_CODE)"
        fi
        
        if [ $ATTEMPT -lt $MAX_RETRIES ] && [ $SUCCESS -eq 0 ]; then
            echo "  Waiting 30 seconds before retry..."
            sleep 30
        fi
    done
    

    if [ $SUCCESS -eq 0 ]; then
        echo ""
        echo "[$CURRENT/$TOTAL] FAILED after $MAX_RETRIES attempts"
        echo "  Moving to next intent..."
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    CURRENT=$((CURRENT + 1))
    
    if [ $CURRENT -le $TOTAL ]; then
        echo ""
        echo "Waiting 10 seconds before next intent..."
        sleep 10
    fi
    
done < "$INTENTS_FILE"

echo ""
echo "Batch Run Complete!"
echo "Job ID: $SLURM_JOB_ID"
echo "Total intents: $TOTAL"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""

if [ $SUCCESS_COUNT -eq 0 ]; then
    echo "All experiments failed!"
    exit 1
fi

echo "✓ Job completed successfully"
exit 0
