
current_dir=$(dirname "$(readlink -f "$0")")
cd $current_dir

# NEW: Use centralized directory (will be handled by integration_helper)
export DOCKER_WORKPLACE_NAME=workplace

export BASE_IMAGES=tjbtech1/metachain:amd64_latest



export COMPLETION_MODEL=anthropic/claude-3-5-sonnet-20241022
export CHEEP_MODEL=anthropic/claude-3-5-haiku-20241022

category=${CATEGORY:-ml_test}
instance_id=${INSTANCE_ID:-geoparse_test}
export GPUS='"device=0,1"'

python run_infer_plan.py --instance_path ../benchmark/final/${category}/${instance_id}.json \
    --container_name test_eval \
    --task_level task1 \
    --model $COMPLETION_MODEL \
    --workplace_name workplace \
    --cache_path cache \
    --port 12380 \
    --max_iter_times 0 \
    --category ${category}