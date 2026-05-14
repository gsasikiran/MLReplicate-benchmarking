import os
import json
import argparse
import re
from datetime import datetime
import logging
import time
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_session(data, gpu_power_kw=0.7, electricity_price=0.12, gpu_cost=25000, gpu_lifetime_hours=5*365*24):
    """
    Analyze GPU session cost and performance from monitoring JSON.


    Args:
        data (dict): JSON log with wall_time_sec, gpu, token_count
        gpu_power_kw (float): Max power draw per GPU in kW (default: 0.7 for H100)
        electricity_price (float): Electricity cost per kWh
        gpu_cost (float): GPU purchase cost in USD
        gpu_lifetime_hours (float): Expected GPU lifetime in hours (default: 5 years)

    Returns:
        dict: Analysis with throughput, utilization, costs
    """
    wall_time_sec = data.get("wall_time_sec", 0)
    session_hours = wall_time_sec / 3600

    # Token statistics - Using placeholder as direct token count from external scripts is not available
    total_tokens = data["token_count"].get("total", 0)
    input_tokens = data["token_count"].get("input", 0)
    output_tokens = data["token_count"].get("output", 0)

    # GPU usage - Placeholder as real GPU data is not available from LLM calls
    # Assuming 1 GPU with 100% utilization during LLM calls for cost estimation
    num_gpus = 1
    active_gpus = 1
    total_power_kw = gpu_power_kw * active_gpus # Assuming full utilization for active GPU

    # Performance
    tokens_per_sec = total_tokens / wall_time_sec if wall_time_sec > 0 else 0
    tokens_per_sec_per_gpu = tokens_per_sec / max(active_gpus, 1)

    # Energy cost
    energy_consumed_kwh = total_power_kw * session_hours
    energy_cost = energy_consumed_kwh * electricity_price

    # Hardware amortization
    gpu_hourly_cost = gpu_cost / gpu_lifetime_hours
    hardware_cost = gpu_hourly_cost * num_gpus * session_hours

    # Total cost
    total_cost = energy_cost + hardware_cost
    cost_per_1k_tokens = (total_cost / total_tokens) * 1000 if total_tokens else 0

    return {
        "wall_time_sec": round(wall_time_sec, 2),
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "tokens_per_sec_per_gpu": round(tokens_per_sec_per_gpu, 2),
        "active_gpus": active_gpus,
        "energy_cost_usd": round(energy_cost, 5),
        "hardware_cost_usd": round(hardware_cost, 5),
        "total_cost_usd": round(total_cost, 5),
        "cost_per_1k_tokens_usd": round(cost_per_1k_tokens, 5),
    }

def read_paper_content(file_path):
    """Reads a markdown paper and extracts title and abstract."""
    with open(file_path, 'r') as f:
        content = f.read()

    title_match = re.search(r'#\s*(.*)', content, re.IGNORECASE)
    abstract_match = re.search(r'##\s*Abstract\s*\n(.*?)(\n##\s*|\Z)', content, re.IGNORECASE | re.DOTALL)

    title = title_match.group(1).strip() if title_match else "Untitled Paper"
    abstract = abstract_match.group(1).strip() if abstract_match else "No abstract found."

    return title, abstract, content

def main():
    parser = argparse.ArgumentParser(description="Process papers to generate ideas and writeups, and calculate costs.")
    parser.add_argument("--model_ideation", type=str, default="gpt-4o-2024-05-13", help="LLM model to use for idea generation.")
    parser.add_argument("--model_writeup", type=str, default="o1-preview-2024-09-12", help="LLM model to use for paper writeup.")
    parser.add_argument("--model_citation", type=str, default="gpt-4o-2024-11-20", help="LLM model to use for citation.")
    parser.add_argument("--model_review", type=str, default="gpt-4o-2024-11-20", help="LLM model to use for review.")
    parser.add_argument("--model_agg_plots", type=str, default="o3-mini-2025-01-31", help="LLM model to use for aggregating plots.")
    parser.add_argument("--num_ideas", type=int, default=20, help="Number of ideas to generate per paper.")
    parser.add_argument("--num_reflections", type=int, default=5, help="Number of reflections for idea generation.")
    parser.add_argument("--num_cite_rounds", type=int, default=20, help="Number of citation rounds for paper generation.")
    parser.add_argument("--electricity_price", type=float, default=0.12, help="Electricity price $/kWh.")
    parser.add_argument("--gpu_cost", type=float, default=25000, help="GPU purchase cost $.")
    parser.add_argument("--gpu_lifetime_hours", type=float, default=5*365*24, help="GPU expected lifetime in hours.")
    args = parser.parse_args()

    base_path = "/nfs/home/keyaf/research/ScAiBench/AI-Scientist-v2"
    modified_datasets_path = os.path.join(base_path, "ai_scientist", "modified_datasets")
    ideas_dir = os.path.join(base_path, "ai_scientist", "modified_datasets")
    os.makedirs(ideas_dir, exist_ok=True)
    log_dir = os.path.join(base_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    for i in range(1,2):
        try:
            paper_filename = f"paper{i}.md"
            paper_path = os.path.join(modified_datasets_path, paper_filename)
            idea_json_path = os.path.join(ideas_dir, f"paper{i}.json")

            if not os.path.exists(paper_path):
                logging.warning(f"Paper file not found: {paper_path}. Skipping.")
                continue

            logging.info(f"Processing {paper_filename}...")

            total_wall_time_sec = 0.0
            generated_ideas = []
            generated_paper_content = ""
            

            if not os.path.exists(idea_json_path):
                # 1. Idea Generation
                logging.info(f"Generating ideas for {paper_filename}...")
                idea_gen_command = (
                    f"python -u ai_scientist/perform_ideation_temp_free.py "
                    f"--workshop-file ai_scientist/modified_datasets/{paper_filename} "
                    f"--model gpt-4o-mini "
                    f"--max-num-generations 20 "
                    f"--num-reflections 5"
                )

                start_time = time.time()
                idea_gen_result = subprocess.run(idea_gen_command,shell=True,cwd=base_path)
                end_time = time.time()
                idea_gen_wall_time = end_time - start_time
                total_wall_time_sec += idea_gen_wall_time
                logging.info(f"Idea generation for {paper_filename} completed in {idea_gen_wall_time:.2f} seconds.")
                logging.info(f"Idea generation stdout: {idea_gen_result.stdout}")
                if idea_gen_result.stderr:
                    logging.error(f"Idea generation stderr: {idea_gen_result.stderr}")

            # Read generated ideas from the JSON file
            if os.path.exists(idea_json_path):
                with open(idea_json_path, 'r') as f:
                    generated_ideas = json.load(f)
                logging.info(f"Loaded {len(generated_ideas)} ideas from {idea_json_path}")
            else:
                logging.warning(f"Idea JSON file not found: {idea_json_path}")
            
            # 2. Experiments + Paper Generation
            logging.info(f"Generating paper for ai_scientist/modified_datasets/paper{i}.json...")
            paper_gen_command = (
                f"python -u launch_scientist_bfts.py "
                f"--load_ideas ai_scientist/modified_datasets/paper{i}.json "
                f"--load_code "
                f"--add_dataset_ref "
                f"--model_writeup gpt-4o-mini "
                f"--model_citation gpt-4o-2024-11-20 "
                f"--model_review gpt-4o-2024-11-20 "
                f"--model_agg_plots o3-mini-2025-01-31 "
                f"--num_cite_rounds 20"
            )


            start_time = time.time()
            paper_gen_result = subprocess.run(paper_gen_command,shell=True,cwd=base_path)
            end_time = time.time()
            paper_gen_wall_time = end_time - start_time
            total_wall_time_sec += paper_gen_wall_time
            logging.info(f"Paper generation for paper{i}.json completed in {paper_gen_wall_time:.2f} seconds.")
            logging.info(f"Paper generation stdout: {paper_gen_result.stdout}")
            if paper_gen_result.stderr:
                logging.error(f"Paper generation stderr: {paper_gen_result.stderr}")

            # Placeholder for generated paper content - actual content would need to be parsed from output or file
            generated_paper_content = f"Paper generation command executed. Output: {paper_gen_result.stdout}"

            # 3. Calculate Cost (simplified due to external script execution)
            session_data = {
                "wall_time_sec": total_wall_time_sec,
                "token_count": {
                    "total": 0, # Placeholder: direct token count not available from external scripts
                    "input": 0,
                    "output": 0,
                },
                "gpu": [{"util_percent": 100, "device": 0, "gpu_mem_delta_MB": 0.0, "mem_util_percent": 100}] # Placeholder
            }

            cost_analysis = analyze_session(
                session_data,
                electricity_price=args.electricity_price,
                gpu_cost=args.gpu_cost,
                gpu_lifetime_hours=args.gpu_lifetime_hours
            )
            logging.info(f"Cost analysis for {paper_filename}: {cost_analysis}")

            # 4. Create Log File
            log_filename = f"{os.path.splitext(paper_filename)[0]}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = os.path.join(log_dir, log_filename)

            log_content = {
                "paper_filename": paper_filename,
                "idea_generation_command": idea_gen_command,
                "idea_generation_wall_time_sec": round(idea_gen_wall_time, 2),
                "generated_ideas": generated_ideas,
                "paper_generation_command": paper_gen_command,
                "paper_generation_wall_time_sec": round(paper_gen_wall_time, 2),
                "generated_paper_content_summary": generated_paper_content, # Summary of output
                "cost_analysis": cost_analysis,
                "note": "Token counts are placeholders as direct token tracking from external scripts is not available. Cost is primarily based on wall time.",
                "timestamp": datetime.now().isoformat()
            }

            with open(log_path, 'w') as f:
                json.dump(log_content, f, indent=4)
            logging.info(f"Log saved to {log_path}")
        except Exception as e:
            logging.error(f"An error occurred while processing {paper_filename}: {e}")

if __name__ == "__main__":
    main()
