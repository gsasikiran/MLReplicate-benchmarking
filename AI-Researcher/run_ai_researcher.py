from main_ai_researcher import main_ai_researcher
import os
import gradio as gr
import time
import json
import logging
from datetime import datetime
from typing import Tuple
import importlib
from dotenv import load_dotenv, set_key, find_dotenv, unset_key
import threading
import queue
import re  # For regular expression operations
import random
import global_state
import base64
import shutil  # Added for workspace bridging
from resource_monitor import ResourceMonitor
from directory_manager import get_directory_manager 
from integration_helper import get_cache_path
from pathlib import Path  

os.environ["PYTHONIOENCODING"] = "utf-8"

# Dictionary containing module descriptions
MODULE_DESCRIPTIONS = {
    "Detailed Idea Description": "At this level, users provide comprehensive descriptions of their specific research ideas. The system processes these detailed inputs to develop implementation strategies based on the user's explicit requirements. Examples 1-2 are the templates of this mode.",
    "Reference-Based Ideation": "This simpler level involves users submitting reference papers without a specific idea in mind. The user query typically follows the format: "'"I have some reference papers, please come up with an innovative idea and implement it with these papers."'" The system then analyzes the provided references to generate and develop novel research concepts. Examples 3-4 are the templates of this mode.",
    "Paper Generation Agent": "Once all research and experimental work is finished, employ this agent for paper generation",
    # "exit": "exit mode"
}


def bridge_workspace_if_needed(research_field=None, instance_id=None):
    research_field = research_field or os.getenv("CATEGORY")
    instance_id = instance_id or os.getenv("INSTANCE_ID")
    if not research_field or not instance_id:
        raise ValueError("research_field and instance_id must be provided")
    """
    Automatically bridge research results to paper generation workspace.
    Uses centralized directory system.
    """
    from integration_helper import get_cache_path
    from directory_manager import get_directory_manager
    
    manager = get_directory_manager()
    
    if not manager.current_run_dir:
        logging.error("No active run directory found")
        return False
    
    # Use centralized paths
    research_cache_dir = manager.get_cache_dir()
    research_workplace_dir = manager.get_workplace_dir()
    
    # Paper target in centralized location
    paper_target_dir = manager.get_paper_dir() / f"{research_field}_{instance_id}"
    paper_workplace_target = paper_target_dir / "workplace"
    
    logging.info(f"Checking workspace bridge for {research_field}/{instance_id}")
    logging.info(f"Research cache: {research_cache_dir}")
    logging.info(f"Research workplace: {research_workplace_dir}")
    logging.info(f"Paper target: {paper_target_dir}")
    
    # Create paper directory
    os.makedirs(paper_target_dir, exist_ok=True)
    logging.info(f"Created paper agent directory: {paper_target_dir}")
    
    # Bridge cache directory - find which cache actually exists
    cache_subdirs = [d for d in os.listdir(research_cache_dir) if d.startswith(f'cache_{instance_id}_')]
    if cache_subdirs:
        # Use the first cache found (there should only be one per instance)
        actual_cache_name = cache_subdirs[0]
        paper_cache_target = paper_target_dir / actual_cache_name
        source_cache = research_cache_dir / actual_cache_name
        
        if source_cache.exists() and not paper_cache_target.exists():
            try:
                os.symlink(str(source_cache.absolute()), str(paper_cache_target))
                logging.info(f"✅ Linked cache: {paper_cache_target} -> {source_cache}")
            except OSError:
                shutil.copytree(source_cache, paper_cache_target)
                logging.info(f"✅ Copied cache to: {paper_cache_target}")
    
    # Bridge workplace directory
    if research_workplace_dir.exists() and not paper_workplace_target.exists():
        try:
            os.symlink(str(research_workplace_dir.absolute()), str(paper_workplace_target))
            logging.info(f"✅ Linked workplace: {paper_workplace_target} -> {research_workplace_dir}")
        except OSError:
            shutil.copytree(research_workplace_dir, paper_workplace_target)
            logging.info(f"✅ Copied workplace to: {paper_workplace_target}")
    
        valid_cache_found = False
    
    if cache_subdirs:
        for cache_name in cache_subdirs:
            paper_cache_target = paper_target_dir / cache_name
            required_paths = [
                paper_cache_target / "agents",
                paper_cache_target / "tools"
            ]
            
            # Check if this cache has all required subdirectories
            if all(p.exists() for p in required_paths):
                logging.info(f"✅ Workspace bridge completed successfully using {cache_name}!")
                valid_cache_found = True
                break  # Found a valid cache, stop searching
            else:
                # This cache is missing files, continue to next
                logging.debug(f"Cache {cache_name} missing required files, checking next...")
        
        # After checking all caches
        if not valid_cache_found:
            logging.error("❌ Workspace bridge failed - no cache with required files found")
            for cache_name in cache_subdirs:
                paper_cache_target = paper_target_dir / cache_name
                logging.error(f"  Checked {cache_name}:")
                for subdir in ["agents", "tools"]:
                    path = paper_cache_target / subdir
                    if not path.exists():
                        logging.error(f"    Missing: {path}")
            return False
        
        return True
    else:
        logging.error("❌ No cache directory found")
        return False

def validate_input(question: str) -> bool:
    """验证用户输入是否有效

    Args:
        question: 用户问题

    Returns:
        bool: 输入是否有效
    """
    # 检查输入是否为空或只包含空格
    if not question or question.strip() == "":
        return False
    return True

def run_ai_researcher(question: str, reference: str, example_module: str) -> Tuple[str, str, str]:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_module = example_module.replace(' ', '_').lower()
    
    instance_id = os.getenv("INSTANCE_ID", "default_instance")

    # Determine phase and model
    phase = "research" if example_module in ["Detailed Idea Description", "Reference-Based Ideation"] else "paper_generation"
    model_name = os.getenv("COMPLETION_MODEL", "gpt-4o")
    
    # Get directory manager for centralized paths
    from directory_manager import get_directory_manager
    manager = get_directory_manager()
    
    # Setup paths
    if manager.current_run_dir:
        metrics_dir = manager.current_run_dir / "logs" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        log_file = metrics_dir / f"ai_researcher_{safe_module}_{timestamp}.log"
        
        # GPU log path (will be created by container if research phase)
        if phase == "research":
            gpu_log_path = metrics_dir / f"gpu_container_{instance_id}.json"
        else:
            gpu_log_path = None
    else:
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        log_file = metrics_dir / f"ai_researcher_{safe_module}_{timestamp}.log"
        gpu_log_path = None
    
    import re
    gpus_env = os.getenv("GPUS", "")
    gpu_id = None
    if gpus_env:
        # Parse: GPUS='"device=0"' -> extract 0
        match = re.search(r'device=(\d+)', gpus_env)
        if match:
            gpu_id = int(match.group(1))

    monitor = ResourceMonitor(
        log_file=str(log_file),
        phase=phase,
        model_name=model_name,
        instance_id=instance_id,
        gpu_log_path=str(gpu_log_path) if gpu_log_path else None,
        gpu_id=gpu_id  # Pass the GPU ID from GPUS env variable
    )
    monitor.start_monitoring()
    
    try:
        # Validate input
        if not validate_input(question):
            logging.warning("User submitted invalid input")
            monitor.stop_monitoring()
            return ("Please enter a valid question", "0", "Error: Invalid input question")

        load_dotenv(find_dotenv(), override=True)
        logging.info(f"Processing question: '{question}', using module: {example_module}")

        if example_module not in MODULE_DESCRIPTIONS:
            logging.error(f"User selected an unsupported module: {example_module}")
            monitor.stop_monitoring()
            return (f"Selected module '{example_module}' is not supported", "0", "Error: Unsupported module")

        # Handle Paper Generation Agent workspace bridging
        if example_module == "Paper Generation Agent":
            # Get from environment or use defaults
            research_field = os.getenv("CATEGORY")
            if not research_field:
                raise ValueError("CATEGORY environment variable must be set")
            instance_id = os.getenv("INSTANCE_ID")
            if not instance_id:
                raise ValueError("INSTANCE_ID environment variable must be set")
            
            logging.info(f"Paper Generation using: field={research_field}, instance={instance_id}")
            
            import glob
            project_root = os.path.dirname(os.path.abspath(__file__))
            completed_runs = glob.glob(os.path.join(project_root, "results", "test_*/"))
            if completed_runs:
                latest_run = max(completed_runs, key=os.path.getmtime)
                manager.current_run_dir = Path(latest_run)
                logging.info(f"Using existing run directory: {latest_run}")
            else:
                logging.error("No completed research runs found")
                monitor.stop_monitoring()
                return ("No completed research runs found. Please run research phase first.", "0", "Error: No research runs")
            
            research_cache = get_cache_path(instance_id, os.getenv("COMPLETION_MODEL"))
            if not os.path.exists(research_cache):
                logging.error("Research phase not completed. Please run research phase first.")
                monitor.stop_monitoring()
                return ("Research phase not completed. Please run research phase first by selecting 'Detailed Idea Description' module.", "0", "Error: Research phase required")
            
            if not bridge_workspace_if_needed(research_field, instance_id):
                logging.error("Failed to bridge workspace for paper generation")
                monitor.stop_monitoring()
                return ("Failed to prepare workspace for paper generation. Please check research results.", "0", "Error: Workspace bridge failed")

        # Run the main AI researcher
        try:
            print(f"Starting AI-Researcher with module: {example_module}")
            answer = main_ai_researcher(question, reference, example_module)
            logging.info("Successfully completed AI Researcher execution")
            
            # Handle post-processing for research phases
            if example_module in ["Detailed Idea Description", "Reference-Based Ideation"]:
                research_field = os.getenv("CATEGORY")
                instance_id = os.getenv("INSTANCE_ID")
                if research_field and instance_id:
                    bridge_workspace_if_needed(research_field, instance_id)
                else:
                    logging.warning("CATEGORY or INSTANCE_ID not set, skipping workspace bridge")
                
        except Exception as e:
            logging.error(f"Error occurred while running Researcher: {str(e)}")
            monitor.stop_monitoring()
            return (f"Error occurred while running Researcher: {str(e)}", "0", f"Error: Run failed - {str(e)}")

        # Stop monitoring and get final metrics
        final_metrics = monitor.stop_monitoring()
        
        # Create info string with actual metrics
        cost = final_metrics.cost_metrics
        sh_cost = final_metrics.self_hosted_costs
        
        info_str = (
            f"Tokens: {cost.total_tokens:,} | "
            f"API Cost: ${cost.total_cost_usd:.4f} | "
            f"Duration: {final_metrics.total_duration:.1f}s | "
            f"Memory: {final_metrics.memory_usage_mb_max:.1f}MB | "
            f"CPU: {final_metrics.cpu_percent_avg:.1f}%"
        )
        
        # Add GPU info if available
        if final_metrics.gpu_metrics.gpu_available:
            gpu = final_metrics.gpu_metrics
            avg_util = sum(gpu.gpu_utilization_avg) / len(gpu.gpu_utilization_avg)
            info_str += f" | GPU: {avg_util:.1f}%"
            
            # Add compute cost if GPUs were used
            if sh_cost.gpus_used > 0:
                info_str += f" | Compute: ${sh_cost.total_compute_cost_usd:.4f}"

        logging.info(f"Processing completed successfully. Duration: {final_metrics.total_duration:.1f}s")

        return (answer, info_str, f"Successfully completed in {final_metrics.total_duration:.1f}s")

    except Exception as e:
        monitor.stop_monitoring()
        logging.error(f"Uncaught error occurred while processing the question: {str(e)}")
        return (f"Error occurred: {str(e)}", "0", f"Error: {str(e)}")

def view_recent_metrics(count: int = 5):
    """View recent AI-Researcher execution metrics"""
    metrics_dir = "metrics"
    if not os.path.exists(metrics_dir):
        print("No metrics directory found")
        return
    
    json_files = [f for f in os.listdir(metrics_dir) if f.endswith('_detailed.json')]
    
    if not json_files:
        print("No metrics files found")
        return
    
    print("\nRECENT AI-RESEARCHER RUNS:")
    print("-" * 100)
    print(f"{'Time':<20} {'Duration':<12} {'Tokens':<10} {'Memory':<12} {'CPU Avg':<10} {'Module':<20}")
    print("-" * 100)
    
    for json_file in sorted(json_files)[-count:]:
        try:
            with open(os.path.join(metrics_dir, json_file), 'r') as f:
                metrics = json.load(f)
            
            module_name = json_file.replace('ai_researcher_', '').replace('_detailed.json', '').split('_')[0]
            
            print(f"{metrics['start_time']:<20} "
                  f"{metrics['total_duration']:>8.1f}s    "
                  f"{metrics['token_usage']['total_tokens']:>8,}  "
                  f"{metrics['memory_usage_mb_max']:>8.1f}MB   "
                  f"{metrics['cpu_percent_avg']:>6.1f}%    "
                  f"{module_name:<20}")
                  
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    print("-" * 100)

if __name__ == "__main__":
    print("="*70)
    print("AI-RESEARCHER: FULL PIPELINE")
    print("="*70)
    
    # Phase 1: Research
    print("\n🔬 PHASE 1: Research & Experimentation")
    research_result = run_ai_researcher(
        "Conformal Prediction as Bayesian Quadrature. Reinterprets conformal prediction through a Bayesian lens, proposing a Bayesian quadrature framework that provides interpretable and flexible uncertainty guarantees. This approach extends traditional conformal prediction by integrating prior knowledge into uncertainty estimation, yielding richer and more informative loss distributions for high-stakes machine learning applications.",
        "Aitchison, J. and Dunsmore, I. R. Statistical Prediction Analysis, Vovk, V., Gammerman, A., and Shafer, G. Algorithmic Learning in a Random World, Angelopoulos, A. N. and Bates, S. Conformal prediction: A gentle introduction, Cockayne, J., Oates, C. J., Sullivan, T. J., and Girolami, M. Bayesian probabilistic numerical methods, Diaconis, P. Bayesian numerical analysis, Hennig, P., Osborne, M. A., and Kersting, H. P. Probabilistic Numerics: Computation as Machine Learning, Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. Conformal risk control, Papadopoulos, H., Proedrou, K., Vovk, V., and Gammerman, A. Inductive confidence machines for regression, O'Hagan, A. Bayes-Hermite quadrature, Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft COCO: Common objects in context, Shafer, G. and Vovk, V. A tutorial on conformal prediction, Barber, R. F., Candès, E. J., Ramdas, A., and Tibshirani, R. J. The limits of distribution-free conditional predictive inference, Gibbs, I., Cherian, J. J., and Candès, E. J. Conformal Prediction With Conditional Guarantees, Ng, K. W., Tian, G.-L., and Tang, M.-L. Dirichlet and Related Distributions: Theory, Methods and Applications, Tukey, J. W. Nonparametric estimation II. Statistically equivalent blocks and tolerance regions—the continuous case.",
        "Reference-Based Ideation"
    )

    print(f"\n✓ Research Phase Complete")
    print(f"Status: {research_result[2]}")
    print(f"Metrics: {research_result[1]}")
    
    # Check if research succeeded
    if "Successfully completed" in research_result[2]:
        print("\n" + "="*70)
        print("📝 PHASE 2: Paper Generation")
        print("="*70)
        
        # Phase 2: Paper Generation
        paper_result = run_ai_researcher(
            "Conformal Prediction as Bayesian Quadrature. Reinterprets conformal prediction through a Bayesian lens, proposing a Bayesian quadrature framework that provides interpretable and flexible uncertainty guarantees. This approach extends traditional conformal prediction by integrating prior knowledge into uncertainty estimation, yielding richer and more informative loss distributions for high-stakes machine learning applications.",
            "Aitchison, J. and Dunsmore, I. R. Statistical Prediction Analysis, Vovk, V., Gammerman, A., and Shafer, G. Algorithmic Learning in a Random World, Angelopoulos, A. N. and Bates, S. Conformal prediction: A gentle introduction, Cockayne, J., Oates, C. J., Sullivan, T. J., and Girolami, M. Bayesian probabilistic numerical methods, Diaconis, P. Bayesian numerical analysis, Hennig, P., Osborne, M. A., and Kersting, H. P. Probabilistic Numerics: Computation as Machine Learning, Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., and Schuster, T. Conformal risk control, Papadopoulos, H., Proedrou, K., Vovk, V., and Gammerman, A. Inductive confidence machines for regression, O'Hagan, A. Bayes-Hermite quadrature, Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft COCO: Common objects in context, Shafer, G. and Vovk, V. A tutorial on conformal prediction, Barber, R. F., Candès, E. J., Ramdas, A., and Tibshirani, R. J. The limits of distribution-free conditional predictive inference, Gibbs, I., Cherian, J. J., and Candès, E. J. Conformal Prediction With Conditional Guarantees, Ng, K. W., Tian, G.-L., and Tang, M.-L. Dirichlet and Related Distributions: Theory, Methods and Applications, Tukey, J. W. Nonparametric estimation II. Statistically equivalent blocks and tolerance regions—the continuous case.",
            "Paper Generation Agent"
        )


        print(f"\n✓ Paper Generation Phase Complete")
        print(f"Status: {paper_result[2]}")
        print(f"Metrics: {paper_result[1]}")
        
    # After both phases complete successfully
        if "Successfully completed" in research_result[2] and "Successfully completed" in paper_result[2]:
            print("\n" + "="*70)
            print("📊 GENERATING COMBINED METRICS")
            print("="*70)
            
            # Create combined metrics summary
            from resource_monitor import create_combined_metrics
            instance_id = os.getenv("INSTANCE_ID", "default_instance")
            combined_file = create_combined_metrics(instance_id)
            
            if combined_file:
                print(f"✓ All metrics saved successfully")

        # View combined metrics
        print("\n" + "="*70)
        print("COMBINED METRICS")
        print("="*70)
        view_recent_metrics(2)
    else:
        print("\n❌ Research phase failed. Skipping paper generation.")
        print(f"Error: {research_result[0]}")

