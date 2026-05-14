import numpy as np
import argparse
import os
import asyncio
import global_state
from dotenv import load_dotenv
from integration_helper import ensure_run_initialized 

load_dotenv()
print(os.getenv("CATEGORY"))
print("Current Working Directory:", os.getcwd())


def init_ai_researcher():
    a = 1

def get_args_research(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=os.getenv("CATEGORY"))
    parser.add_argument("--instance_path", type=str, default=None)
    parser.add_argument('--container_name', type=str, default='paper_eval')
    parser.add_argument("--task_level", type=str, default="task1")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--workplace_name", type=str, default="workplace")
    parser.add_argument("--cache_path", type=str, default="cache")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--max_iter_times", type=int, default=0)
    args = parser.parse_args([])
    return args

def get_args_paper():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research_field", type=str, default=os.getenv("CATEGORY"))
    parser.add_argument("--instance_id", type=str, default=os.getenv("INSTANCE_ID"))
    args = parser.parse_args([])
    return args

def main_ai_researcher(input, reference, mode):
    # For paper generation, use existing directory; otherwise create new one
    if mode == 'Paper Generation Agent':
        from integration_helper import get_directory_manager
        manager = get_directory_manager()
        if not manager.current_run_dir:
            raise RuntimeError("No run directory set for paper generation")
        run_dir = manager.current_run_dir
        print(f"Using existing directory: {run_dir}")
    else:
        # For research modes, create new run directory
        run_dir = ensure_run_initialized(title=input, reference=reference)
        print(f"Using centralized directory: {run_dir}")
        
        # Get manager and store absolute path in environment
        from directory_manager import get_directory_manager
        manager = get_directory_manager()
        # Store ABSOLUTE path so it survives os.chdir
        os.environ['AI_RESEARCHER_RUN_DIR'] = str(manager.current_run_dir.absolute())
        print(f"Set AI_RESEARCHER_RUN_DIR to: {os.environ['AI_RESEARCHER_RUN_DIR']}")

    
    # Get environment variables
    category = os.getenv("CATEGORY")
    instance_id = os.getenv("INSTANCE_ID")
    task_level = os.getenv("TASK_LEVEL")
    container_name = os.getenv("CONTAINER_NAME") 
    workplace_name = os.getenv("WORKPLACE_NAME")
    cache_path = os.getenv("CACHE_PATH")
    port = int(os.getenv("PORT"))
    max_iter_times = int(os.getenv("MAX_ITER_TIMES"))

    try:
        if mode == 'Detailed Idea Description':
            # global INIT_FLAG
            if global_state.INIT_FLAG is False:
                global_state.INIT_FLAG = True
                current_file_path = os.path.realpath(__file__)
                current_dir = os.path.dirname(current_file_path)
                sub_dir = os.path.join(current_dir, "research_agent")
                os.chdir(sub_dir)
                from research_agent.constant import COMPLETION_MODEL
                from research_agent import run_infer_idea, run_infer_plan
                args = get_args_research()
                args.instance_path = os.path.join(current_dir, "benchmark", "final", category, f"{instance_id}.json")
                args.task_level = task_level
                args.model = COMPLETION_MODEL
                args.container_name = container_name
                args.workplace_name = workplace_name
                args.cache_path = cache_path
                args.port = port
                args.max_iter_times = max_iter_times
                args.category = category
                run_infer_plan.main(args, input, reference)
                global_state.INIT_FLAG = False
                
        if mode == 'Reference-Based Ideation':
            if global_state.INIT_FLAG is False:
                global_state.INIT_FLAG = True
                current_file_path = os.path.realpath(__file__)
                current_dir = os.path.dirname(current_file_path)
                sub_dir = os.path.join(current_dir, "research_agent")
                os.chdir(sub_dir)
                from research_agent.constant import COMPLETION_MODEL
                from research_agent import run_infer_idea, run_infer_plan
                args = get_args_research()
                args.instance_path = os.path.join(current_dir, "benchmark", "final", category, f"{instance_id}.json")
                args.container_name = container_name
                args.task_level = task_level
                args.model = COMPLETION_MODEL
                args.workplace_name = workplace_name
                args.cache_path = cache_path
                args.port = port
                args.max_iter_times = max_iter_times
                args.category = category
                run_infer_idea.main(args, reference)
                global_state.INIT_FLAG = False
                
        if mode == 'Paper Generation Agent':
            if global_state.INIT_FLAG is False:
                global_state.INIT_FLAG = True
                from run_ai_researcher import bridge_workspace_if_needed
                bridge_success = bridge_workspace_if_needed(category, instance_id)
                if not bridge_success:
                    print("Failed to bridge workspace. Please run research phase first.")
                    global_state.INIT_FLAG = False
                    return "Research phase required."
            
                from paper_agent import writing
                args = get_args_paper()
                research_field = category
                args.research_field = research_field
                args.instance_id = instance_id
                asyncio.run(writing.writing(args.research_field, args.instance_id))
                global_state.INIT_FLAG = False
        
        # Mark run as completed only if everything succeeded
        from integration_helper import get_directory_manager
        manager = get_directory_manager()
        manager.mark_run_completed()
        print("Research run completed successfully!")
        return "Research completed successfully"
        
    except Exception as e:
        # Don't mark as completed if there was an error
        print(f"Research run failed: {e}")
        global_state.INIT_FLAG = False
        raise