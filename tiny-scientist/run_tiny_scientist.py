#!/usr/bin/env python3


import os
import sys
import json
import time
import threading
import psutil
import torch
import re
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not installed. GPU tracking disabled.")

class OutputCapture:
    """Capture stdout/stderr and parse costs from tiny-scientist output"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.output_lines = []
        self.costs = []
        
    class TeeOutput:
        """Write to both original stream and capture"""
        def __init__(self, original, capture_list, log_file):
            self.original = original
            self.capture_list = capture_list
            self.log_file = log_file
            
        def write(self, text):
            self.original.write(text)
            self.original.flush()
            self.capture_list.append(text)
            # Write to log file immediately
            with open(self.log_file, 'a') as f:
                f.write(text)
                
        def flush(self):
            self.original.flush()

        def isatty(self):
            return self.original.isatty()
            
    def start(self):
        """Start capturing output"""
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.TeeOutput(self.old_stdout, self.output_lines, self.log_file)
        sys.stderr = self.TeeOutput(self.old_stderr, self.output_lines, self.log_file)
        
    def stop(self):
        """Stop capturing and parse costs"""
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self._parse_costs()
        
    def _parse_costs(self):
        full_text = "".join(self.output_lines)
        
        scientist_total_match = re.search(r'TOTAL COST\s*:\s*\$([0-9.]+)', full_text)
        scientist_total = float(scientist_total_match.group(1)) if scientist_total_match else 0.0
        
        drawer_match = re.search(r'Drawer cost:\s*\$([0-9.]+)', full_text)
        drawer_cost = float(drawer_match.group(1)) if drawer_match else 0.0
        
        final_match = re.search(r'FINAL TOTAL.*:\s*\$([0-9.]+)', full_text)
        final_total = float(final_match.group(1)) if final_match else (scientist_total + drawer_cost)
        
        # FIX: Include all modules including Safety Checker and Drawer
        module_costs = {}
        for name, cost in re.findall(r'(Safety Checker|Thinker|Coder|Writer|Reviewer|Drawer)\s*:\s*\$([0-9.]+)', full_text):
            module_costs[name] = float(cost)
        if drawer_cost > 0 and "Drawer" not in module_costs:
            module_costs["Drawer"] = drawer_cost

        # Per-module token counts (printed alongside each module's cost line)
        module_tokens = {}
        for name, in_tok, out_tok in re.findall(
            r'(Safety Checker|Thinker|Coder|Writer|Reviewer|Drawer)\s*:\s*\$[0-9.]+\s*'
            r'\(input_tokens=(\d+),\s*output_tokens=(\d+)\)',
            full_text,
        ):
            module_tokens[name] = {
                "input_tokens": int(in_tok),
                "output_tokens": int(out_tok),
                "total_tokens": int(in_tok) + int(out_tok),
            }

        # Global totals from the TOTAL TOKENS line printed by get_total_cost()
        total_tokens_match = re.search(
            r'TOTAL TOKENS\s*:\s*input=(\d+),\s*output=(\d+),\s*total=(\d+)',
            full_text,
        )
        if total_tokens_match:
            total_input_tokens = int(total_tokens_match.group(1))
            total_output_tokens = int(total_tokens_match.group(2))
            total_tokens = int(total_tokens_match.group(3))
        else:
            total_input_tokens = sum(m["input_tokens"] for m in module_tokens.values())
            total_output_tokens = sum(m["output_tokens"] for m in module_tokens.values())
            total_tokens = total_input_tokens + total_output_tokens

        self.costs = {
            "final_total": final_total,
            "module_costs": module_costs,
            "scientist_total": scientist_total,
            "drawer_cost": drawer_cost,
            "tokens": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "module_tokens": module_tokens,
            },
        }
            
    def get_costs(self) -> Dict:
        """Get parsed cost information"""
        return self.costs

class ResourceMonitor:
    """Monitor CPU/GPU resources in background while tiny-scientist runs"""
    
    def __init__(self):
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring in background"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            sample = {
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3)
            }
            
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        sample["gpu_load"] = gpu.load * 100
                        sample["gpu_memory_percent"] = (gpu.memoryUsed / gpu.memoryTotal) * 100
                        sample["gpu_memory_used_mb"] = gpu.memoryUsed
                        sample["gpu_temp"] = gpu.temperature
                except:
                    pass
                    
            if torch.cuda.is_available():
                try:
                    sample["torch_gpu_memory_mb"] = torch.cuda.memory_allocated(0) / (1024**2)
                except:
                    pass
                    
            self.samples.append(sample)
            time.sleep(5)
            
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.samples:
            return {}
            
        cpu_vals = [s["cpu_percent"] for s in self.samples]
        mem_vals = [s["memory_percent"] for s in self.samples]
        
        summary = {
            "duration_seconds": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"],
            "num_samples": len(self.samples),
            "cpu": {
                "avg": sum(cpu_vals) / len(cpu_vals),
                "max": max(cpu_vals),
                "min": min(cpu_vals)
            },
            "memory": {
                "avg": sum(mem_vals) / len(mem_vals),
                "max": max(mem_vals),
                "min": min(mem_vals)
            }
        }
        
        gpu_load_vals = [s.get("gpu_load") for s in self.samples if "gpu_load" in s]
        if gpu_load_vals:
            summary["gpu"] = {
                "avg_load": sum(gpu_load_vals) / len(gpu_load_vals),
                "max_load": max(gpu_load_vals),
                "min_load": min(gpu_load_vals)
            }
            
            gpu_mem_vals = [s.get("gpu_memory_percent") for s in self.samples if "gpu_memory_percent" in s]
            if gpu_mem_vals:
                summary["gpu"]["avg_memory_percent"] = sum(gpu_mem_vals) / len(gpu_mem_vals)
                summary["gpu"]["max_memory_percent"] = max(gpu_mem_vals)
                
        return summary


def run_with_tracking(intent: str, model: str = "gpt-4o", budget: float = 5.0, 
                      output_base: str = "./tracked_experiments"):
    """
    Run tiny-scientist with resource tracking
    
    Args:
        intent: Research question/goal
        model: LLM model to use
        budget: API budget in dollars
        output_base: Base directory for outputs
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_intent = "".join(c if c.isalnum() else "_" for c in intent[:50])
    output_dir = Path(output_base) / f"{timestamp}_{safe_intent}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "terminal_output.log"
    
    print(f"\n{'='*80}")
    print(f"TINY-SCIENTIST WITH RESOURCE TRACKING")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Intent: {intent}")
    print(f"Model: {model}")
    print(f"Budget: ${budget}")
    print(f"Log file: {log_file}")
    print(f"{'='*80}\n")
    
    # Save configuration
    config = {
        "intent": intent,
        "model": model,
        "budget": budget,
        "timestamp": timestamp,
        "output_dir": str(output_dir)
    }
    
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    

    from tiny_scientist import TinyScientist
    from tiny_scientist.tool import CodeSearchTool, DrawerTool
    
    output_capture = OutputCapture(log_file)
    output_capture.start()
    
    monitor = ResourceMonitor()
    monitor.start()
    
    stage_times = {}
    overall_start = time.time()
    
    scientist = None
    idea = None
    exp_dir = None
    diagrams_generated = False
    
    try:
        # Stage 1: Think
        print(f"\n[1/4] THINK - Generating research idea...")
        stage_start = time.time()
        
        scientist = TinyScientist(
            model=model,
            budget=budget,
            output_dir="./tiny_scientist_output",
            enable_safety_check=True
        )
        
        idea = scientist.think(intent=intent, num_ideas=1)
        
        if not idea or (isinstance(idea, dict) and not idea):
            print("✗ Failed to generate idea")
            stage_times["think"] = time.time() - stage_start
            return None
            
        if isinstance(idea, list) and len(idea) > 0:
            idea = idea[0]
        
        stage_times["think"] = time.time() - stage_start
        
        # Save idea
        idea_file = output_dir / "research_idea.json"
        with open(idea_file, "w") as f:
            json.dump(idea, f, indent=2)
        print(f"✓ Idea saved: {idea_file}")
        
        # Search GitHub for relevant code (optional - requires token)
        github_results = None
        if os.getenv("GITHUB_TOKEN"):
            try:
                print("\nSearching GitHub for relevant code...")
                code_searcher = CodeSearchTool()
                github_results = code_searcher.run(query=intent[:100], search_type="repositories")
                if github_results:
                    github_file = output_dir / "github_code_search.json"
                    with open(github_file, "w") as f:
                        json.dump(github_results, f, indent=2)
                    print(f"✓ GitHub search saved: {github_file}")
            except Exception as e:
                print(f"GitHub search failed (rate limit): {e}")
        else:
            print("⊘ GitHub search skipped (no GITHUB_TOKEN set)")
        
        # Stage 2: Code
        print(f"\n[2/4] CODE - Implementing experiments...")
        stage_start = time.time()
        
        status, exp_dir = scientist.code(idea=idea)
        stage_times["code"] = time.time() - stage_start
        
        if not status:
            print("✗ Experiments failed")
            return None
        
        # Stage 3: Write
        print(f"\n[3/4] WRITE - Generating paper...")
        stage_start = time.time()
        
        if status and exp_dir:
            try:
                pdf_path = scientist.write(idea=idea, experiment_dir=exp_dir)
                stage_times["write"] = time.time() - stage_start
                print(f"✓ Paper generated: {pdf_path}")
            except Exception as e:
                print(f"Paper generation had errors: {e}")
                stage_times["write"] = time.time() - stage_start
        else:
            print("⊘ Skipping write (experiments failed)")
            stage_times["write"] = 0
        
        # Stage 3.5: Draw diagrams
        diagrams_generated = False
        drawer_cost = 0.0
        if status and exp_dir and os.getenv("OPENAI_API_KEY"):
            print("\n[3.5/4] DRAW - Generating diagrams...")
            try:
                latex_file = Path(exp_dir) / "latex" / "acl_latex.tex"
                if latex_file.exists():
                    with open(latex_file, "r") as f:
                        latex_content = f.read()
                    
                    method_match = re.search(
                        r'\\section\{Method\}(.*?)\\section',
                        latex_content,
                        re.DOTALL
                    )
                    
                    if method_match:
                        method_text = method_match.group(1).strip()
                        method_text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', method_text)
                        method_text = re.sub(r'\\[a-zA-Z]+', '', method_text)
                        
                        if len(method_text) > 100:
                            print("  Generating diagram from method section...")
                            
                            drawer = DrawerTool(
                                model="gpt-4o-mini",  
                                temperature=0.75,
                                cost_tracker=scientist.global_cost_tracker
                            )
                            
                            drawer_cost_before = drawer.cost_tracker.get_total_cost()
                            inner_json = json.dumps({
                                "section_name": "Method",
                                "section_content": method_text
                            })
                            diagram_result = drawer.run(json.dumps(inner_json))
                            drawer_cost = drawer.cost_tracker.get_total_cost() - drawer_cost_before
                            print(f"  Drawer cost: ${drawer_cost:.4f}")
                            
                            if diagram_result and diagram_result.get("diagram"):
                                diagram = diagram_result["diagram"]
                                
                                svg_file = Path(exp_dir) / "method_diagram.svg"
                                if diagram.get("svg"):
                                    with open(svg_file, "w") as f:
                                        f.write(diagram["svg"])
                                    print(f"✓ Diagram saved: method_diagram.svg")
                                    diagrams_generated = True
                                
                                if diagram.get("summary"):
                                    summary_file = Path(exp_dir) / "diagram_summary.txt"
                                    with open(summary_file, "w") as f:
                                        f.write(diagram["summary"])
                            else:
                                print("No diagram generated")
                        else:
                            print("Method section too short for diagram")
                    else:
                        print("Could not find method section in paper")
                else:
                    print("LaTeX file not found")
                    
            except Exception as e:
                print(f"Diagram generation failed: {e}")
                print("Continuing without diagrams...")
        
        if drawer_cost > 0:
            print(f"\n Drawer cost: ${drawer_cost:.4f}")

        if scientist:
            scientist_cost = scientist.get_total_cost() 
            total_cost = scientist_cost + drawer_cost
            print(f"\nFINAL TOTAL: ${total_cost:.4f}")
                    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Stop monitoring and output capture
        monitor.stop()
        output_capture.stop()
        
    # Calculate total time
    total_time = time.time() - overall_start
    
    # Extract costs
    costs = output_capture.get_costs()
    
    # Copy tiny-scientist outputs to our directory
    print(f"\n{'='*80}")
    print("Organizing outputs...")
    print(f"{'='*80}\n")
    
    if exp_dir and Path(exp_dir).exists():
        import shutil
        
        dest_exp = output_dir / "tiny_scientist_output"
        if Path(exp_dir).exists():
            shutil.copytree(exp_dir, dest_exp, dirs_exist_ok=True)
            print(f"✓ Copied experiments to: {dest_exp}")
            
    resource_summary = monitor.get_summary()
    
    summary = {
        "config": config,
        "timings": {
            "total_seconds": total_time,
            "stages": stage_times
        },
        "costs": costs,
        "resources": resource_summary,
        "outputs": {
            "idea": str(output_dir / "research_idea.json"),
            "github_search": str(output_dir / "github_code_search.json") if github_results else None,
            "experiments": str(output_dir / "tiny_scientist_output"),
            "diagrams": str(output_dir / "tiny_scientist_output" / "method_diagram.svg") if diagrams_generated else None,
            "terminal_log": str(log_file)
        }
    }
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    with open(output_dir / "resource_samples.json", "w") as f:
        json.dump(monitor.samples, f, indent=2)
    
    summary_text = generate_summary_report(summary, output_dir)
    summary_file = output_dir / "SUMMARY.txt"
    with open(summary_file, "w") as f:
        f.write(summary_text)
    
    print(f"\n{'='*80}")
    print("COMPLETED")
    print(f"{'='*80}")
    print(summary_text)
    print(f"{'='*80}")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"View summary: {summary_file}")
    print(f"View metrics: {metrics_file}")
    print(f"{'='*80}\n")
    
    return output_dir


def generate_summary_report(summary: Dict, output_dir: Path) -> str:
    """Generate human-readable summary"""
    
    config = summary["config"]
    timings = summary["timings"]
    resources = summary["resources"]
    
    report = f"""
TINY-SCIENTIST RUN SUMMARY
{'='*80}

Configuration:
  Intent: {config['intent']}
  Model: {config['model']}
  Budget: ${config['budget']}
  Timestamp: {config['timestamp']}

Timing:
  Total Duration: {timings['total_seconds']:.1f} seconds ({timings['total_seconds']/60:.1f} minutes)
  
  Stage Breakdown:
"""
    
    for stage, duration in timings['stages'].items():
        report += f"    {stage.capitalize()}: {duration:.1f}s\n"
    
    # Add costs section
    costs_data = summary.get('costs', {})
    if costs_data and costs_data.get('final_total', 0) > 0:
        report += f"""
API Costs:
  Total Cost: ${costs_data.get('final_total', 0):.4f}
  Budget: ${summary['config']['budget']}
  Used: {(costs_data.get('final_total', 0) / summary['config']['budget'] * 100):.1f}%
"""

        module_costs = costs_data.get('module_costs', {})
        module_tokens = costs_data.get('tokens', {}).get('module_tokens', {})
        if module_costs:
            sorted_modules = sorted(module_costs.items(), key=lambda x: x[1], reverse=True)
            report += "\n  Module Breakdown:\n"
            for module_name, cost in sorted_modules:
                tok = module_tokens.get(module_name)
                if tok:
                    report += (
                        f"    - {module_name}: ${cost:.4f} "
                        f"(input={tok['input_tokens']}, "
                        f"output={tok['output_tokens']}, "
                        f"total={tok['total_tokens']})\n"
                    )
                else:
                    report += f"    - {module_name}: ${cost:.4f}\n"

        tokens_data = costs_data.get('tokens', {})
        if tokens_data.get('total_tokens'):
            report += f"""
Token Usage:
  Total Input Tokens:  {tokens_data['total_input_tokens']:,}
  Total Output Tokens: {tokens_data['total_output_tokens']:,}
  Total Tokens:        {tokens_data['total_tokens']:,}
"""
    
    if resources:
        report += f"""
Resource Usage (Average):
  CPU: {resources.get('cpu', {}).get('avg', 0):.1f}%
  Memory: {resources.get('memory', {}).get('avg', 0):.1f}%
"""
        if 'gpu' in resources:
            report += f"  GPU Load: {resources['gpu'].get('avg_load', 0):.1f}%\n"
            report += f"  GPU Memory: {resources['gpu'].get('avg_memory_percent', 0):.1f}%\n"
    
    report += f"""
Outputs:
  All files: {output_dir}/
  Research Idea: research_idea.json
  GitHub Code Search: github_code_search.json (if available)
  Method Diagram: tiny_scientist_output/method_diagram.svg (if generated)
  Experiments: tiny_scientist_output/
  Paper (LaTeX): tiny_scientist_output/latex/
  Terminal Log: terminal_output.log
  Detailed Metrics: metrics.json
  Resource Samples: resource_samples.json

Next Steps:
  1. Check tiny_scientist_output/ for all generated files
  2. Review tiny_scientist_output/latex/acl_latex.tex for the paper
  3. Check tiny_scientist_output/experiment.py for experiment code
  4. View metrics.json for detailed resource usage
"""
    
    return report


def main():
    """Main entry point"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set!")
        sys.exit(1)
    

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    
    intent = "Run a small PyTorch matrix multiplication benchmark on GPU"
    

    if len(sys.argv) > 1:
        intent = " ".join(sys.argv[1:])
    
    output_dir = run_with_tracking(
        intent=intent,
        model="gpt-4o",
        budget=10.0,
        output_base="./tracked_experiments"
    )
    if output_dir:
        latex_file = output_dir / "tiny_scientist_output" / "latex" / "acl_latex.tex"
        
        if latex_file.exists():
            print(f"\n✓ SUCCESS: LaTeX paper generated at {latex_file}")
            
            main_output = Path("./tiny_scientist_output")
            if main_output.exists():
                import shutil
                shutil.rmtree(main_output)
                print(f"  Cleaned up: {main_output}")
            
            sys.exit(0)
        else:
            print(f"\n PARTIAL SUCCESS: Outputs saved but no LaTeX generated")
            sys.exit(1)
    else:
        print(f"\n✗ FAILED: Check logs for errors")
        sys.exit(1)


if __name__ == "__main__":
    main()