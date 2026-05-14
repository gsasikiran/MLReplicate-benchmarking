from ai_researcher import CycleResearcher
from ai_researcher.utils import print_paper_summary
from huggingface_hub import login as hf_login
from dotenv import load_dotenv
import os, json, time, psutil, pynvml
from datetime import datetime

# Load environment variables
load_dotenv()
hf_login(token=os.getenv("HUGGINGFACE_TOKEN"))

def track_all_resources(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        # Capture start stats
        start_time = time.perf_counter()
        start_mem = process.memory_info().rss
        start_io = process.io_counters()
        start_gpu_mem = []
        for i in range(device_count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            start_gpu_mem.append(pynvml.nvmlDeviceGetMemoryInfo(h).used)

        # Run the function
        result = func(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            func_result, extra_metrics = result
        else:
            func_result, extra_metrics = result, {}

        # Capture end stats
        end_time = time.perf_counter()
        end_mem = process.memory_info().rss
        end_io = process.io_counters()
        end_gpu_mem = []
        gpu_stats = []
        for i in range(device_count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            end_gpu_mem.append(mem_info.used)
            gpu_stats.append({
                "device": i,
                "gpu_mem_delta_MB": (mem_info.used - start_gpu_mem[i]) / 1024**2,
                "util_percent": util.gpu,
                "mem_util_percent": util.memory
            })

        # Build log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "function": func.__name__,
            "wall_time_sec": end_time - start_time,
            "ram_delta_MB": (end_mem - start_mem) / 1024**2,
            "disk_read_delta_MB": (end_io.read_bytes - start_io.read_bytes) / 1024**2,
            "disk_write_delta_MB": (end_io.write_bytes - start_io.write_bytes) / 1024**2,
            "gpu": gpu_stats,
        }

        # Merge extra metrics
        log_entry.update(extra_metrics)

        # Save log entry to JSON (append mode)
        log_file = "experiments/resource_log.json"
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

        pynvml.nvmlShutdown()
        return func_result
    return wrapper


@track_all_resources
def main(topic, references, filename=None):
    researcher = CycleResearcher(model_size="12B")

    generated_papers = researcher.generate_paper(
        topic=topic,
        references=references,
        n=1
    )
    extra_metrics = {"token_count": researcher.token_count, "filename": filename}
    latex_text = generated_papers[0].get("latex", "")
    # Save generated papers [0] to .txt file

    if generated_papers[0]:
        with open(f"experiments/generated_papers/{filename}", "w") as f:
            json.dump(generated_papers[0], f, indent=2)
        print(f"Generated paper saved to experiments/generated_papers/{filename}")
    if latex_text:
        with open(f"experiments/generated_papers/{filename.replace('.json', '.tex')}", "w") as f:
            f.write(latex_text)
        print(f"LaTeX content saved to experiments/generated_papers/{filename.replace('.json', '.tex')}")
    return generated_papers[0], extra_metrics
