import os
import glob
import json
import time
from datetime import datetime
import psutil
import pynvml
import subprocess

def track_resources(command, output_file):
    """Run a command as a subprocess while tracking CPU, RAM, disk, GPU usage."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    process = psutil.Process(os.getpid())
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    # Capture start stats
    start_time = time.perf_counter()
    start_mem = process.memory_info().rss
    start_io = process.io_counters()
    start_gpu_mem = [pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used
                     for i in range(device_count)]

    # Run the command and capture stdout/stderr
    with open(output_file, "w") as f:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        f.write(result.stdout)

    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        print(result.stdout)

    # Capture end stats
    end_time = time.perf_counter()
    end_mem = process.memory_info().rss
    end_io = process.io_counters()
    gpu_stats = []
    for i in range(device_count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        gpu_stats.append({
            "device": i,
            "gpu_mem_delta_MB": (mem_info.used - start_gpu_mem[i]) / 1024**2,
            "util_percent": util.gpu,
            "mem_util_percent": util.memory
        })

    # Build log entry
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "command": command,
        "returncode": result.returncode,
        "wall_time_sec": end_time - start_time,
        "ram_delta_MB": (end_mem - start_mem) / 1024**2,
        "disk_read_delta_MB": (end_io.read_bytes - start_io.read_bytes) / 1024**2,
        "disk_write_delta_MB": (end_io.write_bytes - start_io.write_bytes) / 1024**2,
        "gpu": gpu_stats,
        "output_file": output_file
    }

    # Save log entry
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
    return result.returncode

def main():
    print("Starting benchmark runs...")
    yaml_folder = "dataset"
    output_folder = "experiments"
    os.makedirs(output_folder, exist_ok=True)

    yaml_files = glob.glob(os.path.join(yaml_folder, "*.yaml"))

    for yaml_file in yaml_files:
        yaml_name = os.path.splitext(os.path.basename(yaml_file))[0]
        output_file = os.path.join(output_folder, f"{yaml_name}.out")
        command = f"uv run ai_lab_repo.py --yaml-location \"{yaml_file}\""
        print(f"Running: {command}")
        ret = track_resources(command, output_file)
        if ret != 0:
            print(f"Warning: command for {yaml_file} exited with {ret}")

if __name__ == "__main__":
    main()
