import json
import argparse

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
    data = data if isinstance(data, dict) else data[0]
    wall_time_sec = data.get("wall_time_sec", 0)
    session_hours = wall_time_sec / 3600

    # Token statistics
    total_tokens = sum(data.get("token_count", {}).values())
    input_tokens = data["token_count"].get("input", 0)
    output_tokens = data["token_count"].get("output", 0)

    # GPU usage
    gpus = data.get("gpu", [])
    num_gpus = len(gpus)

    total_power_kw = 0
    active_gpus = 0
    for gpu in gpus:
        util = gpu["util_percent"] / 100
        if util > 0:
            active_gpus += 1
            total_power_kw += gpu_power_kw * util
        else:
            # Idle GPUs assumed ~10% power
            total_power_kw += gpu_power_kw * 0.1

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

    # Scaling projection: assume all GPUs fully utilized
    projected_tokens_per_sec = tokens_per_sec_per_gpu * num_gpus * (1 / max((gpus[0]["util_percent"]/100), 0.01))

    return {
        "wall_time_sec": round(wall_time_sec, 2),
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "tokens_per_sec_per_gpu": round(tokens_per_sec_per_gpu, 2),
        "active_gpus": active_gpus,
        "avg_gpu_util_percent": round(sum([g["util_percent"] for g in gpus]) / num_gpus, 2),
        "energy_cost_usd": round(energy_cost, 5),
        "hardware_cost_usd": round(hardware_cost, 5),
        "total_cost_usd": round(total_cost, 5),
        "cost_per_1k_tokens_usd": round(cost_per_1k_tokens, 5),
        "projected_tokens_per_sec_all_gpus": round(projected_tokens_per_sec, 2)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate self-hosted GPU inference costs from monitoring JSON.")
    # parser.add_argument("json_file", help="Path to JSON file with GPU and token info")
    parser.add_argument("--electricity", type=float, default=0.45, help="Electricity price $/kWh")
    parser.add_argument("--gpu_cost", type=float, default=25000, help="GPU purchase cost $")
    parser.add_argument("--gpu_lifetime_hours", type=float, default=5*365*24, help="GPU expected lifetime in hours")
    args = parser.parse_args()

    # with open(args.json_file, "r") as f:
    #     data = json.load(f)

    data = {
    "timestamp": "2025-09-29T11:33:05.276223",
    "function": "main",
    "wall_time_sec": 155.68613526783884,
    "ram_delta_MB": 58.26171875,
    "disk_read_delta_MB": 0.0,
    "disk_write_delta_MB": 0.0078125,
    "gpu": [
      {
        "device": 0,
        "gpu_mem_delta_MB": 0.0,
        "util_percent": 31,
        "mem_util_percent": 4
      },
      {
        "device": 1,
        "gpu_mem_delta_MB": 0.0,
        "util_percent": 0,
        "mem_util_percent": 0
      },
      {
        "device": 2,
        "gpu_mem_delta_MB": 0.0,
        "util_percent": 0,
        "mem_util_percent": 0
      },
      {
        "device": 3,
        "gpu_mem_delta_MB": 0.0,
        "util_percent": 0,
        "mem_util_percent": 0
      }
    ],
    "token_count": {
      "system_prompt": 167,
      "user_prompt": 6720,
      "input": 6898,
      "output": 10794
    }
  },
    result = analyze_session(
        data,
        electricity_price=args.electricity,
        gpu_cost=args.gpu_cost,
        gpu_lifetime_hours=args.gpu_lifetime_hours
    )

    print("===== GPU Cost & Utilization Report =====")
    for k, v in result.items():
        print(f"{k}: {v}")
