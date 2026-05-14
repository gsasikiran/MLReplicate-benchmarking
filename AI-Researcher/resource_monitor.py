import time
import psutil
import threading
import json
import os
import logging
from datetime import datetime
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
from pathlib import Path
import pynvml

_active_monitor = None

def get_active_monitor():
    """Get the currently active ResourceMonitor instance"""
    global _active_monitor
    return _active_monitor


@dataclass
class GPUMetrics:
    """GPU usage metrics"""
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_utilization_avg: List[float] = field(default_factory=list)
    gpu_utilization_max: List[float] = field(default_factory=list)
    gpu_memory_used_mb_avg: List[float] = field(default_factory=list)
    gpu_memory_used_mb_max: List[float] = field(default_factory=list)
    gpu_memory_total_mb: List[float] = field(default_factory=list)

@dataclass
class CostMetrics:
    """Cost calculation based on token usage (OpenAI API)"""
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cost_usd: float = 0.0
    completion_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

@dataclass
class SelfHostedCosts:
    """Self-hosted GPU compute costs"""
    gpus_used: int = 0
    gpu_hours: float = 0.0
    electricity_kwh: float = 0.0
    electricity_cost_usd: float = 0.0
    hardware_amortization_usd: float = 0.0
    total_compute_cost_usd: float = 0.0

@dataclass
class ResourceMetrics:
    """Data class to store resource usage metrics"""
    phase: str  # "research" or "paper_generation"
    instance_id: str
    start_time: str
    end_time: str
    total_duration: float
    cpu_percent_avg: float
    cpu_percent_max: float
    memory_usage_mb_avg: float
    memory_usage_mb_max: float
    memory_percent_avg: float
    memory_percent_max: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    token_usage: Dict[str, int]
    cost_metrics: CostMetrics
    self_hosted_costs: SelfHostedCosts
    gpu_metrics: GPUMetrics
    process_count: int
    thread_count: int

class ResourceMonitor:
    """Real-time resource monitoring for AI-Researcher operations"""
    
    # Pricing per 1M tokens (USD) - OpenAI API
    PRICING = {
        "gpt-4o": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "gpt-4o-2024-08-06": {"prompt": 2.50, "completion": 10.00},
        "gpt-4o-mini-2024-07-18": {"prompt": 0.15, "completion": 0.60},
    }
    
    # GPU hardware specifications (RTX 3090)
    GPU_SPECS = {
        "rtx_3090": {
            "max_power_watts": 350,
            "purchase_cost_usd": 1500,
            "lifetime_hours": 5 * 365 * 24,  # 5 years = 43,800 hours
        }
    }
    
    # Electricity pricing (€0.45/kWh ≈ $0.50/kWh)
    ELECTRICITY_PRICE_USD_PER_KWH = 0.50
    
    def __init__(self, log_file: Optional[str] = None, phase: str = "unknown", model_name: str = "gpt-4o", 
                 gpu_log_path: Optional[str] = None, gpu_id: Optional[int] = None, instance_id: Optional[str] = None):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics_thread = None
        self.phase = phase
        self.model_name = model_name
        self.gpu_log_path = gpu_log_path
        self.gpu_id = gpu_id
        self.instance_id = instance_id or "unknown"
        self.log_file = log_file or f"ai_researcher_{phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Metrics storage
        self.start_time = None
        self.end_time = None
        self.cpu_samples = []
        self.memory_samples = []
        self.memory_percent_samples = []
        self.initial_io = None
        self.final_io = None
        self.initial_net = None
        self.final_net = None
        self.token_usage = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        self.token_usage_by_model = {}  # Track tokens per model
        
        # GPU tracking with pynvml
        self.gpu_available = self._check_gpu_available()
        self.gpu_samples = []  # List of dicts with GPU metrics per sample
        
        # Setup logging
        self.setup_logging()
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available via pynvml"""
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            return count > 0
        except Exception as e:
            return False
    
    def _get_gpu_metrics(self) -> Optional[Dict]:
        """Get current GPU metrics using pynvml"""
        if not self.gpu_available:
            return None
        
        try:
            gpu_data = []
            device_count = pynvml.nvmlDeviceGetCount()
            
            # If gpu_id is specified, only monitor that GPU
            gpu_indices = [self.gpu_id] if self.gpu_id is not None else range(device_count)
            
            for i in gpu_indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_data.append({
                    'index': i,
                    'name': name,
                    'utilization': float(util.gpu),
                    'memory_used': float(mem_info.used / (1024 * 1024)),  # Convert to MB
                    'memory_total': float(mem_info.total / (1024 * 1024))
                })
            
            return {'gpus': gpu_data, 'timestamp': time.time()}
        
        except Exception as e:
            self.logger.warning(f"Failed to get GPU metrics: {e}")
            return None
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ResourceMonitor')
    
    def start_monitoring(self):
        """Start resource monitoring in a separate thread"""
        if self.monitoring:
            return
        
        global _active_monitor
        _active_monitor = self    
    
        self.monitoring = True
        self.start_time = datetime.now()
        
        # Get initial measurements
        self.initial_io = psutil.disk_io_counters()
        self.initial_net = psutil.net_io_counters()
        
        # Start monitoring thread
        self.metrics_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.metrics_thread.start()
        
        phase_name = self.phase.replace('_', ' ').title()
        self.logger.info(f"Resource monitoring started for {phase_name} phase")
        print(f"\n{'='*60}")
        print(f"MONITORING: {phase_name} Phase")
        print(f"Instance: {self.instance_id}")
        print(f"Started at: {self.start_time.strftime('%H:%M:%S')}")
        print(f"Model: {self.model_name}")
        print(f"GPU Available: {'Yes' if self.gpu_available else 'No'}")
        print(f"{'='*60}\n")
    
    def stop_monitoring(self) -> ResourceMetrics:
        """Stop monitoring and return collected metrics"""
        if not self.monitoring:
            return None
        
        global _active_monitor
        _active_monitor = None 
        
        self.monitoring = False
        self.end_time = datetime.now()
        
        # Wait for monitoring thread to finish
        if self.metrics_thread:
            self.metrics_thread.join(timeout=2.0)
        
        # Get final measurements
        self.final_io = psutil.disk_io_counters()
        self.final_net = psutil.net_io_counters()
        
        # Shutdown pynvml
        try:
            if self.gpu_available:
                pynvml.nvmlShutdown()
        except:
            pass
        
        # Calculate metrics
        metrics = self._calculate_final_metrics()
        
        # Log and display results
        self._log_metrics(metrics)
        self._display_metrics(metrics)
        
        return metrics
    
    def _monitor_resources(self):
        """Monitor resources in background thread"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = self.process.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)
                
                # Memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                memory_percent = self.process.memory_percent()
                
                self.memory_samples.append(memory_mb)
                self.memory_percent_samples.append(memory_percent)
                
                # GPU metrics
                if self.gpu_available:
                    gpu_data = self._get_gpu_metrics()
                    if gpu_data:
                        self.gpu_samples.append(gpu_data)
                
                time.sleep(2.0)  # Sample every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                break
    
    def _calculate_gpu_metrics(self) -> GPUMetrics:
        """Calculate GPU metrics from samples"""
        if not self.gpu_samples:
            return GPUMetrics(gpu_available=False)
        
        # Extract GPU info from first sample
        first_sample = self.gpu_samples[0]['gpus']
        gpu_count = len(first_sample)
        gpu_names = [gpu['name'] for gpu in first_sample]
        gpu_memory_total = [gpu['memory_total'] for gpu in first_sample]
        
        # Calculate averages and maxes for each GPU
        gpu_util_avg = [0.0] * gpu_count
        gpu_util_max = [0.0] * gpu_count
        gpu_mem_avg = [0.0] * gpu_count
        gpu_mem_max = [0.0] * gpu_count
        
        for sample in self.gpu_samples:
            for i, gpu in enumerate(sample['gpus']):
                gpu_util_avg[i] += gpu['utilization']
                gpu_util_max[i] = max(gpu_util_max[i], gpu['utilization'])
                gpu_mem_avg[i] += gpu['memory_used']
                gpu_mem_max[i] = max(gpu_mem_max[i], gpu['memory_used'])
        
        # Calculate averages
        sample_count = len(self.gpu_samples)
        gpu_util_avg = [u / sample_count for u in gpu_util_avg]
        gpu_mem_avg = [m / sample_count for m in gpu_mem_avg]
        
        return GPUMetrics(
            gpu_available=True,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
            gpu_utilization_avg=gpu_util_avg,
            gpu_utilization_max=gpu_util_max,
            gpu_memory_used_mb_avg=gpu_mem_avg,
            gpu_memory_used_mb_max=gpu_mem_max,
            gpu_memory_total_mb=gpu_memory_total
        )
    
    def _read_training_gpu_logs(self) -> Optional[Dict]:
        """Read GPU logs generated during training inside container"""
        if not self.gpu_log_path or not os.path.exists(self.gpu_log_path):
            return None
        
        try:
            with open(self.gpu_log_path, 'r') as f:
                gpu_data = json.load(f)
            
            if not gpu_data:
                return None
            
            # Aggregate training GPU metrics
            gpu_stats = {}
            for metric in gpu_data:
                gpu_id = metric['gpu']
                if gpu_id not in gpu_stats:
                    gpu_stats[gpu_id] = {
                        'utilization': [],
                        'memory_used': []
                    }
                gpu_stats[gpu_id]['utilization'].append(metric['utilization'])
                gpu_stats[gpu_id]['memory_used'].append(metric['memory_used'])
            
            # Calculate averages and maxes
            aggregated = {}
            for gpu_id, stats in gpu_stats.items():
                aggregated[gpu_id] = {
                    'utilization_avg': sum(stats['utilization']) / len(stats['utilization']),
                    'utilization_max': max(stats['utilization']),
                    'memory_avg_mb': sum(stats['memory_used']) / len(stats['memory_used']),
                    'memory_max_mb': max(stats['memory_used'])
                }
            
            return aggregated
        except Exception as e:
            self.logger.warning(f"Failed to read training GPU logs: {e}")
            return None
    
    def _calculate_cost(self) -> CostMetrics:
        """Calculate OpenAI API cost based on token usage (per-model if available)"""
        total_prompt_cost = 0.0
        total_completion_cost = 0.0
        
        # If we have per-model tracking, calculate cost for each model
        if self.token_usage_by_model:
            for model_name, tokens in self.token_usage_by_model.items():
                pricing = self.PRICING.get(model_name, self.PRICING.get("gpt-4o"))
                prompt_cost = (tokens["prompt_tokens"] / 1_000_000) * pricing["prompt"]
                completion_cost = (tokens["completion_tokens"] / 1_000_000) * pricing["completion"]
                total_prompt_cost += prompt_cost
                total_completion_cost += completion_cost
            
            model_display = f"mixed ({len(self.token_usage_by_model)} models)"
        else:
            # Fallback to single model pricing
            pricing = self.PRICING.get(self.model_name, self.PRICING.get("gpt-4o"))
            total_prompt_cost = (self.token_usage["prompt_tokens"] / 1_000_000) * pricing["prompt"]
            total_completion_cost = (self.token_usage["completion_tokens"] / 1_000_000) * pricing["completion"]
            model_display = self.model_name
        
        return CostMetrics(
            model_name=model_display,
            prompt_tokens=self.token_usage["prompt_tokens"],
            completion_tokens=self.token_usage["completion_tokens"],
            total_tokens=self.token_usage["total_tokens"],
            prompt_cost_usd=total_prompt_cost,
            completion_cost_usd=total_completion_cost,
            total_cost_usd=total_prompt_cost + total_completion_cost
        )
    
    
    def _calculate_self_hosted_costs(self, duration_hours: float, gpu_metrics: GPUMetrics) -> SelfHostedCosts:
        """Calculate self-hosted GPU compute costs (RTX 3090)"""
        if not gpu_metrics.gpu_available or gpu_metrics.gpu_count == 0:
            return SelfHostedCosts()
        
        specs = self.GPU_SPECS["rtx_3090"]
        
        # Calculate GPU hours (accounting for utilization)
        total_gpu_hours = 0.0
        total_power_kwh = 0.0
        
        for i in range(gpu_metrics.gpu_count):
            avg_util = gpu_metrics.gpu_utilization_avg[i] / 100.0  # Convert % to fraction
            
            # GPU hours = actual usage time × utilization
            gpu_hours = duration_hours * avg_util
            total_gpu_hours += gpu_hours
            
            # Power consumption = max_power × utilization × time
            # Idle GPUs still consume ~10% power
            actual_util = max(avg_util, 0.10)
            power_kwh = (specs["max_power_watts"] / 1000) * actual_util * duration_hours
            total_power_kwh += power_kwh
        
        # Calculate costs
        electricity_cost = total_power_kwh * self.ELECTRICITY_PRICE_USD_PER_KWH
        
        # Hardware amortization: (GPU cost / lifetime hours) × GPU hours used
        hourly_gpu_cost = specs["purchase_cost_usd"] / specs["lifetime_hours"]
        hardware_cost = hourly_gpu_cost * total_gpu_hours
        
        total_cost = electricity_cost + hardware_cost
        
        return SelfHostedCosts(
            gpus_used=gpu_metrics.gpu_count,
            gpu_hours=round(total_gpu_hours, 3),
            electricity_kwh=round(total_power_kwh, 3),
            electricity_cost_usd=round(electricity_cost, 4),
            hardware_amortization_usd=round(hardware_cost, 4),
            total_compute_cost_usd=round(total_cost, 4)
        )
    
    def _calculate_final_metrics(self) -> ResourceMetrics:
        """Calculate final metrics from collected samples"""
        duration = (self.end_time - self.start_time).total_seconds()
        duration_hours = duration / 3600.0
        
        # CPU metrics
        cpu_avg = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        cpu_max = max(self.cpu_samples) if self.cpu_samples else 0
        
        # Memory metrics
        memory_avg = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
        memory_max = max(self.memory_samples) if self.memory_samples else 0
        memory_percent_avg = sum(self.memory_percent_samples) / len(self.memory_percent_samples) if self.memory_percent_samples else 0
        memory_percent_max = max(self.memory_percent_samples) if self.memory_percent_samples else 0
        
        # I/O metrics
        disk_read = ((self.final_io.read_bytes - self.initial_io.read_bytes) / (1024 * 1024)) if (self.initial_io and self.final_io) else 0
        disk_write = ((self.final_io.write_bytes - self.initial_io.write_bytes) / (1024 * 1024)) if (self.initial_io and self.final_io) else 0
        
        # Network metrics
        net_sent = ((self.final_net.bytes_sent - self.initial_net.bytes_sent) / (1024 * 1024)) if (self.initial_net and self.final_net) else 0
        net_recv = ((self.final_net.bytes_recv - self.initial_net.bytes_recv) / (1024 * 1024)) if (self.initial_net and self.final_net) else 0
        
        # Process info
        try:
            process_count = len(psutil.pids())
            thread_count = self.process.num_threads()
        except:
            process_count = 0
            thread_count = 0

        # GPU and cost metrics
        gpu_metrics = self._calculate_gpu_metrics()
        training_gpu = self._read_training_gpu_logs()
        
        # Override with training data if available - USE ALL GPUs from container log
        if training_gpu:
            # If we have container GPU logs, they're more accurate than host sampling
            # Rebuild gpu_metrics using container data
            for gpu_id, stats in training_gpu.items():
                if gpu_id < gpu_metrics.gpu_count:
                    # Override this GPU's metrics with container data
                    gpu_metrics.gpu_utilization_avg[gpu_id] = stats['utilization_avg']
                    gpu_metrics.gpu_utilization_max[gpu_id] = stats['utilization_max']
                    gpu_metrics.gpu_memory_used_mb_avg[gpu_id] = stats['memory_avg_mb']
                    gpu_metrics.gpu_memory_used_mb_max[gpu_id] = stats['memory_max_mb']
                elif gpu_metrics.gpu_count == 0:
                    # If host didn't detect GPUs but container logged them, initialize
                    gpu_metrics.gpu_available = True
                    gpu_metrics.gpu_count = max(training_gpu.keys()) + 1
                    gpu_metrics.gpu_names = [f"GPU {i}" for i in range(gpu_metrics.gpu_count)]
                    gpu_metrics.gpu_utilization_avg = [0.0] * gpu_metrics.gpu_count
                    gpu_metrics.gpu_utilization_max = [0.0] * gpu_metrics.gpu_count
                    gpu_metrics.gpu_memory_used_mb_avg = [0.0] * gpu_metrics.gpu_count
                    gpu_metrics.gpu_memory_used_mb_max = [0.0] * gpu_metrics.gpu_count
                    gpu_metrics.gpu_memory_total_mb = [24576.0] * gpu_metrics.gpu_count  # RTX 3090
                    
                    # Now set the values
                    gpu_metrics.gpu_utilization_avg[gpu_id] = stats['utilization_avg']
                    gpu_metrics.gpu_utilization_max[gpu_id] = stats['utilization_max']
                    gpu_metrics.gpu_memory_used_mb_avg[gpu_id] = stats['memory_avg_mb']
                    gpu_metrics.gpu_memory_used_mb_max[gpu_id] = stats['memory_max_mb']
            
            print(f"✓ Using GPU metrics from container logs ({len(training_gpu)} GPUs)")

        # Calculate costs
        cost_metrics = self._calculate_cost()
        self_hosted_costs = self._calculate_self_hosted_costs(duration_hours, gpu_metrics)
        
        return ResourceMetrics(
            phase=self.phase,
            instance_id=self.instance_id,
            start_time=self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
            end_time=self.end_time.strftime('%Y-%m-%d %H:%M:%S'),
            total_duration=duration,
            cpu_percent_avg=cpu_avg,
            cpu_percent_max=cpu_max,
            memory_usage_mb_avg=memory_avg,
            memory_usage_mb_max=memory_max,
            memory_percent_avg=memory_percent_avg,
            memory_percent_max=memory_percent_max,
            disk_io_read_mb=disk_read,
            disk_io_write_mb=disk_write,
            network_sent_mb=net_sent,
            network_recv_mb=net_recv,
            token_usage=self.token_usage,
            cost_metrics=cost_metrics,
            self_hosted_costs=self_hosted_costs,
            gpu_metrics=gpu_metrics,
            process_count=process_count,
            thread_count=thread_count
        )
    
    def update_token_usage(self, completion_tokens: int = 0, prompt_tokens: int = 0, model_name: str = None):
        """Update token usage statistics, optionally tracking per model"""
        # Update totals
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["total_tokens"] = self.token_usage["completion_tokens"] + self.token_usage["prompt_tokens"]
        
        # Track per model if specified
        if model_name:
            if model_name not in self.token_usage_by_model:
                self.token_usage_by_model[model_name] = {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "total_tokens": 0
                }
            self.token_usage_by_model[model_name]["completion_tokens"] += completion_tokens
            self.token_usage_by_model[model_name]["prompt_tokens"] += prompt_tokens
            self.token_usage_by_model[model_name]["total_tokens"] += completion_tokens + prompt_tokens

    def _log_metrics(self, metrics: ResourceMetrics):
        """Log metrics to file in centralized location"""
        try:
            from directory_manager import get_directory_manager
            manager = get_directory_manager()
            
            if manager.current_run_dir:
                metrics_dir = manager.current_run_dir / "logs" / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                output_file = metrics_dir / f"{self.phase}_{self.instance_id}_metrics.json"
            else:
                metrics_dir = Path("metrics")
                metrics_dir.mkdir(exist_ok=True)
                output_file = metrics_dir / f"{self.phase}_{self.instance_id}_metrics.json"
            
            with open(output_file, 'w') as f:
                json.dump(asdict(metrics), f, indent=2)
            
            self.logger.info(f"Metrics saved to {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
            return None
    
    def _display_metrics(self, metrics: ResourceMetrics):
        """Display metrics in terminal"""
        print("\n" + "="*70)
        print(f"AI-RESEARCHER {metrics.phase.upper().replace('_', ' ')} PHASE METRICS")
        print("="*70)
        
        # Time metrics
        print(f"\nEXECUTION TIME:")
        print(f"   Instance: {metrics.instance_id}")
        print(f"   Start: {metrics.start_time}")
        print(f"   End: {metrics.end_time}")
        print(f"   Duration: {metrics.total_duration:.2f}s ({metrics.total_duration/60:.1f} min)")
        
        # Token metrics
        tokens = metrics.token_usage
        if tokens['total_tokens'] > 0:
            tokens_per_sec = tokens['total_tokens'] / metrics.total_duration if metrics.total_duration > 0 else 0
            print(f"\nTOKEN USAGE:")
            print(f"   Total Tokens: {tokens['total_tokens']:,}")
            print(f"   Prompt Tokens: {tokens['prompt_tokens']:,}")
            print(f"   Completion Tokens: {tokens['completion_tokens']:,}")
            print(f"   Throughput: {tokens_per_sec:.2f} tokens/sec")
            
            # Show per-model breakdown if available
            if hasattr(self, 'token_usage_by_model') and self.token_usage_by_model:
                print(f"\n   Per-Model Breakdown:")
                for model, model_tokens in self.token_usage_by_model.items():
                    # Calculate cost for this specific model
                    pricing = self.PRICING.get(model, self.PRICING.get("gpt-4o"))
                    model_cost = (
                        (model_tokens['prompt_tokens'] / 1_000_000) * pricing['prompt'] +
                        (model_tokens['completion_tokens'] / 1_000_000) * pricing['completion']
                    )
                    cost_per_1k = (model_cost / model_tokens['total_tokens']) * 1000 if model_tokens['total_tokens'] > 0 else 0
                    
                    print(f"      {model}:")
                    print(f"         Tokens: {model_tokens['total_tokens']:,} (Prompt: {model_tokens['prompt_tokens']:,} | Completion: {model_tokens['completion_tokens']:,})")
                    print(f"         Cost: ${model_cost:.4f} (${cost_per_1k:.4f} per 1K tokens)")

        # OpenAI API cost
        cost = metrics.cost_metrics
        print(f"\nOPENAI API COSTS (Model: {cost.model_name}):")
        print(f"   Prompt Cost: ${cost.prompt_cost_usd:.4f}")
        print(f"   Completion Cost: ${cost.completion_cost_usd:.4f}")
        print(f"   Total API Cost: ${cost.total_cost_usd:.4f}")
        if tokens['total_tokens'] > 0:
            cost_per_1k = (cost.total_cost_usd / tokens['total_tokens']) * 1000
            
        
        # GPU commpute costs
        gpu_cost = metrics.self_hosted_costs
        if gpu_cost.gpus_used > 0:
            print(f"\nGPU COMPUTE COSTS (RTX 3090):")
            print(f"   GPUs Used: {gpu_cost.gpus_used}")
            print(f"   GPU Hours: {gpu_cost.gpu_hours:.3f}h")
            print(f"   Electricity: {gpu_cost.electricity_kwh:.3f} kWh (${gpu_cost.electricity_cost_usd:.4f})")
            print(f"   Hardware Depreciation: ${gpu_cost.hardware_amortization_usd:.4f}")
            print(f"   Total GPU Cost: ${gpu_cost.total_compute_cost_usd:.4f}")
        
        # Total cost summary
        api_cost = metrics.cost_metrics.total_cost_usd
        if api_cost > 0 and gpu_cost.gpus_used > 0:
            total_cost = api_cost + gpu_cost.total_compute_cost_usd
            print(f"\n TOTAL RESEARCH COST:")
            print(f"   OpenAI API: ${api_cost:.4f}")
            print(f"   GPU Compute: ${gpu_cost.total_compute_cost_usd:.4f}")
            print(f"   Total: ${total_cost:.4f}")
        
        # CPU/Memory
        print(f"\nCPU & MEMORY:")
        print(f"   CPU - Avg: {metrics.cpu_percent_avg:.1f}% | Max: {metrics.cpu_percent_max:.1f}%")
        print(f"   Memory - Avg: {metrics.memory_usage_mb_avg:.1f}MB | Max: {metrics.memory_usage_mb_max:.1f}MB")
        
        # GPU metrics
        gpu = metrics.gpu_metrics
        if gpu.gpu_available:
            print(f"\nGPU USAGE:")
            for i in range(gpu.gpu_count):
                print(f"   GPU {i} ({gpu.gpu_names[i]}):")
                print(f"      Utilization - Avg: {gpu.gpu_utilization_avg[i]:.1f}% | Max: {gpu.gpu_utilization_max[i]:.1f}%")
                print(f"      Memory - Avg: {gpu.gpu_memory_used_mb_avg[i]:.0f}MB | Max: {gpu.gpu_memory_used_mb_max[i]:.0f}MB | Total: {gpu.gpu_memory_total_mb[i]:.0f}MB")
        else:
            print(f"\nGPU USAGE: No GPU detected")
        
        # I/O
        print(f"\nI/O:")
        print(f"   Disk - Read: {metrics.disk_io_read_mb:.1f}MB | Write: {metrics.disk_io_write_mb:.1f}MB")
        print(f"   Network - Sent: {metrics.network_sent_mb:.1f}MB | Received: {metrics.network_recv_mb:.1f}MB")
        
        print("="*70 + "\n")


def create_combined_metrics(instance_id: str, run_dir: Path = None):
    """
    Create a combined metrics summary from research and paper generation phases.
    
    Args:
        instance_id: Instance identifier
        run_dir: Run directory path (if None, uses current run from directory_manager)
    
    Returns:
        Path to combined metrics file
    """
    try:
        from directory_manager import get_directory_manager
        
        if run_dir is None:
            manager = get_directory_manager()
            run_dir = manager.current_run_dir
        
        if not run_dir:
            print("No run directory found")
            return None
        
        metrics_dir = run_dir / "logs" / "metrics"
        
        # Load individual phase metrics
        research_file = metrics_dir / f"research_{instance_id}_metrics.json"
        paper_file = metrics_dir / f"paper_generation_{instance_id}_metrics.json"
        
        research_metrics = None
        paper_metrics = None
        
        if research_file.exists():
            with open(research_file, 'r') as f:
                research_metrics = json.load(f)
        
        if paper_file.exists():
            with open(paper_file, 'r') as f:
                paper_metrics = json.load(f)
        
        if not research_metrics and not paper_metrics:
            print("No phase metrics found")
            return None
        
        # Build combined summary
        combined = {
            "run_id": run_dir.name,
            "instance_id": instance_id,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "phases": {}
        }
        
        total_duration = 0
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_openai_cost = 0.0
        total_compute_cost = 0.0
        
        # Add research phase
        if research_metrics:
            combined["phases"]["research"] = {
                "duration_sec": research_metrics["total_duration"],
                "tokens": research_metrics["token_usage"],
                "openai_cost_usd": research_metrics["cost_metrics"]["total_cost_usd"],
                "compute_cost_usd": research_metrics["self_hosted_costs"]["total_compute_cost_usd"],
                "total_phase_cost_usd": research_metrics["cost_metrics"]["total_cost_usd"] + research_metrics["self_hosted_costs"]["total_compute_cost_usd"]
            }
            total_duration += research_metrics["total_duration"]
            for key in total_tokens:
                total_tokens[key] += research_metrics["token_usage"][key]
            total_openai_cost += research_metrics["cost_metrics"]["total_cost_usd"]
            total_compute_cost += research_metrics["self_hosted_costs"]["total_compute_cost_usd"]
        
        # Add paper generation phase
        if paper_metrics:
            combined["phases"]["paper_generation"] = {
                "duration_sec": paper_metrics["total_duration"],
                "tokens": paper_metrics["token_usage"],
                "openai_cost_usd": paper_metrics["cost_metrics"]["total_cost_usd"],
                "compute_cost_usd": paper_metrics["self_hosted_costs"]["total_compute_cost_usd"],
                "total_phase_cost_usd": paper_metrics["cost_metrics"]["total_cost_usd"] + paper_metrics["self_hosted_costs"]["total_compute_cost_usd"]
            }
            total_duration += paper_metrics["total_duration"]
            for key in total_tokens:
                total_tokens[key] += paper_metrics["token_usage"][key]
            total_openai_cost += paper_metrics["cost_metrics"]["total_cost_usd"]
            total_compute_cost += paper_metrics["self_hosted_costs"]["total_compute_cost_usd"]
        
        # Calculate totals
        tokens_per_sec = total_tokens["total_tokens"] / total_duration if total_duration > 0 else 0
        cost_per_1k_tokens = (total_openai_cost / total_tokens["total_tokens"]) * 1000 if total_tokens["total_tokens"] > 0 else 0
        
        combined["total"] = {
            "duration_sec": total_duration,
            "duration_min": round(total_duration / 60, 2),
            "total_tokens": total_tokens["total_tokens"],
            "prompt_tokens": total_tokens["prompt_tokens"],
            "completion_tokens": total_tokens["completion_tokens"],
            "tokens_per_sec": round(tokens_per_sec, 2),
            "openai_cost_usd": round(total_openai_cost, 4),
            "compute_cost_usd": round(total_compute_cost, 4),
            "total_cost_usd": round(total_openai_cost + total_compute_cost, 4),
            "cost_per_1k_tokens_usd": round(cost_per_1k_tokens, 4)
        }
        
        # Save combined metrics
        combined_file = metrics_dir / f"combined_{instance_id}_metrics.json"
        with open(combined_file, 'w') as f:
            json.dump(combined, f, indent=2)
        
        print(f"\n✓ Combined metrics saved to: {combined_file}")
        
        # Display summary
        print("\n" + "="*70)
        print("COMBINED METRICS SUMMARY")
        print("="*70)
        print(f"Instance: {instance_id}")
        print(f"Total Duration: {combined['total']['duration_min']:.1f} minutes")
        print(f"Total Tokens: {combined['total']['total_tokens']:,}")
        print(f"Throughput: {combined['total']['tokens_per_sec']:.2f} tokens/sec")
        print(f"\nCosts:")
        print(f"  OpenAI API: ${combined['total']['openai_cost_usd']:.4f}")
        print(f"  GPU Compute: ${combined['total']['compute_cost_usd']:.4f}")
        print(f"  Total: ${combined['total']['total_cost_usd']:.4f}")
        print(f"  Cost per 1K tokens: ${combined['total']['cost_per_1k_tokens_usd']:.4f}")
        print("="*70 + "\n")
        
        return combined_file
        
    except Exception as e:
        print(f"Error creating combined metrics: {e}")
        import traceback
        traceback.print_exc()
        return None
    

@contextmanager
def monitor_ai_researcher(log_file: Optional[str] = None, phase: str = "unknown", model_name: str = "gpt-4o"):
    """Context manager for monitoring AI-Researcher execution"""
    monitor = ResourceMonitor(log_file, phase, model_name)
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        metrics = monitor.stop_monitoring()
        return metrics