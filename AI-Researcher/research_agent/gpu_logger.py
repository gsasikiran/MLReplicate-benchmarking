import time
import json
import sys
import signal
import os

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available, GPU logging disabled")

class GPULogger:
    def __init__(self, output_file, interval=0.5, gpu_ids=None):
        """
        Args:
            output_file: Path to save GPU logs
            interval: Seconds between samples (default 0.5s)
            gpu_ids: List of GPU IDs to monitor (None = all GPUs)
        """
        self.output_file = output_file
        self.interval = interval
        self.gpu_ids = gpu_ids
        self.samples = []
        self.running = True
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nGPU Logger: Received signal {signum}, stopping...")
        self.running = False
    
    def start(self):
        """Start logging GPU metrics"""
        if not PYNVML_AVAILABLE:
            print("pynvml not available, skipping GPU logging")
            return
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            # Determine which GPUs to monitor
            if self.gpu_ids is None:
                gpus_to_monitor = list(range(device_count))
            else:
                gpus_to_monitor = self.gpu_ids
            
            print(f"GPU Logger: Monitoring GPUs {gpus_to_monitor} every {self.interval}s")
            print(f"GPU Logger: Saving to {self.output_file}")
            
            sample_count = 0
            start_time = time.time()
            
            while self.running:
                timestamp = time.time()
                
                for gpu_id in gpus_to_monitor:
                    if gpu_id >= device_count:
                        continue
                    
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        sample = {
                            "timestamp": timestamp,
                            "gpu": gpu_id,
                            "utilization": util.gpu,
                            "memory_used": mem.used / (1024**2),  # MB
                            "memory_total": mem.total / (1024**2)
                        }
                        self.samples.append(sample)
                        sample_count += 1
                    except Exception as e:
                        print(f"Error reading GPU {gpu_id}: {e}")
                
                time.sleep(self.interval)
            
            # Save results
            self._save()
            
            duration = time.time() - start_time
            print(f"\nGPU Logger: Collected {sample_count} samples over {duration:.1f}s")
            print(f"GPU Logger: Saved to {self.output_file}")
            
        except Exception as e:
            print(f"GPU Logger error: {e}")
        finally:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
    
    def _save(self):
        """Save samples to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w') as f:
                json.dump(self.samples, f)
        except Exception as e:
            print(f"Error saving GPU log: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python gpu_logger.py <output_file> [interval] [gpu_ids]")
        print("Example: python gpu_logger.py /tmp/gpu_log.json 0.5 0,1")
        sys.exit(1)
    
    output_file = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    gpu_ids = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
    
    logger = GPULogger(output_file, interval, gpu_ids)
    logger.start()