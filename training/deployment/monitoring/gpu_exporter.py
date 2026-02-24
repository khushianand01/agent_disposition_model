import time
import pynvml
from prometheus_client import start_http_server, Gauge

# Metrics definition
GPU_UTIL = Gauge("gpu_utilization_percent", "GPU utilization percent", ["gpu"])
GPU_MEM_USED = Gauge("gpu_memory_used_mb", "GPU memory used in MB", ["gpu"])
GPU_MEM_TOTAL = Gauge("gpu_memory_total_mb", "GPU memory total in MB", ["gpu"])
GPU_AVAILABLE = Gauge("gpu_is_available", "Is GPU available")

def get_gpu_metrics():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        GPU_AVAILABLE.set(1 if device_count > 0 else 0)
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            GPU_UTIL.labels(gpu=str(i)).set(util.gpu)
            GPU_MEM_USED.labels(gpu=str(i)).set(mem.used / 1024 / 1024)
            GPU_MEM_TOTAL.labels(gpu=str(i)).set(mem.total / 1024 / 1024)
            
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"Error getting GPU metrics: {e}")
        GPU_AVAILABLE.set(0)

if __name__ == "__main__":
    # Start Prometheus exporter on port 9400
    start_http_server(9400)
    print("GPU Exporter started on port 9400")
    
    while True:
        get_gpu_metrics()
        time.sleep(5)
