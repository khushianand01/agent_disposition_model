import asyncio
import aiohttp
import json
import time
import random
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import sys
import os

# CONFIG
API_URL = "http://localhost:8005/predict"
DATA_PATH = "/home/ubuntu/disposition_model/data/production/test_best.json"
TOTAL_REQUESTS = 100
CONCURRENCY = 5  # Number of simultaneous requests

class StressTestDashboard:
    def __init__(self, total_expected):
        plt.ion()  # Interactive mode on
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Disposition API Stress Test - Live Dashboard", fontsize=16)
        
        self.latencies = []
        self.timestamps = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Plot 1: Latency over time
        self.ax_latency = self.axs[0, 0]
        self.ax_latency.set_title("Latency over Time")
        self.line_latency, = self.ax_latency.plot([], [], 'b-o', markersize=2)
        self.ax_latency.set_ylabel("Seconds")
        
        # Plot 2: Latency Distribution
        self.ax_dist = self.axs[0, 1]
        self.ax_dist.set_title("Latency Distribution")
        
        # Plot 3: Success vs Error
        self.ax_status = self.axs[1, 0]
        self.ax_status.set_title("Result Status")
        
        # Plot 4: Throughput Trend
        self.ax_tput = self.axs[1, 1]
        self.ax_tput.set_title("Throughput (Req/Sec)")
        self.line_tput, = self.ax_tput.plot([], [], 'g-')
        self.throughput_data = []
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update(self, latency, success):
        self.latencies.append(latency)
        self.timestamps.append(time.time() - self.start_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            current_tput = len(self.latencies) / elapsed
            self.throughput_data.append(current_tput)

        # Update Plots
        # 1. Latency Line
        self.line_latency.set_data(self.timestamps, self.latencies)
        self.ax_latency.relim()
        self.ax_latency.autoscale_view()
        
        # 2. Distribution
        self.ax_dist.clear()
        self.ax_dist.set_title("Latency Distribution")
        if self.latencies:
            self.ax_dist.hist(self.latencies, bins=15, color='skyblue', edgecolor='black')
        
        # 3. Status Bar
        self.ax_status.clear()
        self.ax_status.set_title("Result Status")
        self.ax_status.bar(['Success', 'Error'], [self.success_count, self.error_count], color=['green', 'red'])
        
        # 4. Throughput
        self.line_tput.set_data(self.timestamps, self.throughput_data)
        self.ax_tput.relim()
        self.ax_tput.autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

async def send_request(session, transcript, dashboard):
    payload = {"transcript": transcript}
    start_t = time.time()
    try:
        async with session.post(API_URL, json=payload) as response:
            latency = time.time() - start_t
            if response.status == 200:
                await response.json()
                dashboard.update(latency, True)
            else:
                print(f"Error: {response.status}")
                dashboard.update(latency, False)
    except Exception as e:
        latency = time.time() - start_t
        print(f"Exception: {e}")
        dashboard.update(latency, False)

async def main():
    print(f"Loading data from {DATA_PATH}...")
    try:
        with open(DATA_PATH, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Pick 100 samples
    samples = random.sample(data, min(TOTAL_REQUESTS, len(data)))
    transcripts = []
    for s in samples:
        # Check 'input' or 'transcript' or 'instruction' as fallback
        text = s.get('input') or s.get('transcript') or s.get('instruction') or ""
        transcripts.append(str(text))
    
    print(f"Starting Stress Test: {TOTAL_REQUESTS} total requests, Concurrency: {CONCURRENCY}")
    dashboard = StressTestDashboard(TOTAL_REQUESTS)
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, text in enumerate(transcripts):
            async def wrapped_req(t=text):
                async with semaphore:
                    await send_request(session, t, dashboard)
            tasks.append(wrapped_req())
            
        await asyncio.gather(*tasks)

    print("\n--- STRESS TEST COMPLETE ---")
    print(f"Total Requests: {len(dashboard.latencies)}")
    print(f"Success: {dashboard.success_count}")
    print(f"Errors: {dashboard.error_count}")
    if dashboard.latencies:
        print(f"Avg Latency: {np.mean(dashboard.latencies):.2f}s")
        print(f"P95 Latency: {np.percentile(dashboard.latencies, 95):.2f}s")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())
