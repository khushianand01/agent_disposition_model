import requests
import time
import concurrent.futures
import statistics

API_URL = "http://localhost:8005/predict"

# Sample transcripts varying in length and complexity
TRANSCRIPTS = [
    "Agent: Hello. Borrower: I will pay 500 dollars tomorrow.",
    "Agent: Hello? Borrower: My job is lost, I cannot pay the EMI.",
    "Agent: Pay your dues. Borrower: Send your collection agent to my house, I will pay cash. Agent: Okay.",
    "Agent: Am I speaking to Rahul? Borrower: No, you have the wrong number. I don't know any Rahul.",
    "Agent: Loan payment pending. Borrower: Interest bahut zyada hai. Mujhe kam interest rate chahiye tabhi bharunga. Agent: Sir, we can't change it now. Borrower: Toh fir main baat karunga manager se.",
    "Agent: Namaskar, EMI eppodu kattuveergal? Borrower: Naan next week kattugiren.",
    "Agent: Hehe hello sir payment detail? Borrower: I have enough funds now, I want to foreclosure the account. How much to pay? Agent: One minute sir... it is 4700. Borrower: Fine, bhej do link.",
    "Agent: The number you are trying to reach is currently out of network coverage area."
]

def make_request(request_id):
    transcript = TRANSCRIPTS[request_id % len(TRANSCRIPTS)]
    payload = {"transcript": transcript, "current_date": "2026-02-27"}
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        end_time = time.time()
        latency = end_time - start_time
        
        if response.status_code == 200:
            return {"status": "success", "latency": latency, "id": request_id}
        else:
            return {"status": "error", "error": f"HTTP {response.status_code} - {response.text}", "latency": latency, "id": request_id}
    except Exception as e:
        end_time = time.time()
        return {"status": "error", "error": str(e), "latency": end_time - start_time, "id": request_id}

def run_stress_test(concurrency, total_requests):
    print(f"\n🚀 Running Stress Test: {concurrency} workers, {total_requests} total requests...")
    results = []
    
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks
        future_to_req = {executor.submit(make_request, i): i for i in range(total_requests)}
        
        for future in concurrent.futures.as_completed(future_to_req):
            res = future.result()
            results.append(res)
            # Print minimal progress
            if res["status"] == "success":
                print(f"  [+] Req {res['id']:02d} | Success | {res['latency']:.2f}s")
            else:
                print(f"  [-] Req {res['id']:02d} | Failed  | {res['latency']:.2f}s | {res['error']}")
                
    end_total = time.time()
    
    # Analyze Results
    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    
    success_rate = (len(successes) / total_requests) * 100
    
    print("\n" + "="*50)
    print(f"📊 RESULTS: {concurrency} Concurrent Workers")
    print("="*50)
    print(f"Total Requests Processed : {total_requests}")
    print(f"Total Time Taken         : {end_total - start_total:.2f} seconds")
    print(f"Success Rate             : {success_rate:.1f}% ({len(successes)}/{total_requests})")
    print(f"Error Rate               : {100 - success_rate:.1f}% ({len(errors)}/{total_requests})")
    
    if successes:
        latencies = [r["latency"] for r in successes]
        print(f"\nLatency Metrics (Successful Requests):")
        print(f"  Average (Mean) : {statistics.mean(latencies):.2f}s")
        print(f"  Median (P50)   : {statistics.median(latencies):.2f}s")
        if len(latencies) >= 5:
            print(f"  P90            : {statistics.quantiles(latencies, n=10)[8]:.2f}s")
            print(f"  P95            : {statistics.quantiles(latencies, n=20)[18]:.2f}s")
        print(f"  Min            : {min(latencies):.2f}s")
        print(f"  Max            : {max(latencies):.2f}s")
        
    print("="*50)

if __name__ == "__main__":
    time.sleep(2) # Give it a second
    run_stress_test(concurrency=1, total_requests=5)
    time.sleep(3)
    run_stress_test(concurrency=3, total_requests=10)
    time.sleep(3)
    run_stress_test(concurrency=5, total_requests=15)
