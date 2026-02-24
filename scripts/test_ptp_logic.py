import requests
import json
import time

url = "http://localhost:8005/predict"
headers = {"Content-Type": "application/json"}

tests = [
    # Amounts
    ("6500 रुपये, 25 तारीख", "Agent: नमस्ते, आपका 6500 रुपये का payment pending है। Borrower: 25 तारीख को salary आएगी, उसी दिन 6500 जमा कर दूंगा।"),
    ("रुपये at end", "Agent: Payment due. Borrower: Main 5000 dunga Friday ko."),
    ("Rs format", "Agent: Due pending. Borrower: I will pay Rs. 4500 on Monday."),
    ("₹ format", "Agent: Due pending. Borrower: Main ₹12000 aaj sham tak transfer kar dunga."),
    ("rupaye format", "Customer: Haan, mere paas 3500 rupaye hain, parso de dunga."),
    
    # Dates
    ("Hindi kal", "Customer: Main 1000 rupees kal bhej dunga pakka."),
    ("Hindi aaj", "Customer: Aaj sham tak 2000 rupaye transfer kar deta hoon."),
    ("Hindi parso", "Customer: Parso 3000 ka arrangement ho jayega."),
    ("English Wednesday", "Customer: Main Wednesday tak baaki 4000 de dunga."),
    ("Date format (th)", "Customer: Main 15th ko 5000 rupaye bhar dunga."),
    ("Hindi ko", "Customer: 10 ko meri salary aati hai, tab 2000 dunga."),
    
    # Edge Cases
    ("No amount no date (just PTP)", "Customer: Haan main bhenj dunga paisa."),
    ("Amount but no date", "Customer: Mere paas sirf 500 rupaye hain.")
]

print("=== Running Comprehensive PTP Logic Tests ===\n")
for name, transcript in tests:
    print(f"Test: {name}")
    print(f"T: {transcript}")
    
    payload = {"transcript": transcript, "current_date": "2026-02-20"}
    
    start = time.time()
    response = requests.post(url, headers=headers, json=payload)
    latency = time.time() - start
    
    try:
        data = response.json()
        ptp = data.get("ptp_details", {})
        amt = ptp.get("amount") if ptp else None
        dt = ptp.get("date") if ptp else None
        print(f"Result  -> Amount: {amt} | Date: {dt}")
    except Exception as e:
        print(f"Error   -> {e}")
    print("-" * 50)
