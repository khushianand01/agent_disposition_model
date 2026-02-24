import json
import sys

LOG_FILE = "/home/ubuntu/disposition_model/outputs/qwen_3b_production_best/temp_results.jsonl"

def calculate_metrics():
    try:
        total = 0
        correct_disp = 0
        correct_pay = 0
        
        with open(LOG_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    gold = data.get("gold", {})
                    pred = data.get("prediction", {})
                    
                    if "error" in pred:
                        continue
                        
                    total += 1
                    
                    # Disposition Check
                    if gold.get("disposition") == pred.get("disposition"):
                        correct_disp += 1
                        
                    # Payment Disposition Check
                    if gold.get("payment_disposition") == pred.get("payment_disposition"):
                        correct_pay += 1
                except:
                    continue
        
        if total == 0:
            print("No valid samples yet.")
            return

        print(f"--- Partial Results ({total} samples) ---")
        print(f"Disposition Accuracy: {correct_disp/total*100:.2f}%")
        print(f"Payment Accuracy:     {correct_pay/total*100:.2f}%")
        print("---------------------------------------")

    except FileNotFoundError:
        print("Log file not found.")

if __name__ == "__main__":
    calculate_metrics()
