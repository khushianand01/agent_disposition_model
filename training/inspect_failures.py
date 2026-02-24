import json

LOG_FILE = "/home/ubuntu/disposition_model/outputs/qwen_3b_production_best/temp_results.jsonl"

def inspect_failures():
    print(f"Inspecting failures in {LOG_FILE}...\n")
    try:
        count = 0
        with open(LOG_FILE, "r") as f:
            for line in f:
                if count >= 10: break # Only show first 10 failures
                
                data = json.loads(line)
                gold = data.get("gold", {})
                pred = data.get("prediction", {})
                
                # Check Disposition mismatch
                gold_disp = gold.get("disposition")
                pred_disp = pred.get("disposition")
                
                if gold_disp != pred_disp:
                    print(f"❌ Mismatch #{count+1}")
                    print(f"   Transcript: {data.get('transcript')[:100]}...")
                    print(f"   Gold: {gold_disp}")
                    print(f"   Pred: {pred_disp}")
                    print("-" * 30)
                    count += 1
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_failures()
