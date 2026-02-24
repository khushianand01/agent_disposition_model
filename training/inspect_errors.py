import json

LOG_FILE = "/home/ubuntu/disposition_model/outputs/qwen_3b_production_best/temp_results.jsonl"

def inspect_errors():
    print(f"Inspecting errors in {LOG_FILE}...\n")
    try:
        count = 0
        with open(LOG_FILE, "r") as f:
            for line in f:
                data = json.loads(line)
                pred = data.get("prediction", {})
                
                if "error" in pred:
                    print(f"❌ Error #{count+1}")
                    print(f"   Error Msg: {pred.get('error')}")
                    print(f"   Raw Output: {pred.get('raw_output')[:200]}...") # Print first 200 chars
                    print("-" * 30)
                    count += 1
                    if count >= 10: break
                    
        print(f"\nTotal Errors Found: {count} (showing first 10)")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_errors()
