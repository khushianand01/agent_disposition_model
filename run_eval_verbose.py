import json
import os
import requests
import time

EVAL_DIR = "eval_datasets"
API_URL = "http://localhost:8005/predict"
RESULTS_FILE = "gold_vs_predicted_results.json"

def evaluate_language(filename, all_results):
    filepath = os.path.join(EVAL_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    correct = {"disposition": 0, "payment": 0, "reason": 0, "amount": 0, "date": 0}
    
    print(f"\n--- Evaluating {filename} ---")
    lang_name = filename.replace("_test.json", "")
    
    for item in data:
        transcript = item["transcript"]
        exp = {
            "disp": item.get("expected_disposition"),
            "pay": item.get("expected_payment_disposition"),
            "reason": item.get("expected_reason_for_not_paying"),
            "amt": item.get("expected_amount"),
            "date": item.get("expected_date")
        }
        
        payload = {"transcript": transcript, "current_date": "2026-03-05"}
        
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                pred_ptp = result.get("ptp_details", {}) or {}
                
                pred = {
                    "disp": result.get("disposition"),
                    "pay": result.get("payment_disposition"),
                    "reason": result.get("reason_for_not_paying"),
                    "amt": pred_ptp.get("amount"),
                    "date": pred_ptp.get("date")
                }
                
                matches = {
                    "disposition": pred["disp"] == exp["disp"],
                    "payment": pred["pay"] == exp["pay"] or (pred["pay"] in ["PTP", "PARTIAL_PAYMENT"] and exp["pay"] in ["PTP", "PARTIAL_PAYMENT"]),
                    "reason": pred["reason"] == exp["reason"] or (pred["reason"] in ["None", None] and exp["reason"] in ["None", None]),
                    "amount": str(pred["amt"]) == str(exp["amt"]) or (pred["amt"] in ["None", None] and exp["amt"] in ["None", None]),
                    "date": (pred["date"] is None) == (exp["date"] is None) or (pred["date"] in ["None", None] and exp["date"] in ["None", None])
                }
                
                for k, v in matches.items():
                    if v: correct[k] += 1
                
                all_results.append({
                    "language": lang_name,
                    "transcript": transcript,
                    "gold": exp,
                    "predicted": pred,
                    "is_exact_match": all(matches.values()),
                    "remarks": result.get("remarks")
                })
                
                if not all(matches.values()):
                    pass # We record everything in the json anyway
            else:
                pass
        except Exception as e:
            pass

    acc = {k: (v / total) * 100 if total > 0 else 0 for k, v in correct.items()}
    overall = sum(acc.values()) / len(acc)
    
    return {
        "language": lang_name,
        "total": total,
        "metrics": acc,
        "overall_accuracy": overall
    }

def main():
    if not os.path.exists(EVAL_DIR):
        print(f"Directory {EVAL_DIR} not found.")
        return
        
    files = sorted([f for f in os.listdir(EVAL_DIR) if f.endswith(".json")])
    
    summary_results = []
    all_results = []
    print("Starting Comprehensive Multilingual Evaluation with Gold vs Predicted output...")
    for f in files:
        res = evaluate_language(f, all_results)
        summary_results.append(res)
        
    print("\n=========================================================")
    print(" COMPLEX MULTILINGUAL EVALUATION RESULTS ")
    print("=========================================================\n")
    
    print(f"{'Language':<10} | {'Disp':<6} | {'Pay':<6} | {'Reason':<6} | {'Amt':<6} | {'Date':<6} | {'Overall':<6}")
    print("-" * 65)
    
    summary_results.sort(key=lambda x: x["overall_accuracy"])
    
    for r in summary_results:
        lang = r["language"].capitalize()
        m = r["metrics"]
        print(f"{lang:<10} | {m['disposition']:>5.0f}% | {m['payment']:>5.0f}% | {m['reason']:>5.0f}% | {m['amount']:>5.0f}% | {m['date']:>5.0f}% | {r['overall_accuracy']:>5.0f}%")

    # Write out the verbose results
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"\nFinal verbose results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
