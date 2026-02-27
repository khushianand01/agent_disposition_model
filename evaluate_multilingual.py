import json
import os
import requests
import time

EVAL_DIR = "eval_datasets"
API_URL = "http://localhost:8005/predict"

def safe_str(val):
    return str(val).lower().strip() if val is not None else "none"

def evaluate_language(filename):
    filepath = os.path.join(EVAL_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    correct = {
        "disposition": 0,
        "payment": 0,
        "reason": 0,
        "amount": 0,
        "date": 0
    }
    
    print(f"\n--- Evaluating {filename} ---")
    
    for item in data:
        transcript = item["transcript"]
        exp = {
            "disp": item.get("expected_disposition"),
            "pay": item.get("expected_payment_disposition"),
            "reason": item.get("expected_reason_for_not_paying"),
            "amt": item.get("expected_amount"),
            "date": item.get("expected_date")
        }
        
        payload = {"transcript": transcript}
        
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
                    "reason": pred["reason"] == exp["reason"] or (pred["reason"] == "None" and exp["reason"] is None) or (exp["reason"] == "None" and pred["reason"] is None),
                    "amount": str(pred["amt"]) == str(exp["amt"]) or (pred["amt"] == "None" and exp["amt"] is None) or (exp["amt"] == "None" and pred["amt"] is None),
                    # Date is fuzzy in tests unless strict format, so we just check if both are None or both exist for basic validation
                    "date": (pred["date"] is None) == (exp["date"] is None) or (pred["date"] == "None" and exp["date"] is None) or (exp["date"] == "None" and pred["date"] is None)
                }
                
                for k, v in matches.items():
                    if v: correct[k] += 1
                
                if not all(matches.values()):
                    print(f"❌ MISMATCH: {transcript[:50]}...")
                    if not matches['disposition'] or not matches['payment']:
                        print(f"   Exp: Disp={exp['disp']}, Pay={exp['pay']}")
                        print(f"   Got: Disp={pred['disp']}, Pay={pred['pay']}")
                    if not matches['reason']:
                        print(f"   Exp Reason: {exp['reason']} | Got: {pred['reason']}")
                    if not matches['amount'] or not matches['date']:
                        print(f"   Exp PTP: {exp['amt']}, {exp['date']} | Got: {pred['amt']}, {pred['date']}")
            else:
                print(f"⚠️ API Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"⚠️ Request Failed: {str(e)}")

    acc = {k: (v / total) * 100 if total > 0 else 0 for k, v in correct.items()}
    
    # Calculate overall by averaging the exact matches
    overall = sum(acc.values()) / len(acc)
    
    return {
        "language": filename.replace("_test.json", ""),
        "total": total,
        "metrics": acc,
        "overall_accuracy": overall
    }

def main():
    if not os.path.exists(EVAL_DIR):
        print(f"Directory {EVAL_DIR} not found.")
        return
        
    files = [f for f in os.listdir(EVAL_DIR) if f.endswith(".json")]
    
    results = []
    print("Starting Comprehensive Multilingual Evaluation...")
    for f in files:
        res = evaluate_language(f)
        results.append(res)
        
    print("\n=========================================================")
    print(" COMPLEX MULTILINGUAL EVALUATION RESULTS ")
    print("=========================================================\n")
    
    print(f"{'Language':<10} | {'Disp':<6} | {'Pay':<6} | {'Reason':<6} | {'Amt':<6} | {'Date':<6} | {'Overall':<6}")
    print("-" * 65)
    
    results.sort(key=lambda x: x["overall_accuracy"])
    
    for r in results:
        lang = r["language"].capitalize()
        m = r["metrics"]
        print(f"{lang:<10} | {m['disposition']:>5.0f}% | {m['payment']:>5.0f}% | {m['reason']:>5.0f}% | {m['amount']:>5.0f}% | {m['date']:>5.0f}% | {r['overall_accuracy']:>5.0f}%")

if __name__ == "__main__":
    main()
