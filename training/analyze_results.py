import json
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

LOG_FILE = "/home/ubuntu/disposition_model/outputs/qwen_3b_production_best/temp_results.jsonl"

def analyze():
    print(f"🔍 Analyzing detailed results from {LOG_FILE}...\n")
    
    y_true_disp = []
    y_pred_disp = []
    y_true_pay = []
    y_pred_pay = []
    
    json_errors = 0
    
    mismatches = []
    
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    gold = data.get("gold", {})
                    pred = data.get("prediction", {})
                    
                    if "error" in pred:
                        json_errors += 1
                        continue
                        
                    # Extract Labels
                    g_d = gold.get("disposition", "None")
                    p_d = pred.get("disposition", "None")
                    
                    g_p = gold.get("payment_disposition", "None")
                    p_p = pred.get("payment_disposition", "None")
                    
                    y_true_disp.append(g_d)
                    y_pred_disp.append(p_d)
                    
                    y_true_pay.append(g_p)
                    y_pred_pay.append(p_p)
                    
                    if g_d != p_d:
                        mismatches.append({
                            "transcript": data.get("transcript", "")[:100],
                            "gold": g_d,
                            "pred": p_d
                        })
                        
                except:
                    continue
                    
        # Disposition Stats
        print(f"📊 Total Samples Analyzed: {len(y_true_disp) + json_errors}")
        print(f"✅ JSON Validity Rate:      {(len(y_true_disp))/(len(y_true_disp) + json_errors)*100:.2f}% ({json_errors} errors)")
        
        # Calculate raw accuracy
        correct_disp = sum([1 for i in range(len(y_true_disp)) if y_true_disp[i] == y_pred_disp[i]])
        print(f"\n✅ Disposition Accuracy (Strict): {correct_disp/len(y_true_disp)*100:.2f}%")
        
        print("\n📊 Disposition Classification Report:")
        print(classification_report(y_true_disp, y_pred_disp, zero_division=0))
        
        # Confusion Matrix (Top 10 Errors)
        print("\n📉 Top 10 Confused Pairs (Gold -> Pred):")
        confusions = Counter()
        for i in range(len(y_true_disp)):
            if y_true_disp[i] != y_pred_disp[i]:
                confusions[(y_true_disp[i], y_pred_disp[i])] += 1
                
        for (g, p), count in confusions.most_common(10):
            print(f"   {count}x: {g} -> {p}")

        # Payment Stats
        correct_pay = sum([1 for i in range(len(y_true_pay)) if y_true_pay[i] == y_pred_pay[i]])
        print(f"\n💰 Payment Accuracy (Strict):     {correct_pay/len(y_true_pay)*100:.2f}%")

        print("\n📊 Payment Classification Report:")
        print(classification_report(y_true_pay, y_pred_pay, zero_division=0))
        
        print("\n📉 Top 5 Payment Errors:")
        pay_confusions = Counter()
        for i in range(len(y_true_pay)):
            if y_true_pay[i] != y_pred_pay[i]:
                pay_confusions[(y_true_pay[i], y_pred_pay[i])] += 1
                
        for (g, p), count in pay_confusions.most_common(5):
            print(f"   {count}x: {g} -> {p}")

    except FileNotFoundError:
        print("Log file not found.")

if __name__ == "__main__":
    analyze()
