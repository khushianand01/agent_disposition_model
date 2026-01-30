
import json
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

def calculate_metrics():
    y_true = []
    y_pred = []
    
    # Read the progress file
    with open('ringg_eval_v11_baseline_progress.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line)
            # The progress file structure usually mirrors the eval script's record_result
            # Expected keys: "input_data" (with expected output) and "prediction"
            
            # Correct parsing for progress file format
            # {"index": 0, "prediction": {...}, "ground_truth": "{\"disposition\": ...}"}
            
            try:
                # Parse ground truth (which is a stringified JSON)
                gt_str = item.get("ground_truth", "{}")
                if isinstance(gt_str, str):
                    expected = json.loads(gt_str)
                else:
                    expected = gt_str
                
                prediction = item.get("prediction", {})
                
                # Disposition
                gt_disp = expected.get("disposition", "None")
                if gt_disp: gt_disp = gt_disp.upper()
                else: gt_disp = "None"
                
                pr_disp = prediction.get("disposition", "None")
                if isinstance(pr_disp, str):
                    pr_disp = pr_disp.upper()
                else:
                    pr_disp = "None"
                
                y_true.append(gt_disp)
                y_pred.append(pr_disp)
                
            except Exception as e:
                print(f"Skipping error line: {e}")
                continue

    if not y_true:
        print("No data found.")
        return

    # Disposition Metrics
    print(f"\n--- Partial Metrics (Samples: {len(y_true)}) ---")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Confusion Matrix
    unique_labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    print("\nConfusion Matrix (Top Mistakes):")
    
    # Print only non-diagonal > 0 (mistakes)
    df_cm = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    for i in range(len(unique_labels)):
        for j in range(len(unique_labels)):
            if i != j and df_cm.iloc[i, j] > 0:
                print(f"True: {unique_labels[i]} -> Pred: {unique_labels[j]} (Count: {df_cm.iloc[i, j]})")

if __name__ == "__main__":
    calculate_metrics()
