import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, classification_report

LOG_FILE = "outputs/qwen_3b_production_best/temp_results.jsonl"

def analyze_confidence():
    correct_confidences = []
    error_confidences = []
    
    y_true = []
    y_pred = []
    confidences = []

    print(f"Reading {LOG_FILE}...")
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                gold = data.get("gold", {}).get("payment_disposition", "None")
                pred = data.get("prediction", {})
                pred_label = pred.get("payment_disposition", "None")
                confidence = pred.get("confidence_score", 0.0)
                
                # Filter just PTP for this analysis as it's critical
                if pred_label == "PTP":
                    y_true.append(gold == "PTP") # True if actually PTP, False if hallucinated
                    confidences.append(confidence)
                    
                    if gold == "PTP":
                        correct_confidences.append(confidence)
                    else:
                        error_confidences.append(confidence)
            except:
                continue
                
    print(f"\n--- Confidence Analysis for PTP (Promise to Pay) ---")
    print(f"Total PTP Predictions: {len(confidences)}")
    print(f"True Positives: {len(correct_confidences)}")
    print(f"False Positives: {len(error_confidences)}")
    
    if correct_confidences:
        print(f"Avg Confidence (Correct): {sum(correct_confidences)/len(correct_confidences):.4f}")
    if error_confidences:
        print(f"Avg Confidence (Errors):  {sum(error_confidences)/len(error_confidences):.4f}")
        
    # Simulate Thresholds
    print("\n--- Impact of Confidence Thresholding ---")
    for threshold in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95]:
        filtered_true = [t for t, c in zip(y_true, confidences) if c >= threshold]
        
        if not filtered_true:
            print(f"Threshold {threshold}: No samples left")
            continue
            
        precision = sum(filtered_true) / len(filtered_true)
        recall_impact = len(filtered_true) / len(y_true) # Percentage of volume kept
        
        print(f"Threshold > {threshold}: Precision = {precision*100:.1f}% | Retained Volume = {recall_impact*100:.1f}%")

if __name__ == "__main__":
    analyze_confidence()
