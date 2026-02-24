import json
import os
from collections import Counter, defaultdict

RESULTS_FILE = "inference/temp_results.jsonl"

def str_to_json(s):
    if isinstance(s, dict):
        return s
    if s is None:
        return {}
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1:
            return json.loads(s[start:end+1])
        return json.loads(s)
    except:
        return {}

def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def calculate_metrics():
    if not os.path.exists(RESULTS_FILE):
        print(f"File {RESULTS_FILE} not found.")
        return

    results = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass
    
    total = len(results)
    if total == 0:
        print("No results found.")
        return

    print(f"\n[INFO] Loaded {total} samples from {RESULTS_FILE}")
    
    # Global Counters
    em_count = 0
    emnr_count = 0
    valid_json_count = 0
    
    fields = ["disposition", "payment_disposition", "reason_for_not_paying", 
              "ptp_amount", "ptp_date", "followup_date", "remarks"]
    
    field_correct = Counter()
    field_total = Counter()
    
    # Classification Metrics Structures
    # { field_name: { label: {tp: 0, fp: 0, fn: 0} } }
    class_stats = {
        "disposition": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        "payment_disposition": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    }
    
    hallucinations = Counter() # Count when pred is not null but gold is null

    for row in results:
        gold_json = str_to_json(row.get("gold", {}))
        pred_dict = row.get("prediction", {})
        
        # Check if prediction is a valid extraction (not an error dict)
        is_error = "error" in pred_dict
        if not is_error:
            valid_json_count += 1

        # Normalize Gold
        gold_norm = {}
        for f in fields:
            val = gold_json.get(f)
            if val is None or str(val).lower() in ["null", "none", "nan", ""]:
                gold_norm[f] = "null"
            else:
                gold_norm[f] = str(val).strip().upper() if f in ["disposition", "payment_disposition"] else str(val).strip().lower()

        # Normalize Pred
        pred_norm = {}
        for f in fields:
            if is_error:
                pred_norm[f] = "error"
            else:
                val = pred_dict.get(f)
                if val is None or str(val).lower() in ["null", "none", "nan", ""]:
                    pred_norm[f] = "null"
                else:
                    pred_norm[f] = str(val).strip().upper() if f in ["disposition", "payment_disposition"] else str(val).strip().lower()

        # Update Classification Stats for Disposition & Payment Disposition
        for cf in ["disposition", "payment_disposition"]:
            g_lab = gold_norm[cf]
            p_lab = pred_norm[cf]
            
            if g_lab == p_lab:
                if g_lab != "null": # We only track actual labels for F1, or include null? Usually include all.
                    class_stats[cf][g_lab]["tp"] += 1
            else:
                if g_lab != "null":
                    class_stats[cf][g_lab]["fn"] += 1
                if p_lab != "null" and p_lab != "error":
                    class_stats[cf][p_lab]["fp"] += 1

        # Field Accuracy & Hallucination
        for f in fields:
            field_total[f] += 1
            if gold_norm[f] == pred_norm[f]:
                field_correct[f] += 1
            
            # Hallucination check: Gold is null, but Pred extracted something
            if gold_norm[f] == "null" and pred_norm[f] not in ["null", "error"]:
                hallucinations[f] += 1

        # Exact Match logic
        if not is_error:
            if all(gold_norm[f] == pred_norm[f] for f in fields):
                em_count += 1
            if all(gold_norm[f] == pred_norm[f] for f in fields if f != "remarks"):
                emnr_count += 1

    # --- REPORTING ---
    print("\n" + "█"*60)
    print("  PERFECT EVALUATION REPORT (DIAGNOSTIC SUITE)")
    print("█"*60)
    
    print(f"\n[GLOBAL METRICS]")
    print(f"{'Total Samples':<25}: {total}")
    print(f"{'Valid JSON / Format':<25}: {valid_json_count/total:>6.1%} ({valid_json_count}/{total})")
    print(f"{'Exact Match (All)':<25}: {em_count/total:>6.1%} ({em_count}/{total})")
    print(f"{'Exact Match (No Remarks)':<25}: {emnr_count/total:>6.1%} ({emnr_count}/{total})")

    # Field Accuracy Table
    print(f"\n[FIELD-LEVEL EXTRACTION ACCURACY]")
    print(f"{'-'*60}")
    print(f"{'Field':<25} | {'Accuracy':<10} | {'Hallucination Rate'}")
    print(f"{'-'*60}")
    for f in fields:
        acc = field_correct[f] / field_total[f]
        h_rate = hallucinations[f] / field_total[f]
        print(f"{f:<25} | {acc:>9.1%} | {h_rate:>6.1%}")

    # Per-Label Analysis for Disposition
    for cf in ["disposition", "payment_disposition"]:
        print(f"\n[PER-LABEL PERFORMANCE: {cf.upper()}]")
        print(f"{'-'*85}")
        print(f"{'Label':<40} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'Support'}")
        print(f"{'-'*85}")
        
        labels = sorted(class_stats[cf].keys())
        macro_f1 = 0
        weighted_f1 = 0
        total_support = 0
        
        for lab in labels:
            stats = class_stats[cf][lab]
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            p, r, f1 = calculate_f1(tp, fp, fn)
            support = tp + fn
            
            print(f"{lab[:40]:<40} | {p:>6.1%} | {r:>6.1%} | {f1:>6.1%} | {support}")
            
            macro_f1 += f1
            weighted_f1 += (f1 * support)
            total_support += support
            
        if labels:
            macro_f1 /= len(labels)
            weighted_f1 = weighted_f1 / total_support if total_support > 0 else 0
            print(f"{'-'*85}")
            print(f"{'MACRO AVERAGE':<40} | {'-':<7} | {'-':<7} | {macro_f1:>6.1%} | {total_support}")
            print(f"{'WEIGHTED AVERAGE':<40} | {'-':<7} | {'-':<7} | {weighted_f1:>6.1%} | {total_support}")

    print("\n" + "█"*60)
    print("  END OF REPORT")
    print("█"*60 + "\n")

if __name__ == "__main__":
    calculate_metrics()
