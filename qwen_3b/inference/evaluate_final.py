import json
import torch
import gc
import sys
import os
import hashlib
from collections import Counter
from tqdm import tqdm

# Add project root to path (Insert at 0 to prioritize over script dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from inference.inference import get_model
except ImportError as e:
    print(f"Error: Could not import inference module. {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

TEST_DATA_PATH = "data/splits/test_final.json"
OUTPUT_REPORT = "inference/final_evaluation_report.json"

def str_to_json(s):
    if isinstance(s, dict):
        return s
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1:
            return json.loads(s[start:end+1])
        return json.loads(s)
    except:
        return None

def evaluate_final():
    print(f"Loading test data from {TEST_DATA_PATH}...")
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use a subset for faster feedback if needed, but comprehensive eval uses all
    # data = data # Optional: Uncomment for speed
    
    print("Loading Model...")
    model = get_model()
    
    # Load transcript date map for grounding
    date_map_path = "data/processed/transcript_date_map.json"
    date_map = {}
    if os.path.exists(date_map_path):
        print(f"Loading date map from {date_map_path}...")
        with open(date_map_path, "r") as f:
            date_map = json.load(f)
    
    def get_hash(text):
        return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

    results = []
    
    print(f"Running Inference on {len(data)} samples...")
    
    # Check for existing partial results
    temp_results_path = "inference/temp_results.jsonl"
    processed_indices = set()
    if os.path.exists(temp_results_path):
        print(f"Resuming from {temp_results_path}...")
        with open(temp_results_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                try:
                    res = json.loads(line)
                    # Use the stored index if available, otherwise fallback to index from file count
                    idx = res.get("index")
                    if idx is not None:
                        processed_indices.add(idx)
                    else:
                        # Fallback for old results: find the index in 'data' by matching transcript
                        # This is slow but only happens once during resume
                        transcript = res.get("transcript")
                        for d_idx, d_row in enumerate(data):
                            if d_row.get("input") == transcript and d_idx not in processed_indices:
                                processed_indices.add(d_idx)
                                res["index"] = d_idx
                                break
                    results.append(res)
                except:
                    pass
        print(f"Found {len(processed_indices)} already processed samples.")

    print(f"Starting inference loop on {len(data)} total samples...")
    import time

    with open(temp_results_path, "w", encoding="utf-8") as f_out:
        # Rewrite existing results to include indices if they were missing, 
        # and to ensure the file is in a clean state with readable Hindi.
        for res in results:
            f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
        f_out.flush()

        for i, row in enumerate(data):
            if i in processed_indices:
                continue

            transcript = row.get("input", "")
            gold = row.get("output", "")

            try:
                start_time = time.time()
                # Predict with grounded date for consistent extraction
                # We use the original call date found in mapping, or fallback to strict today
                transcript_hash = get_hash(transcript)
                reference_date = date_map.get(transcript_hash, None) # Let inference.py handle default to today
                
                prediction = model.predict(transcript, current_date=reference_date)
                end_time = time.time()
                elapsed = end_time - start_time
                
                # Parse gold if it's a string to keep the JSONL clean and readable
                gold_obj = str_to_json(gold)
                
                result_item = {
                    "index": i,
                    "transcript": transcript,
                    "gold": gold_obj,
                    "prediction": prediction
                }
                
                # Save to JSONL immediately with flush
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()
                
                results.append(result_item)
                processed_indices.add(i)
                
                print(f"[{i+1}/{len(data)}] Processed sample in {elapsed:.2f}s. Confidence: {prediction.get('confidence_score', 'N/A')}", flush=True)

                # Proactive OOM Management: Force clear cache and collect garbage
                if (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                prediction = {"error": str(e)}
                result_item = {
                    "transcript": transcript,
                    "gold": gold,
                    "prediction": prediction
                }
                f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                f_out.flush()
                results.append(result_item)
    
    # Reload results from JSONL to ensure we have everything (including resumed ones)
    results = []
    with open(temp_results_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass


    # Metrics Calculation
    total = len(results)
    valid_json_count = 0
    exact_match_count = 0
    exact_match_no_remarks_count = 0
    
    fields = ["disposition", "payment_disposition", "reason_for_not_paying", 
              "ptp_amount", "ptp_date", "followup_date", "remarks"]
    
    field_correct = Counter()
    field_total = Counter()
    
    # Calibration Buckets
    confidence_buckets = {
        "Low (<50%)": {"correct": 0, "total": 0},
        "Medium (50-80%)": {"correct": 0, "total": 0},
        "High (80-95%)": {"correct": 0, "total": 0},
        "Very High (>95%)": {"correct": 0, "total": 0}
    }

    for row in results:
        gold_str = row["gold"]
        pred_dict = row["prediction"] # This is already a dict from inference.py
        
        gold_json = str_to_json(gold_str)
        
        # Handle the case where prediction failed to parse (dict with error) or is valid
        is_valid = "error" not in pred_dict
        
        if is_valid:
            valid_json_count += 1
            conf = pred_dict.get("confidence_score", 0.0)
            
            # Normalize gold for comparison
            if gold_json:
                gold_compare = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in gold_json.items() if k in fields}
            else:
                gold_compare = {k: "null" for k in fields}
            
            # Normalize pred for comparison
            pred_compare = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in pred_dict.items() if k in fields}
            
            # Exact Match
            is_exact_match = True
            for field in fields:
                g_val = gold_compare.get(field)
                p_val = pred_compare.get(field)
                if g_val is None: g_val = "null"
                if p_val is None: p_val = "null"
                if str(g_val) != str(p_val):
                    is_exact_match = False
                    break
            
            if is_exact_match:
                exact_match_count += 1

            # Exact Match (No Remarks)
            is_exact_match_no_remarks = True
            for field in [f for f in fields if f != "remarks"]:
                g_val = gold_compare.get(field)
                p_val = pred_compare.get(field)
                if g_val is None: g_val = "null"
                if p_val is None: p_val = "null"
                if str(g_val) != str(p_val):
                    is_exact_match_no_remarks = False
                    break
            
            if is_exact_match_no_remarks:
                exact_match_no_remarks_count += 1
                
            # Calibration Check
            if conf < 0.5: bucket = "Low (<50%)"
            elif conf < 0.8: bucket = "Medium (50-80%)"
            elif conf < 0.95: bucket = "High (80-95%)"
            else: bucket = "Very High (>95%)"
            
            confidence_buckets[bucket]["total"] += 1
            if is_exact_match:
                confidence_buckets[bucket]["correct"] += 1
        else:
            pass # Invalid count as failure already handled in reporting loop

    # Printing Report
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION REPORT")
    print("="*50)
    print(f"Total Test Samples: {total}")
    print(f"Valid JSON Output:  {valid_json_count} ({valid_json_count/total:.1%})")
    print(f"Exact Match (All Fields): {exact_match_count} ({exact_match_count/total:.1%})")
    print(f"Exact Match (No Remarks): {exact_match_no_remarks_count} ({exact_match_no_remarks_count/total:.1%})")
    print("-" * 50)
    
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    import pandas as pd

    print("FIELD-LEVEL METRICS (Macro Average):")
    print(f"{'Field':<25} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10}")
    
    for field in fields:
        y_true = []
        y_pred = []
        
        for row in results:
            gold_str = row["gold"]
            gold_json = str_to_json(gold_str)
            pred_dict = row["prediction"]
            
            # Gold Value
            g_val = gold_json.get(field) if gold_json else None
            if g_val is None: g_val = "null"
            if isinstance(g_val, str): g_val = g_val.strip().lower()
            
            # Predicted Value
            p_val = pred_dict.get(field) if "error" not in pred_dict else "error"
            if p_val is None: p_val = "null"
            if isinstance(p_val, str): p_val = p_val.strip().lower()
            
            y_true.append(str(g_val))
            y_pred.append(str(p_val))
            
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        print(f"{field:<25} {precision:.1%}      {recall:.1%}      {f1:.1%}      {acc:.1%}")

    print("-" * 50)
    print("CONFIDENCE CALIBRATION (Accuracy by Confidence Level):")
    for bucket, stats in confidence_buckets.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"{bucket:<20}: {acc:.1%} Accuracy (based on {stats['total']} samples)")
        else:
            print(f"{bucket:<20}: N/A (No samples)")
    print("="*50)

if __name__ == "__main__":
    if not os.path.exists("outputs/qwen3_8b_lora_production"):
        # For testing purposes, we might want to run even without model if we mock?
        # But usually valid.
        print("WARNING: Model not found at outputs/qwen3_8b_lora_production.")
        print("Please wait for training to finish before running this script.")
    else:
        evaluate_final()
