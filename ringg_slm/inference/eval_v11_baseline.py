import torch
from unsloth import FastLanguageModel
import json
import os
import random
from tqdm import tqdm
from datetime import date
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
from collections import defaultdict

# =========================
# CONFIG
# =========================
MODEL_NAME = "RinggAI/Transcript-Analytics-SLM1.5b"
TEST_DATA_PATH = "data/baseline_test.json"
MAX_SEQ_LEN = 2048
DTYPE = None
LOAD_IN_4BIT = True
PROGRESS_FILE = "ringg_eval_v11_baseline_progress.jsonl"
OUTPUT_REPORT = "ringg_eval_results_v11_baseline.json"
CURRENT_DATE = "2026-01-30"

class RinggEvaluator:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading Ringg model for evaluation: {model_name}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LEN,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded.")

    VALID_DISPOSITIONS = [
        "ANSWERED", "CALL_BACK_LATER", "ANSWERED_BY_FAMILY_MEMBER", 
        "CALL_DISCONNECTED_BY_CUSTOMER", "ANSWERED_DISCONNECTED", 
        "ANSWERED_VOICE_ISSUE", "SILENCE_ISSUE", "LANGUAGE_BARRIER", 
        "AUTOMATED_VOICE", "WRONG_NUMBER", "CUSTOMER_ABUSIVE", 
        "AGENT_BUSY_ON_ANOTHER_CALL", "FORWARDED_CALL", "RINGING", 
        "RINGING_DISCONNECTED", "WILL_ASK_TO_PAY", "DO_NOT_KNOW_THE_PERSON", 
        "OTHERS", "BUSY", "SWITCHED_OFF", "NOT_IN_CONTACT_ANYMORE", "CUSTOMER_PICKED",
        "NO_INCOMING_CALL_RECORDED"
    ]

    VALID_PAYMENT_DISPOSITIONS = [
        "PTP", "PAID", "SETTLEMENT", "PARTIAL_PAYMENT", 
        "NO_PAYMENT_COMMITMENT", "DENIED_TO_PAY", "NO_PROOF_GIVEN", 
        "WILL_PAY_AFTER_VISIT", "WANT_FORECLOSURE", "WANTS_TO_RENEGOTIATE_LOAN_TERMS", "DISPUTE", "None"
    ]

    def format_prompt(self, transcript):
        instruction = f"""Analyze the following call transcript and extract structured data.
Valid Dispositions: {', '.join(self.VALID_DISPOSITIONS)}
Valid Payment Dispositions: {', '.join(self.VALID_PAYMENT_DISPOSITIONS)}
Current Date: {CURRENT_DATE}
Return ONLY valid JSON with 'disposition' and 'payment_disposition' fields."""
        
        return f"""You are a call transcript analyst. {instruction}

### Transcript:
{transcript}

### Response (JSON):
"""

    @torch.inference_mode()
    def predict(self, transcript):
        prompt = self.format_prompt(transcript)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Handle the way Ringg model might output (concatenated or not)
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            # Look for JSON block
            if "### Response (JSON):" in full_output:
                json_part = full_output.split("### Response (JSON):")[1].strip()
            else:
                json_part = full_output.strip()

            json_start = json_part.find('{')
            json_end = json_part.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                clean_json = json_part[json_start:json_end]
                return json.loads(clean_json)
            return None
        except:
            return None

def main():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: {TEST_DATA_PATH} not found.")
        return

    with open(TEST_DATA_PATH, "r") as f:
        eval_data = json.load(f)

    print(f"Evaluating on {len(eval_data)} examples (Fixed Baseline Set)...")

    # Load existing progress
    completed_indices = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed_indices.add(data["index"])
                except: continue
        print(f"Found existing progress. Skipping {len(completed_indices)} items.")

    if len(completed_indices) < len(eval_data):
        evaluator = RinggEvaluator()
        
        for i, item in enumerate(tqdm(eval_data)):
            if i in completed_indices: continue
                
            transcript = item["input"]
            ground_truth = item["output"]
            
            prediction = evaluator.predict(transcript)
            
            with open(PROGRESS_FILE, "a") as f:
                f.write(json.dumps({
                    "index": i, 
                    "prediction": prediction, 
                    "ground_truth": ground_truth
                }, ensure_ascii=False) + "\n")

    # Reporting
    y_true_disp, y_pred_disp = [], []
    y_true_pay, y_pred_pay = [], []
    invalid_json_count = 0
    invalid_label_count = 0
    total_processed = 0

    valid_disps = set(RinggEvaluator.VALID_DISPOSITIONS)
    valid_pays = set(RinggEvaluator.VALID_PAYMENT_DISPOSITIONS)

    with open(PROGRESS_FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            prediction = data["prediction"]
            gt = json.loads(data["ground_truth"])
            total_processed += 1

            t_disp = str(gt.get("disposition", "null"))
            t_pay = str(gt.get("payment_disposition", "null"))
            
            y_true_disp.append(t_disp)
            y_true_pay.append(t_pay)

            if not prediction:
                invalid_json_count += 1
                y_pred_disp.append("INVALID_JSON")
                y_pred_pay.append("INVALID_JSON")
            else:
                p_disp = str(prediction.get("disposition", "null"))
                p_pay = str(prediction.get("payment_disposition", "null"))
                
                if p_disp not in valid_disps or (p_pay not in valid_pays and p_pay != "null"):
                    invalid_label_count += 1
                
                y_pred_disp.append(p_disp)
                y_pred_pay.append(p_pay)

    print("\n" + "="*50)
    print("RINGG BASELINE EVALUATION REPORT")
    print("="*50)
    print(f"Total Samples: {total_processed}")
    print(f"Invalid JSON %: {(invalid_json_count/total_processed)*100:.2f}%")
    print(f"Invalid Labels %: {(invalid_label_count/total_processed)*100:.2f}%")

    print("\n--- 📞 DISPOSITION METRICS ---")
    print(f"Accuracy: {accuracy_score(y_true_disp, y_pred_disp)*100:.2f}%")
    # Macro F1 and Recall are in the report
    print(classification_report(y_true_disp, y_pred_disp, zero_division=0))

    # Payment Metrics
    y_true_pay_filt, y_pred_pay_filt = [], []
    for t, p in zip(y_true_pay, y_pred_pay):
        if t != "None":
            y_true_pay_filt.append(t)
            y_pred_pay_filt.append(p)

    print("\n--- 💰 PAYMENT METRICS (WHERE APPLICABLE) ---")
    if y_true_pay_filt:
        print(f"Accuracy: {accuracy_score(y_true_pay_filt, y_pred_pay_filt)*100:.2f}%")
        print(classification_report(y_true_pay_filt, y_pred_pay_filt, zero_division=0))
        
        labels = sorted(list(set(y_true_pay_filt) | set(y_pred_pay_filt)))
        cm = confusion_matrix(y_true_pay_filt, y_pred_pay_filt, labels=labels)
        print("\nConfusion Matrix (Payment):")
        print(pd.DataFrame(cm, index=labels, columns=labels))
    else:
        print("No payment cases found.")

    results = {
        "timestamp": str(date.today()),
        "invalid_json_pct": invalid_json_count/total_processed,
        "invalid_label_pct": invalid_label_count/total_processed,
        "disposition_report": classification_report(y_true_disp, y_pred_disp, output_dict=True, zero_division=0),
        "payment_report": classification_report(y_true_pay_filt, y_pred_pay_filt, output_dict=True, zero_division=0) if y_true_pay_filt else None
    }
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nFinal results saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
