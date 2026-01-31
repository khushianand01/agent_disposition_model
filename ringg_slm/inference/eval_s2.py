import torch
from unsloth import FastLanguageModel
import json
import os
from tqdm import tqdm
from datetime import date
from sklearn.metrics import classification_report, accuracy_score

# =========================
# CONFIG
# =========================
MODEL_PATH = "/home/ubuntu/Disposition_model2-main/ringg_slm/outputs/ringg_slm_stage2"
VAL_DATA_PATH = "data/splits/val_v11_s2_balanced.json"
MAX_SEQ_LEN = 2048
DTYPE = None
LOAD_IN_4BIT = True
PROGRESS_FILE = "ringg_slm/results/ringg_eval_s2_progress.jsonl"
OUTPUT_REPORT = "ringg_slm/results/ringg_eval_results_s2.json"

VALID_REASONS = [
    "CUSTOMER_NOT_TELLING_REASON", "CLAIMING_PAYMENT_IS_COMPLETED", "OTHER_REASONS",
    "JOB_CHANGED_WAITING_FOR_SALARY", "MEDICAL_ISSUE", "DEATH_IN_FAMILY",
    "CLAIMING_FRAUD", "SERVICE_ISSUE", "LOST_JOB", "PENALTY_ISSUE",
    "LOAN_TAKEN_BY_KNOWN_PARTY", "LOAN_CLOSURE_MISCOMMUNICATION", "GRIEVANCE_FRAUD",
    "MULTIPLE_LOANS", "BUSINESS_LOSS", "GRIEVANCE_LOAN_AMOUNT_DISPUTE",
    "BUSINESS_CLOSED", "FINANCIAL_DIFFICULTY"
]

alpaca_prompt = """### Instruction:
{}

### Input:
{}

### Response:
{}"""

class RinggEvaluatorS2:
    def __init__(self, model_path=MODEL_PATH):
        print(f"Loading Ringg Stage 2 model from {model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LEN,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded.")

    @torch.inference_mode()
    def predict(self, instruction, input_dict):
        # Stage 2 input is a serialized JSON
        input_text = json.dumps(input_dict, ensure_ascii=False)
        prompt = alpaca_prompt.format(instruction, input_text, "")
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            if "### Response:" in full_output:
                json_part = full_output.split("### Response:")[1].strip()
            else:
                json_part = full_output.strip()

            for token in ["<|im_end|>", "###", "</s>"]:
                json_part = json_part.split(token)[0].strip()

            json_start = json_part.find('{')
            json_end = json_part.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                clean_json = json_part[json_start:json_end]
                return json.loads(clean_json)
            return None
        except:
            return None

def main():
    if not os.path.exists(VAL_DATA_PATH):
        print(f"Error: {VAL_DATA_PATH} not found.")
        return

    with open(VAL_DATA_PATH, "r") as f:
        eval_data = json.load(f)

    print(f"Evaluating Stage 2 on {len(eval_data)} examples...")

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
        evaluator = RinggEvaluatorS2()
        
        for i, item in enumerate(tqdm(eval_data)):
            if i in completed_indices: continue
                
            prediction = evaluator.predict(item["instruction"], item["input"])
            
            with open(PROGRESS_FILE, "a") as f:
                f.write(json.dumps({
                    "index": i, 
                    "prediction": prediction, 
                    "ground_truth": item["output"]
                }, ensure_ascii=False) + "\n")

    # Metrics calculation
    y_true_reason, y_pred_reason = [], []
    invalid_json_count = 0
    total_processed = 0

    with open(PROGRESS_FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            prediction = data["prediction"]
            gt = data["ground_truth"]
            
            total_processed += 1
            t_reason = str(gt.get("reason_for_not_paying", "null"))
            y_true_reason.append(t_reason)

            if not prediction:
                invalid_json_count += 1
                y_pred_reason.append("INVALID_JSON")
            else:
                p_reason = str(prediction.get("reason_for_not_paying", "null"))
                y_pred_reason.append(p_reason)

    print("\n" + "="*50)
    print("RINGG STAGE 2 EVALUATION REPORT")
    print("="*50)
    print(f"Total Samples: {total_processed}")
    print(f"Invalid JSON %: {(invalid_json_count/total_processed)*100:.2f}%")

    print("\n--- 📝 REASON FOR NOT PAYING METRICS ---")
    print(f"Accuracy: {accuracy_score(y_true_reason, y_pred_reason)*100:.2f}%")
    print(classification_report(y_true_reason, y_pred_reason, zero_division=0))

    results = {
        "timestamp": str(date.today()),
        "total_processed": total_processed,
        "invalid_json_pct": invalid_json_count/total_processed,
        "reason_accuracy": accuracy_score(y_true_reason, y_pred_reason),
        "reason_report": classification_report(y_true_reason, y_pred_reason, output_dict=True, zero_division=0)
    }
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nFinal results saved to {OUTPUT_REPORT}")

if __name__ == "__main__":
    main()
