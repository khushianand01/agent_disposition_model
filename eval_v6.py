
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
CHECKPOINT_PATH = "outputs/qwen3_8b_v6_balanced/checkpoint-2566"
TEST_DATA_PATH = "data/splits/test_v6.json"
MAX_SEQ_LEN = 2048
DTYPE = None
LOAD_IN_4BIT = True
SAMPLE_SIZE = None # None means process full dataset

class MetaEvaluator:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        print(f"Loading model for evaluation from {checkpoint_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=MAX_SEQ_LEN,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.model)
        print("Model loaded.")

    def format_prompt(self, transcript, current_date="2026-01-27"):
        instruction = f"You are an AI assistant that extracts structured call disposition data.\nGiven a call transcript between an agent and a borrower, extract the following fields. Return ONLY valid JSON. Do not explain.:\ndisposition, payment_disposition, reason_for_not_paying, ptp_amount, ptp_date, followup_date, remarks.\nCurrent Date: {current_date}\nIf a field is not present, return null."
        
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{transcript}

### Response:
"""

    @torch.inference_mode()
    def predict(self, transcript):
        prompt = self.format_prompt(transcript)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        try:
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                return json.loads(generated_text[json_start:json_end])
            return None
        except:
            return None

def main():
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Test data not found at {TEST_DATA_PATH}")
        return

    with open(TEST_DATA_PATH, "r") as f:
        test_data = json.load(f)

    print(f"Total test examples: {len(test_data)}")
    eval_data = test_data[:SAMPLE_SIZE] if SAMPLE_SIZE else test_data
    print(f"Evaluating on {len(eval_data)} examples...")

    evaluator = MetaEvaluator()
    
    y_true_disposition = []
    y_pred_disposition = []
    y_true_pay_disp = []
    y_pred_pay_disp = []
    
    valid_json_count = 0

    for item in tqdm(eval_data):
        transcript = item["input"]
        ground_truth = item["output"]
        
        prediction = evaluator.predict(transcript)
        
        if prediction:
            valid_json_count += 1
            
            # Extract fields safely
            y_true_disposition.append(str(ground_truth.get("disposition", "null")))
            y_pred_disposition.append(str(prediction.get("disposition", "null")))
            
            y_true_pay_disp.append(str(ground_truth.get("payment_disposition", "null")))
            y_pred_pay_disp.append(str(prediction.get("payment_disposition", "null")))
        else:
            # If JSON failed, we treat it as "null" or "error" for metrics
            y_true_disposition.append(str(ground_truth.get("disposition", "null")))
            y_pred_disposition.append("JSON_ERROR")
            
            y_true_pay_disp.append(str(ground_truth.get("payment_disposition", "null")))
            y_pred_pay_disp.append("JSON_ERROR")

    # REPORTING
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    json_validity = valid_json_count/len(eval_data)
    print(f"JSON Validity Rate: {json_validity*100:.1f}%")
    
    disp_report = classification_report(y_true_disposition, y_pred_disposition, output_dict=True, zero_division=0)
    pay_disp_report = classification_report(y_true_pay_disp, y_pred_pay_disp, output_dict=True, zero_division=0)
    
    print("\n--- Disposition Classification Report ---")
    print(classification_report(y_true_disposition, y_pred_disposition, zero_division=0))
    
    print("\n--- Payment Disposition Classification Report ---")
    print(classification_report(y_true_pay_disp, y_pred_pay_disp, zero_division=0))
    
    overall_acc = accuracy_score(y_true_disposition, y_pred_disposition)
    print(f"\nOverall Disposition Accuracy: {overall_acc*100:.2f}%")

    # Save to JSON
    results = {
        "timestamp": str(date.today()),
        "model_checkpoint": CHECKPOINT_PATH,
        "sample_size": len(eval_data),
        "json_validity_rate": json_validity,
        "overall_disposition_accuracy": overall_acc,
        "disposition_report": disp_report,
        "payment_disposition_report": pay_disp_report
    }
    
    with open("eval_results_v6.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to eval_results_v6.json")

if __name__ == "__main__":
    main()
