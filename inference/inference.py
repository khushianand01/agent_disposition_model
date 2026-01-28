
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import json
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
import sys

# =========================
# CONFIG
# =========================
MODEL_PATH = "outputs/qwen_production_v5_perfect" # Path to the fine-tuned adapter
MAX_SEQ_LEN = 2048
DTYPE = None # Auto
LOAD_IN_4BIT = True

class DispositionModel:
    def __init__(self, model_path=MODEL_PATH):
        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LEN,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference
        print("Model loaded successfully.")

    def format_prompt(self, transcript, current_date=None):
        date_context = f"\nCurrent Date: {current_date}" if current_date else ""
        instruction = f"You are an AI assistant that extracts structured call disposition data.\nGiven a call transcript between an agent and a borrower, extract the following fields Return ONLY valid JSON. Do not explain.:\ndisposition, payment_disposition, reason_for_not_paying, ptp_amount, ptp_date, followup_date, remarks.{date_context}\nIf a field is not present, return null."
        
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{transcript}

### Response:
"""

    def clean_output(self, result: dict, transcript: str, current_date: str) -> dict:
        """
        Post-processing logic to fix common model hallucinations and date issues.
        """
        if not isinstance(result, dict): return result

        # 1. Fix Future PTP Dates (If date < current, assume next month)
        if current_date:
            try:
                curr_dt = datetime.strptime(current_date.split(' ')[0], "%Y-%m-%d").date()
                
                for date_field in ["ptp_date", "followup_date"]:
                    val = result.get(date_field)
                    if val and isinstance(val, str):
                        try:
                            # Parse date (Assuming ISO format from model)
                            pred_dt = datetime.strptime(val.split(' ')[0], "%Y-%m-%d").date()
                            
                            # If date is in the past (e.g. said "25th" on the 26th), move to next month
                            if pred_dt < curr_dt:
                                # Add 1 month
                                new_dt = pred_dt + relativedelta(months=1)
                                result[date_field] = f"{new_dt} 00:00:00"
                        except:
                            pass # Ignore parse errors
            except:
                pass



        return result

    @torch.inference_mode()
    def predict(self, transcript, current_date=None):
        # Strict Date Logic: Use provided date or default to system date
        if current_date is None:
            current_date = str(date.today())
            
        prompt = self.format_prompt(transcript, current_date=current_date)
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            use_cache=True,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        # Decode text
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[-1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Calculate Confidence Score (Average Probability)
        confidence_score = 0.0
        if outputs.scores:
            probs = [torch.nn.functional.softmax(score, dim=-1) for score in outputs.scores]
            # Get the probability of the token that was actually chosen (greedy)
            token_probs = []
            for i, token_id in enumerate(generated_ids):
                if i < len(probs):
                    token_probs.append(probs[i][0, token_id].item())
            
            if token_probs:
                confidence_score = sum(token_probs) / len(token_probs)

        try:
            # Extract JSON substring - find FIRST valid JSON block
            json_start = generated_text.find('{')
            if json_start != -1:
                # Naive matching of braces to find the end
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(generated_text)):
                    if generated_text[i] == '{':
                        brace_count += 1
                    elif generated_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end != -1:
                    json_text = generated_text[json_start:json_end]
                    result = json.loads(json_text)
                else:
                     raise json.JSONDecodeError("Unbalanced braces", generated_text, 0)
            else:
                raise json.JSONDecodeError("No JSON found", generated_text, 0)

            if isinstance(result, dict):
                # Apply Cleaning / Logic Layer
                result = self.clean_output(result, transcript, current_date)
                
                result["confidence_score"] = round(confidence_score, 4)
            return result
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse JSON", 
                "raw_output": generated_text, 
                "confidence_score": round(confidence_score, 4)
            }

# Singleton instance for easy import
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = DispositionModel()
    return _model_instance

if __name__ == "__main__":
    # Test
    pass
