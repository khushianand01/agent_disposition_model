
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
MODEL_PATH = "outputs/qwen3_8b_v6_balanced" # Latest evaluation checkpoint
MAX_SEQ_LEN = 4096
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
        instruction = f"""You are an AI assistant that extracts structured call disposition data.
Given a call transcript between an agent and a borrower, extract the following fields:
- disposition
- payment_disposition
- reason_for_not_paying
- ptp_amount (Extract as numerical value)
- ptp_date (Format: YYYY-MM-DD. If relative date like 'today' or 'tomorrow' is used, resolve it using {current_date})
- followup_date (Format: YYYY-MM-DD)
- remarks (Include descriptive summary like "will pay one EMI" or reason for delay)

STRICT RULES:
1. Return ONLY valid JSON.
2. If payment_disposition is 'PTP', you MUST populate 'ptp_amount' and 'ptp_date' if mentioned. 
3. NEVER hallucinate numbers, dates, or details. Only extract what is explicitly stated in the transcript.
4. If an amount or date is not mentioned, return null. Do not guess.
5. Do not only put dates/amounts in 'remarks'; they MUST be in their respective fields.
6. Current Date for reference: {current_date}
7. If a field is not mentioned, return null.
8. If the borrower says "full amount" or "clear dues", you MAY use the outstanding amount mentioned by the Agent.
9. Extract relative dates (e.g. "10th", "next week") into 'ptp_date'. Do not leave them only in remarks."""
    
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
        Ensures dates are in YYYY-MM-DD format and verifies data against transcript.
        """
        if not isinstance(result, dict): return result

        # Rescue missing Date from Remarks if needed
        global_date_regex = re.compile(r"\b(\d{1,2}(?:st|nd|rd|th)?)\b", re.IGNORECASE)
        if not result.get("ptp_date") and result.get("remarks"):
            # Check for date-like digits in remarks (e.g. "pay on 10th")
            if global_date_regex.search(str(result["remarks"])):
                 result["ptp_date"] = result["remarks"]

        # Current reference date
        curr_dt = date(2026, 1, 29)
        if current_date:
            try:
                dt_part = current_date.split(' ')[0]
                curr_dt = datetime.strptime(dt_part, "%Y-%m-%d").date()
            except: pass
        
        curr_year = curr_dt.year
        curr_month = curr_dt.month

        # 1. Amount Verification (Anti-Hallucination)
        if "ptp_amount" in result and result["ptp_amount"] is not None:
            try:
                amt_val = float(re.sub(r"[^\d.]", "", str(result["ptp_amount"])))
                # Check if this specific number (or a close variant) exists in the transcript
                # We search for the integer part or the full string
                amt_str = str(int(amt_val))
                if amt_str not in transcript:
                    # If not found, check if maybe it was mentioned in words (simplified check)
                    # For now, if it's a "clean" number not in text, it's likely a hallucination
                    if amt_val > 0:
                        result["ptp_amount"] = None
            except:
                pass

        # 2. Date Verification and Standardization
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        
        # Detect mentioned month in transcript
        mentioned_month = None
        lower_transcript = transcript.lower()
        for m_name, m_num in month_map.items():
            if m_name in lower_transcript:
                mentioned_month = m_num
                break

        for date_field in ["ptp_date", "followup_date"]:
            val = result.get(date_field)
            if not val: continue
            
            val_str = str(val).strip()
            if not val_str or val_str.lower() == "null":
                result[date_field] = None
                continue

            parsed_dt = None

            # a. Handle YYYYMMDD format (e.g. 20260121)
            if re.match(r"^\d{8}$", val_str):
                try:
                    parsed_dt = datetime.strptime(val_str, "%Y%m%d").date()
                except: pass

            # b. Handle YYYY-MM-DD (standard)
            elif re.match(r"^\d{4}-\d{2}-\d{2}", val_str):
                try:
                    parsed_dt = datetime.strptime(val_str[:10], "%Y-%m-%d").date()
                except: pass

            # c. Handle just the day (e.g. "25")
            elif val_str.isdigit() and len(val_str) <= 2:
                try:
                    day = int(val_str)
                    parsed_dt = date(curr_year, curr_month, day)
                except: pass

            # d. Fallback: Try natural language and ISO-ish formats
            if not parsed_dt:
                try:
                    # Clean the string (remove th, st, etc.)
                    clean_val = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", val_str).lower()
                    
                    # Try "Day Month" (e.g. 7 feb)
                    for m_name, m_num in month_map.items():
                        if m_name in clean_val:
                            day_match = re.search(r"(\d{1,2})", clean_val)
                            if day_match:
                                day = int(day_match.group(1))
                                parsed_dt = date(curr_year, m_num, day)
                                break
                    
                    if not parsed_dt:
                        # Try YYYY-MM-DD anywhere
                        match = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", clean_val)
                        if match:
                            parsed_dt = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
                except: pass

            # 3. Post-Parsing Corrections
            if parsed_dt:
                # If a specific month was mentioned in transcript but model returned current month/different month
                if mentioned_month and parsed_dt.month != mentioned_month:
                    try:
                        # Only override if the year is current or future
                        if parsed_dt.year >= curr_year:
                             parsed_dt = date(parsed_dt.year, mentioned_month, parsed_dt.day)
                    except ValueError: # e.g. Feb 31st
                         pass

                # If date is in the past (e.g. said "7th" on the 29th), move to next month
                if parsed_dt < curr_dt:
                    # Only auto-shift if the transcript DOES NOT mention a specific month
                    # Or if it's the current month but a past day
                    if not mentioned_month or mentioned_month == curr_month:
                        parsed_dt = parsed_dt + relativedelta(months=1)
                
                result[date_field] = parsed_dt.strftime("%Y-%m-%d")

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

        # Calculate Smart Confidence Score (Weighted Average)
        confidence_score = 0.0
        if outputs.scores:
            probs = [torch.nn.functional.softmax(score, dim=-1) for score in outputs.scores]
            token_probs = []
            for i, token_id in enumerate(generated_ids):
                if i < len(probs):
                    token_probs.append(probs[i][0, token_id].item())
            
            if token_probs:
                # Weighted average: First 50% of tokens (usually categories) weight 70%
                # Last 50% of tokens (usually remarks) weight 30%
                mid_point = max(1, len(token_probs) // 2)
                first_half = token_probs[:mid_point]
                second_half = token_probs[mid_point:]
                
                avg_first = sum(first_half) / len(first_half)
                avg_second = sum(second_half) / len(second_half) if second_half else avg_first
                
                # Combine: Categorization (first half) is prioritized
                confidence_score = (avg_first * 0.7) + (avg_second * 0.3)

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
