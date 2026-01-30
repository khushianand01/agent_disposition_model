import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import json
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU
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
4. If an amount or date is not mentioned, return null. Do not guess. NEVER make up an amount like 10000 or 5000 if it is not in the transcript.
5. Do not only put dates/amounts in 'remarks'; they MUST be in their respective fields.
6. Current Date for reference: {current_date}
7. If a field is not mentioned, return null.
8. If the borrower says "full amount" or "clear dues" AND the Agent mentions an amount, you MAY use that amount. Otherwise, return null.
9. Extract relative dates (e.g. "Friday", "Monday", "10th", "next week") into 'ptp_date'. Do not leave them only in remarks.
10. FINAL WARNING: If a date or amount is NOT explicitly stated in the transcript, you MUST return null. Do NOT guess or provide 'dummy' values.
11. If the borrower says they are a family member (wife, husband, son, etc.), set 'disposition' to 'ANSWERED_BY_FAMILY_MEMBER'.
12. If the borrower asks to call back later (e.g. "baad mein phone karna", "busy hoon"), set 'disposition' to 'CALL_BACK_LATER'.
13. If the borrower is unsure or refuses to give a commitment date (e.g. "Abhi kuch keh nahi sakta", "kuch keh nahi sakta"), set 'payment_disposition' to 'NO_PAYMENT_COMMITMENT' and 'disposition' to 'ANSWERED'."""
    
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

        # Current reference date
        curr_dt = date(2026, 1, 29)
        if current_date:
            try:
                dt_part = current_date.split(' ')[0]
                curr_dt = datetime.strptime(dt_part, "%Y-%m-%d").date()
            except: pass

        curr_year = curr_dt.year
        curr_month = curr_dt.month

        # Rescue missing Date from Remarks if needed
        if result.get("ptp_date") and result.get("ptp_date") == result.get("remarks"):
                 result["ptp_date"] = result["remarks"]

        # 2. Date Rescue Logic (Transcript Scan + Remarks Scan)
        # We scan for specific patterns like "5 Feb" or "Feb 5"
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        
        sources_to_check = [
            str(result.get("remarks", "")).lower(),
            transcript.lower() 
        ]

        found_date = None
        for source in sources_to_check:
            if found_date: break
            
            # Pattern: 5 Feb, 5th February, 05-Feb
            for m_name, m_num in month_map.items():
                pattern = r"(\b\d{1,2})(?:st|nd|rd|th)?\s+" + m_name + r"\b"
                match = re.search(pattern, source)
                if not match:
                    # Pattern: February 5th, Feb 5
                    pattern = r"\b" + m_name + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b"
                    match = re.search(pattern, source)
                
                if match:
                    day = int(match.group(1))
                    if 1 <= day <= 31:
                        try:
                            found_date = date(curr_year, m_num, day)
                            if m_num < curr_month:
                                found_date = found_date.replace(year=curr_year + 1)
                            print(f"DEBUG: Found date via rescue: {found_date} from pattern with {m_name}")
                            break
                        except Exception as e:
                            print(f"DEBUG: Error building date: {e}")
                            pass

        # Track if we rescued the date
        date_rescued = False
        if found_date:
            res_str = found_date.strftime("%Y-%m-%d")
            # Override if model date is missing or wrong
            if result.get("ptp_date") != res_str:
                print(f"DEBUG: Overriding ptp_date from {result.get('ptp_date')} to {res_str}")
                result["ptp_date"] = res_str
                date_rescued = True

        # Weekday and Hindi Relative Date Logic (Fallback)
        if not date_rescued:
            # ... (rest of logic same)
            # Original weekday logic
            weekday_map = {
                "monday": MO, "tuesday": TU, "wednesday": WE, "thursday": TH, 
                "friday": FR, "saturday": SA, "sunday": SU,
                "mon": MO, "tue": TU, "wed": WE, "thu": TH, "fri": FR, "sat": SA, "sun": SU
            }
            
            for source in sources_to_check:
                if date_rescued: break
                
                if "next week" in source:
                    result["ptp_date"] = (curr_dt + timedelta(days=7)).strftime("%Y-%m-%d")
                    date_rescued = True
                    break
                
                for day_name, day_obj in weekday_map.items():
                    if day_name in source:
                        parsed_dt = curr_dt + relativedelta(weekday=day_obj(+1))
                        result["ptp_date"] = parsed_dt.strftime("%Y-%m-%d")
                        date_rescued = True
                        break
            
            # Kal/Aaj logic
            t_lower = transcript.lower()
            if not date_rescued:
                if "kal " in t_lower or " kal" in t_lower:
                    result["ptp_date"] = (curr_dt + timedelta(days=1)).strftime("%Y-%m-%d")
                    date_rescued = True
                elif "aaj " in t_lower or " aaj" in t_lower:
                    result["ptp_date"] = curr_dt.strftime("%Y-%m-%d")
                    date_rescued = True
        
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
                        # SANITIZE REMARKS: If the fake amount is in remarks, scrub it
                        if amt_str in str(result.get("remarks", "")):
                             result["remarks"] = "Customer mentioned full amount (exact figure not in text)"
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
            
            # Simplified check: If it's a model output that looks like a day number ("25", "10th"), check if it exists.
            # If it's a full date YYYY-MM-DD, we can't check easily because transcript says "10th".
            # So, we rely on the fact that we previously scanned for validation.
            
            # Let's add a specific check for the "model made up a number" case.
            # If valid, clean_val usually extracts a day.
            
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

            # c. Handle just the day (e.g. "25") - use rescue logic first if ptp_date was null
            elif val_str.isdigit() and len(val_str) <= 2:
                try:
                    day = int(val_str)
                    # If it's ptp_date and we rescued it from remarks, use current month/year first
                    if date_field == "ptp_date" and result.get("ptp_date") == result.get("remarks"):
                        parsed_dt = date(curr_year, curr_month, day)
                    else: # Otherwise assume current year
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
            
            # Update the field with parsed object (temporary) or keep string if failed
            if parsed_dt:
                 # Fix hallucinated past year (e.g. model returns 2025 when it's 2026)
                 if parsed_dt.year < curr_year:
                      parsed_dt = parsed_dt.replace(year=curr_year)
                 
                 # STRICT DAY VERIFICATION (New)
                 # If this date was NOT rescued (i.e. it came from model generation), ensure the day number exists in text
                 elif (date_field == "ptp_date" and not date_rescued) or (date_field == "followup_date"):
                      day_str = str(parsed_dt.day)
                      # Check for "25", "25th", "25-02", etc.
                      # Regex: word boundary 25 word boundary OR 25th/st/rd/nd
                      # Also check if it's the "current_date" (sometimes model echoes current date) -> Allow current date
                      
                      # Simplified: Just check if the day number appears in transcript digits
                      # NOTE: This might match "2500" rupees. So we want \b25\b
                      if day_str not in transcript:
                           # Try with suffixes
                           suffixes = ["th", "st", "nd", "rd"]
                           has_match = False
                           # 1. Exact match e.g. "on 25"
                           if re.search(r"\b" + day_str + r"\b", transcript): has_match = True
                           # 2. Suffix match e.g. "25th"
                           if not has_match:
                                for s in suffixes:
                                     if (day_str + s) in transcript:
                                          has_match = True; break
                           
                           # 3. Allow if it matches current date (Agent: "Today is 25th")
                           if parsed_dt == curr_dt: has_match = True
                           
                           if not has_match:
                                result[date_field] = None # Kill hallucination (e.g. 25th not in text)
                                parsed_dt = None # Invalidate so we don't process it below
                           else:
                                result[date_field] = parsed_dt
                      else:
                           # It's in the text (simple, maybe part of 2500 but risky to filter too hard)
                           # Let's enforce boundary check for safety
                           if re.search(r"\b" + day_str + r"(?:th|st|nd|rd)?\b", transcript):
                                result[date_field] = parsed_dt
                           else:
                                if parsed_dt == curr_dt:
                                     result[date_field] = parsed_dt
                                else:
                                     result[date_field] = None
                                     parsed_dt = None
                 else:
                      result[date_field] = parsed_dt
            else:
                 # If we couldn't parse it despite it being a string, kill it to be safe
                 result[date_field] = None

        # 3. Post-Parsing Corrections
        for date_field in ["ptp_date", "followup_date"]: # Re-iterate to apply corrections to the parsed_dt objects
            parsed_dt = result.get(date_field)
            if not isinstance(parsed_dt, date): # Only process if it's a date object (i.e., successfully parsed)
                continue

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
            
            result[date_field] = parsed_dt # Store the date object back

        # FINAL SYNC: If PTP Date exists but Followup is missing (or killed above), usually they recall on same day
        if result.get("ptp_date") and not result.get("followup_date"):
             result["followup_date"] = result["ptp_date"]
             
        # Convert all date objects back to strings
        for f in ["ptp_date", "followup_date"]:
             val = result.get(f)
             if isinstance(val, date):
                  result[f] = val.strftime("%Y-%m-%d")

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
        print(f"\n--- DEBUG: RAW GENERATED TEXT ---\n{generated_text}\n---------------------------------")

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
