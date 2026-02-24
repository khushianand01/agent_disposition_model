import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
import json
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU
import re
import sys
import os
import threading

class StopOnJson(StoppingCriteria):
    """Stop generation when the outermost JSON '{}' is closed (brace depth returns to 0)."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Pre-compute which token IDs contain '{' or '}'
        self.open_ids = set()
        self.close_ids = set()
        for tid in range(tokenizer.vocab_size):
            tok = tokenizer.decode([tid])
            if '{' in tok:
                self.open_ids.add(tid)
            if '}' in tok:
                self.close_ids.add(tid)
        self.reset()

    def reset(self):
        self.depth = 0
        self.started = False

    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0, -1].item()
        if last_token in self.open_ids:
            self.depth += 1
            self.started = True
        if last_token in self.close_ids:
            self.depth -= 1
        # Stop only when we've opened at least one brace and depth is back to 0
        return self.started and self.depth <= 0

# =========================
# CONFIG
# =========================
MODEL_PATH = os.getenv("QWEN_MODEL", "khushianand01/disposition_model")
MAX_SEQ_LEN = 8192 # Expanded from 4096 to handle long transcripts
DTYPE = None # Auto
LOAD_IN_4BIT = True

class DispositionModel:
    def __init__(self, model_path=MODEL_PATH):
        self.lock = threading.Lock()
        print(f"Loading model from {model_path}...")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This server requires a GPU to run.")
        self.device = "cuda"
        print(f"Using device: {self.device}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=MAX_SEQ_LEN,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.model)
        self.stop_criteria = StoppingCriteriaList([StopOnJson(self.tokenizer)])
        print("Model loaded successfully.")

    def format_prompt(self, transcript, current_date=None):
        instruction = (
            "You are an AI assistant that extracts structured call disposition data.\n"
            "Fields: disposition, payment_disposition, reason_for_not_paying, ptp_details, remarks, confidence_score.\n"
            "\n"
            "EXAMPLES:\n"
            "1. Transcript: 'Hello? Haan Mamata ji ke devar bol raha hoon. Wo ghar pe nahi hain.'\n"
            "   Output: {\"disposition\": \"ANSWERED_BY_FAMILY_MEMBER\", \"payment_disposition\": null, \"reason_for_not_paying\": null, \"ptp_details\": {\"amount\": null, \"date\": null}, \"remarks\": \"talked to brother-in-law\", \"confidence_score\": 0.98}\n"
            "\n"
            "2. Transcript: 'Haan main parso 5000 jama kar dunga.' Current Date: 2026-01-27\n"
            "   Output: {\"disposition\": \"ANSWERED\", \"payment_disposition\": \"PTP\", \"reason_for_not_paying\": \"CUSTOMER_NOT_TELLING_REASON\", \"ptp_details\": {\"amount\": 5000, \"date\": \"2026-01-29\"}, \"remarks\": \"will pay day after tomorrow\", \"confidence_score\": 0.95}\n"
            "\n"
            "3. Transcript: 'Wrong number hai, main nahi jaanta kisi Rahul ko.'\n"
            "   Output: {\"disposition\": \"WRONG_NUMBER\", \"payment_disposition\": null, \"reason_for_not_paying\": null, \"ptp_details\": {\"amount\": null, \"date\": null}, \"remarks\": \"wrong number\", \"confidence_score\": 0.99}\n"
            "\n"
            "RULES:\n"
            "- A 'PTP' (Promise to Pay) occurs when a customer commits to pay a specific amount on a specific date.\n"
            "- If the customer is vague (e.g., 'I will try'), use 'NO_PAYMENT_COMMITMENT'.\n"
            "- DATE CALCULATION: 'Kal' = Tomorrow (Today + 1), 'Parso' = Day After Tomorrow (Today + 2). February has 28 days.\n"
            "- confidence_score should be between 0.0 and 1.0 based on how clear the transcript is.\n"
            "- Return ONLY valid JSON."
        )
    
        return f"""### Instruction:
{instruction}

### Input:
Context: Current Date is {current_date}
Transcript: {transcript}

### Response:
"""

    def clean_output(self, result: dict, transcript: str, current_date: str) -> dict:
        if not isinstance(result, dict): return {"error": "Invalid format", "raw": str(result)}

        call_labels = [
            "ANSWERED", "ANSWERED_BY_FAMILY_MEMBER", "CUSTOMER_PICKED", "AGENT_BUSY_ON_ANOTHER_CALL",
            "SILENCE_ISSUE", "LANGUAGE_BARRIER", "ANSWERED_VOICE_ISSUE", "CUSTOMER_ABUSIVE",
            "AUTOMATED_VOICE", "FORWARDED_CALL", "RINGING", "BUSY", "SWITCHED_OFF",
            "WRONG_NUMBER", "DO_NOT_KNOW_THE_PERSON", "NOT_IN_CONTACT_ANYMORE", "OUT_OF_NETWORK", "OUT_OF_SERVICES",
            "CALL_BACK_LATER", "WILL_ASK_TO_PAY", "GAVE_ALTERNATE_NUMBER",
            "ANSWERED_DISCONNECTED", "CALL_DISCONNECTED_BY_CUSTOMER", "NOT_AVAILABLE", "WRONG_PERSON", 
            "NO_INCOMING_CALLS", "RINGING_DISCONNECTED", "OTHERS"
        ]
        
        pay_labels = [
            "PAID", "PTP", "PARTIAL_PAYMENT", "SETTLEMENT", "WILL_PAY_AFTER_VISIT",
            "DENIED_TO_PAY", "NO_PAYMENT_COMMITMENT", "NO_PROOF_GIVEN", "WANT_FORECLOSURE", "WANTS_TO_RENEGOTIATE_LOAN_TERMS",
            "None"
        ]

        # 1. FUZZY Label Mapping (Don't be too strict)
        disp = str(result.get("disposition", "OTHERS")).upper().replace(" ", "_")
        if disp not in call_labels:
            if "FAMILY" in disp: result["disposition"] = "ANSWERED_BY_FAMILY_MEMBER"
            elif "BUSY" in disp: result["disposition"] = "BUSY"
            elif "WRONG" in disp: result["disposition"] = "WRONG_NUMBER"
            elif "ANSWER" in disp: result["disposition"] = "ANSWERED"
            elif len(disp) > 2 and disp.replace("_", "").isalnum():
                # If it looks like a valid label, let it through
                result["disposition"] = disp
            else:
                result["disposition"] = "OTHERS"
        else:
            result["disposition"] = disp

        p_disp = str(result.get("payment_disposition", "None")).upper().replace(" ", "_")
        if p_disp not in pay_labels:
            if "CLAIM" in p_disp: result["payment_disposition"] = "PAID"
            elif "PROMISE" in p_disp or "PTP" in p_disp: result["payment_disposition"] = "PTP"
            elif "REFUSE" in p_disp or "DENY" in p_disp: result["payment_disposition"] = "DENIED_TO_PAY"
            else: result["payment_disposition"] = "None"
        else:
            result["payment_disposition"] = p_disp

        # 2. Reason Mapping (Fuzzy)
        reason = str(result.get("reason_for_not_paying", "None")).upper().replace(" ", "_")
        if "JOB" in reason and ("LOSS" in reason or "REH GAYA" in reason): 
            result["reason_for_not_paying"] = "JOB_CHANGED_WAITING_FOR_SALARY"

        # 3. PTP Details Rescue & Validation
        ptp = result.get("ptp_details", {})
        if not isinstance(ptp, dict): ptp = {"amount": None, "date": None}
        
        # Amount Validation (Intelligent Match)
        amt = ptp.get("amount")
        if amt:
            try:
                # Clean amount for comparison (5,000 -> 5000)
                clean_amt = str(int(float(str(amt).replace(',', ''))))
                found = False
                # Check for direct or fuzzy digits in transcript
                if clean_amt in transcript.replace(',', ''):
                    found = True
                
                if not found: ptp["amount"] = None # Still verify it's supported by text
                else: ptp["amount"] = clean_amt
            except: ptp["amount"] = None

        # 4. Date Validation & "Parso" Correction
        p_date = ptp.get("date")
        if p_date and current_date:
            try:
                import calendar
                from datetime import datetime, timedelta, date as dt_date
                c_y, c_m, c_d = map(int, current_date.split('-'))
                dt_today = dt_date(c_y, c_m, c_d)
                
                # Intelligent "Parso" Recovery
                l_trans = transcript.lower()
                if "parso" in l_trans:
                    # Force Today + 2
                    ptp["date"] = str(dt_today + timedelta(days=2))
                elif "kal" in l_trans and ptp.get("date") == current_date:
                    # If model said Today but transcript said Kal, fix it to Today + 1
                    ptp["date"] = str(dt_today + timedelta(days=1))

                # Structural cleanup (Feb 30 fix)
                try:
                    datetime.strptime(str(ptp["date"]), "%Y-%m-%d")
                except ValueError:
                    # If invalid date (e.g. Feb 30), Cap it at month end
                    parts = str(ptp["date"]).split('-')
                    if len(parts) == 3:
                        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                        # Get max days in that month/year
                        _, max_days = calendar.monthrange(y, m)
                        if d > max_days: 
                            ptp["date"] = f"{y:04d}-{m:02d}-{max_days:02d}"
            except:
                pass

        # 5. Balanced PTP Enforcement (Negative Rules)
        # Rule: Only downgrade if BOTH vague AND lacks specific commitment details
        if result.get("payment_disposition") == "PTP":
            lower_t = transcript.lower()
            commitment_words = ["pay", "paid", "amount", "rupaye", "kal", "aaj", "parso", "tarikh", "send", "karunga", "dena"]
            # Be very careful with "vague" words - in Hinglish, "koshish" is often polite commitment
            # Only downgrade if it's truly non-committal
            is_non_committal = "sochunga" in lower_t or "dekhunga" in lower_t
            has_strong_keyword = any(w in lower_t for w in commitment_words)

            if is_non_committal and not has_strong_keyword:
                # Only downgrade if NO date/amount were found as well
                if ptp.get("date") is None and ptp.get("amount") is None:
                    result["payment_disposition"] = "NO_PAYMENT_COMMITMENT"
                    result["remarks"] = result.get("remarks", "") + " (Policy Downgrade: Non-committal)"
            
        result["ptp_details"] = ptp
        # Ensure confidence_score is never null in a successful response
        try:
            val = float(result.get("confidence_score", 0.85))
            result["confidence_score"] = min(max(val, 0.0), 1.0)
        except:
            result["confidence_score"] = 0.85
            
        return result

    @torch.inference_mode()
    def predict(self, transcript, current_date=None):
        with self.lock:
            if current_date is None: current_date = str(date.today())
            
            # Handle cases where transcript might be a dict (from raw test data)
            if isinstance(transcript, dict):
                transcript = transcript.get("transcript", str(transcript))
            else:
                transcript = str(transcript)

            # Hard Truncation to prevent CUDA Illegal Memory Access
            # 8192 is the absolute max. We truncate transcript to ~7500 tokens.
            # Character count is a safe proxy (1 token ~ 3 chars). 22,000 chars is safe.
            if len(transcript) > 22000:
                transcript = transcript[:22000] + "... [TRUNCATED]"

            prompt = self.format_prompt(transcript, current_date=current_date)
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            
            # Additional safety: hard truncate input_ids if they still exceed context
            if inputs["input_ids"].shape[1] > MAX_SEQ_LEN:
                 inputs["input_ids"] = inputs["input_ids"][:, :MAX_SEQ_LEN]
                 inputs["attention_mask"] = inputs["attention_mask"][:, :MAX_SEQ_LEN]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                do_sample=False,
                stopping_criteria=self.stop_criteria,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            try:
                json_start = generated_text.find('{')
                json_end = generated_text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    result = json.loads(generated_text[json_start:json_end])
                else:
                    raise ValueError("No JSON found")
                
                return self.clean_output(result, transcript, current_date)
            except Exception as e:
                return {"error": str(e), "raw": generated_text}

_model_instance = None
def get_model():
    global _model_instance
    if _model_instance is None: _model_instance = DispositionModel()
    return _model_instance
