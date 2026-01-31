import torch
from unsloth import FastLanguageModel
import json
import sys

# =========================
# CONFIG
# =========================
S1_MODEL_PATH = "/home/ubuntu/Disposition_model2-main/ringg_slm/outputs/ringg_slm_stage1"
S2_MODEL_PATH = "/home/ubuntu/Disposition_model2-main/ringg_slm/outputs/ringg_slm_stage2_v2"
MAX_SEQ_LEN = 2048
DTYPE = None
LOAD_IN_4BIT = True

s1_prompt = """### Instruction:
Analyze the following call transcript and classify the call outcome. Return ONLY valid JSON with 'disposition' and 'payment_disposition' fields.

PAYMENT_DISPOSITION RULES:
- PTP (Promise to Pay): Customer promises to pay in the FUTURE. If they mention a specific amount or date for a FUTURE payment, it is PTP.
  Examples: 
  * "5 tareekh ko 4000 de dunga" → PTP
  * "salary aane pe pay kar dunga" → PTP
  
- CLAIMING_PAYMENT_IS_COMPLETED: Customer claims they have ALREADY paid.
  Examples:
  * "maine kal hi payment kar diya" (Past Tense) → CLAIMING_PAYMENT_IS_COMPLETED
  * "bhej diya hai" (Past Tense) → CLAIMING_PAYMENT_IS_COMPLETED

- NO_PAYMENT_COMMITMENT: Customer is vague, refusing, or avoiding commitment.
  Examples:
  * "abhi paise nahi hain" → NO_PAYMENT_COMMITMENT
  * "baad mein dekhunga" → NO_PAYMENT_COMMITMENT
  * "pata nahi kab de paunga" → NO_PAYMENT_COMMITMENT

CRITICAL TENSE LOGIC:
1. FUTURE ("dunga", "karunga", "paunga") -> PTP (if amount/date mentioned or implied).
2. PAST ("diya", "hua", "gaya") -> CLAIMING_PAYMENT_IS_COMPLETED.
3. VAGUE/REFUSAL ("nahi hai", "dekhunga", "baad mein") -> NO_PAYMENT_COMMITMENT.

DO NOT return 'PAID'. Use 'CLAIMING_PAYMENT_IS_COMPLETED'.

### Input:
{}

### Response:
{}"""

s2_prompt = """### Instruction:
Extract structured payment-related information from the call transcript. 

1. Specific Fields:
   - "reason_for_not_paying": Select the most appropriate enum from the following list:
     [BUSINESS_CLOSED, BUSINESS_LOSS, CLAIMING_FRAUD, CLAIMING_PAYMENT_IS_COMPLETED, CUSTOMER_EXPIRED, CUSTOMER_NOT_TELLING_REASON, CUSTOMER_PLANS_TO_VISIT_BRANCH, DEATH_IN_FAMILY, FUND_ISSUE, GRIEVANCE_APP_ISSUE, GRIEVANCE_CALLER_MISCONDUCT, GRIEVANCE_FRAUD, GRIEVANCE_LOAN_AMOUNT_DISPUTE, JOB_CHANGED_WAITING_FOR_SALARY, LOAN_CLOSURE_MISCOMMUNICATION, LOAN_TAKEN_BY_KNOWN_PARTY, LOST_JOB, MEDICAL_ISSUE, MULTIPLE_LOANS, OTHER_PERSON_TAKEN, OTHER_REASONS, PENALTY_ISSUE, SERVICE_ISSUE, TRUST_ISSUE]
     Default to CUSTOMER_NOT_TELLING_REASON if no specific reason is given.
   - "ptp_amount": The exact amount (number) promised for future payment.
   - "ptp_date": The future date promised (strictly YYYY-MM-DD format).
   - "remarks": A short summary of the outcome.

2. Extraction Priorities:
   - PRIORITY: If multiple amounts are mentioned, extract the one promised for the FUTURE (e.g., "baaki 2500 dunga" -> 2500).
   - REASONING: If the customer mentions job loss, financial crisis, or app issues, map it to the corresponding enum (e.g., "job chali gayi" -> LOST_JOB).
   - TENSE: Extract "ptp_amount" and "ptp_date" IF AND ONLY IF the customer mentions them in a FUTURE context (e.g., "dunga", "karunga").

3. Date Logic:
   - Use provided "current_date" to calculate "ptp_date".
   - "Aaj" = current_date, "Kal" = +1 day, "Parso" = +2 days, "Next week" = +7 days.
   - Return dates strictly in YYYY-MM-DD format.

REQUIRED OUTPUT FORMAT (ALL 5 FIELDS MUST BE PRESENT):
{{
  "reason_for_not_paying": "enum_value",
  "ptp_amount": number or null,
  "ptp_date": "YYYY-MM-DD" or null,
  "followup_date": "YYYY-MM-DD" or null,
  "remarks": "brief note" or null
}}

### Examples:
Transcript: "Agent: EMI delay kyu ho rahi hai? Borrower: Job chali gayi hai meri is month, payment nahi de paunga."
Response: {{"reason_for_not_paying": "LOST_JOB", "ptp_amount": null, "ptp_date": null, "followup_date": null, "remarks": "customer lost his job and cannot pay this month"}}

Transcript: "Agent: Payment kab karoge? Borrower: Kal 4000 pay kar dunga."
Response: {{"reason_for_not_paying": "CUSTOMER_NOT_TELLING_REASON", "ptp_amount": 4000, "ptp_date": "2026-02-01", "followup_date": "2026-02-02", "remarks": "customer promised 4k tomorrow"}}

### Input:
{}

### Response:
{}"""



class RinggPipeline:
    def __init__(self):
        print(f"[1/2] Loading Stage 1 Model...")
        self.s1_model, self.s1_tokenizer = FastLanguageModel.from_pretrained(
            model_name=S1_MODEL_PATH,
            max_seq_length=MAX_SEQ_LEN,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.s1_model)

        print(f"[2/2] Loading Stage 2 Model...")
        self.s2_model, self.s2_tokenizer = FastLanguageModel.from_pretrained(
            model_name=S2_MODEL_PATH,
            max_seq_length=MAX_SEQ_LEN,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.s2_model)
        print("Pipeline Ready!\n")

    @torch.inference_mode()
    def run_inference(self, transcript):
        # --- Stage 1 ---
        prompt1 = s1_prompt.format(transcript, "")
        inputs1 = self.s1_tokenizer([prompt1], return_tensors="pt").to("cuda")
        outputs1 = self.s1_model.generate(
            **inputs1, max_new_tokens=64, use_cache=True, do_sample=False,
            eos_token_id=self.s1_tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Calculate Stage 1 Confidence
        s1_scores = torch.stack(outputs1.scores, dim=1)  # (batch, seq_len, vocab)
        s1_probs = torch.softmax(s1_scores, dim=-1)
        # For greedy decoding (do_sample=False), the generated token is the argmax
        s1_token_probs, _ = torch.max(s1_probs, dim=-1)
        s1_confidence = s1_token_probs[0].mean().item()

        res1_text = self.s1_tokenizer.decode(outputs1.sequences[0], skip_special_tokens=True)
        s1_json = self._extract_json(res1_text)
        
        # --- Strict Enum Mapping ---
        # The model sometimes outputs 'PAID' or 'ALREADY_PAID' which are valid in training data but not in schema
        if s1_json.get("payment_disposition") in ["PAID", "ALREADY_PAID"]:
            s1_json["payment_disposition"] = "CLAIMING_PAYMENT_IS_COMPLETED"


        if not s1_json:
            return {"error": "Stage 1 failed", "raw": res1_text}

        from datetime import datetime
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        
        # --- Stage 2 ---
        s2_input_dict = {
            "transcript": transcript,
            "disposition": s1_json.get("disposition"),
            "payment_disposition": s1_json.get("payment_disposition"),
            "current_date": current_date_str
        }
        prompt2 = s2_prompt.format(json.dumps(s2_input_dict, ensure_ascii=False), "")
        inputs2 = self.s2_tokenizer([prompt2], return_tensors="pt").to("cuda")
        outputs2 = self.s2_model.generate(
            **inputs2, max_new_tokens=256, use_cache=True, do_sample=False,
            eos_token_id=self.s2_tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Calculate Stage 2 Confidence
        s2_scores = torch.stack(outputs2.scores, dim=1)
        s2_probs = torch.softmax(s2_scores, dim=-1)
        s2_token_probs, _ = torch.max(s2_probs, dim=-1)
        s2_confidence = s2_token_probs[0].mean().item()

        res2_text = self.s2_tokenizer.decode(outputs2.sequences[0], skip_special_tokens=True)
        print(f"[DEBUG] Raw Stage 2 output: {res2_text[-200:]}")  # Last 200 chars
        s2_json = self._extract_json(res2_text)
        
        # Robustness: ensure s2_json is a dict
        if s2_json is None:
            print("[DEBUG] Stage 2 extraction failed, using empty dict")
            s2_json = {}
        
        print(f"[DEBUG] Extracted JSON: {s2_json}")


        
        # Fix date using comprehensive date mapper
        # IMPORTANT: Run the mapper UNCONDITIONALLY to catch dates the model misses
        extracted_date = s2_json.get("ptp_date") or s2_json.get("payment_date")
        
        import sys
        import os
        
        # Force reload to get latest code
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if 'utils.date_mapper' in sys.modules:
            del sys.modules['utils.date_mapper']
        from utils.date_mapper import fix_ptp_date
        
        fixed_date = fix_ptp_date(
            extracted_date, 
            current_date_str, 
            transcript
        )
        
        if fixed_date:
            print(f"[DEBUG] Date mapping success: {extracted_date} -> {fixed_date}")
            s2_json["payment_date"] = fixed_date
            # Clean up internal key
            if "ptp_date" in s2_json:
                del s2_json["ptp_date"]
        else:
            # If no date found anywhere, ensure payment_date is null
            s2_json["payment_date"] = None
            if "ptp_date" in s2_json:
                del s2_json["ptp_date"]


        # --- Smart Post-Processing Logic ---
        # Ensure all required fields are present and named correctly
        # The user specifically requested 'payment_date' and 'reason_for_not_paying'
        
        # Step 1: Handle Date Mapping (ptp_date -> payment_date)
        if "ptp_date" in s2_json:
            if not s2_json.get("payment_date"):
                s2_json["payment_date"] = s2_json["ptp_date"]
            del s2_json["ptp_date"] # Remove internal key
            
        # Step 2: Ensure all 5 keys exist
        required_keys = ["reason_for_not_paying", "ptp_amount", "payment_date", "followup_date", "remarks"]
        final_s2 = {}
        for key in required_keys:
            final_s2[key] = s2_json.get(key, None)
        
        # Step 3: Validate and Map reason_for_not_paying enums
        VALID_REASONS = [
            "BUSINESS_CLOSED", "BUSINESS_LOSS", "CLAIMING_FRAUD", "CLAIMING_PAYMENT_IS_COMPLETED",
            "CUSTOMER_EXPIRED", "CUSTOMER_NOT_TELLING_REASON", "CUSTOMER_PLANS_TO_VISIT_BRANCH",
            "DEATH_IN_FAMILY", "FUND_ISSUE", "GRIEVANCE_APP_ISSUE", "GRIEVANCE_CALLER_MISCONDUCT",
            "GRIEVANCE_FRAUD", "GRIEVANCE_LOAN_AMOUNT_DISPUTE", "JOB_CHANGED_WAITING_FOR_SALARY",
            "LOAN_CLOSURE_MISCOMMUNICATION", "LOAN_TAKEN_BY_KNOWN_PARTY", "LOST_JOB", "MEDICAL_ISSUE",
            "MULTIPLE_LOANS", "OTHER_PERSON_TAKEN", "OTHER_REASONS", "PENALTY_ISSUE", "SERVICE_ISSUE", "TRUST_ISSUE"
        ]
        
        # Synonym mapping
        SYNONYMS = {
            "JOB_LOST": "LOST_JOB",
            "JOB_CHALI_GAYI": "LOST_JOB",
            "PAISE_NAHI_HAI": "FUND_ISSUE",
            "FINANCIAL_CRISIS": "FUND_ISSUE"
        }
        
        reason = final_s2.get("reason_for_not_paying")
        if reason:
            reason_upper = str(reason).upper().replace(" ", "_").strip()
            # Try synonyms first
            if reason_upper in SYNONYMS:
                final_s2["reason_for_not_paying"] = SYNONYMS[reason_upper]
            elif reason_upper in VALID_REASONS:
                final_s2["reason_for_not_paying"] = reason_upper
            elif reason_upper in ["NULL", "NONE"]:
                final_s2["reason_for_not_paying"] = "CUSTOMER_NOT_TELLING_REASON"
            else:
                print(f"[DEBUG] Invalid reason enum: {reason}, defaulting to OTHER_REASONS")
                final_s2["reason_for_not_paying"] = "OTHER_REASONS"
        else:
            final_s2["reason_for_not_paying"] = "CUSTOMER_NOT_TELLING_REASON"

        # Step 4: Global PTP Override
        # If Stage 2 extracted valid PTP details, force Stage 1 to PTP
        if final_s2.get("ptp_amount") or final_s2.get("payment_date"):
            s1_json["payment_disposition"] = "PTP"
            
        # Update s2_json for return
        s2_json = final_s2

        # --- Confidence Metadata ---
        overall_confidence = (s1_confidence + s2_confidence) / 2
        
        metadata = {
            "overall_confidence": round(overall_confidence, 4),
            "stage_confidences": {
                "stage1": round(s1_confidence, 4),
                "stage2": round(s2_confidence, 4)
            }
        }

        # --- Merge ---
        final_result = {
            "stage1": s1_json,
            "stage2": s2_json if s2_json else {"error": "Stage 2 failed", "raw": res2_text},
            "metadata": metadata
        }
        return final_result

    def _extract_json(self, text):
        try:
            if "### Response:" in text:
                json_part = text.split("### Response:")[1].strip()
            else:
                json_part = text.strip()

            for token in ["<|im_end|>", "###", "</s>"]:
                json_part = json_part.split(token)[0].strip()

            json_start = json_part.find('{')
            json_end = json_part.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                return json.loads(json_part[json_start:json_end])
            return None
        except:
            return None

if __name__ == "__main__":
    print("\n--- Ringg Unified Pipeline Test ---")
    
    if len(sys.argv) > 1:
        transcript = sys.argv[1]
    else:
        print("\nEnter transcript:")
        transcript = sys.stdin.readline().strip()
        
    if not transcript:
        print("Error: No transcript entered.")
        sys.exit(1)
        
    pipeline = RinggPipeline()
    result = pipeline.run_inference(transcript)
    
    print("\n" + "="*50)
    print(json.dumps(result, indent=4))
    print("="*50)
