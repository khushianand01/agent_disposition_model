"""
Stage-2 Prompt Templates

These templates define the prompts for Stage-2 inference (full 7-field extraction).
"""

# ============================================================================
# QWEN STAGE-2 PROMPT
# ============================================================================

QWEN_STAGE2_SYSTEM_PROMPT = """You are an expert call transcript analyzer for a loan collection agency. Your task is to extract structured information from Hindi/English call transcripts.

Extract the following 7 fields in valid JSON format:

1. **disposition** - Call outcome (ANSWERED, CALL_BACK_LATER, RINGING, etc.)
2. **payment_disposition** - Payment commitment (PTP, PAID, NO_PAYMENT_COMMITMENT, etc.)
3. **reason_for_not_paying** - Why customer cannot pay (use ONLY these values):
   - Financial: JOB_CHANGED_WAITING_FOR_SALARY, LOST_JOB, BUSINESS_LOSS, BUSINESS_CLOSED, FINANCIAL_DIFFICULTY, MULTIPLE_LOANS
   - Medical/Family: MEDICAL_ISSUE, DEATH_IN_FAMILY, FAMILY_ISSUE
   - Disputes: CLAIMING_PAYMENT_IS_COMPLETED, CLAIMING_FRAUD, GRIEVANCE_FRAUD, GRIEVANCE_LOAN_AMOUNT_DISPUTE, PENALTY_ISSUE
   - Service: SERVICE_ISSUE, LOAN_CLOSURE_MISCOMMUNICATION, LOAN_TAKEN_BY_KNOWN_PARTY
   - Other: OUT_OF_STATION, CUSTOMER_NOT_TELLING_REASON, OTHER_REASONS
4. **ptp_amount** - Promised payment amount (numeric, null if not mentioned)
5. **ptp_date** - Promised payment date (YYYY-MM-DD format, null if not mentioned)
6. **followup_date** - Next followup date (YYYY-MM-DD format, null if not mentioned)
7. **remarks** - Any additional important notes (null if none)

CRITICAL RULES:
- Return ONLY valid JSON, no explanations
- Use null for missing/unclear values, NEVER hallucinate
- Dates MUST be in YYYY-MM-DD format
- Convert relative dates: "kal" -> tomorrow's date, "next Monday" -> actual date
- Convert Hindi numbers: "teen hazaar" -> 3000
- If customer doesn't give a reason, use CUSTOMER_NOT_TELLING_REASON
- If reason doesn't fit any category, use OTHER_REASONS
- Remarks should be brief and factual, null if nothing important"""

QWEN_STAGE2_USER_TEMPLATE = """Transcript:
{transcript}

Extract the 7 fields as JSON:"""


# ============================================================================
# RINGG STAGE-2 INSTRUCTION
# ============================================================================

RINGG_STAGE2_INSTRUCTION = """Analyze the following call transcript and extract structured information. Return ONLY valid JSON with these 7 fields:

1. disposition - Call outcome
2. payment_disposition - Payment commitment
3. reason_for_not_paying - Why customer cannot pay (use enum: JOB_CHANGED_WAITING_FOR_SALARY, LOST_JOB, BUSINESS_LOSS, BUSINESS_CLOSED, FINANCIAL_DIFFICULTY, MULTIPLE_LOANS, MEDICAL_ISSUE, DEATH_IN_FAMILY, FAMILY_ISSUE, CLAIMING_PAYMENT_IS_COMPLETED, CLAIMING_FRAUD, GRIEVANCE_FRAUD, GRIEVANCE_LOAN_AMOUNT_DISPUTE, PENALTY_ISSUE, SERVICE_ISSUE, LOAN_CLOSURE_MISCOMMUNICATION, LOAN_TAKEN_BY_KNOWN_PARTY, OUT_OF_STATION, CUSTOMER_NOT_TELLING_REASON, OTHER_REASONS)
4. ptp_amount - Promised amount (number or null)
5. ptp_date - Promised date (YYYY-MM-DD or null)
6. followup_date - Next followup (YYYY-MM-DD or null)
7. remarks - Additional notes (string or null)

RULES:
- Return valid JSON only
- Use null for missing values
- Convert relative dates to YYYY-MM-DD
- Convert Hindi numbers to numeric values
- Never hallucinate information"""


# ============================================================================
# EXAMPLE OUTPUTS (for reference)
# ============================================================================

STAGE2_EXAMPLE_1 = {
    "disposition": "ANSWERED",
    "payment_disposition": "PTP",
    "reason_for_not_paying": "JOB_CHANGED_WAITING_FOR_SALARY",
    "ptp_amount": 5000,
    "ptp_date": "2024-02-05",
    "followup_date": "2024-02-04",
    "remarks": "Customer waiting for salary on 5th, will pay full amount"
}

STAGE2_EXAMPLE_2 = {
    "disposition": "ANSWERED",
    "payment_disposition": "NO_PAYMENT_COMMITMENT",
    "reason_for_not_paying": "CLAIMING_PAYMENT_IS_COMPLETED",
    "ptp_amount": None,
    "ptp_date": None,
    "followup_date": None,
    "remarks": "Customer claims payment done via UPI, no proof provided"
}

STAGE2_EXAMPLE_3 = {
    "disposition": "ANSWERED_BY_FAMILY_MEMBER",
    "payment_disposition": "None",
    "reason_for_not_paying": None,
    "ptp_amount": None,
    "ptp_date": None,
    "followup_date": "2024-02-01",
    "remarks": "Spoke to wife, customer will call back tomorrow"
}


# ============================================================================
# VALIDATION SCHEMA (for post-processing)
# ============================================================================

STAGE2_SCHEMA = {
    "type": "object",
    "required": ["disposition", "payment_disposition", "reason_for_not_paying", 
                 "ptp_amount", "ptp_date", "followup_date", "remarks"],
    "properties": {
        "disposition": {"type": "string"},
        "payment_disposition": {"type": ["string", "null"]},
        "reason_for_not_paying": {"type": ["string", "null"]},
        "ptp_amount": {"type": ["number", "null"]},
        "ptp_date": {"type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
        "followup_date": {"type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
        "remarks": {"type": ["string", "null"]}
    }
}


if __name__ == "__main__":
    print("=== QWEN STAGE-2 PROMPT ===")
    print(QWEN_STAGE2_SYSTEM_PROMPT)
    print("\n" + "="*60 + "\n")
    print(QWEN_STAGE2_USER_TEMPLATE.format(transcript="[Sample transcript here]"))
    
    print("\n\n=== RINGG STAGE-2 INSTRUCTION ===")
    print(RINGG_STAGE2_INSTRUCTION)
    
    print("\n\n=== EXAMPLE OUTPUTS ===")
    import json
    print("Example 1 (PTP):")
    print(json.dumps(STAGE2_EXAMPLE_1, indent=2))
    print("\nExample 2 (Dispute):")
    print(json.dumps(STAGE2_EXAMPLE_2, indent=2))
    print("\nExample 3 (Family Member):")
    print(json.dumps(STAGE2_EXAMPLE_3, indent=2))
