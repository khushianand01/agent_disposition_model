import json
from collections import Counter

MASTER_FILE = "/home/ubuntu/disposition_model/data/master_production_data.json"

CALL_DISPOSITIONS = [
    "ANSWERED", "ANSWERED_BY_FAMILY_MEMBER", "CUSTOMER_PICKED", "AGENT_BUSY_ON_ANOTHER_CALL",
    "SILENCE_ISSUE", "LANGUAGE_BARRIER", "ANSWERED_VOICE_ISSUE", "CUSTOMER_ABUSIVE",
    "AUTOMATED_VOICE", "FORWARDED_CALL", "RINGING", "BUSY", "SWITCHED_OFF",
    "WRONG_NUMBER", "DO_NOT_KNOW_THE_PERSON", "NOT_IN_CONTACT_ANYMORE", "OUT_OF_NETWORK", "OUT_OF_SERVICES",
    "CALL_BACK_LATER", "WILL_ASK_TO_PAY", "GAVE_ALTERNATE_NUMBER",
    "ANSWERED_DISCONNECTED", "CALL_DISCONNECTED_BY_CUSTOMER", "NOT_AVAILABLE", "WRONG_PERSON", "OTHERS"
]

PAYMENT_DISPOSITIONS = [
    "PAID", "PTP", "PARTIAL_PAYMENT", "SETTLEMENT", "WILL_PAY_AFTER_VISIT",
    "DENIED_TO_PAY", "NO_PAYMENT_COMMITMENT", "NO_PROOF_GIVEN", "WANT_FORECLOSURE", "WANTS_TO_RENEGOTIATE_LOAN_TERMS",
    "None"
]

REASONS = [
    "FUNDS_ISSUE", "TECHNICAL_ISSUE", "HEALTH_ISSUE", "SALARY_NOT_CREDITED", 
    "LOAN_ALREADY_PAID", "DISPUTE_ON_INTEREST", "CUSTOMER_OUT_OF_STATION",
    "CUSTOMER_NOT_TELLING_REASON", "JOB_CHANGED_WAITING_FOR_SALARY", "SERVICE_ISSUE", "OTHER_REASONS",
    None
]

def audit():
    print(f"Auditing {MASTER_FILE}...")
    try:
        with open(MASTER_FILE, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"FAILED to read master file: {e}")
        return

    total = len(data)
    schema_errors = 0
    invalid_call_disp = []
    invalid_pay_disp = []
    invalid_reason = []
    missing_ptp_details = 0
    
    for item in data:
        output = item.get('output', {})
        
        # 1. Schema Check
        required_fields = ["disposition", "payment_disposition", "reason_for_not_paying", "ptp_details", "remarks"]
        if not all(f in output for f in required_fields):
            schema_errors += 1
            continue
            
        # 2. Label Validity
        if output['disposition'] not in CALL_DISPOSITIONS:
            invalid_call_disp.append(output['disposition'])
            
        if output['payment_disposition'] not in PAYMENT_DISPOSITIONS:
            invalid_pay_disp.append(output['payment_disposition'])
            
        if output['reason_for_not_paying'] not in REASONS:
            invalid_reason.append(output['reason_for_not_paying'])
            
        # 3. PTP Details Check
        ptp = output.get('ptp_details', {})
        if not isinstance(ptp, dict) or "amount" not in ptp or "date" not in ptp:
            missing_ptp_details += 1

    print(f"\n--- Audit Results ---")
    print(f"Total Samples: {total}")
    print(f"Schema Consistent: {total - schema_errors}/{total}")
    print(f"Invalid Call Dispos: {len(invalid_call_disp)} (Top: {Counter(invalid_call_disp).most_common(3)})")
    print(f"Invalid Pay Dispos: {len(invalid_pay_disp)} (Top: {Counter(invalid_pay_disp).most_common(3)})")
    print(f"Invalid Reasons: {len(invalid_reason)} (Top: {Counter(invalid_reason).most_common(3)})")
    print(f"PTP Details Valid: {total - missing_ptp_details}/{total}")
    
    if schema_errors == 0 and not invalid_call_disp and not invalid_pay_disp:
        print("\nSUMMARY: Data is 100% compliant with production labels and schema.")
    else:
        print("\nSUMMARY: Issues found. Needs fixing.")

if __name__ == "__main__":
    audit()
