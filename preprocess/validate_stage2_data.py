"""
Validate Stage-2 Training Data

Ensures data quality before training:
1. Stage-1 labels (disposition, payment_disposition) are present and valid
2. Stage-2 eligibility criteria are met
3. All enum values match the locked schema

If validation fails, the sample is REMOVED from training data.
"""

import json
from collections import Counter

INPUT_FILE = "data/stage2_train_cleaned.json"
OUTPUT_FILE = "data/stage2_train_validated.json"
REJECTED_FILE = "data/stage2_rejected_samples.json"

# Valid Stage-1 enums
VALID_DISPOSITIONS = {
    "ANSWERED", "CALL_BACK_LATER", "ANSWERED_BY_FAMILY_MEMBER", 
    "CALL_DISCONNECTED_BY_CUSTOMER", "ANSWERED_DISCONNECTED", 
    "ANSWERED_VOICE_ISSUE", "SILENCE_ISSUE", "LANGUAGE_BARRIER", 
    "AUTOMATED_VOICE", "WRONG_NUMBER", "CUSTOMER_ABUSIVE", 
    "AGENT_BUSY_ON_ANOTHER_CALL", "FORWARDED_CALL", "RINGING", 
    "RINGING_DISCONNECTED", "WILL_ASK_TO_PAY", "DO_NOT_KNOW_THE_PERSON", 
    "OTHERS", "BUSY", "SWITCHED_OFF", "NOT_IN_CONTACT_ANYMORE", "CUSTOMER_PICKED"
}

VALID_PAYMENT_DISPOSITIONS = {
    "PTP", "PAID", "SETTLEMENT", "PARTIAL_PAYMENT", 
    "NO_PAYMENT_COMMITMENT", "DENIED_TO_PAY", "NO_PROOF_GIVEN", 
    "WILL_PAY_AFTER_VISIT", "WANT_FORECLOSURE", "WANTS_TO_RENEGOTIATE_LOAN_TERMS",
    "None", None
}

# Stage-2 eligibility
STAGE2_ELIGIBLE_DISPOSITIONS = {
    "ANSWERED",
    "ANSWERED_BY_FAMILY_MEMBER",
    "CALL_BACK_LATER"
}

# Locked Stage-2 reason enum (20 values)
VALID_REASONS = {
    'JOB_CHANGED_WAITING_FOR_SALARY', 'LOST_JOB', 'BUSINESS_LOSS', 'BUSINESS_CLOSED', 
    'FINANCIAL_DIFFICULTY', 'MULTIPLE_LOANS', 'MEDICAL_ISSUE', 'DEATH_IN_FAMILY', 
    'FAMILY_ISSUE', 'CLAIMING_PAYMENT_IS_COMPLETED', 'CLAIMING_FRAUD', 'GRIEVANCE_FRAUD',
    'GRIEVANCE_LOAN_AMOUNT_DISPUTE', 'PENALTY_ISSUE', 'SERVICE_ISSUE', 
    'LOAN_CLOSURE_MISCOMMUNICATION', 'LOAN_TAKEN_BY_KNOWN_PARTY', 'OUT_OF_STATION',
    'CUSTOMER_NOT_TELLING_REASON', 'OTHER_REASONS', None
}

def validate_stage2_data():
    print(f"Loading cleaned data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    valid_samples = []
    rejected_samples = []
    rejection_reasons = Counter()
    
    for idx, item in enumerate(data):
        input_data = item.get('input', {})
        output_data = item.get('output', {})
        
        disposition = input_data.get('disposition')
        payment_disposition = input_data.get('payment_disposition')
        reason = output_data.get('reason_for_not_paying')
        
        rejection_reason = None
        
        # Validation 1: Stage-1 labels present
        if not disposition:
            rejection_reason = "missing_disposition"
        elif not payment_disposition:
            rejection_reason = "missing_payment_disposition"
        
        # Validation 2: Stage-1 labels valid
        elif disposition not in VALID_DISPOSITIONS:
            rejection_reason = f"invalid_disposition: {disposition}"
        elif payment_disposition not in VALID_PAYMENT_DISPOSITIONS:
            rejection_reason = f"invalid_payment_disposition: {payment_disposition}"
        
        # Validation 3: Stage-2 eligibility
        elif disposition not in STAGE2_ELIGIBLE_DISPOSITIONS:
            rejection_reason = f"not_stage2_eligible_disposition: {disposition}"
        elif payment_disposition in ["None", None]:
            rejection_reason = "payment_disposition_is_none"
        
        # Validation 4: Stage-2 reason enum
        elif reason and reason not in VALID_REASONS:
            rejection_reason = f"invalid_reason_enum: {reason}"
        
        # Accept or reject
        if rejection_reason:
            rejected_samples.append({
                "index": idx,
                "reason": rejection_reason,
                "sample": item
            })
            rejection_reasons[rejection_reason] += 1
        else:
            valid_samples.append(item)
    
    # Save valid samples
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(valid_samples, f, indent=2, ensure_ascii=False)
    
    # Save rejected samples
    with open(REJECTED_FILE, 'w', encoding='utf-8') as f:
        json.dump(rejected_samples, f, indent=2, ensure_ascii=False)
    
    # Report
    print("\n" + "="*60)
    print("VALIDATION RESULTS:")
    print("="*60)
    print(f"✅ Valid samples:    {len(valid_samples):6d} ({len(valid_samples)/len(data)*100:5.1f}%)")
    print(f"❌ Rejected samples: {len(rejected_samples):6d} ({len(rejected_samples)/len(data)*100:5.1f}%)")
    
    if rejection_reasons:
        print("\n" + "="*60)
        print("REJECTION BREAKDOWN:")
        print("="*60)
        for reason, count in rejection_reasons.most_common():
            print(f"  {reason:50s} {count:6d}")
    
    print("\n" + "="*60)
    print("FILES SAVED:")
    print("="*60)
    print(f"✅ Valid:    {OUTPUT_FILE}")
    print(f"❌ Rejected: {REJECTED_FILE}")
    
    if len(rejected_samples) > 0:
        print("\n⚠️  WARNING: Some samples were rejected. Review rejected_samples.json")
    else:
        print("\n✅ All samples passed validation!")
    
    return len(valid_samples), len(rejected_samples)

if __name__ == "__main__":
    validate_stage2_data()
