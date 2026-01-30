import json
import random
import os
import re
from collections import Counter

# Configuration
INPUT_FILE = 'data/calls_data.json'
OUTPUT_FILE = 'data/calls_data_v11_s2_master.json'

# Filters
MIN_WORDS = 15

# Balancing
MAX_CLASS_CAP = 1200
UNIQUE_THRESHOLD = 100
MINORITY_REPLICATION_CAP = 5

# Stage-2 ENUMS (LOCKED)
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

# NEW: Stage-2 reason_for_not_paying enum (20 values)
VALID_REASONS = {
    # Financial (6)
    "JOB_CHANGED_WAITING_FOR_SALARY", "LOST_JOB", "BUSINESS_LOSS", 
    "BUSINESS_CLOSED", "FINANCIAL_DIFFICULTY", "MULTIPLE_LOANS",
    # Medical/Family (3)
    "MEDICAL_ISSUE", "DEATH_IN_FAMILY", "FAMILY_ISSUE",
    # Disputes (5)
    "CLAIMING_PAYMENT_IS_COMPLETED", "CLAIMING_FRAUD", "GRIEVANCE_FRAUD",
    "GRIEVANCE_LOAN_AMOUNT_DISPUTE", "PENALTY_ISSUE",
    # Service (3)
    "SERVICE_ISSUE", "LOAN_CLOSURE_MISCOMMUNICATION", "LOAN_TAKEN_BY_KNOWN_PARTY",
    # Other (3)
    "OUT_OF_STATION", "CUSTOMER_NOT_TELLING_REASON", "OTHER_REASONS",
    "None", None
}

def standardize_reason(reason):
    """Map old reason values to new 20-value enum"""
    if not reason or str(reason).strip().lower() == 'none':
        return None
    
    reason = str(reason).upper()
    
    # Direct matches
    if reason in VALID_REASONS:
        return reason
    
    # Mapping old values to new enum
    mappings = {
        "CUSTOMER_EXPIRED": "DEATH_IN_FAMILY",
        "LOAN_TAKEN_BY_KNOWN_PARTY": "LOAN_TAKEN_BY_KNOWN_PARTY",
        "GRIEVANCE_CALLER_MISCONDUCT": "SERVICE_ISSUE",
        "GRIEVANCE_APP_ISSUE": "SERVICE_ISSUE",
        "AWAITING_STATEMENT": "SERVICE_ISSUE",
        "TRUST_ISSUE": "SERVICE_ISSUE",
        "FUND_ISSUE": "FINANCIAL_DIFFICULTY",
    }
    
    if reason in mappings:
        return mappings[reason]
    
    # Default to OTHER_REASONS for unmapped values
    return "OTHER_REASONS"

def validate_date(date_str):
    """Validate YYYY-MM-DD format"""
    if not date_str:
        return None
    try:
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if re.match(pattern, str(date_str)):
            return str(date_str)
    except:
        pass
    return None

def validate_amount(amount):
    """Validate numeric amount"""
    if amount is None:
        return None
    try:
        val = float(amount)
        return val if val > 0 else None
    except:
        return None

def clean_and_balance_s2():
    print(f"Loading raw data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. Cleaning & Stage-2 Schema Enforcement
    cleaned_data = []
    for item in data:
        transcript = item.get('input', '')
        output = item.get('output', {})
        if not output:
            continue
        
        disp = str(output.get('disposition')).upper()
        p_disp = str(output.get('payment_disposition'))
        if p_disp != "None" and p_disp is not None:
            p_disp = p_disp.upper() if p_disp else None
        
        # Filter by minimum words and valid disposition
        if len(str(transcript).split()) >= MIN_WORDS and disp in VALID_DISPOSITIONS:
            # Stage-2: 7-field schema
            item['output'] = {
                "disposition": disp,
                "payment_disposition": p_disp if p_disp in VALID_PAYMENT_DISPOSITIONS else "None",
                "reason_for_not_paying": standardize_reason(output.get("reason_for_not_paying")),
                "ptp_amount": validate_amount(output.get("ptp_amount")),
                "ptp_date": validate_date(output.get("ptp_date")),
                "followup_date": validate_date(output.get("followup_date")),
                "remarks": output.get("remarks") if output.get("remarks") else None
            }
            cleaned_data.append(item)
    
    print(f"Cleaned samples: {len(cleaned_data)}")
    
    # 2. Group by Joint Key (disposition + payment_disposition)
    grouped_joint = {}
    for item in cleaned_data:
        joint_key = f"{item['output']['disposition']} | {item['output']['payment_disposition']}"
        if joint_key not in grouped_joint:
            grouped_joint[joint_key] = []
        grouped_joint[joint_key].append(item)
    
    # 3. Balancing
    balanced_data = []
    print("\n--- Balancing (Stage-2) ---")
    for joint_key, samples in grouped_joint.items():
        count = len(samples)
        if count < 3:
            continue
        
        if count > MAX_CLASS_CAP:
            processed = random.sample(samples, MAX_CLASS_CAP)
            print(f"  - {joint_key}: {count} -> {MAX_CLASS_CAP} (Capped)")
        elif count >= UNIQUE_THRESHOLD:
            processed = samples
            print(f"  - {joint_key}: {count} -> {count} (Unique)")
        else:
            target = min(UNIQUE_THRESHOLD, count * MINORITY_REPLICATION_CAP)
            processed = (samples * (target // count + 1))[:target]
            print(f"  - {joint_key}: {count} -> {len(processed)} (Replicated)")
        
        balanced_data.extend(processed)
    
    random.shuffle(balanced_data)
    
    # 4. Statistics
    print(f"\nFinal Records: {len(balanced_data)}")
    unique_tx = len(set(i['input'] for i in balanced_data))
    print(f"Duplicate Ratio: {((len(balanced_data)-unique_tx)/len(balanced_data))*100:.1f}%")
    
    # Reason distribution
    reasons = [item['output']['reason_for_not_paying'] for item in balanced_data if item['output']['reason_for_not_paying']]
    reason_counts = Counter(reasons)
    print(f"\nTop 10 Reasons:")
    for reason, count in reason_counts.most_common(10):
        print(f"  {reason}: {count}")
    
    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, indent=2, ensure_ascii=True)
    
    print(f"\nSuccess! Master file saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    random.seed(42)
    clean_and_balance_s2()
