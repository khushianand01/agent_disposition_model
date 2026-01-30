import json
import random
import os
import re
from collections import Counter

# Configuration
INPUT_FILE = 'data/calls_data.json'
BOOST_FILE = 'data/failed_cases_boost.json'
OUTPUT_FILE = 'data/calls_data_v11_master.json'

# Filters
MIN_WORDS = 15

# Balancing Targets (Overfitting Prevention)
# Majorities = 100% Unique. Minorities = Max 5x Replication.
UNIQUE_THRESHOLD = 100
MINORITY_REPLICATION_CAP = 5
MAX_CLASS_CAP = 1200 # Lower cap for even better balance

# ENUMS
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

def clean_and_balance_v11():
    print(f"Loading raw data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 1. Cleaning & Enum Standardization
    cleaned_data = []
    for item in data:
        transcript = item.get('input', '')
        output = item.get('output' , {})
        if not output: continue
        
        disp = str(output.get('disposition')).upper()
        p_disp = str(output.get('payment_disposition'))
        if p_disp != "None" and p_disp is not None:
             p_disp = p_disp.upper() if p_disp else None
        
        # Filter and Standardize
        if len(str(transcript).split()) >= MIN_WORDS and disp in VALID_DISPOSITIONS:
            # Enforce 4-key schema
            item['output'] = {
                "disposition": disp,
                "payment_disposition": p_disp if p_disp in VALID_PAYMENT_DISPOSITIONS else "None",
                "ptp_amount": output.get("ptp_amount"),
                "ptp_date": output.get("ptp_date")
            }
            cleaned_data.append(item)
    
    # 2. Quality Audit: Filter Lazy PTP
    amount_pattern = re.compile(r'(₹|rs|rupees|amount|pay)\s*\d+', re.I)
    date_pattern = re.compile(r'(\d{1,2}(st|nd|rd|th)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|date))', re.I)
    
    final_cleaned = []
    for item in cleaned_data:
        is_ptp = item['output']['payment_disposition'] == 'PTP'
        if is_ptp:
            t = item['input']
            o = item['output']
            if ((o['ptp_amount'] is None and amount_pattern.search(t)) or 
                (o['ptp_date'] is None and date_pattern.search(t))):
                continue
        final_cleaned.append(item)
    
    print(f"Cleaned unique samples: {len(final_cleaned)}")

    # 3. Group by Joint Key
    grouped_joint = {}
    for item in final_cleaned:
        joint_key = f"{item['output']['disposition']} | {item['output']['payment_disposition']}"
        if joint_key not in grouped_joint: grouped_joint[joint_key] = []
        grouped_joint[joint_key].append(item)

    # 4. Balancing with Overfitting Prevention
    balanced_data = []
    print("\n--- Balancing (v11) ---")
    for joint_key, samples in grouped_joint.items():
        count = len(samples)
        if count < 3: continue # Remove noise

        if count > MAX_CLASS_CAP:
            processed = random.sample(samples, MAX_CLASS_CAP)
            print(f"  - {joint_key}: {count} -> {MAX_CLASS_CAP} (Unique Downsample)")
        elif count >= UNIQUE_THRESHOLD:
            processed = samples
            print(f"  - {joint_key}: {count} -> {count} (Unique No Dups)")
        else:
            # Limited Replication
            target = min(UNIQUE_THRESHOLD, count * MINORITY_REPLICATION_CAP)
            processed = (samples * (target // count + 1))[:target]
            print(f"  - {joint_key}: {count} -> {len(processed)} (Max {MINORITY_REPLICATION_CAP}x Rep)")
        balanced_data.extend(processed)

    # 5. Error Boosting (Interleaved)
    if os.path.exists(BOOST_FILE):
        with open(BOOST_FILE, 'r') as f:
            boost_data = json.load(f)
        print(f"\nAdding {len(boost_data)} failure cases (2x visibility)...")
        balanced_data.extend(boost_data * 2)

    random.shuffle(balanced_data)
    
    # Final check for Unicode Safety & Export
    print(f"\nFinal Records: {len(balanced_data)}")
    unique_tx = len(set(i['input'] for i in balanced_data))
    print(f"Duplicate Ratio: {((len(balanced_data)-unique_tx)/len(balanced_data))*100:.1f}%")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # ensure_ascii=True fixes the "unicode error" in notebooks
        json.dump(balanced_data, f, indent=2, ensure_ascii=True)
    
    print(f"Success! Master file saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    random.seed(42)
    clean_and_balance_v11()
