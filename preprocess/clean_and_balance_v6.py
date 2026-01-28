import json
import random
import os
from collections import Counter

# Configuration
INPUT_FILE = 'data/calls_data.json'
OUTPUT_FILE = 'data/calls_data_balanced_v6.json'

# Filters
MIN_WORDS = 15

# Balancing Caps (Tiered)
# Tier 1: Non-Payment Dispositions (None payment_disposition)
DISP_CAP_FOR_NONE_PAYMENT = 1000

# Tier 2: Result-Oriented Dispositions
PTP_CAP = 2500
MAJOR_CLASS_CAP = 2500 # For NO_PAYMENT_COMMITMENT and DENIED_TO_PAY
MINORITY_CLASS_CAP = 1000 # For PARTIAL_PAYMENT, SETTLEMENT

def clean_and_balance():
    print(f"Loading data from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_initial = len(data)
    print(f"Initial samples: {total_initial}")

    # 1. Cleaning
    cleaned_data = []
    print(f"Cleaning data (Filters: words >= {MIN_WORDS}, non-null disposition)...")
    for item in data:
        transcript = item.get('input', '')
        output = item.get('output' , {})
        disposition = output.get('disposition') if output else None

        # Word count check
        word_count = len(str(transcript).split())
        
        if word_count >= MIN_WORDS and disposition is not None:
            cleaned_data.append(item)
    
    total_cleaned = len(cleaned_data)
    print(f"Cleaned samples: {total_cleaned} (Removed {total_initial - total_cleaned})")

    # 2. Balancing
    # Shuffle for fair sampling
    random.seed(42)
    random.shuffle(cleaned_data)

    # Tier 1: Payment Disposition is None
    none_payment_data = [item for item in cleaned_data if item['output'].get('payment_disposition') is None]
    
    grouped_none = {}
    for item in none_payment_data:
        disp = str(item['output'].get('disposition'))
        if disp not in grouped_none:
            grouped_none[disp] = []
        grouped_none[disp].append(item)
    
    balanced_none = []
    print("\nBalancing 'None' payment disposition (Tier 1)...")
    for disp, samples in grouped_none.items():
        capped_samples = samples[:DISP_CAP_FOR_NONE_PAYMENT]
        print(f"  - {disp}: {len(samples)} -> {len(capped_samples)}")
        balanced_none.extend(capped_samples)
    
    # Tier 2: Payment Disposition is NOT None
    result_payment_data = [item for item in cleaned_data if item['output'].get('payment_disposition') is not None]
    
    grouped_results = {}
    for item in result_payment_data:
        p_disp = str(item['output'].get('payment_disposition'))
        if p_disp not in grouped_results:
            grouped_results[p_disp] = []
        grouped_results[p_disp].append(item)
    
    balanced_results = []
    print("\nBalancing Result-Oriented dispositions (Tier 2)...")
    for p_disp, samples in grouped_results.items():
        if p_disp == "PTP":
            cap = PTP_CAP
        elif p_disp in ["NO_PAYMENT_COMMITMENT", "DENIED_TO_PAY"]:
            cap = MAJOR_CLASS_CAP
        elif p_disp in ["PARTIAL_PAYMENT", "SETTLEMENT"]:
            cap = MINORITY_CLASS_CAP
        else:
            cap = None # Keep all
        
        capped_samples = samples[:cap] if cap else samples
        print(f"  - {p_disp}: {len(samples)} -> {len(capped_samples)}")
        balanced_results.extend(capped_samples)
    
    # Final Merge and Shuffle
    final_data = balanced_none + balanced_results
    random.shuffle(final_data)
    
    print(f"\nFinal Balanced Dataset size: {len(final_data)}")
    
    # Saving
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    print("Success!")

if __name__ == "__main__":
    clean_and_balance()
