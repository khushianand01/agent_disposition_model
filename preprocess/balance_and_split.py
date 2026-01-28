import json
import random
from collections import Counter
import os

# File paths
INPUT_FILE = 'data/calls_data.json'
OUTPUT_BALANCED_V5 = 'data/calls_data_balanced_v5.json'

# Balancing Caps
DISP_CAP_FOR_NONE_PAYMENT = 1000
PTP_CAP = 2500
MAJOR_CLASS_CAP = 2500 # For NO_PAYMENT_COMMITMENT and DENIED_TO_PAY
MINORITY_CLASS_CAP = 1000 # For P/S (synthetic + real > 1000)

def main():
    # ... (loading code is above) ...
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loading synthetic data from data/synthetic_data.json...")
    try:
        with open('data/synthetic_data.json', 'r', encoding='utf-8') as f:
            syn_data = json.load(f)
            data.extend(syn_data)
            print(f"Merged {len(syn_data)} synthetic samples.")
    except Exception as e:
        print(f"Warning: Could not load synthetic data: {e}")
    
    print(f"Total samples: {len(data)}")
    
    # Shuffle for randomness
    random.seed(42)
    random.shuffle(data)
    
    # Tier 1: Process Payment Disposition = None (Failure cases)
    # Group by underlying 'disposition' to preserve diversity in failure reasons
    none_payment_data = [item for item in data if item['output'].get('payment_disposition') is None]
    
    grouped_none = {}
    for item in none_payment_data:
        disp = str(item['output'].get('disposition'))
        if disp not in grouped_none:
            grouped_none[disp] = []
        grouped_none[disp].append(item)
    
    balanced_none = []
    print("\nBalancing 'None' payment disposition by sub-dispositions...")
    for disp, samples in grouped_none.items():
        capped_samples = samples[:DISP_CAP_FOR_NONE_PAYMENT]
        print(f"- {disp}: {len(samples)} -> {len(capped_samples)}")
        balanced_none.extend(capped_samples)
    
    # Tier 2: Process Result-Oriented Classes (Payment Disposition != None)
    result_payment_data = [item for item in data if item['output'].get('payment_disposition') is not None]
    
    grouped_results = {}
    for item in result_payment_data:
        p_disp = str(item['output'].get('payment_disposition'))
        if p_disp not in grouped_results:
            grouped_results[p_disp] = []
        grouped_results[p_disp].append(item)
    
    balanced_results = []
    print("\nBalancing result-oriented payment dispositions...")
    for p_disp, samples in grouped_results.items():
        if p_disp == "PTP":
            capped_samples = samples[:PTP_CAP]
        elif p_disp in ["NO_PAYMENT_COMMITMENT", "DENIED_TO_PAY"]:
            capped_samples = samples[:MAJOR_CLASS_CAP]
        elif p_disp in ["PARTIAL_PAYMENT", "SETTLEMENT"]:
             # These now have ~1200+ samples (real + synthetic). Cap them to ensure they don't over-dominate if generated too much.
             capped_samples = samples[:MINORITY_CLASS_CAP]
        else:
            # Keep all others (PAID, NO_PROOF_GIVEN etc.)
            capped_samples = samples
        print(f"- {p_disp}: {len(samples)} -> {len(capped_samples)}")
        balanced_results.extend(capped_samples)
    
    # Final Merge and Shuffle
    final_balanced = balanced_none + balanced_results
    random.shuffle(final_balanced)
    
    print(f"\nFinal v5 Balanced Dataset size: {len(final_balanced)}")
    
    # Saving
    print(f"Saving to {OUTPUT_BALANCED_V5}...")
    with open(OUTPUT_BALANCED_V5, 'w', encoding='utf-8') as f:
        json.dump(final_balanced, f, indent=2, ensure_ascii=False)
    
    # Cleanup old balanced/split files
    print("Cleaning up old files...")
    for f in ["data/calls_data_balanced.json", "data/calls_data_balanced_v2.json", 
              "data/train_production_v3_balanced.json", "data/val_production_v3.json", "data/test_production_v3.json"]:
        if os.path.exists(f): 
            os.remove(f)
            print(f"Deleted {f}")

    print("Success! v3 Balancing complete.")

if __name__ == "__main__":
    main()
