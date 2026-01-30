import json
import random
import os
from collections import defaultdict

# Configuration
INPUT_FILE = 'data/splits/test_v11_s1.json'
OUTPUT_FILE = 'data/baseline_test.json'
SAMPLE_SIZE = 400

def create_baseline_test():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading test split from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Stratified Sampling logic
    strata = defaultdict(list)
    for item in test_data:
        try:
            out = json.loads(item['output'])
            disp = out.get('disposition', 'None')
            p_disp = out.get('payment_disposition', 'None')
            
            if disp == "ANSWERED" and p_disp != "None":
                cat = "ANSWERED_WITH_PAYMENT"
            elif disp in ["RINGING", "RINGING_DISCONNECTED", "BUSY", "SWITCHED_OFF"]:
                cat = "NON_CONNECTED"
            elif disp == "LANGUAGE_BARRIER":
                cat = "LANGUAGE_BARRIER"
            elif disp == "ANSWERED_BY_FAMILY_MEMBER":
                cat = "FAMILY_MEMBER"
            else:
                cat = "OTHER"
            strata[cat].append(item)
        except:
            strata["OTHER"].append(item)

    eval_data = []
    random.seed(42)
    
    # Target counts to ensure representation
    target_per_cat = SAMPLE_SIZE // 4
    for cat, samples in strata.items():
        if cat == "OTHER": continue
        random.shuffle(samples)
        selected = samples[:target_per_cat]
        eval_data.extend(selected)
        print(f"  - {cat}: selected {len(selected)} samples")
    
    remaining = SAMPLE_SIZE - len(eval_data)
    if remaining > 0:
        other_samples = strata["OTHER"]
        random.shuffle(other_samples)
        selected_other = other_samples[:remaining]
        eval_data.extend(selected_other)
        print(f"  - OTHER: selected {len(selected_other)} samples")

    # Final shuffle
    random.shuffle(eval_data)
    print(f"Total baseline test set: {len(eval_data)} samples")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    create_baseline_test()
