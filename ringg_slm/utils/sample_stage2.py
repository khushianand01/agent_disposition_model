import json
import random
from collections import Counter

# Set seed for reproducibility
random.seed(42)

MASTER_DATA = "data/stage2_train_validated.json"
TRAIN_OUTPUT = "data/splits/train_v11_s2_balanced.json"
VAL_OUTPUT = "data/splits/val_v11_s2_balanced.json"

TARGET_DISTRIBUTION = {
    "NO_PAYMENT_COMMITMENT": 275,
    "PTP": 275,
    "DENIED_TO_PAY": 225,
    "NO_PROOF_GIVEN": 225,
    "PAID": 175,
    "SETTLEMENT": 125,
    "PARTIAL_PAYMENT": 65,
    "WILL_PAY_AFTER_VISIT": 40,
    "WANT_FORECLOSURE": 40,
    "WANTS_TO_RENEGOTIATE_LOAN_TERMS": 30 # Added based on stats
}

def sample_data():
    with open(MASTER_DATA, "r") as f:
        data = json.load(f)

    # Group by payment_disposition
    grouped = {}
    for item in data:
        pd = item["input"].get("payment_disposition")
        if pd not in grouped:
            grouped[pd] = []
        grouped[pd].append(item)

    sampled_pool = []
    
    print("--- Sampling Results ---")
    for pd, target in TARGET_DISTRIBUTION.items():
        available = grouped.get(pd, [])
        num_to_sample = min(len(available), target)
        selection = random.sample(available, num_to_sample)
        sampled_pool.extend(selection)
        print(f"{pd:<35}: {len(selection)} (Target: {target})")

    # Shuffle the final pool
    random.shuffle(sampled_pool)

    # Split 85/15
    split_idx = int(len(sampled_pool) * 0.85)
    train_data = sampled_pool[:split_idx]
    val_data = sampled_pool[split_idx:]

    print(f"\nTotal Sampled: {len(sampled_pool)}")
    print(f"Train Set: {len(train_data)}")
    print(f"Val Set: {len(val_data)}")

    with open(TRAIN_OUTPUT, "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(VAL_OUTPUT, "w") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved splits to {TRAIN_OUTPUT} and {VAL_OUTPUT}")

if __name__ == "__main__":
    sample_data()
