import json
import random
import os
from collections import defaultdict

# Configuration
INPUT_FILE = 'data/calls_data_balanced_v6.json'
OUTPUT_DIR = 'data/splits'
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
# TEST_RATIO will be the remainder (approx 0.05)

def split_data():
    print(f"Loading balanced data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Stratify by disposition and payment_disposition
    # We combine them into a single strata key
    strata = defaultdict(list)
    for item in data:
        disp = item['output'].get('disposition', 'None')
        p_disp = item['output'].get('payment_disposition', 'None')
        key = f"{disp}|{p_disp}"
        strata[key].append(item)

    train_set = []
    val_set = []
    test_set = []

    random.seed(42)
    
    for key, samples in strata.items():
        random.shuffle(samples)
        n = len(samples)
        
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Ensure at least 1 in val/test if n is small but > 2
        if n >= 3 and n_val == 0:
            n_val = 1
        
        # Determine splits
        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]
        
        train_set.extend(train_samples)
        val_set.extend(val_samples)
        test_set.extend(test_samples)

    # Final shuffle
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    print(f"Split completed:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")
    print(f"  Total: {len(train_set) + len(val_set) + len(test_set)} (Original: {len(data)})")

    # Create directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Saving
    files = {
        'train_v6.json': train_set,
        'val_v6.json': val_set,
        'test_v6.json': test_set
    }

    for filename, dataset in files.items():
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved {path}")

if __name__ == "__main__":
    split_data()
