import json
import random
import os
from collections import defaultdict

# Configuration
INPUT_FILE = 'data/calls_data_v11_s1_master.json'
OUTPUT_DIR = 'data/splits'
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
# TEST_RATIO will be the remainder (0.05)

def split_data():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading Stage-1 master data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Stratify by disposition and payment_disposition
    # Note: stage-1 'output' is a JSON string
    strata = defaultdict(list)
    for item in data:
        try:
            output_obj = json.loads(item['output'])
            disp = output_obj.get('disposition', 'None')
            p_disp = output_obj.get('payment_disposition', 'None')
            key = f"{disp}|{p_disp}"
            strata[key].append(item)
        except:
            # Fallback if parsing fails
            strata["unknown"].append(item)

    train_set = []
    val_set = []
    test_set = []

    random.seed(42)
    
    for key, samples in strata.items():
        random.shuffle(samples)
        n = len(samples)
        
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        
        # Ensure at least 1 in val/test if n is sufficient
        if n >= 10:
            if n_val == 0: n_val = 1
            if (n - n_train - n_val) == 0: n_train -= 1
        
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

    print(f"Split completed for {len(data)} items:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Val:   {len(val_set)} samples")
    print(f"  Test:  {len(test_set)} samples")

    # Create directory if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Saving
    files = {
        'train_v11_s1.json': train_set,
        'val_v11_s1.json': val_set,
        'test_v11_s1.json': test_set
    }

    for filename, dataset in files.items():
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Saved {path}")

if __name__ == "__main__":
    split_data()
