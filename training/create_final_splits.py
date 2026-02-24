import json
import random
import os

MASTER_FILE = "/home/ubuntu/disposition_model/data/master_production_data.json"
OUTPUT_DIR = "/home/ubuntu/disposition_model/data/production"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_best.json")
VAL_FILE = os.path.join(OUTPUT_DIR, "val_best.json")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_best.json")

def create_splits():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(MASTER_FILE, 'r') as f:
        data = json.load(f)
    
    random.shuffle(data)
    
    # 80/10/10 split
    n = len(data)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Total samples: {n}")
    print(f"Training samples: {len(train_data)} (80%)")
    print(f"Validation samples: {len(val_data)} (10%)")
    print(f"Test samples: {len(test_data)} (10%)")
    
    with open(TRAIN_FILE, 'w') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
        
    with open(VAL_FILE, 'w') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    with open(TEST_FILE, 'w') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"3-Way Splits saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    create_splits()
