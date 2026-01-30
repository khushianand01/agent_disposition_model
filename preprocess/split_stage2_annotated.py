"""
Split Stage-2 Annotated Data into Train/Val

Once you have ~300-500 manually annotated samples, use this script
to create train/val splits.

Split ratio: 85% train, 15% val
"""

import json
import random
from sklearn.model_selection import train_test_split

INPUT_FILE = 'data/stage2_train_validated.json'
OUTPUT_TRAIN = 'data/splits/train_v11_s2.json'
OUTPUT_VAL = 'data/splits/val_v11_s2.json'

TRAIN_RATIO = 0.85
VAL_RATIO = 0.15

def split_stage2_data():
    print(f"Loading annotated data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"Total annotated samples: {len(data)}")
    
    if len(data) < 50:
        print("⚠️  Warning: Less than 50 samples. Recommend at least 300-500 for good training.")
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Split
    train_data, val_data = train_test_split(
        data,
        test_size=VAL_RATIO,
        random_state=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    
    # Save
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved train: {OUTPUT_TRAIN}")
    
    with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved val: {OUTPUT_VAL}")
    
    print("\n🎯 Ready for Stage-2 training!")

if __name__ == "__main__":
    split_stage2_data()
