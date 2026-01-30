import json
import random
from sklearn.model_selection import train_test_split

INPUT_FILE = 'data/calls_data_v11_s2_ringg.json'
OUTPUT_DIR = 'data/splits/'

# Split ratios
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

def split_data():
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # First split: train vs (val+test)
    train_data, temp_data = train_test_split(
        data, 
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=42
    )
    
    # Second split: val vs test
    val_data, test_data = train_test_split(
        temp_data,
        test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO),
        random_state=42
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
    
    # Save splits
    splits = {
        'train_v11_s2.json': train_data,
        'val_v11_s2.json': val_data,
        'test_v11_s2.json': test_data
    }
    
    for filename, split_data in splits.items():
        output_path = OUTPUT_DIR + filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_path}")
    
    print("\nStage-2 splits created successfully!")

if __name__ == "__main__":
    split_data()
