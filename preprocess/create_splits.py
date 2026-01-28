import json
import random
import os

INPUT_FILE = 'data/calls_data_balanced_v5.json'
OUTPUT_DIR = 'data/splits'

def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total = len(data)
    print(f"Total samples: {total}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Split 90 / 5 / 5
    train_end = int(total * 0.90)
    val_end = int(total * 0.95)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Train: {len(train_data)} ({len(train_data)/total:.1%})")
    print(f"Val:   {len(val_data)} ({len(val_data)/total:.1%})")
    print(f"Test:  {len(test_data)} ({len(test_data)/total:.1%})")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save
    with open(f'{OUTPUT_DIR}/train_v5.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
        
    with open(f'{OUTPUT_DIR}/val_v5.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
        
    with open(f'{OUTPUT_DIR}/test_v5.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
        
    print(f"Splits saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
