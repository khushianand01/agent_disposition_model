import json
import random
from collections import Counter

DATA_PATH = "data/stage2_train_formatted.json"
OUTPUT_PATH = "data/splits/train_v11_s2_extraction_heavy.json"
VAL_OUTPUT_PATH = "data/splits/val_v11_s2_extraction_heavy.json"

with open(DATA_PATH, "r") as f:
    data = json.load(f)

# Sort into buckets
extraction_cases = [] # Has amount or date
logic_cases = []      # No amount/date, but has a specific reason (not null/other)
simple_cases = []     # The rest (Answered, Wrong Number, etc)

for item in data:
    output = item["output"]
    # If it's a string (old format), parse it
    if isinstance(output, str):
        output = json.loads(output)
        
    has_extraction = output.get("ptp_amount") is not None or output.get("ptp_date") is not None
    has_logic = output.get("reason_for_not_paying") not in [None, "OTHER_REASONS", "CUSTOMER_NOT_TELLING_REASON"]
    
    if has_extraction:
        extraction_cases.append(item)
    elif has_logic:
        logic_cases.append(item)
    else:
        simple_cases.append(item)

print(f"Extraction Cases: {len(extraction_cases)}")
print(f"Logic Cases: {len(logic_cases)}")
print(f"Simple Cases: {len(simple_cases)}")

# THE STRATEGY: 
# 1. Use 100% of Extraction Cases (approx 700)
# 2. Use 1000 Logic Cases (Boosted)
# 3. Use 1000 Simple Cases (for balance)

final_train = []
final_train.extend(extraction_cases)
final_train.extend(random.sample(logic_cases, min(1000, len(logic_cases))))
final_train.extend(random.sample(simple_cases, min(1000, len(simple_cases))))

random.shuffle(final_train)

# Split 90/10
split_idx = int(len(final_train) * 0.9)
train_set = final_train[:split_idx]
val_set = final_train[split_idx:]

with open(OUTPUT_PATH, "w") as f:
    json.dump(train_set, f, indent=2)
with open(VAL_OUTPUT_PATH, "w") as f:
    json.dump(val_set, f, indent=2)

print(f"\nFinal Train Set Size: {len(train_set)}")
print(f"Final Val Set Size: {len(val_set)}")
