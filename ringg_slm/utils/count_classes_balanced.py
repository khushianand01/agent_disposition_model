import json
from collections import Counter

train = json.load(open('data/splits/train_v11_s2_balanced.json'))
val = json.load(open('data/splits/val_v11_s2_balanced.json'))
data = train + val

print('\n--- PAYMENT DISPOSITION (INPUT) ---')
counts_pay = Counter(item['input'].get('payment_disposition') for item in data)
for k, v in sorted(counts_pay.items(), key=lambda x: x[1], reverse=True):
    print(f'{str(k):<35} : {v}')

print('\n--- REASON FOR NOT PAYING (OUTPUT) ---')
counts_reason = Counter(item['output'].get('reason_for_not_paying') for item in data)
# Sort by count, with None at the end
for k, v in sorted(counts_reason.items(), key=lambda x: (x[0] is None, x[1] if x[1] is not None else 0), reverse=True):
    print(f'{str(k):<35} : {v}')
