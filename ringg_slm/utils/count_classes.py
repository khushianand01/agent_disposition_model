import json
from collections import Counter

with open('data/stage2_train_validated.json', 'r') as f:
    data = json.load(f)

print('\n--- REASON FOR NOT PAYING ---')
counts = Counter(item['output'].get('reason_for_not_paying') for item in data)
for k, v in sorted(counts.items(), key=lambda x: (x[0] is None, x[1] if x[1] is not None else 0), reverse=True):
    print(f'{str(k):<35} : {v}')

print('\n--- INPUT DISPOSITION ---')
counts_disp = Counter(item['input'].get('disposition') for item in data)
for k, v in sorted(counts_disp.items(), key=lambda x: x[1], reverse=True):
    print(f'{str(k):<35} : {v}')

print('\n--- INPUT PAYMENT DISPOSITION ---')
counts_pay = Counter(item['input'].get('payment_disposition') for item in data)
for k, v in sorted(counts_pay.items(), key=lambda x: x[1], reverse=True):
    print(f'{str(k):<35} : {v}')
