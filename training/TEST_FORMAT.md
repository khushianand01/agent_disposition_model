# How to Test the Disposition Model

## 1. Environment Setup (One Time)
Make sure you are in the correct environment:
```bash
source /home/ubuntu/miniconda3/bin/activate disposition_v3
```

## 2. Testing Script
Use the provided testing script (or `test_single.py` if available). When providing input, you can specify the `current_date`.

### Input Format
The model expects a call transcript. The `current_date` logic is now strict:
- **Case A: Date Provided**: The model uses the date you give it.
- **Case B: No Date**: The model automatically uses **today's date**.

### Example Command (Python)
```python
from inference.inference import get_model

# Load Model
model = get_model() # Loads from 'outputs/qwen3_8b_lora_production'

# Case A: Explicit Date (Historical Testing)
transcript = "Agent: Hello... Borrower: I will pay on the 25th..."
date_input = "2024-01-01" # The date the call supposedly happened
result = model.predict(transcript, current_date=date_input)
print(result)
# Result 'ptp_date' will be calculated relative to Jan 1st, 2024.

# Case B: Live Call (No Date Provided)
# This will default to TODAY'S date automatically.
result = model.predict(transcript)
print(result)
```

## 3. JSON Output Format
The model returns a JSON object with these fields:
```json
{
  "disposition": "Promise to Pay",
  "payment_disposition": "PROMISE_TO_PAY",
  "reason_for_not_paying": null,
  "ptp_amount": "5000",
  "ptp_date": "2024-02-25 00:00:00",
  "followup_date": null,
  "remarks": "Customer agreed to pay on 25th.",
  "confidence_score": 0.98
}
```
