# Stage-2 Data Preparation Pipeline

This directory contains scripts to prepare **Stage-2** training data with full 7-field extraction.

## Stage-2 Schema (7 fields)

```json
{
  "disposition": "ENUM",
  "payment_disposition": "ENUM",
  "reason_for_not_paying": "ENUM (20 values)",
  "ptp_amount": "number | null",
  "ptp_date": "YYYY-MM-DD | null",
  "followup_date": "YYYY-MM-DD | null",
  "remarks": "string | null"
}
```

## Pipeline Steps

### 1. Create Master Data
```bash
python3 preprocess/create_master_v11_s2.py
```
**Output**: `data/calls_data_v11_s2_master.json`

**What it does**:
- Loads raw `data/calls_data.json`
- Enforces 7-field schema
- Standardizes `reason_for_not_paying` to 20-value enum
- Validates dates (YYYY-MM-DD) and amounts (numeric)
- Applies balancing (caps at 1200, replicates minorities)

### 2. Convert to Model-Specific Formats

#### For Ringg:
```bash
python3 preprocess/convert_to_ringg_s2.py
```
**Output**: `data/calls_data_v11_s2_ringg.json`

**Format**: `{instruction, input, output}` where output is JSON string

#### For Qwen:
```bash
python3 preprocess/convert_to_qwen_s2.py
```
**Output**: `data/calls_data_v11_s2_qwen.json`

**Format**: `{input, output}` where output is dict

### 3. Create Train/Val/Test Splits
```bash
python3 preprocess/split_v11_s2.py
```
**Output**: 
- `data/splits/train_v11_s2.json` (90%)
- `data/splits/val_v11_s2.json` (5%)
- `data/splits/test_v11_s2.json` (5%)

## reason_for_not_paying Enum (20 values)

**Financial**: JOB_CHANGED_WAITING_FOR_SALARY, LOST_JOB, BUSINESS_LOSS, BUSINESS_CLOSED, FINANCIAL_DIFFICULTY, MULTIPLE_LOANS

**Medical/Family**: MEDICAL_ISSUE, DEATH_IN_FAMILY, FAMILY_ISSUE

**Disputes**: CLAIMING_PAYMENT_IS_COMPLETED, CLAIMING_FRAUD, GRIEVANCE_FRAUD, GRIEVANCE_LOAN_AMOUNT_DISPUTE, PENALTY_ISSUE

**Service**: SERVICE_ISSUE, LOAN_CLOSURE_MISCOMMUNICATION, LOAN_TAKEN_BY_KNOWN_PARTY

**Other**: OUT_OF_STATION, CUSTOMER_NOT_TELLING_REASON, OTHER_REASONS

## Usage Timeline

**Current**: Scripts are ready but NOT executed yet.

**After Stage-1 Training**: Run the full pipeline to generate Stage-2 data.
