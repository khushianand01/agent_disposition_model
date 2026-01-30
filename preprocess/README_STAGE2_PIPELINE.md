# Stage-2 Data Preparation - IMPORTANT

## ⚠️ CRITICAL: Stage-2 Data Flow

**Stage-2 data MUST come from Stage-1 model predictions, NOT from raw data ground truth.**

### Correct Pipeline

```
1. Train Stage-1 Model
   ↓
2. Run Stage-1 Inference on Dataset
   ↓
3. Filter Stage-1 Predictions:
   - disposition ∈ {ANSWERED, ANSWERED_BY_FAMILY_MEMBER, CALL_BACK_LATER}
   - AND payment_disposition ≠ None
   ↓
4. Extract Stage-2 Fields from Raw Data for Eligible Samples
   ↓
5. Prepare Stage-2 Training Data
```

## Why This Matters

In production:
1. Stage-1 model runs first on all calls
2. Only calls that Stage-1 classifies as eligible go to Stage-2
3. Stage-2 training should mirror this production flow

## Current Status

**Stage-1 Training**: In progress (Ringg, ~2 hours remaining)

**Stage-2 Preparation**: ON HOLD until Stage-1 completes

## Scripts Ready (Do NOT Run Yet)

These scripts are ready but should only be executed AFTER Stage-1 training:

1. `extract_stage2_candidates.py` - **NEEDS UPDATE** to use Stage-1 predictions
2. `format_stage2_training.py` - Ready
3. `clean_stage2_enum.py` - Ready
4. `validate_stage2_data.py` - Ready
5. `split_stage2_annotated.py` - Ready

## Next Steps (After Stage-1 Training)

1. Run Stage-1 model inference on train/val data
2. Save Stage-1 predictions
3. Update `extract_stage2_candidates.py` to filter based on predictions
4. Run full Stage-2 pipeline
5. Train Stage-2 models

## Schema Reference

See `stage2_schema.md` for the locked 20-value enum and field definitions.
