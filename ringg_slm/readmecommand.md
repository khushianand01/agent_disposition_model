# Ringg SLM Commands

This file contains the core commands for training and running inference on the Ringg Small Language Model.

## Environment Setup
Ensure you are using the `disposition_v2` conda environment:
```bash
conda activate disposition_v2
```

## Stage 1 Training
To start or resume the Stage 1 training:
```bash
/home/ubuntu/miniconda3/envs/disposition_v2/bin/python ringg_slm/train_ringg_slm.py \
    --data_path data/splits/train_v11_s1.json \
    --val_data_path data/splits/val_v11_s1.json \
    --output_dir ringg_slm/outputs/ringg_slm_stage1
```

## Stage 2 Training
To fine-tune the Stage 2 model for detailed extraction:
```bash
/home/ubuntu/miniconda3/envs/disposition_v2/bin/python ringg_slm/train_ringg_slm_s2.py \
    --data_path data/splits/train_v11_s2_balanced.json \
    --val_data_path data/splits/val_v11_s2_balanced.json \
    --base_model_path ringg_slm/outputs/ringg_slm_stage1 \
    --output_dir ringg_slm/outputs/ringg_slm_stage2
```

## 🚀 Unified Pipeline (Recommended)
This script runs **Stage 1** (Classification) and **Stage 2** (Extraction) automatically in sequence.
```bash
/home/ubuntu/miniconda3/envs/disposition_v2/bin/python ringg_slm/inference/ringg_pipeline.py
```

## Individual Testers
Use these to check a specific stage model on single transcripts.

### Test Stage 1
```bash
/home/ubuntu/miniconda3/envs/disposition_v2/bin/python ringg_slm/inference/test_single_s1.py
```

### Test Stage 2
```bash
/home/ubuntu/miniconda3/envs/disposition_v2/bin/python ringg_slm/inference/test_single_s2.py
```

## Full Evaluation
To run full evaluation on the test set:

### Stage 1 Evaluation
```bash
/home/ubuntu/miniconda3/envs/disposition_v2/bin/python ringg_slm/inference/eval_test_v11.py
```

### Stage 2 Evaluation
```bash
/home/ubuntu/miniconda3/envs/disposition_v2/bin/python ringg_slm/inference/eval_s2.py
```
