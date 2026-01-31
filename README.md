# 📞 Disposition Extraction Model v6

A two-stage AI pipeline for extracting structured **Call Dispositions** and **Payment Details** from conversational transcripts using **Qwen-2.5-3B-Instruct** and **Ringg-1.5B**.

![Status](https://img.shields.io/badge/Status-Active-success)
![Models](https://img.shields.io/badge/Models-Qwen%202.5%203B%20%7C%20Ringg%201.5B-blue)
![Stack](https://img.shields.io/badge/Stack-Unsloth%20%7C%20FastAPI-blueviolet)

## 🚀 Features

### Model Architecture

**Qwen-2.5-3B-Instruct**: Single-stage extraction
- Extracts all 7 fields in one inference call
- Higher accuracy, larger model (~3GB VRAM)
- Fields: disposition, payment_disposition, reason_for_not_paying, ptp_amount, ptp_date, followup_date, remarks

**Ringg-1.5B**: Two-stage approach
- **Stage-1**: Classification (disposition + payment_disposition) - 2 fields
- **Stage-2**: Detailed extraction (reason, amounts, dates, remarks) - 5 additional fields
- Faster inference, smaller footprint (~1.5GB VRAM per stage)

### Production Ready
- **FastAPI** service for real-time inference (port 8080)
- **Confidence scoring** for quality control
- **Unsloth** optimization (2x faster training, 60% less memory)
- **Normalization helpers** for dates and amounts (Hindi/Hinglish support)

---

## 📁 Project Structure

```
Disposition_model2-main/
├── qwen_3b/                    # Qwen model (3B parameters)
│   ├── outputs/                # Trained model checkpoint
│   ├── inference/              # Inference scripts
│   ├── deployment/             # FastAPI server (app.py)
│   └── README_COMMANDS.md      # Qwen-specific commands
│
├── ringg_slm/                  # Ringg model (1.5B parameters)
│   ├── outputs/                # Trained model checkpoints
│   ├── inference/              # Evaluation scripts
│   ├── data/splits/            # Ringg-format data
│   └── train_ringg_slm.py      # Training script
│
├── data/                       # Shared data
│   └── splits/                 # Train/val/test splits
│       ├── train_v11_s1.json   # Stage-1 (19K samples)
│       ├── val_v11_s1.json     # Stage-1 validation
│       ├── train_v11_s2.json   # Stage-2 (5.8K samples)
│       └── val_v11_s2.json     # Stage-2 validation
│
└── preprocess/                 # Data preparation
    ├── normalization_helpers.py
    ├── stage2_prompts.py
    └── README_STAGE2_PIPELINE.md
```

---

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/khushianand01/Disposition_model_v6.git
cd Disposition_model_v6
```

### 2. Install Dependencies
```bash
# Create conda environment
conda create -n disposition_v3 python=3.10
conda activate disposition_v3

# Install requirements
pip install -r requirements.txt
```

---

## 🏃‍♂️ Quick Start

### Run Ringg Unified Pipeline (Two-Stage)
This runs BOTH Stage 1 (Classification) and Stage 2 (Extraction) automatically.

```bash
/home/ubuntu/miniconda3/envs/disposition_v2/bin/python ringg_slm/inference/ringg_pipeline.py
```

### Run Qwen FastAPI Server
```bash
# Start server (runs in background)
cd qwen_3b/deployment
nohup /home/ubuntu/miniconda3/envs/disposition_v2/bin/python app.py > qwen_api.log 2>&1 &
```
# Test the API
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"transcript": "Agent: When can you pay? Borrower: I will pay 3000 on 7th Feb."}'

# Stop server
pkill -f "python app.py"
```

**Note**: Server takes ~60 seconds to load model on startup.

---

## 📊 Model Performance

### Qwen-2.5-3B-Instruct (Single-Stage)
- **Approach**: Extracts all 7 fields in one inference
- **Training**: 19,867 samples (balanced dataset v6)
- **Evaluation** (1,173 test samples):
  - **Overall Accuracy**: 62.9% (disposition) + 47.5% (payment_disp) = **55.2% average**
  - Disposition Accuracy: **62.9%**
  - Disposition F1 (weighted): **0.57**
    - ANSWERED: 0.79
    - WRONG_NUMBER: 0.74
    - LANGUAGE_BARRIER: 0.62
  - Payment Disposition Accuracy: **47.5%**
  - Payment Disposition F1 (weighted): **0.52**
    - PTP: 0.52
    - NO_PROOF_GIVEN: 0.42
    - SETTLEMENT: 0.46
  - JSON Validity: **96.6%**
- **Use case**: Higher accuracy requirements, sufficient GPU memory

### Ringg-1.5B (Two-Stage)
- **Stage-1 Baseline**: 
  - Disposition Accuracy: **43.04%** (Improvement from ~29%)
  - Payment Disposition Accuracy: **30.96%** (Improvement from ~9%)
  - Training: 19,867 samples
- **Stage-2 Extraction**: 
  - Status: **Trained & Verified**
  - Training: 1,475 balanced samples
  - Fields: reason_for_not_paying, ptp_amount, ptp_date, followup_date, remarks
- **Use case**: Resource-constrained environments, faster inference

---

## 🔧 Training

### Train Ringg Stage-1
```bash
cd ringg_slm
./run_ringg_stage1.sh
```

### Train Qwen Stage-1
```bash
cd qwen_3b
python train_production.py
```

See `qwen_3b/README_COMMANDS.md` and `ringg_slm/README.md` for detailed commands.

---

## 📈 GPU Requirements

- **Training**: 1x Tesla T4 (16GB) or better
- **Inference**: 
  - Qwen API: ~4-5 GB VRAM
  - Ringg: ~2-3 GB VRAM
- **Note**: Cannot run both Ringg training + Qwen API simultaneously on T4 (OOM)

---

## 📧 Contact

**Project**: Disposition Model v6  
**Repository**: https://github.com/khushianand01/Disposition_model_v6  
**Maintainer**: Khushi Anand
