# Useful Commands for Disposition Model Training

Here is a quick reference for the commands you need to manage and monitor the training process.

## 1. Monitor Training Progress (Logs)
To see the live output of the training script:
```bash
tail -f training_production.log
```
*Press `Ctrl+C` to exit the tail view (this does NOT stop the training).*

To see just the last 50 lines:
```bash
tail -n 50 training_production.log
```

## 2. Check Process Status
To check if the training script is currently running:
```bash
ps aux | grep train_production.py
```

To see GPU usage and memory:
```bash
nvidia-smi
```
*Run `watch -n 1 nvidia-smi` updates this view every second.*

## 3. Manage Environment
To activate the correct Conda environment:
```bash
source /home/ubuntu/miniconda3/bin/activate disposition_v3
```

## 4. Run Training
To start the training (this script handles environment activation and background execution):
```bash
./run_production.sh
```

## 5. Stop Training
To kill the running training process:
```bash
pkill -f train_production.py
```
## 6. Run Production Demo
To test a single transcript using the production model:
```bash
# Option 1: Activate and Run
source /home/ubuntu/miniconda3/bin/activate disposition_v3
python production_demo.py

# Option 2: One-liner
conda run -n disposition_v3 python production_demo.py
```
