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

### Production API (FastAPI)

**1. Start the Production API Server:**
```bash
# From the deployment directory
cd /home/ubuntu/Disposition_model2-main/qwen_3b/deployment
nohup conda run --no-capture-output -n disposition_v3 python app.py > qwen_api.log 2>&1 &

# Check server status
ps aux | grep app.py

# View logs
tail -f qwen_api.log
```

**Server Details:**
- Port: 8080
- Model: `/home/ubuntu/Disposition_model2-main/qwen_3b/outputs`
- Startup time: ~60 seconds (loading checkpoint shards)

**2. Test the API (from another terminal):**
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "transcript": "Agent: When can you pay? Borrower: I will pay one EMI on 7th February."
     }'
```

**3. Stop the API Server:**
```bash
pkill -f "python app.py"
```
