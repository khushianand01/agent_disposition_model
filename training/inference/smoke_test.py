import json
import torch
import sys
import os
import time
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.inference import get_model

# Constants
TEST_DATA_PATH = "/home/ubuntu/disposition_model/data/production/test_dummy.json"
MODEL_PATH = "/home/ubuntu/disposition_model/qwen_3b/outputs/qwen_3b_production_best"

def run_smoke_test():
    print(f"🚀 Starting Smoke Test...")
    print(f"📂 Model: {MODEL_PATH}")
    print(f"📄 Data: {TEST_DATA_PATH}")

    # 1. Load Model
    print("\n[1/3] Loading Model...")
    t0 = time.time()
    try:
        model = get_model()
        print(f"✅ Model Loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"❌ FATAL: Model load failed: {e}")
        return

    # 2. Load Data
    print("\n[2/3] Loading Test Data...")
    try:
        with open(TEST_DATA_PATH, "r") as f:
            data = json.load(f)
        # Take first 5 samples only
        samples = data[:5]
        print(f"✅ Loaded {len(data)} samples. Running on first {len(samples)}...")
    except Exception as e:
        print(f"❌ FATAL: Data load failed: {e}")
        return

    # 3. Run Inference
    print("\n[3/3] Running Inference...")
    passed = 0
    for i, item in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        transcript = item["input"]
        print(f"Input: {transcript[:100]}...")
        
        try:
            # Simulate date injection if needed
            current_date = "2024-02-19" 
            prediction = model.predict(transcript, current_date=current_date)
            print(f"Output: {json.dumps(prediction, indent=2)}")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")

    print(f"\n✅ Smoke Test Complete: {passed}/{len(samples)} passed individual checks.")

if __name__ == "__main__":
    run_smoke_test()
