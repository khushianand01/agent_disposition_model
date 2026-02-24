import json
import os

REPORT_PATH = "/home/ubuntu/disposition_model/outputs/qwen_3b_production_best/evaluation_report.json"

def main():
    if not os.path.exists(REPORT_PATH):
        print(f"❌ Report not found at {REPORT_PATH}")
        return

    try:
        with open(REPORT_PATH, "r") as f:
            data = json.load(f)
        
        metrics = data.get("metrics", {})
        
        print("\n" + "="*40)
        print("📊 FINAL EVALUATION REPORT")
        print("="*40)
        print(f"✅ Samples:      {metrics.get('total_samples', 0)}")
        print(f"❌ Errors:       {metrics.get('parsing_errors', 0)}")
        print("-" * 40)
        print(f"🎯 Disp Acc:     {metrics.get('disposition_accuracy', 0):.2f}%")
        print(f"💰 Pay Acc:      {metrics.get('payment_disposition_accuracy', 0):.2f}%")
        print("-" * 40)
        
        print("\n🔍 Confusion Matrix (Disposition):")
        for k, v in metrics.get("disposition_confusion_matrix", {}).items():
            if v > 0: print(f"  {k}: {v}")

    except Exception as e:
        print(f"❌ Error parsing report: {e}")

if __name__ == "__main__":
    main()
