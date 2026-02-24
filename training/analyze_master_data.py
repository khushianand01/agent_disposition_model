import json
from collections import Counter

# File paths
master_file = '../data/calls_data_balanced_v6.json'
eval_file = 'eval_results_v6.json'

def analyze_data():
    print("--- Master Data Analysis ---")
    with open(master_file, 'r') as f:
        data = json.load(f)
    
    # Joint Distribution
    joint_pairs = []
    for item in data:
        if 'output' in item and isinstance(item['output'], dict):
            disp = str(item['output'].get('disposition'))
            p_disp = str(item['output'].get('payment_disposition'))
            joint_pairs.append(f"{disp} | {p_disp}")
    
    print(f"Total entries: {len(data)}")
    print("\nTop 20 Joint Distribution (Disposition | Payment Disposition):")
    for k, v in Counter(joint_pairs).most_common(20):
        print(f"  {k}: {v}")

    print("\n--- Evaluation Failure Analysis ---")
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)
    
    failed_cases = []
    for item in eval_data.get('results', []):
        pred_obj = item.get('prediction')
        pred = pred_obj.get('disposition') if pred_obj else "JSON_ERROR"
        gt = item['ground_truth']['disposition']
        if pred != gt:
            failed_cases.append({
                'gt': gt,
                'pred': pred,
                'transcript': item.get('transcript', 'N/A')
            })
            
    print(f"Total failed cases in eval: {len(failed_cases)}")
    
    # GT frequency in failures
    gt_counts = Counter([c['gt'] for c in failed_cases])
    print("\nTop Ground Truth labels that model missed:")
    for k, v in gt_counts.most_common(10):
        print(f"  {k}: {v} misses")

if __name__ == "__main__":
    analyze_data()
