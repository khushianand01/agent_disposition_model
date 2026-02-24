import json
from collections import Counter

def analyze():
    file_path = '/home/ubuntu/disposition_model/data/master_production_data.json'
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total items: {len(data)}")
    
    d_counts = Counter([item['output']['disposition'] for item in data])
    p_counts = Counter([item['output']['payment_disposition'] for item in data])
    
    print("\n--- Call Disposition Distribution ---")
    for k, v in d_counts.most_common():
        print(f"{k}: {v} ({v/len(data)*100:.1f}%)")
        
    print("\n--- Payment Disposition Distribution ---")
    for k, v in p_counts.most_common():
        print(f"{k}: {v} ({v/len(data)*100:.1f}%)")
        
    ptp_samples = [item for item in data if item['output']['payment_disposition'] in ['PTP', 'PARTIAL_PAYMENT']]
    if not ptp_samples:
        print("\nNo PTP/PARTIAL_PAYMENT samples found.")
    else:
        ptp_with_amt = len([s for s in ptp_samples if s['output']['ptp_details'].get('amount') is not None])
        ptp_with_date = len([s for s in ptp_samples if s['output']['ptp_details'].get('date') is not None])
        print(f"\nPTP Quality Analysis (Total: {len(ptp_samples)}):")
        print(f"  PTP with Amount: {ptp_with_amt} ({ptp_with_amt/len(ptp_samples)*100:.1f}%)")
        print(f"  PTP with Date: {ptp_with_date} ({ptp_with_date/len(ptp_samples)*100:.1f}%)")

if __name__ == "__main__":
    analyze()
