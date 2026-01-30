"""
Extract Stage-2 Candidates from Stage-1 Training Data (OPTIMIZED)

Filters Stage-1 training data for Stage-2 eligibility, then extracts
the full 7-field output from raw data using hash map for O(1) lookup.

Eligibility:
- disposition ∈ {ANSWERED, ANSWERED_BY_FAMILY_MEMBER, CALL_BACK_LATER}
- AND payment_disposition ≠ None
"""

import json
import os
from collections import Counter

# Stage-1 training data (Ringg format)
STAGE1_TRAIN = "ringg_slm/data/splits/train_v11_s1.json"
STAGE1_VAL = "ringg_slm/data/splits/val_v11_s1.json"

# Raw data with all 7 fields
RAW_DATA = "data/calls_data.json"

# Output
OUTPUT_FILE = "data/stage2_from_stage1.json"
STATS_FILE = "data/stage2_from_stage1_stats.txt"

# Stage-2 eligibility
STAGE2_ELIGIBLE_DISPOSITIONS = {
    "ANSWERED",
    "ANSWERED_BY_FAMILY_MEMBER",
    "CALL_BACK_LATER"
}

def extract_stage2_from_stage1():
    print("Loading Stage-1 training data...")
    with open(STAGE1_TRAIN, 'r') as f:
        stage1_train = json.load(f)
    
    with open(STAGE1_VAL, 'r') as f:
        stage1_val = json.load(f)
    
    stage1_data = stage1_train + stage1_val
    print(f"Total Stage-1 samples: {len(stage1_data)}")
    
    # Load raw data and build hash map
    print("\nLoading raw data and building hash map...")
    with open(RAW_DATA, 'r') as f:
        raw_data = json.load(f)
    
    print(f"Total raw samples: {len(raw_data)}")
    
    # Build transcript → raw output hash map (O(n))
    raw_index = {}
    for item in raw_data:
        transcript = item.get('input', '').strip()
        raw_index[transcript] = item.get('output', {})
    
    print(f"Indexed {len(raw_index)} unique raw transcripts")
    
    # Extract Stage-2 candidates (O(m) with O(1) lookups)
    print("\nExtracting Stage-2 candidates...")
    eligible = []
    stats = {
        'total_stage1': len(stage1_data),
        'eligible': 0,
        'matched_in_raw': 0,
        'disposition_counts': Counter(),
        'payment_disp_counts': Counter()
    }
    
    for item in stage1_data:
        transcript = item.get('input', '').strip()
        output_str = item.get('output', '{}')
        
        try:
            stage1_output = json.loads(output_str)
        except:
            stage1_output = output_str
        
        disposition = stage1_output.get('disposition', '')
        payment_disposition = stage1_output.get('payment_disposition', 'None')
        
        stats['disposition_counts'][disposition] += 1
        stats['payment_disp_counts'][payment_disposition] += 1
        
        # Check eligibility
        if (
            disposition in STAGE2_ELIGIBLE_DISPOSITIONS
            and payment_disposition not in ['None', None]
        ):
            stats['eligible'] += 1
            
            # O(1) lookup in hash map
            if transcript in raw_index:
                stats['matched_in_raw'] += 1
                raw_output = raw_index[transcript]
                
                eligible.append({
                    "transcript": transcript,
                    "stage1_labels": {
                        "disposition": disposition,
                        "payment_disposition": payment_disposition
                    },
                    "stage2_fields": {
                        "reason_for_not_paying": raw_output.get('reason_for_not_paying'),
                        "ptp_amount": raw_output.get('ptp_amount'),
                        "ptp_date": raw_output.get('ptp_date'),
                        "followup_date": raw_output.get('followup_date'),
                        "remarks": raw_output.get('remarks')
                    }
                })
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(eligible, f, indent=2, ensure_ascii=False)
    
    # Generate report
    report = []
    report.append("=" * 60)
    report.append("STAGE-2 EXTRACTION FROM STAGE-1 DATA")
    report.append("=" * 60)
    report.append(f"\nTotal Stage-1 Samples: {stats['total_stage1']}")
    report.append(f"Stage-2 Eligible: {stats['eligible']} ({stats['eligible']/stats['total_stage1']*100:.1f}%)")
    report.append(f"Matched in Raw Data: {stats['matched_in_raw']}")
    
    if stats['eligible'] != stats['matched_in_raw']:
        report.append(f"⚠️  Missing: {stats['eligible'] - stats['matched_in_raw']} samples not found in raw data")
    
    report.append(f"\n{'Disposition Distribution (Stage-1):'}")
    report.append("-" * 40)
    for disp, count in stats['disposition_counts'].most_common(10):
        marker = "✓" if disp in STAGE2_ELIGIBLE_DISPOSITIONS else " "
        report.append(f"{marker} {disp:40s} {count:6d}")
    
    report.append(f"\n{'Payment Disposition Distribution:'}")
    report.append("-" * 40)
    for p_disp, count in stats['payment_disp_counts'].most_common(10):
        marker = "✓" if p_disp not in ['None', None] else " "
        report.append(f"{marker} {str(p_disp):40s} {count:6d}")
    
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    with open(STATS_FILE, 'w') as f:
        f.write(report_text)
    
    print(f"\n✅ Saved {len(eligible)} Stage-2 candidates to: {OUTPUT_FILE}")
    print(f"📊 Statistics saved to: {STATS_FILE}")

if __name__ == "__main__":
    extract_stage2_from_stage1()
