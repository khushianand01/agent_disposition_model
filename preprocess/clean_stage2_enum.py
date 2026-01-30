"""
Clean Stage-2 Data to Match Locked Enum

Maps invalid reason_for_not_paying values to the locked 20-value enum.
"""

import json

INPUT_FILE = "data/stage2_train_formatted.json"
OUTPUT_FILE = "data/stage2_train_cleaned.json"

# Locked enum (20 values) - DO NOT CHANGE
LOCKED_ENUM = {
    'JOB_CHANGED_WAITING_FOR_SALARY', 'LOST_JOB', 'BUSINESS_LOSS', 'BUSINESS_CLOSED', 
    'FINANCIAL_DIFFICULTY', 'MULTIPLE_LOANS', 'MEDICAL_ISSUE', 'DEATH_IN_FAMILY', 
    'FAMILY_ISSUE', 'CLAIMING_PAYMENT_IS_COMPLETED', 'CLAIMING_FRAUD', 'GRIEVANCE_FRAUD',
    'GRIEVANCE_LOAN_AMOUNT_DISPUTE', 'PENALTY_ISSUE', 'SERVICE_ISSUE', 
    'LOAN_CLOSURE_MISCOMMUNICATION', 'LOAN_TAKEN_BY_KNOWN_PARTY', 'OUT_OF_STATION',
    'CUSTOMER_NOT_TELLING_REASON', 'OTHER_REASONS'
}

# Mapping from invalid values to locked enum
REASON_MAPPING = {
    'CUSTOMER_EXPIRED': 'DEATH_IN_FAMILY',
    'CUSTOMER_PLANS_TO_VISIT_BRANCH': 'OTHER_REASONS',
    'TRUST_ISSUE': 'SERVICE_ISSUE',
    'GRIEVANCE_APP_ISSUE': 'SERVICE_ISSUE',
    'GRIEVANCE_CALLER_MISCONDUCT': 'SERVICE_ISSUE',
    'AWAITING_STATEMENT': 'SERVICE_ISSUE',
    'FUND_ISSUE': 'FINANCIAL_DIFFICULTY',
    'OTHER_PERSON_TAKEN': 'LOAN_TAKEN_BY_KNOWN_PARTY',
}

def clean_stage2_data():
    print(f"Loading formatted data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Clean data
    cleaned_count = 0
    mapping_stats = {}
    
    for item in data:
        reason = item['output'].get('reason_for_not_paying')
        
        if reason and reason not in LOCKED_ENUM:
            # Map to locked enum
            if reason in REASON_MAPPING:
                new_reason = REASON_MAPPING[reason]
                item['output']['reason_for_not_paying'] = new_reason
                
                if reason not in mapping_stats:
                    mapping_stats[reason] = {'count': 0, 'mapped_to': new_reason}
                mapping_stats[reason]['count'] += 1
                cleaned_count += 1
            else:
                # Unmapped value - set to OTHER_REASONS
                print(f"⚠️  Unmapped value: {reason} -> OTHER_REASONS")
                item['output']['reason_for_not_paying'] = 'OTHER_REASONS'
                cleaned_count += 1
    
    # Save cleaned data
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Cleaned {cleaned_count} samples")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Show mapping statistics
    if mapping_stats:
        print("\n" + "="*60)
        print("MAPPING STATISTICS:")
        print("="*60)
        for old_value, stats in sorted(mapping_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"{old_value:40s} -> {stats['mapped_to']:30s} ({stats['count']:4d} samples)")
    
    # Verify all values are now valid
    print("\n" + "="*60)
    print("VERIFICATION:")
    print("="*60)
    invalid_count = 0
    for item in data:
        reason = item['output'].get('reason_for_not_paying')
        if reason and reason not in LOCKED_ENUM:
            invalid_count += 1
            print(f"⚠️  Still invalid: {reason}")
    
    if invalid_count == 0:
        print("✅ All reason_for_not_paying values are now valid!")
    else:
        print(f"❌ {invalid_count} invalid values remain")

if __name__ == "__main__":
    clean_stage2_data()
