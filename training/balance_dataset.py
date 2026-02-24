import json
import random

MASTER_FILE = "/home/ubuntu/disposition_model/data/master_production_data.json"
BALANCED_FILE = "/home/ubuntu/disposition_model/data/master_production_data.json" # Overwrite for "Best"

def balance():
    with open(MASTER_FILE, 'r') as f:
        data = json.load(f)
    
    # 1. Separate items by Call Disposition
    # We want to preserve ALL rare call statuses
    by_call = {}
    for item in data:
        label = item['output']['disposition']
        if label not in by_call: by_call[label] = []
        by_call[label].append(item)
    
    final_data = []
    
    # Rare Call statuses (Keep ALL)
    RARE_CALLS = [
        "CUSTOMER_ABUSIVE", "AGENT_BUSY_ON_ANOTHER_CALL", "FORWARDED_CALL", "RINGING", 
        "SWITCHED_OFF", "NOT_IN_CONTACT_ANYMORE", "CUSTOMER_PICKED", "BUSY", 
        "OUT_OF_SERVICES", "GAVE_ALTERNATE_NUMBER", "OUT_OF_NETWORK", "OTHERS",
        "WILL_ASK_TO_PAY", "DO_NOT_KNOW_THE_PERSON", "WRONG_NUMBER",
        "ANSWERED_BY_FAMILY_MEMBER", "LANGUAGE_BARRIER", "CALL_BACK_LATER",
        "SILENCE_ISSUE", "AUTOMATED_VOICE", "ANSWERED_DISCONNECTED", 
        "CALL_DISCONNECTED_BY_CUSTOMER", "ANSWERED_VOICE_ISSUE", "WRONG_PERSON"
    ]
    
    for label in RARE_CALLS:
        if label in by_call:
            final_data.extend(by_call[label])
    
    # 2. Process ANSWERED calls by their PAYMENT disposition
    answered_items = by_call.get("ANSWERED", [])
    by_pay = {}
    for item in answered_items:
        p_label = item['output']['payment_disposition']
        if p_label not in by_pay: by_pay[p_label] = []
        by_pay[p_label].append(item)
    
    # Balancing Payment within ANSWERED
    PAY_CAPS = {
        "None": 3000,
        "PTP": 3000,
        "NO_PAYMENT_COMMITMENT": 2500,
        "DENIED_TO_PAY": 2500,
        "NO_PROOF_GIVEN": 2500
    }
    
    for p_label, items in by_pay.items():
        cap = PAY_CAPS.get(p_label, 1500) # Default cap for others
        if len(items) > cap:
            print(f"Capping Payment {p_label} in ANSWERED: {len(items)} -> {cap}")
            random.shuffle(items)
            final_data.extend(items[:cap])
        else:
            final_data.extend(items)
            
    print(f"Final items after smart balancing: {len(final_data)}")
    
    with open(MASTER_FILE, 'w') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    balance()
