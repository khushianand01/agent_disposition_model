import json
import os
import re
import glob

# File Paths
DATA_DIR = "/home/ubuntu/disposition_model/data/splits"
MASTER_OUTPUT = "/home/ubuntu/disposition_model/data/master_production_data.json"

# Label Definitions
CALL_DISPOSITIONS = [
    "ANSWERED", "ANSWERED_BY_FAMILY_MEMBER", "CUSTOMER_PICKED", "AGENT_BUSY_ON_ANOTHER_CALL",
    "SILENCE_ISSUE", "LANGUAGE_BARRIER", "ANSWERED_VOICE_ISSUE", "CUSTOMER_ABUSIVE",
    "AUTOMATED_VOICE", "FORWARDED_CALL", "RINGING", "BUSY", "SWITCHED_OFF",
    "WRONG_NUMBER", "DO_NOT_KNOW_THE_PERSON", "NOT_IN_CONTACT_ANYMORE", "OUT_OF_NETWORK", "OUT_OF_SERVICES",
    "CALL_BACK_LATER", "WILL_ASK_TO_PAY", "GAVE_ALTERNATE_NUMBER",
    "ANSWERED_DISCONNECTED", "CALL_DISCONNECTED_BY_CUSTOMER", "NOT_AVAILABLE", "WRONG_PERSON", "OTHERS"
]

PAYMENT_DISPOSITIONS = [
    "PAID", "PTP", "PARTIAL_PAYMENT", "SETTLEMENT", "WILL_PAY_AFTER_VISIT",
    "DENIED_TO_PAY", "NO_PAYMENT_COMMITMENT", "NO_PROOF_GIVEN", "WANT_FORECLOSURE", "WANTS_TO_RENEGOTIATE_LOAN_TERMS",
    "None"
]

REASONS_FOR_NOT_PAYING = [
    "FUNDS_ISSUE", "TECHNICAL_ISSUE", "HEALTH_ISSUE", "SALARY_NOT_CREDITED", 
    "LOAN_ALREADY_PAID", "DISPUTE_ON_INTEREST", "CUSTOMER_OUT_OF_STATION",
    "CUSTOMER_NOT_TELLING_REASON", "JOB_CHANGED_WAITING_FOR_SALARY", "SERVICE_ISSUE", "OTHER_REASONS"
]

def map_disposition(val):
    if not val: return "OTHERS"
    val = str(val).upper().strip()
    if val in CALL_DISPOSITIONS: return val
    
    # Common mappings
    mapping = {
        "RNG": "RINGING",
        "CALL_DISCONNECTED": "ANSWERED_DISCONNECTED",
        "DISCONNECTED": "ANSWERED_DISCONNECTED",
        "W_N": "WRONG_NUMBER",
        "WRONG_NO": "WRONG_NUMBER",
        "SWITCHED OFF": "SWITCHED_OFF",
        "SILENCE": "SILENCE_ISSUE",
        "WRONG PERSON": "WRONG_PERSON",
        "DO NOT KNOW THE PERSON": "DO_NOT_KNOW_THE_PERSON",
    }
    return mapping.get(val, "OTHERS")

def map_payment_disposition(val):
    if not val or val == "null" or val == "NULL": return "None"
    val = str(val).upper().strip()
    if val in PAYMENT_DISPOSITIONS: return val
    if val == "PAYMENT_DONE": return "PAID"
    return "None"

def map_reason(val):
    if not val or val == "null" or val == "NULL": return None
    val = str(val).upper().strip().replace(" ", "_")
    if val in REASONS_FOR_NOT_PAYING: return val
    return "OTHER_REASONS"

def is_high_quality(transcript, output):
    """Filter out noise and dummy data."""
    if len(transcript) < 50: return False
    
    # Filter out dummy/test remarks
    remarks = str(output.get("remarks", "")).lower()
    if any(x in remarks for x in ["test", "dummy", "trial", "testing", "asdf"]):
        return False
        
    return True

def process_data_folder(folder_path):
    print(f"Scanning folder: {folder_path}")
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    processed_count = 0
    duplicate_count = 0
    unique_transcripts = {} # transcript_hash -> item
    
    # Using a simple hash if transcripts are very long to save memory
    import hashlib
    
    for file_path in json_files:
        print(f"  Processing {os.path.basename(file_path)}...")
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                # 1. Extract Transcript
                input_data = item.get('input', '')
                source_input_labels = {}
                if isinstance(input_data, dict):
                    transcript = input_data.get('transcript', '').strip()
                    # Backup labels if output is empty
                    source_input_labels = {
                        "disposition": input_data.get("disposition"),
                        "payment_disposition": input_data.get("payment_disposition")
                    }
                else:
                    transcript = str(input_data).strip()
                
                if not transcript: continue
                
                # Quality Filtering
                if not is_high_quality(transcript, item.get('output', {})):
                    continue
                
                if transcript in unique_transcripts:
                    duplicate_count += 1
                    continue
                
                # Standardize instruction
                item['instruction'] = "You are an AI assistant that extracts structured call disposition data.\n" \
                                     "Fields: disposition, payment_disposition, reason_for_not_paying, ptp_details, remarks.\n" \
                                     "Return ONLY valid JSON."

                # 2. Extract Output Data
                output_raw = item.get('output', {})
                if isinstance(output_raw, str):
                    try: 
                        loaded = json.loads(output_raw)
                        if isinstance(loaded, dict):
                            output_raw = loaded
                        else:
                            output_raw = {}
                    except: 
                        output_raw = {}
                
                if not isinstance(output_raw, dict): 
                    output_raw = {}
                
                # Merge source_input_labels as fallback if output is sparse
                final_disposition = output_raw.get("disposition") or source_input_labels.get("disposition")
                final_payment = output_raw.get("payment_disposition") or source_input_labels.get("payment_disposition")
                
                # 3. New Schema Mapping
                new_output = {
                    "disposition": map_disposition(final_disposition),
                    "payment_disposition": map_payment_disposition(final_payment),
                    "reason_for_not_paying": map_reason(output_raw.get("reason_for_not_paying")),
                    "ptp_details": {
                        "amount": output_raw.get("ptp_amount") or output_raw.get("ptp_details", {}).get("amount") if isinstance(output_raw.get("ptp_details"), dict) else output_raw.get("ptp_amount"),
                        "date": output_raw.get("ptp_date") or output_raw.get("ptp_details", {}).get("date") if isinstance(output_raw.get("ptp_details"), dict) else output_raw.get("ptp_date")
                    },
                    "remarks": output_raw.get("remarks")
                }
                
                item['output'] = new_output
                unique_transcripts[transcript] = item
                processed_count += 1
            
            # Explicitly clear data from memory
            del data
                
        except Exception as e:
            print(f"    Error processing {file_path}: {e}")
            
    print(f"Total processed Unique items: {processed_count}")
    print(f"Total duplicates skipped: {duplicate_count}")
    return list(unique_transcripts.values())

def main():
    cleaned_data = process_data_folder(DATA_DIR)
    
    print(f"Saving {len(cleaned_data)} unique items to {MASTER_OUTPUT}...")
    with open(MASTER_OUTPUT, 'w') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    print("Master data creation complete.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
