import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import os
import random
import re

# File paths
INPUT_MAIN = 'data/raw/missing_data.xlsx'
INPUT_AUGMENT = 'data/raw/calls_raw.xlsx'
OUTPUT_JSON = 'data/calls_data.json'

def clean_val(val):
    """Deep cleaning of a single value for JSON serializability."""
    if pd.isna(val) or str(val).strip().lower() in ['nat', 'nan', 'null', 'none', '']:
        return None
    if isinstance(val, (pd.Timestamp, np.datetime64)) or 'datetime' in str(type(val)).lower():
        try:
            return str(val).split(' ')[0]
        except:
            return None
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val

def format_transcript(text):
    if not isinstance(text, str):
        return ""
    # Standardize Speaker labels
    text = text.replace("Speaker 0:", "Agent:")
    text = text.replace("Speaker 1:", "Borrower:")
    # Remove any other Speaker labels
    text = re.sub(r"Speaker \d+:", "", text)
    # Clean redundant spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def map_payment_disposition(val):
    if pd.isna(val): return None
    v = str(val).upper().strip()
    
    # PTP Mapping
    if v in ['PTP', 'WILLING_TO_PAY', 'PROMISE_TO_PAY']:
        return 'PTP'
    
    # PAID Mapping
    if v in ['PAID', 'ALREADY_PAID_BEFORE_CALL', 'FULL_PAID_ON_CALL', 'PAYMENT_COMPLETED']:
        return 'PAID'
    
    # SETTLEMENT Mapping
    if v in ['SETTLEMENT', 'WANT_TO_SETTLE', 'SETTLED_AFTER_CALL', 'ALREADY_SETTLED_BEFORE_CALL', 'SETTLEMENT_NEGOTIATION']:
        return 'SETTLEMENT'
    
    # PARTIAL_PAYMENT Mapping
    if v in ['PARTIAL_PAYMENT', 'PARTIALLY_PAID_AFTER_CALL']:
        return 'PARTIAL_PAYMENT'
    
    return val

def clean_remarks(remarks):
    if not remarks or not isinstance(remarks, str) or "Synthetic" in remarks:
        return None
    
    # Replace variations like 'cx', 'cm', 'sm', 'cs', 'cus', 'cust' with 'customer'
    # Removed 'ch' from here because it conflicts with Hindi words (e.g., 'chuka', 'chal')
    remarks = re.sub(r'\b(cx|cm|sm|cs|cus|cust)\b', 'customer', remarks, flags=re.IGNORECASE)
    
    # Specific safe replacement for 'ch' when followed by context clues
    # e.g., "ch said", "ch bol", "ch is", "ch ko"
    remarks = re.sub(r'\bch\s+(?=(said|bol|say|ask|denied|pay|is|ko|ne|ka|ki))\b', 'customer ', remarks, flags=re.IGNORECASE)
    
    return remarks.strip()

INSTRUCTION = """You are an AI assistant that extracts structured call disposition data.
Given a call transcript between an agent and a borrower, extract the following fields. Return ONLY valid JSON. Do not explain.:
disposition, payment_disposition, reason_for_not_paying, ptp_amount, ptp_date, followup_date, remarks.
Current Date: 2026-01-27
If a field is not present, return null."""

def process_dataframe(df, source_name):
    print(f"Processing {len(df)} rows from {source_name}...")
    processed_entries = []
    dropped_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Clean basic fields
        disposition = clean_val(row['disposition'])
        transcription = format_transcript(row['transcription'])
        
        # 2. FILTER: Remove if disposition OR transcription is null/empty
        if not disposition or not transcription or transcription.strip() == "":
            dropped_count += 1
            continue

        # 3. Clean remaining fields
        output_obj = {
            "disposition": disposition,
            "payment_disposition": map_payment_disposition(row['payment_disposition']),
            "reason_for_not_paying": clean_val(row['reason_for_not_paying']),
            "ptp_amount": clean_val(row['ptp_amount']),
            "ptp_date": clean_val(row['ptp_date']),
            "followup_date": clean_val(row['followup_date']),
            "remarks": clean_remarks(str(row['remarks'])) if pd.notna(row['remarks']) else None
        }
        
        # Ensure all values in output_obj are JSON serializable
        output_obj = {k: clean_val(v) for k, v in output_obj.items()}
        
        entry = {
            "instruction": INSTRUCTION,
            "input": transcription,
            "output": output_obj
        }
        processed_entries.append(entry)
        
    print(f"[{source_name}] Dropped {dropped_count} rows due to null transcripts or dispositions.")
    print(f"[{source_name}] Preserved: {len(processed_entries)}")
    return processed_entries

def main():
    # 1. Load Main Data
    if os.path.exists(INPUT_MAIN):
        print(f"Loading {INPUT_MAIN}...")
        df_main = pd.read_excel(INPUT_MAIN)
        entries_main = process_dataframe(df_main, "Main Data")
    else:
        print(f"Error: {INPUT_MAIN} not found.")
        return

    # 2. Load Augment Data
    if os.path.exists(INPUT_AUGMENT):
        print(f"Loading {INPUT_AUGMENT}...")
        df_aug = pd.read_excel(INPUT_AUGMENT)
        entries_aug = process_dataframe(df_aug, "Augment Data")
    else:
        print(f"Warning: {INPUT_AUGMENT} not found. Skipping augmentation.")
        entries_aug = []

    # 3. Merge and Deduplicate
    print("\nMerging and Deduplicating...")
    seen_transcripts = set()
    final_data = []
    
    # Add main data first (priority)
    for entry in entries_main:
        t = entry['input']
        # Use a hash or just the string if not too long. String is safer for exact match.
        # Normalize slightly more to catch near-duplicates? No, exact match of cleaned text is safer.
        if t not in seen_transcripts:
            seen_transcripts.add(t)
            final_data.append(entry)
    
    main_count = len(final_data)
    print(f"Main data unique entries: {main_count}")
    
    # Add augment data if unique
    aug_added_count = 0
    for entry in entries_aug:
        t = entry['input']
        if t not in seen_transcripts:
            seen_transcripts.add(t)
            final_data.append(entry)
            aug_added_count += 1
            
    print(f"Augmented with {aug_added_count} unique rows from {INPUT_AUGMENT}")
    print(f"Total Combined Dataset: {len(final_data)}")

    # 4. Save
    print(f"Saving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False, default=lambda x: None if pd.isna(x) else str(x))

    print("Process completed successfully.")

if __name__ == "__main__":
    main()
