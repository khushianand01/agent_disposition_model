#!/usr/bin/env python3
import argparse
import json
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import sys
import os

# Add parent dir to path to import api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.inference import get_model

def get_normalized_disposition(response_dict):
    """Safely extracts the main disposition string."""
    if not isinstance(response_dict, dict):
        return "UNKNOWN"
    disp = response_dict.get('disposition', "UNKNOWN")
    if disp is None:
        return "UNKNOWN"
    return str(disp).strip().upper().replace(" ", "_")

def get_equivalent_disposition(disp):
    """Groups similar labels to allow for 'less strict' fair scoring."""
    disp = disp.upper().replace(" ", "_")
    
    # Group 1: System/Automated picks
    if disp in ["AUTOMATED_VOICE", "FORWARDED_CALL", "RINGING_DISCONNECTED"]:
        return "SYSTEM_PICKUP"
    
    # Group 2: Wrong Person / Wrong Number
    if disp in ["WRONG_NUMBER", "DO_NOT_KNOW_THE_PERSON", "WRONG_PERSON", "NOT_IN_CONTACT_ANYMORE"]:
        return "WRONG_PERSON_GROUP"
    
    # Group 3: Availability Issues
    if disp in ["BUSY", "RINGING", "NOT_AVAILABLE", "ANSWERED_DISCONNECTED"]:
        return "UNAVAILABLE_GROUP"
    
    # Group 4: Network issues
    if disp in ["SWITCHED_OFF", "OUT_OF_NETWORK", "OUT_OF_SERVICES"]:
        return "OFFLINE_GROUP"

    return disp

def normalize_amount(x):
    """Normalizes amount to integer for exact match."""
    if x is None:
        return None
    try:
        return int(float(str(x).replace(',', '')))
    except:
        return None

def normalize_date(x):
    """Normalizes date string."""
    if not x:
        return None
    return str(x).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default='data/production/test_best.json')
    parser.add_argument('--samples', type=int, default=200, help='Number of samples to evaluate')
    args = parser.parse_args()

    print(f"Loading test data from {args.test}...")
    with open(args.test, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use Random Sampling to prevent ordering bias
    n = min(len(data), args.samples)
    test_samples = random.sample(data, n)
    print(f"Evaluating {n} RANDOM samples for Production Readiness.\n")

    print('Loading Disposition Model...')
    model = get_model()

    # Metric Containers
    y_true_disp = []
    y_pred_disp = []
    
    y_true_pay = []
    y_pred_pay = []

    y_true_reason = []
    y_pred_reason = []

    # KPI Accumulators
    json_valid_count = 0
    amt_matches = 0
    amt_possible = 0
    date_matches = 0
    date_possible = 0

    # Collection for Audit
    audit_data = []

    print("\nRunning Production Grade Inference Engine...")
    for i, d in enumerate(test_samples):
        # Ground Truth
        gold = d['output']
        g_disp = get_normalized_disposition(gold)
        g_pay = str(gold.get('payment_disposition', "NONE")).strip().upper().replace(" ", "_")
        g_reason = str(gold.get('reason_for_not_paying', "NONE")).strip().upper().replace(" ", "_")
        
        g_ptp = gold.get('ptp_details', {})
        g_amt = normalize_amount(g_ptp.get('amount'))
        g_date = normalize_date(g_ptp.get('date'))

        # Model Prediction
        try:
            pred = model.predict(d['input'])
            if not isinstance(pred, dict) or "error" in pred:
                pred_raw = pred
                pred = {}
            else:
                json_valid_count += 1
                pred_raw = pred
        except Exception as e:
            pred = {} 
            pred_raw = {"error": str(e)}

        p_disp = get_normalized_disposition(pred)
        p_pay = str(pred.get('payment_disposition', "NONE")).strip().upper().replace(" ", "_")
        p_reason = str(pred.get('reason_for_not_paying', "NONE")).strip().upper().replace(" ", "_")
        
        # Apply Logic Equivalence for Fair Scoring
        g_disp_equiv = get_equivalent_disposition(g_disp)
        p_disp_equiv = get_equivalent_disposition(p_disp)

        p_ptp = pred.get('ptp_details', {})
        p_amt = normalize_amount(p_ptp.get('amount'))
        p_date = normalize_date(p_ptp.get('date'))

        # Audit Storage
        audit_data.append({
            "id": i + 1,
            "input": d['input'],
            "ground_truth": gold,
            "prediction": pred_raw,
            "match_disp": g_disp_equiv == p_disp_equiv
        })

        # Save Audit Results Incrementally (Every 5 samples)
        if (i + 1) % 5 == 0 or (i + 1) == n:
            audit_file = "docs/evaluation_results_audit.json"
            with open(audit_file, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False)

        # Collect for Classification Reports (Using Equivalence)
        y_true_disp.append(g_disp_equiv)
        y_pred_disp.append(p_disp_equiv)
        
        y_true_pay.append(g_pay)
        y_pred_pay.append(p_pay)

        y_true_reason.append(g_reason)
        y_pred_reason.append(p_reason)

        # Amount/Date Accuracy (Only check if Gold has a value)
        if g_amt is not None:
            amt_possible += 1
            if g_amt == p_amt:
                amt_matches += 1
        
        if g_date is not None:
            date_possible += 1
            if g_date == p_date:
                date_matches += 1

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{n} conversations processed...")

    print(f"\n✅ Final Audit results saved to: evaluation_results_audit.json")

    # Binary PTP KPI Calculation
    ptp_true_bin = [1 if x == "PTP" else 0 for x in y_true_pay]
    ptp_pred_bin = [1 if x == "PTP" else 0 for x in y_pred_pay]

    # Overall Results
    print("\n" + "="*65)
    print("              FINAL PRODUCTION READINESS REPORT              ")
    print("="*65 + "\n")

    json_validity = (json_valid_count / n) * 100
    print(f"🧩 JSON VALIDITY RATE: {json_validity:.2f}%  (Target: 100%)")
    
    disp_acc = accuracy_score(y_true_disp, y_pred_disp) * 100
    print(f"🎯 DISPOSITION ACCURACY: {disp_acc:.1f}%  (Target: >90%)")

    ptp_prec = precision_score(ptp_true_bin, ptp_pred_bin, zero_division=0) * 100
    ptp_rec = recall_score(ptp_true_bin, ptp_pred_bin, zero_division=0) * 100
    print(f"🔥 PTP PRECISION (KPI): {ptp_prec:.1f}%")
    print(f"🔥 PTP RECALL (KPI):    {ptp_rec:.1f}%  (Target: >92%)")

    if amt_possible > 0:
        amt_acc = (amt_matches / amt_possible) * 100
        print(f"💰 AMOUNT ACCURACY:     {amt_acc:.1f}%  (Target: >85%)")
    
    if date_possible > 0:
        date_acc = (date_matches / date_possible) * 100
        print(f"📅 DATE ACCURACY:       {date_acc:.1f}%  (Target: >85%)")

    print("\n--- 📂 DETAILED CLASSIFICATION REPORTS ---")
    print("\n[DISPOSITION]")
    print(classification_report(y_true_disp, y_pred_disp, zero_division=0))
    
    print("\n[PAYMENT STATUS]")
    print(classification_report(y_true_pay, y_pred_pay, zero_division=0))

    print("\n[REASON CLASSIFICATION]")
    print(classification_report(y_true_reason, y_pred_reason, zero_division=0))

    print("\n" + "="*65)
    print("            STATUS: VERIFY F1-SCORES FOR DEPLOYMENT            ")
    print("="*65)

if __name__ == '__main__':
    main()
