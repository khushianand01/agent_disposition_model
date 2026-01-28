import json
import random
import datetime

OUTPUT_FILE = 'data/synthetic_data.json'
TARGET_COUNT = 1000

# Templates for PARTIAL_PAYMENT
partial_templates = [
    {
        "agent": ["Hello sir, payment due hai aapka.", "Loan ki EMI pending hai sir.", "Amount pay kar dijiye aaj last date hai."],
        "borrower": ["Haan pata hai, par abhi full amount nahi hai.", "Thoda problem chal raha hai, aadha hi de paunga.", "Salary late aayi hai, 5000 abhi le lo."],
        "agent_response": ["Thik hai, abhi itna kara do.", "Ok receive kar lenge, baki kab doge?", "Chalo thik hai, screenshot bhej dena."],
        "disposition": "PARTIAL_PAYMENT",
        "remarks_pool": ["customer agreed for partial payment", "customer will pay remaining amount next week", "part payment received", "customer has financial issue, paying half"]
    },
    {
        "agent": ["Sir legal action lena padega agar pay nahi kiya.", "Aapka case manager bol raha hu, payment chahiye."],
        "borrower": ["Darao mat sir, main bhag nahi raha. Thoda paisa hai abhi.", "Sirf part payment ho payega aaj.", "Pura nahi hai, aadha adjust kar lo."],
        "agent_response": ["Kab tak karoge baki?", "Aaj 50% kar do phir.", "Ok note kar raha hu part payment."],
        "disposition": "PARTIAL_PAYMENT",
        "remarks_pool": ["customer paying 50% only", "agreed to partial payment", "customer request to adjust amount"]
    },
    {
         "agent": ["Good morning sir, loan department se.", "Recovery team se bol raha hu."],
         "borrower": ["Sir medical emergency thi, pura paisa kharch ho gaya. Thoda le lo.", "Abhi partial amount hi arrange hua hai."],
         "agent_response": ["Kitna bhej rahe ho?", "Thik hai partial update kar deta hu."],
         "disposition": "PARTIAL_PAYMENT",
         "remarks_pool": ["medical issue partial payment", "customer arranging funds", "will pay partial amount"]
    }
]

# Templates for SETTLEMENT
settlement_templates = [
    {
        "agent": ["Sir loan close karna hai ya nahi?", "Bohot overdue ho gaya hai aapka account.", "Payment kyu nahi aa raha sir?"],
        "borrower": ["Sir close karna hai par interest bohot lag gaya. Settlement chahiye.", "Main one time settlement karna chahta hu.", "Discount milega toh aaj hi close kar dunga."],
        "agent_response": ["Kitne mein close karna hai?", "Thik hai main manager se baat karta hu settlement ke liye.", "Ok settlement process initiate karte hain."],
        "disposition": "SETTLEMENT",
        "remarks_pool": ["customer wants settlement", "requesting one time settlement", "asking for discount to close loan"]
    },
    {
        "agent": ["Notice bhej diya hai humne.", "Police bhejni padegi kya sir?"],
        "borrower": ["Sir dhamki mat do. Mere paas job nahi hai. Settlement kara do.", "Principal amount le lo aur kissa khatam karo.", "Bina byaj ke settlement ho sakta hai?"],
        "agent_response": ["Proposal dijiye apna.", "Ok 50k mein settlement try karte hain.", "Request dal raha hu settlement ki."],
        "disposition": "SETTLEMENT",
        "remarks_pool": ["job loss settlement request", "customer wants to pay principal only", "settlement proposal discussion"]
    }
]

def generate_conversation(templates):
    t = random.choice(templates)
    
    agent = random.choice(t["agent"])
    borrower = random.choice(t["borrower"])
    agent_res = random.choice(t["agent_response"])
    remark = random.choice(t["remarks_pool"])
    
    # Construct Transcript
    transcript = f"Agent: {agent}\nBorrower: {borrower}\nAgent: {agent_res}"
    
    # Generate dates/amounts dynamically based on context for realism
    ptp_amt = None
    reason = None
    
    if t["disposition"] == "PARTIAL_PAYMENT":
        ptp_amt = str(random.choice([2000, 5000, 10000, 15000]))
        reason = "Financial Issue"
    elif t["disposition"] == "SETTLEMENT":
        ptp_amt = str(random.choice([50000, 75000, 100000]))
        reason = "Wants Settlement"
        
    return {
        "instruction": "Extract the call disposition from the transcript.",
        "input": transcript,
        "output": {
            "disposition": "ANSWERED",
            "payment_disposition": t["disposition"],
            "reason_for_not_paying": reason,
            "ptp_amount": ptp_amt,
            "ptp_date": None, # Could add date logic if needed
            "followup_date": None,
            "remarks": remark
        }
    }

def main():
    synthetic_data = []
    
    # Generate PARTIAL_PAYMENT
    print(f"Generating {TARGET_COUNT} PARTIAL_PAYMENT samples...")
    for _ in range(TARGET_COUNT):
        synthetic_data.append(generate_conversation(partial_templates))
        
    # Generate SETTLEMENT
    print(f"Generating {TARGET_COUNT} SETTLEMENT samples...")
    for _ in range(TARGET_COUNT):
        synthetic_data.append(generate_conversation(settlement_templates))
    
    print(f"Total synthetic samples: {len(synthetic_data)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(synthetic_data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
