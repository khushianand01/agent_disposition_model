import json
import random
import os

MASTER_FILE = "/home/ubuntu/disposition_model/data/master_production_data.json"
AUGMENTED_FILE = "/home/ubuntu/disposition_model/data/master_production_data_augmented.json"

TEMPLATES = {
    "PARTIAL_PAYMENT": [
        "Agent: Hello, I am calling from MobiKwik regarding your EMI. Borrower: Yes, I know. But I can't pay the full amount right now. Can I pay 500 today? Agent: Sir, your total due is 2500. Borrower: I understand, but I only have 500. I will pay the rest next week. Agent: Okay, please pay 500 now to avoid penalty.",
        "Agent: Namaste, aapka payment pending hai. Borrower: Haan, main abhi partial payment kar raha hoon 1000 rupaye ka. Baaki baad mein doonga. Agent: Theek hai sir, link bhej raha hoon.",
        "Agent: Hello, your account is in overdue. Borrower: I can only pay a small part today. Maybe 300 rupees? Agent: Sir, please try to pay more. Borrower: No, only 300 possible now."
    ],
    "WILL_PAY_AFTER_VISIT": [
        "Agent: Hello, payment kab tak hoga? Borrower: Aap ghar pe kisi ko bhejo, main unko hi cash doonga. Agent: Sir, online pay kar dijiye. Borrower: Nahi, mere paas cash hai, visit karwao tabhi doonga.",
        "Agent: Payment due hai sir. Borrower: Haan, office se koi aayega tab payment hoga. Agent: Agent bheju? Borrower: Haan, bhej do kal subah.",
        "Agent: Payment pending. Borrower: Main online nahi karta. Mere ghar aao, paise le jao. Agent: Okay sir, visit schedule kar rahe hain."
    ],
    "WANT_FORECLOSURE": [
        "Agent: Hello, EMI pending. Borrower: Mujhe pura loan ek saath khatam karna hai. Kitna lagega foreclosure ke liye? Agent: Sir, let me check your closing balance. Borrower: Haan, calculate karke batao, main pura pay kar doonga.",
        "Agent: EMI payment? Borrower: Main pura loan band karwana chahta hoon. Process batao. Agent: Foreclosure charges lagenge sir. Borrower: Chalta hai, par loan close karo.",
        "Agent: Payment due. Borrower: I want to close my loan account permanently. How much total amount to pay today? Agent: One moment sir, calculating foreclosure amount."
    ],
    "WANTS_TO_RENEGOTIATE_LOAN_TERMS": [
        "Agent: Payment kab kar rahe hain? Borrower: Meri EMI bahut zyada hai. Isko kam kar sakte hain? Agent: Sir, terms fixed hain. Borrower: Nahi, agar EMI kam nahi ki toh main pay nahi kar paunga. Settlement ya restructure karo.",
        "Agent: Loan payment pending. Borrower: Interest bahut zyada hai. Mujhe kam interest rate chahiye tabhi bharunga. Agent: Sir, we can't change it now. Borrower: Toh fir main baat karunga manager se.",
        "Agent: Hello. Borrower: Mere ko loan ki terms pasand nahi aa rahi, thoda change karwao ya koi dusra scheme batao."
    ],
    "PAID": [
        "Agent: Hello, EMI payment update? Borrower: Haan, maine subah hi pay kar diya hai. Agent: Okay sir, receipt number batayenge? Borrower: Haan, message mein aaya hai. Main WhatsApp kar raha hoon.",
        "Agent: Namaste, payment pending. Borrower: Are main toh pehle hi bhar chuka hoon, check karo. Agent: System mein update nahi hai. Borrower: Maine PhonePe se kiya tha, success ho gaya.",
        "Agent: Payment update? Borrower: Payment is already done from my side yesterday."
    ],
    "SETTLEMENT": [
        "Agent: Settlement amount 5000 pay kar dijiye. Borrower: Haan, main settlement ke liye ready hoon. Itne mein close ho jayega na? Agent: Haan sir, 5000 mein no dues ho jayega. Borrower: Okay, link bhejo main abhi settle karta hoon.",
        "Agent: Settlement offer? Borrower: Haan mera settlement karwa do. Final kitna dena hoga? Agent: Sir 3000 lagega. Borrower: Done, bhej do payment option.",
        "Agent: Settlement process? Borrower: Mujhe loan settle karna hai, EMI nahi bhar paunga. Agent: Okay, settlement rate check karte hain."
    ],
    "DO_NOT_KNOW_THE_PERSON": [
        "Agent: Hello, Rahul baat kar rahe hain? Borrower: Nahi, galat number hai. Main kisi Rahul ko nahi janta. Agent: Sir yeh number unka diya hua hai. Borrower: Bhai main nahi janta, phone rakho.",
        "Agent: Hello, is it Mr. Sharma? Borrower: No, you have the wrong person. I don't know anyone by that name."
    ],
    "WRONG_NUMBER": [
        "Agent: Hello, Amit ji? Borrower: Kaun Amit? Yeh mera number hai, Amit koi nahi hai yahan. Agent: Okay, wrong number.",
        "Agent: Hello, is this the borrower? Borrower: No, this is a different person. Incorrect number."
    ],
    "OUT_OF_NETWORK": [
        "The number you are trying to reach is currently out of network coverage area.",
        "Aapka dial kiya hua number network kshetra se bahar hai."
    ],
    "SWITCHED_OFF": [
        "The number you are trying to reach is currently switched off.",
        "Aapka dial kiya gaya number abhi band hai."
    ],
    "GAVE_ALTERNATE_NUMBER": [
        "Agent: Hello, Anita ji? Borrower: Unka yeh number nahi lag raha, aap 9876543210 pe call karo. Agent: Okay, alternate number mil gaya.",
        "Agent: Hello. Borrower: Unka dusra number hai, woh number note kar lo. Agent: Haan bataiye."
    ]
}

# Additional templates for more variety
TEMPLATES.update({
    "WANT_FORECLOSURE": [
        "Agent: Hello, calling from MobiKwik for your loan. Borrower: I want to close my loan fully today. What is the final amount? Agent: Let me check... sir, including foreclosure charges it is {amount}. Borrower: Okay, send the link, I will clear it now.",
        "Agent: Namaste, aapka EMI pending hai. Borrower: Mujhe EMI nahi bharna, pura loan close karwana hai. Agent: Foreclosure karana chahte hain? Borrower: Haan, pura payment ek saath karna hai. Amount batado.",
        "Agent: Hello sir, payment detail? Borrower: I have enough funds now, I want to foreclosure the account. How much to pay? Agent: One minute sir... it is {amount}. Borrower: Fine, bhej do link.",
        "Agent: Hello. Borrower: Mere ko apna loan account abhi band karna hai permanently. Process kya hai? Agent: Sir foreclosure request dalni hogi.",
        "Agent: Regarding your MobiKwik loan... Borrower: Can I pay everything today and close it? I don't want more EMIs. Agent: Yes sir, foreclosure is possible."
    ],
    "SETTLEMENT": [
        "Agent: Sir, settlement offer available hai {amount} mein. Borrower: Itna zyada? 3000 mein karlo. Agent: Nahi sir, {amount} final hai. Borrower: Okay theek hai, kar do close.",
        "Agent: Hello, your account is in deep overdue. Borrower: I can't pay full. Any settlement? Agent: Yes, pay {amount} for one-time settlement. Borrower: Done, send me the details.",
        "Agent: Hi, regarding your loan... Borrower: Mujhe settle karna hai balance. Kitna discount milega? Agent: Sir {amount} pay karna hoga settlement ke liye.",
        "Agent: Namaste. Borrower: Kya mera settlement ho sakta hai? Agent: Haan sir, {amount} bharna hoga."
    ]
})

REASONS = ["FUNDS_ISSUE", "TECHNICAL_ISSUE", "OTHER_REASONS", "SALARY_NOT_CREDITED"]
NAMES = ["Rahul", "Amit", "Suresh", "Anita", "Priya", "Vikram", "Deepak", "Sunita"]
DATES = ["tomorrow", "next Monday", "by Friday", "within 2 days", "next week", "25th of this month"]

def mutate_text(text):
    """Adds random variation to template text."""
    # Replace amount
    amt = str(random.randint(1, 50) * 100)
    text = text.replace("{amount}", amt)
    
    # Randomly inject names or change phrasing slightly
    if "{name}" in text:
        text = text.replace("{name}", random.choice(NAMES))
    
    # Occasionally append a common filler
    fillers = ["", " Please check.", " Urgent.", " Let me know.", " Thank you."]
    text += random.choice(fillers)
    
    return text

def augment():
    with open(MASTER_FILE, 'r') as f:
        data = json.load(f)
    
    # Clean out old synthetic entries if re-running
    data = [i for i in data if "Synthetic data" not in str(i.get('remarks', ''))]
    
    new_items = []
    
    # Priority Targets
    TARGETS = {
        "PARTIAL_PAYMENT": 1000,
        "WILL_PAY_AFTER_VISIT": 1000,
        "WANT_FORECLOSURE": 1000,
        "WANTS_TO_RENEGOTIATE_LOAN_TERMS": 1000,
        "PAID": 1500,
        "SETTLEMENT": 1500,
        "DO_NOT_KNOW_THE_PERSON": 500,
        "WRONG_NUMBER": 1200,
        "OUT_OF_NETWORK": 300,
        "SWITCHED_OFF": 300,
        "GAVE_ALTERNATE_NUMBER": 300,
    }

    for label, target in TARGETS.items():
        texts = TEMPLATES.get(label, ["Transcript for " + label])
        current_count = len([i for i in data if (i['output']['payment_disposition'] == label or i['output']['disposition'] == label)])
        to_add = max(0, target - current_count)
        
        print(f"Augmenting {label}: current {current_count}, adding {to_add}")
        
        seen_texts = set()
        for _ in range(to_add):
            # Try to produce a unique mutation
            max_tries = 10
            template = random.choice(texts)
            mutated = mutate_text(template)
            while mutated in seen_texts and max_tries > 0:
                mutated = mutate_text(template)
                max_tries -= 1
            
            seen_texts.add(mutated)
            
            # Determine which field to set based on label type
            call_disp = "ANSWERED"
            pay_disp = "None"
            
            if label in ["PAID", "PTP", "PARTIAL_PAYMENT", "SETTLEMENT", "WILL_PAY_AFTER_VISIT", "DENIED_TO_PAY", "NO_PAYMENT_COMMITMENT", "NO_PROOF_GIVEN", "WANT_FORECLOSURE", "WANTS_TO_RENEGOTIATE_LOAN_TERMS"]:
                pay_disp = label
            else: 
                call_disp = label
            
            # Generate Natural Remarks
            remark_options = {
                "PARTIAL_PAYMENT": [f"Customer paid {pay_disp} of {random.randint(500, 2000)}", "Partial payment made, promised rest later", "Borrower only had limited funds for today"],
                "WILL_PAY_AFTER_VISIT": ["Wants executive to visit for cash collection", "Requested doorstep service for payment", "Will pay only after home visit is done"],
                "WANT_FORECLOSURE": ["Customer wants to close the loan account fully", "Inquired about foreclosure charges and closing", "Requested total closing amount to settle all EMIs"],
                "WANTS_TO_RENEGOTIATE_LOAN_TERMS": ["Borrower is asking for EMI reduction", "Requested term extension or interest rate change", "Wants to renegotiate loan terms due to high EMI"],
                "PAID": ["Payment already completed via online channel", "Customer confirmed payment success", "Borrower says payment is done, check system"],
                "SETTLEMENT": [f"Agreed to settlement for {random.randint(2000, 5000)}", "One-time settlement requested by customer", "Wants to settle account with discount"],
                "DO_NOT_KNOW_THE_PERSON": ["Wrong person reached, knows no such borrower", "Incorrect contact details, recipient is a stranger", "Person says they don't know the borrower"],
                "WRONG_NUMBER": ["Reached an incorrect phone number", "Wrong number, Amit/Rahul not at this contact", "Not the intended recipient"],
                "OUT_OF_NETWORK": ["Call failed: Subscriber out of network", "Automated: Recipient out of coverage area"],
                "SWITCHED_OFF": ["Phone is currently switched off/unavailable", "Handset is powered off"],
                "GAVE_ALTERNATE_NUMBER": ["Provided another contact number for borrower", "Requested to call on a different mobile number"]
            }
            
            natural_remark = random.choice(remark_options.get(label, [f"Conversation about {label} completed"]))
            
            item = {
                "instruction": "You are an AI assistant that extracts structured call disposition data.\nFields: disposition, payment_disposition, reason_for_not_paying, ptp_details, remarks.\nReturn ONLY valid JSON.",
                "input": {
                    "transcript": mutated,
                    "disposition": call_disp,
                    "payment_disposition": pay_disp
                },
                "output": {
                    "disposition": call_disp,
                    "payment_disposition": pay_disp,
                    "reason_for_not_paying": random.choice(REASONS) if pay_disp != "None" else None,
                    "ptp_details": {
                        "amount": random.randint(100, 5000) if pay_disp == "PARTIAL_PAYMENT" else None,
                        "date": random.choice(DATES) if pay_disp == "PARTIAL_PAYMENT" else None
                    },
                    "remarks": natural_remark
                }
            }
            new_items.append(item)
    
    full_data = data + new_items
    print(f"Total items after augmentation: {len(full_data)}")
    
    with open(MASTER_FILE, 'w') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    augment()
