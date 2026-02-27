import json
import random
import os
from datetime import datetime

# Define the languages
LANGUAGES = [
    "hindi", "english", "bengali", "marathi", "telugu",
    "tamil", "gujarati", "kannada", "malayalam", "punjabi"
]

# Output directory for datasets
EVAL_DIR = "eval_datasets"
os.makedirs(EVAL_DIR, exist_ok=True)

# Valid labels defined from the model's expected outputs
VALID_DISPOSITIONS = ["ANSWERED", "ANSWERED_BY_FAMILY_MEMBER", "WRONG_NUMBER", "DO_NOT_KNOW_THE_PERSON"]
VALID_PAY_DISPOSITIONS = ["PTP", "PARTIAL_PAYMENT", "PAID", "DENIED_TO_PAY", "WILL_PAY_AFTER_VISIT", "WANTS_TO_RENEGOTIATE_LOAN_TERMS", "None"]
VALID_REASONS = ["FUNDS_ISSUE", "TECHNICAL_ISSUE", "UNEMPLOYED", "OTHER_REASONS", "MEDICAL_ISSUE", "JOB_CHANGED_WAITING_FOR_SALARY", "RATE_OF_INTEREST_ISSUES", "None"]

# Helper to generate randomized data per language
def generate_samples(language, count=200):
    samples = []
    
    # Simple templates for each language showing basic structure to generate permutations
    templates = {
        "hindi": {
            "PTP": "Agent: नमस्कार, તમારું પેમેન્ટ ક્યારે આવશે? Borrower: मैं {date} को {amount} रुपये दे दूंगा।",
            "PAID": "Agent: हेलो, लोन अमाउंट पेंडिंग है। Borrower: मैंने तो कल ही पेमेंट कर दिया था, वो कट भी गया।",
            "WRONG_NUMBER": "Agent: क्या मैं राहुल से बात कर रहा हूँ? Borrower: नहीं, यह गलत नंबर है।",
            "DENIED": "Agent: सर आपकी EMI पेंडिंग है। Borrower: मेरी नौकरी चली गई है, मैं कुछ नहीं दे सकता।",
            "VISIT": "Agent: पेमेंट कब करेंगे? Borrower: आप किसी को घर भेज दो, मैं कैश दे दूंगा।"
        },
        "english": {
            "PTP": "Agent: Hello, when can we expect the payment? Borrower: I will pay {amount} on {date}.",
            "PAID": "Agent: Your EMI is due. Borrower: I already transferred the money yesterday through GPay.",
            "WRONG_NUMBER": "Agent: Am I speaking to Rahul? Borrower: No, you have the wrong number.",
            "DENIED": "Agent: Sir, please pay your dues. Borrower: I lost my job and have no money. I won't pay.",
            "VISIT": "Agent: How will you pay? Borrower: Send your collection agent to my house, I will pay cash."
        },
        "bengali": {
            "PTP": "Agent: নমস্কার, পেমেন্ট কবে করবেন? Borrower: আমি {date} তারিখে {amount} টাকা দেব।",
            "PAID": "Agent: আপনার কিস্তি বাকি আছে। Borrower: আমি তো কালকেই টাকা পাঠিয়ে দিয়েছি।",
            "WRONG_NUMBER": "Agent: রাহুল বলছেন? Borrower: না, এটা ভুল নম্বর।",
            "DENIED": "Agent: পেমেন্ট কবে পাব? Borrower: আমার চাকরি নেই, আমি টাকা দিতে পারব না।",
            "VISIT": "Agent: কিভাবে টাকা দেবেন? Borrower: আমার বাড়িতে লোক পাঠান, আমি ক্যাশ দিয়ে দেব।"
        },
        "marathi": {
            "PTP": "Agent: नमस्कार, तुमचा हप्ता कधी भरणार? Borrower: मी {date} ला {amount} रुपये भरीन.",
            "PAID": "Agent: तुमचे लोन पेंडिंग आहे. Borrower: मी कालच Google Pay वरून पैसे भरले आहेत.",
            "WRONG_NUMBER": "Agent: मी राहुलशी बोलत आहे का? Borrower: नाही, हा चुकीचा नंबर आहे.",
            "DENIED": "Agent: पैसे कधी भरणार? Borrower: माझी नोकरी गेली आहे, मी पैसे भरू शकत नाही.",
            "VISIT": "Agent: पेमेंट कसे करणार? Borrower: कोणालातरी घरी पाठवा, मी रोख रक्कम देईन."
        },
        "telugu": {
            "PTP": "Agent: నమస్కారం, మీరు EMI ఎప్పుడు కడతారు? Borrower: నేను {date} నాటికి {amount} రూపాయలు కడతాను.",
            "PAID": "Agent: మీ పేమెంట్ బాకీ ఉంది. Borrower: నేను నిన్ననే ఆన్‌లైన్ లో కట్టేసాను.",
            "WRONG_NUMBER": "Agent: ఇది రాహుల్ నెంబరా? Borrower: కాదు, రాంగ్ నెంబర్.",
            "DENIED": "Agent: మీరు డబ్బులు ఎప్పుడు కడతారు? Borrower: నా ఉద్యోగం పోయింది, నేను కట్టలేను.",
            "VISIT": "Agent: పేమెంట్ ఎలా చేస్తారు? Borrower: ఇంటికి ఎవరినైనా పంపండి, నేను క్యాష్ ఇస్తాను."
        },
        "tamil": {
            "PTP": "Agent: வணக்கம், EMI எப்போது கட்டுவீர்கள்? Borrower: நான் {date} அன்று {amount} ரூபாய் கட்டுகிறேன்.",
            "PAID": "Agent: உங்கள் லோன் நிலுவையில் உள்ளது. Borrower: நான் நேற்றே GPay மூலம் கட்டிவிட்டேன்.",
            "WRONG_NUMBER": "Agent: ராகுல் பேசுகிறீர்களா? Borrower: இல்லை, இது ராங் நம்பர்.",
            "DENIED": "Agent: பணம் எப்போது கட்டுவீர்கள்? Borrower: எனக்கு வேலை போய்விட்டது, என்னால் கட்ட முடியாது.",
            "VISIT": "Agent: எப்படி பணம் கட்டுவீர்கள்? Borrower: வீட்டிற்கு ஆள் அனுப்புங்கள், நான் ரொக்கமாக தருகிறேன்."
        },
        "gujarati": {
            "PTP": "Agent: નમસ્તે, તમે EMI ક્યારે ભરશો? Borrower: હું {date} ના રોજ {amount} રૂપિયા ભરીશ.",
            "PAID": "Agent: તમારું પેમેન્ટ બાકી છે. Borrower: મેં ગઈકાલે જ ઓનલાઇન ભરી દીધું છે.",
            "WRONG_NUMBER": "Agent: શું આ રાહુલ નો નંબર છે? Borrower: ના, આ ખોટો નંબર છે.",
            "DENIED": "Agent: તમે પૈસા ક્યારે ભરશો? Borrower: મારી નોકરી જતી રહી છે, હું નહિ ભરી શકું.",
            "VISIT": "Agent: તમે પેમેન્ટ કેવી રીતે કરશો? Borrower: ઘરે કોઈને મોકલી દો, હું રોકડા આપી દઈશ."
        },
        "kannada": {
            "PTP": "Agent: ನಮಸ್ಕಾರ, ನೀವು EMI ಯಾವಾಗ ಕಟ್ಟುತೀರಾ? Borrower: ನಾನು {date} ರಂದು {amount} ರೂಪಾಯಿ ಕಟ್ಟುತೀನಿ.",
            "PAID": "Agent: ನಿಮ್ಮ ಸಾಲ ಬಾಕಿ ಇದೆ. Borrower: ನಾನು ನಿನ್ನೆನೇ GPay ಮೂಲಕ ಕಟ್ಟಿದ್ದೀನಿ.",
            "WRONG_NUMBER": "Agent: ರಾಹುಲ್ ಅವರೇನಾ? Borrower: ಇಲ್ಲ, ಇದು ರಾಂಗ್ ನಂಬರ್.",
            "DENIED": "Agent: ದುಡ್ಡು ಯಾವಾಗ ಕಟ್ಟುತೀರಾ? Borrower: ನನ್ನ ಕೆಲಸ ಹೋಗಿದೆ, ನಾನು ಕಟ್ಟೋಕೆ ಆಗಲ್ಲ.",
            "VISIT": "Agent: ಪೇಮೆಂಟ್ ಹೇಗೆ ಮಾಡ್ತೀರಾ? Borrower: ಮನೆಗೆ ಯಾರನ್ನಾದ್ರೂ ಕಳಿಸಿ, ನಾನು ಕ್ಯಾಶ್ ಕೊಡ್ತೀನಿ."
        },
        "malayalam": {
            "PTP": "Agent: നമസ്കാരം, നിങ്ങൾ EMI എന്ന് അടയ്ക്കും? Borrower: ഞാൻ {date} ന് {amount} രൂപ അടയ്ക്കാം.",
            "PAID": "Agent: നിങ്ങളുടെ ലോൺ കുടിശ്ശികയാണ്. Borrower: ഞാൻ ഇന്നലെ തന്നെ ഓൺലൈനായി അടച്ചു.",
            "WRONG_NUMBER": "Agent: രാഹുൽ ആണോ? Borrower: അല്ല, ഇത് റോങ്ങ് നമ്പർ ആണ്.",
            "DENIED": "Agent: എപ്പോഴാണ് പണം അടയ്ക്കുക? Borrower: എനിക്ക് ജോലി നഷ്ടപ്പെട്ടു, എനിക്ക് അടയ്ക്കാൻ കഴിയില്ല.",
            "VISIT": "Agent: എങ്ങനെയാണ് പണം അടയ്ക്കുക? Borrower: വീട്ടിലേക്ക് ആളെ വിടൂ, ഞാൻ ക്യാഷ് ആയി തരാം."
        },
        "punjabi": {
            "PTP": "Agent: ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ EMI ਕਦੋਂ ਭਰੋਗੇ? Borrower: ਮੈਂ {date} ਨੂੰ {amount} ਰੁਪਏ ਭਰਾਂਗਾ।",
            "PAID": "Agent: ਤੁਹਾਡਾ ਲੋਨ ਬਕਾਇਆ ਹੈ। Borrower: ਮੈਂ ਕੱਲ੍ਹ ਹੀ ਆਨਲਾਈਨ ਭਰ ਦਿੱਤਾ ਹੈ।",
            "WRONG_NUMBER": "Agent: ਕੀ ਰਾਹੁਲ ਗੱਲ ਕਰ ਰਿਹਾ ਹੈ? Borrower: ਨਹੀਂ, ਇਹ ਗਲਤ ਨੰਬਰ ਹੈ।",
            "DENIED": "Agent: ਤੁਸੀਂ ਪੈਸੇ ਕਦੋਂ ਦਵੋਗੇ? Borrower: ਮੇਰੀ ਨੌਕਰੀ ਚਲੀ ਗਈ ਹੈ, ਮੈਂ ਨਹੀਂ ਭਰ ਸਕਦਾ।",
            "VISIT": "Agent: ਤੁਸੀਂ ਪੇਮੈਂਟ ਕਿਵੇਂ ਕਰੋਗੇ? Borrower: ਘਰ ਕਿਸੇ ਨੂੰ ਭੇਜ ਦਿਓ, ਮੈਂ ਨਕਦ ਦੇ ਦੇਵਾਂਗਾ।"
        }
    }
    
    # fallback to english if language missing template
    lang_templates = templates.get(language, templates["english"])
    
    # Generate requested number of samples combining variations
    for i in range(count):
        # Pick a random scenario type based on weights to create a realistic distribution
        scenario_type = random.choices(
            ["PTP", "PAID", "WRONG_NUMBER", "DENIED", "VISIT"],
            weights=[40, 20, 10, 20, 10],
            k=1
        )[0]
        
        amount = random.choice([1500, 2500, 4000, 5000, 10000])
        date_str = f"2026-03-{random.randint(10, 28):02d}"
        
        transcript = lang_templates[scenario_type]
        transcript = transcript.replace("{amount}", str(amount)).replace("{date}", date_str)
        
        # Map scenario to expected ground truth outputs
        if scenario_type == "PTP":
            exp_disp = "ANSWERED"
            exp_pay = "PTP"
            exp_reason = "FUNDS_ISSUE"
            exp_amt = amount
            exp_date = date_str
        elif scenario_type == "PAID":
            exp_disp = "ANSWERED"
            exp_pay = "PAID"
            exp_reason = "TECHNICAL_ISSUE"
            exp_amt = None
            exp_date = None
        elif scenario_type == "WRONG_NUMBER":
            exp_disp = "WRONG_NUMBER"
            exp_pay = "None"
            exp_reason = None
            exp_amt = None
            exp_date = None
        elif scenario_type == "DENIED":
            exp_disp = "ANSWERED"
            exp_pay = "DENIED_TO_PAY"
            exp_reason = "UNEMPLOYED"
            exp_amt = None
            exp_date = None
        elif scenario_type == "VISIT":
            exp_disp = "ANSWERED"
            exp_pay = "WILL_PAY_AFTER_VISIT"
            exp_reason = "OTHER_REASONS"
            exp_amt = None
            exp_date = None
            
        sample = {
            "id": f"{language}_{i}",
            "transcript": transcript,
            "expected_disposition": exp_disp,
            "expected_payment_disposition": exp_pay,
            "expected_reason_for_not_paying": exp_reason,
            "expected_amount": exp_amt,
            "expected_date": exp_date,
            "expected_remarks": "Generated sample"
        }
        samples.append(sample)
        
    return samples

def main():
    print(f"Generating 200 samples per language for {len(LANGUAGES)} languages...")
    
    for lang in LANGUAGES:
        samples = generate_samples(lang, count=200)
        file_path = os.path.join(EVAL_DIR, f"{lang}_test.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=4)
        print(f"✅ Generated {len(samples)} samples for {lang} -> {file_path}")
        
if __name__ == "__main__":
    main()
