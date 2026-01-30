"""
Stage-2 Normalization Helpers

These functions normalize dates, amounts, and enforce null values
to ensure clean, consistent training data.
"""

import re
from datetime import datetime, timedelta
from typing import Optional, Union

# ============================================================================
# DATE NORMALIZATION
# ============================================================================

# Hindi/Hinglish relative date mappings
RELATIVE_DATES = {
    # Today/Tomorrow/Yesterday
    'aaj': 0, 'today': 0, 'आज': 0,
    'kal': 1, 'tomorrow': 1, 'कल': 1,
    'parso': 2, 'day after tomorrow': 2, 'परसों': 2,
    'yesterday': -1, 'कल': -1,  # Note: "kal" can mean tomorrow or yesterday (context-dependent)
    
    # Days of week (Hindi)
    'somwar': 'monday', 'सोमवार': 'monday',
    'mangalwar': 'tuesday', 'मंगलवार': 'tuesday',
    'budhwar': 'wednesday', 'बुधवार': 'wednesday',
    'guruwar': 'thursday', 'गुरुवार': 'thursday', 'brihaspatiwar': 'thursday',
    'shukrawar': 'friday', 'शुक्रवार': 'friday',
    'shaniwar': 'saturday', 'शनिवार': 'saturday',
    'raviwar': 'sunday', 'रविवार': 'sunday', 'itwaar': 'sunday',
    
    # Week references
    'next week': 7, 'अगले हफ्ते': 7, 'agle hafte': 7,
    'this week': 3, 'इस हफ्ते': 3, 'is hafte': 3,
}

WEEKDAY_MAP = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
    'friday': 4, 'saturday': 5, 'sunday': 6
}

def normalize_date(date_input: Union[str, None], reference_date: Optional[datetime] = None) -> Optional[str]:
    """
    Normalize date to YYYY-MM-DD format.
    
    Handles:
    - Relative dates: "kal", "tomorrow", "next Monday"
    - Hindi dates: "परसों", "अगले हफ्ते"
    - Absolute dates: "2024-01-30", "30/01/2024"
    - Null/None values
    
    Args:
        date_input: Date string or None
        reference_date: Reference date for relative calculations (default: today)
    
    Returns:
        Normalized date string (YYYY-MM-DD) or None
    """
    if not date_input or str(date_input).strip().lower() in ['none', 'null', '']:
        return None
    
    date_str = str(date_input).strip().lower()
    ref_date = reference_date or datetime.now()
    
    # 1. Check if already in YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    
    # 2. Handle relative dates (aaj, kal, parso, etc.)
    for key, offset in RELATIVE_DATES.items():
        if isinstance(offset, int) and key in date_str:
            target_date = ref_date + timedelta(days=offset)
            return target_date.strftime('%Y-%m-%d')
    
    # 3. Handle "next [weekday]" or "[weekday]"
    for hindi_day, eng_day in RELATIVE_DATES.items():
        if isinstance(eng_day, str) and eng_day in WEEKDAY_MAP:
            if hindi_day in date_str or eng_day in date_str:
                target_weekday = WEEKDAY_MAP[eng_day]
                current_weekday = ref_date.weekday()
                
                # Calculate days until target weekday
                days_ahead = target_weekday - current_weekday
                if days_ahead <= 0:  # Target day already passed this week
                    days_ahead += 7
                
                # If "next" is mentioned, add another week
                if 'next' in date_str or 'agle' in date_str:
                    days_ahead += 7
                
                target_date = ref_date + timedelta(days=days_ahead)
                return target_date.strftime('%Y-%m-%d')
    
    # 4. Try parsing common formats (DD/MM/YYYY, DD-MM-YYYY)
    for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d']:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # 5. Extract numeric date if present (e.g., "30th January" -> try to parse)
    # For now, return None if we can't parse
    return None


# ============================================================================
# AMOUNT NORMALIZATION
# ============================================================================

# Hindi number words
HINDI_NUMBERS = {
    # Units (0-9)
    'zero': 0, 'शून्य': 0,
    'ek': 1, 'एक': 1, 'one': 1,
    'do': 2, 'दो': 2, 'two': 2,
    'teen': 3, 'तीन': 3, 'three': 3,
    'char': 4, 'चार': 4, 'four': 4,
    'paanch': 5, 'पांच': 5, 'five': 5, 'panch': 5,
    'chhe': 6, 'छह': 6, 'six': 6, 'chhah': 6,
    'saat': 7, 'सात': 7, 'seven': 7,
    'aath': 8, 'आठ': 8, 'eight': 8,
    'nau': 9, 'नौ': 9, 'nine': 9,
    
    # Tens (10-90)
    'das': 10, 'दस': 10, 'ten': 10,
    'bees': 20, 'बीस': 20, 'twenty': 20, 'bis': 20,
    'tees': 30, 'तीस': 30, 'thirty': 30, 'tis': 30,
    'chalis': 40, 'चालीस': 40, 'forty': 40,
    'pachas': 50, 'पचास': 50, 'fifty': 50,
    'saath': 60, 'साठ': 60, 'sixty': 60,
    'sattar': 70, 'सत्तर': 70, 'seventy': 70,
    'assi': 80, 'अस्सी': 80, 'eighty': 80,
    'nabbe': 90, 'नब्बे': 90, 'ninety': 90,
    
    # Hundreds
    'sau': 100, 'सौ': 100, 'hundred': 100,
    
    # Thousands
    'hazaar': 1000, 'हजार': 1000, 'thousand': 1000, 'hazar': 1000,
    
    # Lakhs
    'lakh': 100000, 'लाख': 100000, 'lac': 100000,
    
    # Crores
    'crore': 10000000, 'करोड़': 10000000, 'karod': 10000000,
}

def normalize_amount(amount_input: Union[str, int, float, None]) -> Optional[float]:
    """
    Normalize amount to numeric value.
    
    Handles:
    - Number words: "teen hazaar" -> 3000
    - Mixed formats: "3000 rs", "₹5000", "5k"
    - Hindi numbers: "तीन हजार" -> 3000
    - Null/None values
    
    Args:
        amount_input: Amount string, number, or None
    
    Returns:
        Normalized amount (float) or None
    """
    if amount_input is None or str(amount_input).strip().lower() in ['none', 'null', '']:
        return None
    
    # If already numeric
    if isinstance(amount_input, (int, float)):
        return float(amount_input) if amount_input > 0 else None
    
    amount_str = str(amount_input).strip().lower()
    
    # Remove currency symbols and common suffixes
    amount_str = re.sub(r'[₹$,\s]+', ' ', amount_str)
    amount_str = amount_str.replace('rs', '').replace('rupees', '').replace('rupee', '').strip()
    
    # 1. Try direct numeric parsing
    try:
        # Handle "k" suffix (e.g., "5k" -> 5000)
        if 'k' in amount_str:
            num = float(amount_str.replace('k', '').strip())
            return num * 1000
        
        value = float(amount_str)
        return value if value > 0 else None
    except ValueError:
        pass
    
    # 2. Parse Hindi/English number words
    total = 0
    current = 0
    
    words = amount_str.split()
    for word in words:
        word = word.strip()
        if word in HINDI_NUMBERS:
            num = HINDI_NUMBERS[word]
            
            # Multipliers (sau, hazaar, lakh, crore)
            if num >= 100:
                if current == 0:
                    current = 1
                current *= num
                if num >= 1000:  # hazaar, lakh, crore
                    total += current
                    current = 0
            else:
                current += num
    
    total += current
    
    return float(total) if total > 0 else None


# ============================================================================
# NULL ENFORCEMENT
# ============================================================================

def enforce_null(value: any, allowed_values: Optional[set] = None) -> any:
    """
    Enforce null for invalid/empty values.
    
    Args:
        value: Input value
        allowed_values: Set of allowed non-null values (optional)
    
    Returns:
        Value or None if invalid
    """
    # Check for explicit null indicators
    if value is None:
        return None
    
    str_val = str(value).strip().lower()
    if str_val in ['none', 'null', '', 'n/a', 'na', 'nil']:
        return None
    
    # If allowed_values specified, validate
    if allowed_values is not None:
        if value not in allowed_values:
            return None
    
    return value


def clean_remarks(remarks: Optional[str]) -> Optional[str]:
    """
    Clean and normalize remarks field.
    
    Args:
        remarks: Raw remarks string
    
    Returns:
        Cleaned remarks or None
    """
    if not remarks:
        return None
    
    remarks_str = str(remarks).strip()
    
    # Remove if too short or generic
    if len(remarks_str) < 3 or remarks_str.lower() in ['none', 'null', 'na', 'n/a', 'nil', '-']:
        return None
    
    return remarks_str


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=== DATE NORMALIZATION TESTS ===")
    test_dates = [
        "kal", "tomorrow", "परसों", "next monday", "somwar", 
        "agle hafte", "2024-01-30", "30/01/2024", None
    ]
    for date in test_dates:
        print(f"{str(date):20s} -> {normalize_date(date)}")
    
    print("\n=== AMOUNT NORMALIZATION TESTS ===")
    test_amounts = [
        "teen hazaar", "3000 rs", "₹5000", "5k", "paanch lakh",
        "तीन हजार", "50", 3000, None, "do sau"
    ]
    for amount in test_amounts:
        print(f"{str(amount):20s} -> {normalize_amount(amount)}")
    
    print("\n=== NULL ENFORCEMENT TESTS ===")
    test_values = ["valid", None, "null", "", "n/a", 123]
    for val in test_values:
        print(f"{str(val):20s} -> {enforce_null(val)}")
