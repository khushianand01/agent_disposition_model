"""
Date Mapping Utility for Ringg Pipeline

Handles Hindi date references and converts them to proper YYYY-MM-DD format.
"""

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re

# Weekday mappings (English and Hindi)
WEEKDAYS = {
    'monday': 0, 'somwar': 0, 'somvaar': 0,
    'tuesday': 1, 'mangalwar': 1, 'mangalvaar': 1,
    'wednesday': 2, 'budhwar': 2, 'budhvaar': 2,
    'thursday': 3, 'guruwar': 3, 'guruvaar': 3, 'brihaspatiwar': 3,
    'friday': 4, 'shukrawar': 4, 'shukravaar': 4,
    'saturday': 5, 'shaniwar': 5, 'shanivaar': 5,
    'sunday': 6, 'raviwar': 6, 'ravivaar': 6, 'itwaar': 6
}

# Month mappings (English and Hindi)
MONTHS = {
    'january': 1, 'jan': 1, 'janavari': 1,
    'february': 2, 'feb': 2, 'farwari': 2, 'farvari': 2,
    'march': 3, 'mar': 3, 'march': 3,
    'april': 4, 'apr': 4, 'april': 4,
    'may': 5, 'mai': 5,
    'june': 6, 'jun': 6, 'joon': 6,
    'july': 7, 'jul': 7, 'julai': 7,
    'august': 8, 'aug': 8, 'agast': 8,
    'september': 9, 'sep': 9, 'sitambar': 9,
    'october': 10, 'oct': 10, 'aktoobar': 10,
    'november': 11, 'nov': 11, 'navambar': 11,
    'december': 12, 'dec': 12, 'disambar': 12
}


def map_hindi_date_to_absolute(text, current_date_str):
    """
    Maps Hindi date references to absolute dates.
    
    Args:
        text: The transcript text or date string
        current_date_str: Current date in YYYY-MM-DD format
        
    Returns:
        Absolute date in YYYY-MM-DD format or None
    """
    if not text:
        return None
        
    current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
    text_lower = text.lower()
    
    # Direct mappings
    if "aaj" in text_lower or "today" in text_lower:
        return current_date.strftime("%Y-%m-%d")
    
    if "kal" in text_lower and "parso" not in text_lower:
        # Tomorrow
        return (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
    
    if "parso" in text_lower:
        # Day after tomorrow
        return (current_date + timedelta(days=2)).strftime("%Y-%m-%d")
    
    # Check for weekday names (e.g., "Monday", "next Friday", "somwar")
    for weekday_name, weekday_num in WEEKDAYS.items():
        if weekday_name in text_lower:
            current_weekday = current_date.weekday()
            days_ahead = weekday_num - current_weekday
            
            # If the weekday has passed this week, go to next week
            if days_ahead <= 0:
                days_ahead += 7
            
            target_date = current_date + timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d")
    
    # Check for month names (e.g., "February mein", "March tak")
    for month_name, month_num in MONTHS.items():
        # Use word boundary to avoid matching "mai" in "main"
        if re.search(r'\b' + re.escape(month_name) + r'\b', text_lower):

            # Extract day if present (e.g., "5 February")
            day_match = re.search(r'(\d{1,2})\s*' + month_name, text_lower)
            if day_match:
                day = int(day_match.group(1))
            else:
                day = 1  # Default to 1st of the month
            
            try:
                # Determine the year (current or next)
                target_year = current_date.year
                if month_num < current_date.month or (month_num == current_date.month and day < current_date.day):
                    target_year += 1
                
                target_date = datetime(target_year, month_num, day)
                return target_date.strftime("%Y-%m-%d")
            except ValueError:
                pass
    
    
    # Extract day number (e.g., "5 tareekh", "15 ko", "20th")
    # IMPORTANT: Match numbers that are followed by date keywords to avoid matching amounts
    day_match = re.search(r'(\d{1,2})\s*(?:tareekh|tarikh|ko)\b', text_lower)
    if not day_match:
        # Fallback: try English ordinals (5th, 15th)
        day_match = re.search(r'(\d{1,2})(?:th|st|nd|rd)\b', text_lower)
    
    if day_match:
        day = int(day_match.group(1))
        print(f"[DATE_MAPPER] Extracted day: {day}")
        if 1 <= day <= 31:
            try:
                # Try current month first
                target_date = current_date.replace(day=day)
                print(f"[DATE_MAPPER] Current month attempt: {target_date}")
                # If it's in the past, move to next month
                if target_date <= current_date:
                    print(f"[DATE_MAPPER] Date is in past, adding 1 month")
                    target_date = target_date + relativedelta(months=1)
                    print(f"[DATE_MAPPER] Final date: {target_date}")
                return target_date.strftime("%Y-%m-%d")
            except ValueError as e:
                print(f"[DATE_MAPPER] ValueError: {e}")
                # Invalid day for current month, try next month
                next_month = current_date + relativedelta(months=1)
                try:
                    target_date = next_month.replace(day=day)
                    return target_date.strftime("%Y-%m-%d")
                except ValueError:
                    pass

    
    # Week references (e.g., "next week", "agle hafte", "ek hafte baad")
    if re.search(r'\b(next week|agle? hafte?|ek hafte|1 hafte)\b', text_lower):
        return (current_date + timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Month references (e.g., "next month", "agle mahine")
    if re.search(r'\b(next month|agle? mahine?|1 mahine)\b', text_lower):
        return (current_date + relativedelta(months=1)).strftime("%Y-%m-%d")
    
    return None


def fix_ptp_date(ptp_date_str, current_date_str, transcript=""):
    """
    Post-processes a PTP date to ensure it's in the future.
    
    Args:
        ptp_date_str: The date string from model output
        current_date_str: Current date in YYYY-MM-DD format
        transcript: Original transcript for context
        
    Returns:
        Corrected date in YYYY-MM-DD format
    """
    if not ptp_date_str and not transcript:
        return None
    
    # PRIORITY 1: Try to extract date directly from transcript
    # This is more reliable than trusting the model's output
    if transcript:
        mapped_date = map_hindi_date_to_absolute(transcript, current_date_str)
        print(f"[DATE_MAPPER DEBUG] Transcript mapping result: {mapped_date}")
        if mapped_date:
            return mapped_date
    
    # PRIORITY 2: If no date found in transcript, try to fix model's output
    if not ptp_date_str:
        return None
    
    try:
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
        ptp_date = datetime.strptime(ptp_date_str, "%Y-%m-%d")
        
        # If date is in the past, move to next valid occurrence
        if ptp_date <= current_date:
            # If it's just a day mismatch, move to next month
            if ptp_date.day != current_date.day:
                ptp_date = current_date.replace(day=ptp_date.day)
                if ptp_date <= current_date:
                    ptp_date = ptp_date + relativedelta(months=1)
                return ptp_date.strftime("%Y-%m-%d")
        
        return ptp_date_str
    except:
        return ptp_date_str
