import unicodedata
from typing import List, Tuple, Optional, Set

from .constants import (
    VIETNAMESE_CHARS_WITH_DIACRITICS,
    TONE_MARKS,
    MODIFIER_MARKS,
    CHAR_CONFUSIONS,
)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def decompose_vietnamese_char(char: str) -> Tuple[str, List[str]]:
    if len(char) != 1:
        return char, []
    
    nfd = unicodedata.normalize('NFD', char)
    base = ''
    diacritics = []
    
    for c in nfd:
        category = unicodedata.category(c)
        if category == 'Mn':
            diacritics.append(c)
        else:
            base += c
    
    return base, diacritics


def is_valid_vietnamese_char(char: str) -> bool:
    if len(char) != 1:
        return False
    
    if char.isascii():
        return True
    
    if char in VIETNAMESE_CHARS_WITH_DIACRITICS:
        return True
    
    base, diacritics = decompose_vietnamese_char(char)
    
    if not base:
        return False
    
    for diac in diacritics:
        if diac in MODIFIER_MARKS:
            modifier_type = MODIFIER_MARKS[diac]
            if modifier_type == 'circumflex' and base.lower() not in 'aeo':
                return False
            if modifier_type == 'breve' and base.lower() != 'a':
                return False
            if modifier_type == 'horn' and base.lower() not in 'ou':
                return False
    
    return True


def check_diacritic_validity(word: str) -> Tuple[bool, List[str]]:
    issues = []
    word = normalize_unicode(word)
    tone_count = 0
    
    for i, char in enumerate(word):
        base, diacritics = decompose_vietnamese_char(char)
        
        for diac in diacritics:
            if diac in TONE_MARKS:
                tone_count += 1
        
        for diac in diacritics:
            if diac in MODIFIER_MARKS:
                modifier_type = MODIFIER_MARKS[diac]
                base_lower = base.lower()
                
                if modifier_type == 'circumflex' and base_lower not in 'aeo':
                    issues.append(f"Invalid circumflex on '{base}' at position {i}")
                elif modifier_type == 'breve' and base_lower != 'a':
                    issues.append(f"Invalid breve on '{base}' at position {i}")
                elif modifier_type == 'horn' and base_lower not in 'ou':
                    issues.append(f"Invalid horn on '{base}' at position {i}")
    
    return len(issues) == 0, issues


def compute_edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return compute_edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def find_closest_word(word: str, dictionary: Set[str], max_distance: int = 2) -> Optional[str]:
    word_lower = word.lower()
    
    if word_lower in dictionary:
        return word
    
    best_match = None
    best_distance = max_distance + 1
    
    for dict_word in dictionary:
        if abs(len(dict_word) - len(word_lower)) > max_distance:
            continue
        
        distance = compute_edit_distance(word_lower, dict_word)
        if distance < best_distance:
            best_distance = distance
            best_match = dict_word
    
    if best_distance <= max_distance and best_match:
        if word[0].isupper():
            return best_match[0].upper() + best_match[1:]
        return best_match
    
    return None


def get_diacritic_variants(char: str) -> List[str]:
    if char in CHAR_CONFUSIONS:
        return CHAR_CONFUSIONS[char]
    return []
