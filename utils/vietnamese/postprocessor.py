import re
import unicodedata
import logging
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from .constants import (
    COMMON_VIETNAMESE_WORDS,
    COMMON_OCR_CORRECTIONS,
    MODIFIER_MARKS,
)
from .utils import (
    normalize_unicode,
    decompose_vietnamese_char,
    is_valid_vietnamese_char,
    check_diacritic_validity,
    compute_edit_distance,
    get_diacritic_variants,
)

logger = logging.getLogger(__name__)


class VietnameseOCRPostProcessor:
    def __init__(
        self,
        dictionary: Optional[Set[str]] = None,
        custom_corrections: Optional[Dict[str, str]] = None,
        max_edit_distance: int = 2,
        enable_spell_check: bool = True,
        enable_pattern_correction: bool = True,
        enable_diacritic_validation: bool = True
    ):
        self.dictionary = dictionary or COMMON_VIETNAMESE_WORDS
        self.custom_corrections = custom_corrections or {}
        self.max_edit_distance = max_edit_distance
        self.enable_spell_check = enable_spell_check
        self.enable_pattern_correction = enable_pattern_correction
        self.enable_diacritic_validation = enable_diacritic_validation
        
        self.corrections = {**COMMON_OCR_CORRECTIONS, **self.custom_corrections}
        self._build_correction_index()
        
        logger.info(f"VietnameseOCRPostProcessor initialized with {len(self.dictionary)} dictionary words")
    
    def _build_correction_index(self):
        self.dict_by_first_char = defaultdict(set)
        for word in self.dictionary:
            if word:
                self.dict_by_first_char[word[0].lower()].add(word)
    
    def process(self, text: str) -> Tuple[str, Dict]:
        text = normalize_unicode(text)
        
        metadata = {
            'original': text,
            'corrections_made': [],
            'invalid_chars_found': [],
            'spell_corrections': [],
            'pattern_corrections': []
        }
        
        if self.enable_pattern_correction:
            text, pattern_corrections = self._apply_pattern_corrections(text)
            metadata['pattern_corrections'] = pattern_corrections
        
        words = self._tokenize(text)
        corrected_words = []
        
        for word in words:
            corrected_word = word
            
            invalid_chars = self._find_invalid_chars(word)
            if invalid_chars:
                metadata['invalid_chars_found'].extend(invalid_chars)
            
            if self.enable_diacritic_validation:
                is_valid, issues = check_diacritic_validity(word)
                if not is_valid:
                    corrected_word = self._fix_diacritics(word)
                    if corrected_word != word:
                        metadata['corrections_made'].append({
                            'original': word,
                            'corrected': corrected_word,
                            'type': 'diacritic_fix',
                            'issues': issues
                        })
            
            if self.enable_spell_check and corrected_word.isalpha():
                spell_corrected = self._spell_check(corrected_word)
                if spell_corrected and spell_corrected != corrected_word:
                    metadata['spell_corrections'].append({
                        'original': corrected_word,
                        'corrected': spell_corrected
                    })
                    corrected_word = spell_corrected
            
            corrected_words.append(corrected_word)
        
        corrected_text = self._reconstruct_text(words, corrected_words, text)
        metadata['final'] = corrected_text
        
        return corrected_text, metadata
    
    def _tokenize(self, text: str) -> List[str]:
        return text.split()
    
    def _reconstruct_text(self, original_words: List[str], corrected_words: List[str], original_text: str) -> str:
        return ' '.join(corrected_words)
    
    def _apply_pattern_corrections(self, text: str) -> Tuple[str, List[Dict]]:
        corrections = []
        
        for pattern, replacement in self.corrections.items():
            if pattern in text:
                text = text.replace(pattern, replacement)
                corrections.append({
                    'pattern': pattern,
                    'replacement': replacement
                })
        
        return text, corrections
    
    def _find_invalid_chars(self, word: str) -> List[Dict]:
        invalid = []
        for i, char in enumerate(word):
            if not is_valid_vietnamese_char(char) and not char.isspace() and not char in '.,;:!?-()[]{}"\'/':
                invalid.append({
                    'char': char,
                    'position': i,
                    'unicode': f'U+{ord(char):04X}'
                })
        return invalid
    
    def _fix_diacritics(self, word: str) -> str:
        result = []
        for char in word:
            base, diacritics = decompose_vietnamese_char(char)
            
            valid_diacritics = []
            for diac in diacritics:
                is_valid = True
                if diac in MODIFIER_MARKS:
                    modifier_type = MODIFIER_MARKS[diac]
                    base_lower = base.lower() if base else ''
                    if modifier_type == 'circumflex' and base_lower not in 'aeo':
                        is_valid = False
                    elif modifier_type == 'breve' and base_lower != 'a':
                        is_valid = False
                    elif modifier_type == 'horn' and base_lower not in 'ou':
                        is_valid = False
                
                if is_valid:
                    valid_diacritics.append(diac)
            
            if valid_diacritics:
                fixed_char = unicodedata.normalize('NFC', base + ''.join(valid_diacritics))
            else:
                fixed_char = base
            
            result.append(fixed_char)
        
        return ''.join(result)
    
    def _spell_check(self, word: str) -> Optional[str]:
        word_lower = word.lower()
        
        if word_lower in self.dictionary:
            return word
        
        first_char = word_lower[0] if word_lower else ''
        candidates = self.dict_by_first_char.get(first_char, set())
        
        for variant in get_diacritic_variants(first_char):
            candidates = candidates.union(self.dict_by_first_char.get(variant, set()))
        
        best_match = None
        best_distance = self.max_edit_distance + 1
        
        for candidate in candidates:
            if abs(len(candidate) - len(word_lower)) > self.max_edit_distance:
                continue
            
            distance = compute_edit_distance(word_lower, candidate)
            if distance < best_distance:
                best_distance = distance
                best_match = candidate
        
        if best_distance <= self.max_edit_distance and best_match:
            if word[0].isupper():
                return best_match[0].upper() + best_match[1:]
            return best_match
        
        return word
    
    def add_to_dictionary(self, words: List[str]):
        for word in words:
            word_lower = word.lower()
            self.dictionary.add(word_lower)
            if word_lower:
                self.dict_by_first_char[word_lower[0]].add(word_lower)
    
    def add_correction_pattern(self, pattern: str, replacement: str):
        self.corrections[pattern] = replacement


def detect_ocr_confidence_issues(text: str, confidence_scores: Optional[List[float]] = None) -> List[Dict]:
    issues = []
    
    repeated_pattern = re.compile(r'(.)\1{2,}')
    for match in repeated_pattern.finditer(text):
        issues.append({
            'type': 'repeated_chars',
            'position': match.start(),
            'text': match.group(),
            'suggestion': match.group()[0]
        })
    
    invalid_clusters = re.compile(r'[bcdfghjklmnpqrstvwxz]{3,}', re.IGNORECASE)
    for match in invalid_clusters.finditer(text):
        issues.append({
            'type': 'invalid_cluster',
            'position': match.start(),
            'text': match.group()
        })
    
    return issues


def process_ocr_results_batch(
    results: List[Dict],
    text_key: str = 'predicted_text',
    post_processor: Optional[VietnameseOCRPostProcessor] = None
) -> List[Dict]:
    if post_processor is None:
        post_processor = VietnameseOCRPostProcessor()
    
    processed_results = []
    
    for result in results:
        if text_key in result:
            text = result[text_key]
            corrected_text, metadata = post_processor.process(text)
            
            processed_result = result.copy()
            processed_result['corrected_text'] = corrected_text
            processed_result['correction_metadata'] = metadata
            processed_results.append(processed_result)
        else:
            processed_results.append(result)
    
    return processed_results


def load_dictionary_from_file(filepath: str) -> Set[str]:
    dictionary = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    dictionary.add(word)
        logger.info(f"Loaded {len(dictionary)} words from {filepath}")
    except FileNotFoundError:
        logger.warning(f"Dictionary file not found: {filepath}")
    except Exception as e:
        logger.error(f"Error loading dictionary: {e}")
    
    return dictionary


def save_dictionary_to_file(dictionary: Set[str], filepath: str):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for word in sorted(dictionary):
                f.write(word + '\n')
        logger.info(f"Saved {len(dictionary)} words to {filepath}")
    except Exception as e:
        logger.error(f"Error saving dictionary: {e}")
