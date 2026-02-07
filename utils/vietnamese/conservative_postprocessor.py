import re
import unicodedata
import logging
from typing import List, Dict, Tuple

from .constants import VIETNAMESE_CONSONANTS, VIETNAMESE_VOWELS_ALL

logger = logging.getLogger(__name__)

REPEATED_CHAR_END_PATTERN = re.compile(r'(.)\1+$')
TRAILING_ARTIFACT_PATTERN = re.compile(r'\s+[a-zA-Z]\s*$')


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def has_tone_mark(char: str) -> bool:
    if len(char) != 1:
        return False
    nfd = unicodedata.normalize('NFD', char)
    tone_marks = {'\u0300', '\u0301', '\u0303', '\u0309', '\u0323'}   #grave, acute, tilde, hook, dot
    return any(c in tone_marks for c in nfd)


def get_char_base_and_diacritics(char: str) -> Tuple[str, List[str]]:
    if len(char) != 1:
        return char, []
    
    nfd = unicodedata.normalize('NFD', char)
    base = ''
    diacritics = []
    
    for c in nfd:
        if unicodedata.category(c) == 'Mn':
            diacritics.append(c)
        else:
            base += c
    
    return base, diacritics


def is_valid_vietnamese_syllable(syllable: str) -> Tuple[bool, str]:
    if not syllable:
        return True, ""
    
    syllable_lower = syllable.lower()
    
    match = REPEATED_CHAR_END_PATTERN.search(syllable_lower)
    if match:
        return False, f"repeated_char_end:{match.group()}"
    
    if re.search(r'[^aeiouăâêôơưq]iy', syllable_lower):
        return False, "invalid_iy_pattern"
    
    y_match = re.search(r'([^qguaeiouăâêôơư])y(?![aeiouăâêôơư])', syllable_lower)
    if y_match:
        return False, f"invalid_y_after:{y_match.group(1)}"
    
    return True, ""


def fix_repeated_characters(word: str) -> Tuple[str, bool]:
    if len(word) < 2:
        return word, False

    original = word
    
    while len(word) >= 2 and word[-1] == word[-2]:
        word = word[:-1]
    
    return word, word != original


def fix_invalid_y_pattern(word: str, use_lm: bool = False, lm_model=None) -> Tuple[str, bool]:
    word_lower = word.lower()
    
    match = re.search(r'([bcdfghjklmnprstvx])iy$', word_lower)
    if match:
        if use_lm and lm_model:
            candidates = [
                word[:-2] + 'ì',
                word[:-2] + 'í',
                word[:-2] + 'ỉ',
                word[:-2] + 'ĩ',
                word[:-2] + 'ị',
                word[:-2] + 'ỳ',
                word[:-2] + 'ý',
                word[:-2] + 'ỷ',
                word[:-2] + 'ỹ',
                word[:-2] + 'ỵ',
            ]
            
            best_candidate = word[:-2] + 'ỳ'
            best_score = float('-inf')
            
            for candidate in candidates:
                try:
                    score = lm_model.score_sentence(candidate)
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate
                except:
                    pass
            
            return best_candidate, True
        else:
            return word[:-1], True
    
    return word, False


def fix_trailing_artifacts(text: str) -> Tuple[str, bool]:
    original = text
    text = re.sub(r'\s+[bcdfghjklmnpqrstvwxz]\s*$', '', text, flags=re.IGNORECASE)
    return text.strip(), text.strip() != original.strip()


def check_diacritic_validity(char: str) -> Tuple[bool, str]:
    base, diacritics = get_char_base_and_diacritics(char)
    
    if not diacritics:
        return True, ""
    
    base_lower = base.lower()
    
    for diac in diacritics:
        if diac == '\u0302':
            if base_lower not in 'aeo':
                return False, f"invalid_circumflex_on_{base}"
        
        if diac == '\u0306':
            if base_lower != 'a':
                return False, f"invalid_breve_on_{base}"
        
        if diac == '\u031B':
            if base_lower not in 'ou':
                return False, f"invalid_horn_on_{base}"
    
    return True, ""


def fix_malformed_diacritics(char: str) -> Tuple[str, bool]:
    base, diacritics = get_char_base_and_diacritics(char)
    
    if len(diacritics) <= 1:
        return char, False
    
    shape_marks = {'\u0302', '\u0306', '\u031B'}   #circumflex, breve, horn
    tone_marks = {'\u0300', '\u0301', '\u0303', '\u0309', '\u0323'}   #grave, acute, tilde, hook, dot
    
    shape_mark = None
    tone_mark = None
    
    for diac in diacritics:
        if diac in shape_marks:
            if shape_mark is None:
                shape_mark = diac
        elif diac in tone_marks:
            if tone_mark is None:
                tone_mark = diac
    new_char = base
    if shape_mark:
        new_char += shape_mark
    if tone_mark:
        new_char += tone_mark
    
    new_char = unicodedata.normalize('NFC', new_char)
    
    changed = new_char != char
    return new_char, changed


def fix_duplicate_tone_marks(word: str) -> Tuple[str, bool]:
    #Mộẹ -> Mộ
    if len(word) < 2:
        return word, False
    
    vietnamese_vowels = set('aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵ')
    
    result = []
    i = 0
    changed = False
    
    while i < len(word):
        current_char = word[i]
        result.append(current_char)
        
        if current_char.lower() in vietnamese_vowels and has_tone_mark(current_char):
            j = i + 1
            while j < len(word) and word[j].lower() in vietnamese_vowels and has_tone_mark(word[j]):
                changed = True
                j += 1
            i = j
        else:
            i += 1
    
    return ''.join(result), changed


def fix_inconsistent_capitalization(word: str) -> Tuple[str, bool]:
    if len(word) < 2:
        return word, False
    
    original_word = word
    if word[0].lower() == word[1].lower() and word[0].islower() and word[1].isupper():
        word = word[1:].lower()
        return word, True
    has_mixed_case = False
    for i in range(len(word) - 1):
        if word[i].islower() and word[i+1].isupper():
            has_mixed_case = True
            break
    
    if has_mixed_case:
        if word[0].islower() and word[1].isupper():
            word = word.lower()
        else:
            word = word[0] + word[1:].lower()
    
    return word, word != original_word


class VietnameseConservativePostProcessor:
    def __init__(
        self,
        enable_repeated_char_fix: bool = True,
        enable_invalid_pattern_fix: bool = True,
        enable_trailing_artifact_fix: bool = True,
        enable_diacritic_check: bool = True,
        enable_diacritic_fix: bool = True,
        enable_capitalization_fix: bool = True,
        use_lm_for_ambiguous: bool = False,
        lm_model=None
    ):
        self.enable_repeated_char_fix = enable_repeated_char_fix
        self.enable_invalid_pattern_fix = enable_invalid_pattern_fix
        self.enable_trailing_artifact_fix = enable_trailing_artifact_fix
        self.enable_diacritic_check = enable_diacritic_check
        self.enable_diacritic_fix = enable_diacritic_fix
        self.enable_capitalization_fix = enable_capitalization_fix
        self.use_lm_for_ambiguous = use_lm_for_ambiguous
        self.lm_model = lm_model
        
        logger.info("VietnameseConservativePostProcessor initialized")
    
    def process(self, text: str) -> Tuple[str, Dict]:
        text = normalize_unicode(text)
        
        metadata = {
            'original': text,
            'corrections': [],
            'checks_passed': [],
            'checks_failed': []
        }
        
        if self.enable_trailing_artifact_fix:
            text, changed = fix_trailing_artifacts(text)
            if changed:
                metadata['corrections'].append({
                    'type': 'trailing_artifact',
                    'original': metadata['original'],
                    'fixed': text
                })
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            corrected_word = word
            if self.enable_capitalization_fix:
                corrected_word, changed = fix_inconsistent_capitalization(corrected_word)
                if changed:
                    metadata['corrections'].append({
                        'type': 'inconsistent_capitalization',
                        'original': word,
                        'fixed': corrected_word
                    })
            if self.enable_diacritic_fix:
                corrected_word, changed = fix_duplicate_tone_marks(corrected_word)
                if changed:
                    metadata['corrections'].append({
                        'type': 'duplicate_tone_marks',
                        'original': word,
                        'fixed': corrected_word
                    })
            if self.enable_diacritic_fix:
                fixed_chars = []
                word_changed = False
                for char in corrected_word:
                    fixed_char, changed = fix_malformed_diacritics(char)
                    fixed_chars.append(fixed_char)
                    if changed:
                        word_changed = True
                
                if word_changed:
                    old_word = corrected_word
                    corrected_word = ''.join(fixed_chars)
                    metadata['corrections'].append({
                        'type': 'malformed_diacritics',
                        'original': old_word,
                        'fixed': corrected_word
                    })
            
            if self.enable_repeated_char_fix:
                corrected_word, changed = fix_repeated_characters(corrected_word)
                if changed:
                    metadata['corrections'].append({
                        'type': 'repeated_char',
                        'original': word,
                        'fixed': corrected_word
                    })
            
            if self.enable_invalid_pattern_fix:
                is_valid, reason = is_valid_vietnamese_syllable(corrected_word)
                if not is_valid:
                    if 'invalid_iy_pattern' in reason or 'invalid_y_after' in reason:
                        old_word = corrected_word
                        corrected_word, changed = fix_invalid_y_pattern(
                            corrected_word,
                            use_lm=self.use_lm_for_ambiguous,
                            lm_model=self.lm_model
                        )
                        if changed:
                            metadata['corrections'].append({
                                'type': 'invalid_pattern',
                                'reason': reason,
                                'original': old_word,
                                'fixed': corrected_word
                            })
                    metadata['checks_failed'].append({
                        'word': word,
                        'reason': reason
                    })
                else:
                    metadata['checks_passed'].append(word)
            
            if self.enable_diacritic_check:
                for char in corrected_word:
                    is_valid, reason = check_diacritic_validity(char)
                    if not is_valid:
                        metadata['checks_failed'].append({
                            'word': corrected_word,
                            'char': char,
                            'reason': reason
                        })
            
            corrected_words.append(corrected_word)
        
        corrected_text = ' '.join(corrected_words)
        metadata['final'] = corrected_text
        metadata['was_modified'] = corrected_text != metadata['original']
        
        return corrected_text, metadata
    
    def set_language_model(self, lm_model):
        self.lm_model = lm_model
        self.use_lm_for_ambiguous = lm_model is not None


def create_conservative_postprocessor(
    use_lm: bool = False,
    lm_type: str = 'ngram'
) -> VietnameseConservativePostProcessor:
    lm_model = None
    
    if use_lm:
        if lm_type == 'phobert':
            try:
                from .phobert_model import PhoBERTLanguageModel
                lm_model = PhoBERTLanguageModel(device='cpu')
                logger.info("Using PhoBERT for ambiguous corrections")
            except Exception as e:
                logger.warning(f"Could not load PhoBERT: {e}")
        elif lm_type == 'ngram':
            try:
                from .ngram_model import VietnameseNGramModel, create_sample_vietnamese_corpus
                corpus = create_sample_vietnamese_corpus()
                lm_model = VietnameseNGramModel(n=3, smoothing='laplace')
                lm_model.train(corpus)
                logger.info("Using n-gram LM for ambiguous corrections")
            except Exception as e:
                logger.warning(f"Could not create n-gram model: {e}")
    
    return VietnameseConservativePostProcessor(
        enable_repeated_char_fix=True,
        enable_invalid_pattern_fix=True,
        enable_trailing_artifact_fix=True,
        enable_diacritic_check=True,
        enable_diacritic_fix=True,
        enable_capitalization_fix=True,
        use_lm_for_ambiguous=use_lm,
        lm_model=lm_model
    )
