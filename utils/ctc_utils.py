# utils/ctc_utils.py
import torch
import itertools
import logging
import unicodedata
from collections import defaultdict

logger = logging.getLogger(__name__)

class CTCDecoder:
    def __init__(self, idx_to_char_map, blank_idx=0, output_delimiter=''):
        self.idx_to_char = idx_to_char_map
        self.blank_idx = blank_idx
        self.output_delimiter = output_delimiter

    def __call__(self, logits):
        if logits.ndim != 3: raise ValueError(f"Logits 3D, got {logits.shape}")
        logits = logits.cpu().detach()
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded_batch = []
        for pred_seq in predicted_ids:
            merged = [k for k, _ in itertools.groupby(pred_seq.tolist())]
            cleaned = [k for k in merged if k != self.blank_idx]
            try:
                decoded_elements = [self.idx_to_char.get(idx, '?') for idx in cleaned]
                decoded_string = self.output_delimiter.join(decoded_elements)
            except Exception as e:
                logger.warning(f"Decode map error: {e}. Sequence: {cleaned}")
                decoded_string = "<DECODE_ERROR>"
            decoded_batch.append(decoded_string)
        return decoded_batch


def build_ctc_vocab(char_list, add_blank=True, add_unk=True, unk_token='[UNK]'):
    vocab = []
    if add_blank: vocab.append('<blank>')
    processed_char_list = []
    seen_chars = set(vocab)
    for char_item in char_list:
        if char_item not in seen_chars:
            processed_char_list.append(char_item)
            seen_chars.add(char_item)
    
    unique_chars_from_input = sorted(list(set(processed_char_list)))
    vocab.extend(unique_chars_from_input)

    if add_unk:
        if unk_token not in vocab:
            vocab.append(unk_token)

    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for idx, char in enumerate(vocab)}
    return vocab, char_to_idx, idx_to_char

def build_combined_vietnamese_charset(include_basic_latin=True, include_digits=True, include_punctuation=True, additional_chars=""):
    charset = set()
    vowels_base = "aeiouy" 

    diacritics_map = {
        'acute': "\u0301",      # Sắc
        'grave': "\u0300",      # Huyền
        'hook': "\u0309",       # Hỏi
        'tilde': "\u0303",      # Ngã
        'dot': "\u0323",        # Nặng
        'circumflex': "\u0302", # Â, Ê, Ô
        'breve': "\u0306",      # Ă
        'horn': "\u031b",       # Ơ, Ư
        'stroke': "\u0336" # d with stroke (đ), though 'đ' is often treated as a base. Using for completeness if d + stroke exists.
                        # 'đ', unicodedata.normalize('NFD', 'đ') gives 'd\u0336'
    }
    
    for v_char_base in vowels_base:
        for case_func in [str.lower, str.upper]:
            bv_cased = case_func(v_char_base)
            charset.add(bv_cased)
            for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
                combined = unicodedata.normalize('NFC', bv_cased + diacritics_map[tone_name])
                charset.add(combined)
    # Ă (a + breve)
    for case_func in [str.lower, str.upper]:
        a_breve_base = unicodedata.normalize('NFC', case_func('a') + diacritics_map['breve']) # ă, Ă
        charset.add(a_breve_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', a_breve_base + diacritics_map[tone_name]))
            
    # Â (a + circumflex)
    for case_func in [str.lower, str.upper]:
        a_circ_base = unicodedata.normalize('NFC', case_func('a') + diacritics_map['circumflex']) # â, Â
        charset.add(a_circ_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', a_circ_base + diacritics_map[tone_name]))

    # Ê (e + circumflex)
    for case_func in [str.lower, str.upper]:
        e_circ_base = unicodedata.normalize('NFC', case_func('e') + diacritics_map['circumflex']) # ê, Ê
        charset.add(e_circ_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', e_circ_base + diacritics_map[tone_name]))

    # Ô (o + circumflex)
    for case_func in [str.lower, str.upper]:
        o_circ_base = unicodedata.normalize('NFC', case_func('o') + diacritics_map['circumflex']) # ô, Ô
        charset.add(o_circ_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', o_circ_base + diacritics_map[tone_name]))
            
    # Ơ (o + horn)
    for case_func in [str.lower, str.upper]:
        o_horn_base = unicodedata.normalize('NFC', case_func('o') + diacritics_map['horn']) # ơ, Ơ
        charset.add(o_horn_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', o_horn_base + diacritics_map[tone_name]))

    # Ư (u + horn)
    for case_func in [str.lower, str.upper]:
        u_horn_base = unicodedata.normalize('NFC', case_func('u') + diacritics_map['horn']) # ư, Ư
        charset.add(u_horn_base)
        for tone_name in ['acute', 'grave', 'hook', 'tilde', 'dot']:
            charset.add(unicodedata.normalize('NFC', u_horn_base + diacritics_map[tone_name]))
    
    if include_basic_latin:
        consonants_from_base_vocab = "bcdfghklmnpqrstvxzfjw"
        for c in consonants_from_base_vocab: 
            charset.add(c)
            charset.add(c.upper())

    charset.add('đ'); charset.add('Đ')

    if include_digits:
        for d in "0123456789": charset.add(d)

    if include_punctuation:
        punct = " .,-_()[]{}:;\"'/\\?!@#$%^&*+=<>|" 
        for p in punct: charset.add(p)
        charset.add(' ')
    for char in additional_chars: charset.add(char)
    charset.discard("")
    
    return sorted(list(charset))


VIETNAMESE_DIACRITIC_UNICODE_TO_NAME = {
    '\u0301': 'acute',      # Sắc
    '\u0300': 'grave',      # Huyền
    '\u0309': 'hook',       # Hỏi (hook above)
    '\u0303': 'tilde',      # Ngã
    '\u0323': 'dot',        # Nặng (dot below)
    '\u0302': 'circumflex', # Â, Ê, Ô
    '\u0306': 'breve',      # Ă
    '\u031b': 'horn',       # Ơ, Ư
    '\u0336': 'stroke'      # For đ
}
VIETNAMESE_DIACRITIC_NAME_TO_UNICODE = {v: k for k, v in VIETNAMESE_DIACRITIC_UNICODE_TO_NAME.items()}
VIETNAMESE_TONE_NAMES = {'acute', 'grave', 'hook', 'tilde', 'dot'}
VIETNAMESE_MODIFIER_NAMES = {'circumflex', 'breve', 'horn', 'stroke'}

def decompose_vietnamese_char(char_in):
    if not char_in or len(char_in) != 1:
        return (char_in, 'no_diacritic', None)

    nfd_char = unicodedata.normalize('NFD', char_in)
    
    base_char = ""
    combining_diacritics_unicode = []

    for ch_part in nfd_char:
        if unicodedata.category(ch_part) != 'Mn': # Mn = Mark, Nonspacing (combining characters)
            base_char += ch_part
        else:
            combining_diacritics_unicode.append(ch_part)

    if not base_char:
        return (char_in, 'no_diacritic', None)

    base_char_nfc = unicodedata.normalize('NFC', base_char)

    if not combining_diacritics_unicode:
        return (base_char_nfc, 'no_diacritic', None)

    tone_unicode_parts = []
    modifier_unicode_parts = []

    for diac_unicode in combining_diacritics_unicode:
        diac_name_from_map = VIETNAMESE_DIACRITIC_UNICODE_TO_NAME.get(diac_unicode)
        if diac_name_from_map:
            if diac_name_from_map in VIETNAMESE_TONE_NAMES:
                tone_unicode_parts.append(diac_unicode)
            elif diac_name_from_map in VIETNAMESE_MODIFIER_NAMES:
                modifier_unicode_parts.append(diac_unicode)

    tone_name_only = None
    if tone_unicode_parts:
        tone_name_only = VIETNAMESE_DIACRITIC_UNICODE_TO_NAME.get(tone_unicode_parts[0])
        
    modifier_names_list = sorted([VIETNAMESE_DIACRITIC_UNICODE_TO_NAME.get(m) for m in modifier_unicode_parts if VIETNAMESE_DIACRITIC_UNICODE_TO_NAME.get(m)])
    
    diacritic_name_parts = []
    diacritic_name_parts.extend(modifier_names_list)
    if tone_name_only:
        diacritic_name_parts.append(tone_name_only)
    
    if not diacritic_name_parts:
        diacritic_name_combined = 'no_diacritic'
    else:
        if len(modifier_names_list) > 1: 
            logger.debug(f"Multiple modifiers found for '{char_in}': {modifier_names_list}. This might need specific handling.")

        final_diac_name_parts = []
        if 'stroke' in modifier_names_list:
            final_diac_name_parts = ['stroke']
        else:
            primary_modifier = None
            if 'circumflex' in modifier_names_list: primary_modifier = 'circumflex'
            elif 'breve' in modifier_names_list: primary_modifier = 'breve'
            elif 'horn' in modifier_names_list: primary_modifier = 'horn'
            
            if primary_modifier:
                final_diac_name_parts.append(primary_modifier)
            if tone_name_only:
                final_diac_name_parts.append(tone_name_only)
        
        if not final_diac_name_parts:
            diacritic_name_combined = tone_name_only if tone_name_only else 'no_diacritic'
        else:
            diacritic_name_combined = "_".join(final_diac_name_parts)

        if not diacritic_name_combined:
            diacritic_name_combined = 'no_diacritic'

    return base_char_nfc, diacritic_name_combined, tone_name_only


def get_char_type(char_in):
    if not char_in or len(char_in) != 1:
        return "symbol"
    nfd_char = unicodedata.normalize('NFD', char_in)
    base_form = ""
    for ch_part in nfd_char:
        if unicodedata.category(ch_part) != 'Mn':
            base_form += ch_part
    base_form_nfc = unicodedata.normalize('NFC', base_form) # NFC of the base
    
    base_lower = base_form_nfc.lower()

    vietnamese_vowels_base_lower = {'a', 'e', 'i', 'o', 'u', 'y', 'ă', 'â', 'ê', 'ô', 'ơ', 'ư'}
    vietnamese_consonants_base_lower = {'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'f', 'j', 'w', 'z'}
    digits = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

    if base_lower in vietnamese_vowels_base_lower:
        return "vowel"
    elif base_lower in vietnamese_consonants_base_lower:
        return "consonant"
    elif base_lower in digits:
        return "digit"
    else:
        return "symbol"

if __name__ == "__main__":
    test_chars = ['a', 'à', 'ă', 'ằ', 'â', 'ấ', 'đ', 'd', 'ệ', 'k', ' ', '1', '!', 'ườ']
    for char_to_test in test_chars:
        base, diac_name, tone_name = decompose_vietnamese_char(char_to_test)
        char_type = get_char_type(char_to_test)
        print(f"Char: '{char_to_test}' (Type: {char_type}) -> Base: '{base}', CombinedDiacName: '{diac_name}', ToneOnlyName: '{tone_name}'")

    full_charset = build_combined_vietnamese_charset()
    print(f"Generated {len(full_charset)} combined Vietnamese characters. First 10: {full_charset[:10]}, Last 10: {full_charset[-10:]}")
    expected_chars = ['ệ', 'ườ', 'ặ', 'ẫ']
    for ec in expected_chars:
        if ec in full_charset:
            print(f"'{ec}' found in generated charset.")
        else:
            print(f"ERROR: '{ec}' NOT found in generated charset.")
    sample_chars = ['a', 'b', 'c', 'a']
    vocab, c2i, i2c = build_ctc_vocab(sample_chars)
    print(f"Sample vocab: {vocab}")
    print(f"Char to Idx: {c2i}")


def extract_diacritics(text):
    diacritics_map = {}
    diacritics = {
        'á': ('a', 'acute'), 'é': ('e', 'acute'), 'í': ('i', 'acute'), 
        'ó': ('o', 'acute'), 'ú': ('u', 'acute'), 'ý': ('y', 'acute'),
        'Á': ('A', 'acute'), 'É': ('E', 'acute'), 'Í': ('I', 'acute'),
        'Ó': ('O', 'acute'), 'Ú': ('U', 'acute'), 'Ý': ('Y', 'acute'),
        'à': ('a', 'grave'), 'è': ('e', 'grave'), 'ì': ('i', 'grave'),
        'ò': ('o', 'grave'), 'ù': ('u', 'grave'), 'ỳ': ('y', 'grave'),
        'À': ('A', 'grave'), 'È': ('E', 'grave'), 'Ì': ('I', 'grave'),
        'Ò': ('O', 'grave'), 'Ù': ('U', 'grave'), 'Ỳ': ('Y', 'grave'),
        'ả': ('a', 'hook'), 'ẻ': ('e', 'hook'), 'ỉ': ('i', 'hook'),
        'ỏ': ('o', 'hook'), 'ủ': ('u', 'hook'), 'ỷ': ('y', 'hook'),
        'Ả': ('A', 'hook'), 'Ẻ': ('E', 'hook'), 'Ỉ': ('I', 'hook'),
        'Ỏ': ('O', 'hook'), 'Ủ': ('U', 'hook'), 'Ỷ': ('Y', 'hook'),
        'ã': ('a', 'tilde'), 'ẽ': ('e', 'tilde'), 'ĩ': ('i', 'tilde'),
        'õ': ('o', 'tilde'), 'ũ': ('u', 'tilde'), 'ỹ': ('y', 'tilde'),
        'Ã': ('A', 'tilde'), 'Ẽ': ('E', 'tilde'), 'Ĩ': ('I', 'tilde'),
        'Õ': ('O', 'tilde'), 'Ũ': ('U', 'tilde'), 'Ỹ': ('Y', 'tilde'),
        'ạ': ('a', 'dot'), 'ẹ': ('e', 'dot'), 'ị': ('i', 'dot'),
        'ọ': ('o', 'dot'), 'ụ': ('u', 'dot'), 'ỵ': ('y', 'dot'),
        'Ạ': ('A', 'dot'), 'Ẹ': ('E', 'dot'), 'Ị': ('I', 'dot'),
        'Ọ': ('O', 'dot'), 'Ụ': ('U', 'dot'), 'Ỵ': ('Y', 'dot'),
        'ấ': ('a', 'circumflex_acute'), 'ầ': ('a', 'circumflex_grave'), 
        'ẩ': ('a', 'circumflex_hook'), 'ẫ': ('a', 'circumflex_tilde'), 
        'ậ': ('a', 'circumflex_dot'),
        'Ấ': ('A', 'circumflex_acute'), 'Ầ': ('A', 'circumflex_grave'), 
        'Ẩ': ('A', 'circumflex_hook'), 'Ẫ': ('A', 'circumflex_tilde'), 
        'Ậ': ('A', 'circumflex_dot'),
        'ế': ('e', 'circumflex_acute'), 'ề': ('e', 'circumflex_grave'), 
        'ể': ('e', 'circumflex_hook'), 'ễ': ('e', 'circumflex_tilde'), 
        'ệ': ('e', 'circumflex_dot'),
        'Ế': ('E', 'circumflex_acute'), 'Ề': ('E', 'circumflex_grave'), 
        'Ể': ('E', 'circumflex_hook'), 'Ễ': ('E', 'circumflex_tilde'), 
        'Ệ': ('E', 'circumflex_dot'),
        'ố': ('o', 'circumflex_acute'), 'ồ': ('o', 'circumflex_grave'), 
        'ổ': ('o', 'circumflex_hook'), 'ỗ': ('o', 'circumflex_tilde'), 
        'ộ': ('o', 'circumflex_dot'),
        'Ố': ('O', 'circumflex_acute'), 'Ồ': ('O', 'circumflex_grave'), 
        'Ổ': ('O', 'circumflex_hook'), 'Ỗ': ('O', 'circumflex_tilde'), 
        'Ộ': ('O', 'circumflex_dot'),
        'ắ': ('a', 'breve_acute'), 'ằ': ('a', 'breve_grave'), 
        'ẳ': ('a', 'breve_hook'), 'ẵ': ('a', 'breve_tilde'), 
        'ặ': ('a', 'breve_dot'),
        'Ắ': ('A', 'breve_acute'), 'Ằ': ('A', 'breve_grave'), 
        'Ẳ': ('A', 'breve_hook'), 'Ẵ': ('A', 'breve_tilde'), 
        'Ặ': ('A', 'breve_dot'),
        'ớ': ('o', 'horn_acute'), 'ờ': ('o', 'horn_grave'), 
        'ở': ('o', 'horn_hook'), 'ỡ': ('o', 'horn_tilde'), 
        'ợ': ('o', 'horn_dot'),
        'Ớ': ('O', 'horn_acute'), 'Ờ': ('O', 'horn_grave'), 
        'Ở': ('O', 'horn_hook'), 'Ỡ': ('O', 'horn_tilde'), 
        'Ợ': ('O', 'horn_dot'),
        'ứ': ('u', 'horn_acute'), 'ừ': ('u', 'horn_grave'), 
        'ử': ('u', 'horn_hook'), 'ữ': ('u', 'horn_tilde'), 
        'ự': ('u', 'horn_dot'),
        'Ứ': ('U', 'horn_acute'), 'Ừ': ('U', 'horn_grave'), 
        'Ử': ('U', 'horn_hook'), 'Ữ': ('U', 'horn_tilde'), 
        'Ự': ('U', 'horn_dot'),
        'â': ('a', 'circumflex'), 'ê': ('e', 'circumflex'), 'ô': ('o', 'circumflex'),
        'Â': ('A', 'circumflex'), 'Ê': ('E', 'circumflex'), 'Ô': ('O', 'circumflex'),
        'ă': ('a', 'breve'), 'Ă': ('A', 'breve'),
        'ơ': ('o', 'horn'), 'ư': ('u', 'horn'),
        'Ơ': ('O', 'horn'), 'Ư': ('U', 'horn'),
        'đ': ('d', 'stroke'), 'Đ': ('D', 'stroke'),
    }
    for i, char in enumerate(text):
        if char in diacritics:
            base_char, diacritic = diacritics[char]
            diacritics_map[i] = {'char': char, 'base': base_char, 'diacritic': diacritic}
    return diacritics_map


def analyze_diacritic_accuracy(ground_truths, predictions):
    diacritic_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'examples': []})
    for gt, pred in zip(ground_truths, predictions):
        if len(gt) != len(pred): 
            continue
        gt_diacritics = extract_diacritics(gt)
        pred_diacritics = extract_diacritics(pred)
        for i in gt_diacritics:
            if i < len(pred):
                gt_diac = gt_diacritics[i]['diacritic']
                diacritic_stats[gt_diac]['total'] += 1
                if i in pred_diacritics and pred_diacritics[i]['diacritic'] == gt_diac:
                    diacritic_stats[gt_diac]['correct'] += 1
                else:
                    error_example = {
                        'gt_char': gt_diacritics[i]['char'],
                        'pred_char': pred[i] if i < len(pred) else '',
                        'gt_context': gt[max(0, i-5):min(len(gt), i+6)],
                        'pred_context': pred[max(0, i-5):min(len(pred), i+6)],
                    }
                    if len(diacritic_stats[gt_diac]['examples']) < 10:
                        diacritic_stats[gt_diac]['examples'].append(error_example)
    
    results = {}
    for diac, stats in diacritic_stats.items():
        if stats['total'] > 0:
            results[diac] = {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total'],
                'examples': stats['examples']
            }
    return results