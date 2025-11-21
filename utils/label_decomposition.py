
# utils/label_decomposition.py

import torch
from utils.ctc_utils import decompose_vietnamese_char

def decompose_labels_for_multitask(combined_labels, combined_vocab, base_vocab, diacritic_vocab):
    batch_size, max_len = combined_labels.shape
    device = combined_labels.device

    base_char_to_idx = {char: idx for idx, char in enumerate(base_vocab)}
    diacritic_to_idx = {diac: idx for idx, diac in enumerate(diacritic_vocab)}
    
    base_labels = torch.zeros_like(combined_labels)
    diacritic_labels = torch.zeros_like(combined_labels)
    
    for batch_idx in range(batch_size):
        for pos_idx in range(max_len):
            combined_idx = combined_labels[batch_idx, pos_idx].item()
            
            if combined_idx < len(combined_vocab):
                combined_char = combined_vocab[combined_idx]
                base_char, diacritic_name, _ = decompose_vietnamese_char(combined_char)
                base_idx = base_char_to_idx.get(base_char, base_char_to_idx.get('[UNK]', 1))
                diacritic_idx = diacritic_to_idx.get(diacritic_name, diacritic_to_idx.get('no_diacritic', 1))
                
                base_labels[batch_idx, pos_idx] = base_idx
                diacritic_labels[batch_idx, pos_idx] = diacritic_idx
            else:
                base_labels[batch_idx, pos_idx] = combined_idx if combined_idx < len(base_vocab) else 0
                diacritic_labels[batch_idx, pos_idx] = 0
    
    return base_labels, diacritic_labels