# utils/ocr_metrics.py

import editdistance


def calculate_cer(prediction: str, ground_truth: str) -> float:
    if not ground_truth:
        return 1.0 if prediction else 0.0
    
    distance = editdistance.eval(prediction, ground_truth)
    return distance / len(ground_truth)


def calculate_wer(prediction: str, ground_truth: str) -> float:
    pred_words = prediction.split()
    gt_words = ground_truth.split()
    
    if not gt_words:
        return 1.0 if pred_words else 0.0
    
    distance = editdistance.eval(pred_words, gt_words)
    return distance / len(gt_words)


def calculate_accuracy(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction.strip() == ground_truth.strip() else 0.0
