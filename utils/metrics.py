# utils/metrics.py
import jiwer
import logging
import wandb

logger = logging.getLogger(__name__)

def calculate_cer_wer(predictions, references):
    if not predictions or not references or len(predictions) != len(references):
        logger.warning("Invalid input for CER/WER calculation. Predictions or references empty or mismatched lengths.")
        return {'cer': 1.0, 'wer': 1.0}

    try:
        wer = jiwer.wer(references, predictions)
        total_chars = 0
        total_char_errors = 0
        
        for ref, pred in zip(references, predictions):
            ref_clean = ref.strip().lower()
            pred_clean = pred.strip().lower()
            ref_len = len(ref_clean)
            pred_len = len(pred_clean)
            dp = [[0] * (pred_len + 1) for _ in range(ref_len + 1)]
            for i in range(ref_len + 1):
                dp[i][0] = i
            for j in range(pred_len + 1):
                dp[0][j] = j
            for i in range(1, ref_len + 1):
                for j in range(1, pred_len + 1):
                    if ref_clean[i-1] == pred_clean[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            
            char_errors = dp[ref_len][pred_len]
            total_char_errors += char_errors
            total_chars += ref_len
        
        cer = total_char_errors / total_chars if total_chars > 0 else 1.0

    except Exception as e:
        total_chars = 0
        total_char_errors = 0
        word_matches = 0
        total_words = 0
        
        for ref, pred in zip(references, predictions):
            ref_clean = ref.strip().lower()
            pred_clean = pred.strip().lower()
            ref_len = len(ref_clean)
            pred_len = len(pred_clean)
            max_len = max(ref_len, pred_len)
            
            char_errors = 0
            for i in range(max_len):
                if i >= ref_len or i >= pred_len:
                    char_errors += 1
                elif ref_clean[i] != pred_clean[i]:
                    char_errors += 1
            
            total_char_errors += char_errors
            total_chars += ref_len
            ref_words = ref_clean.split()
            pred_words = pred_clean.split()
            word_matches += sum(1 for rw, pw in zip(ref_words, pred_words) if rw == pw)
            total_words += max(len(ref_words), len(pred_words))
        
        cer = total_char_errors / total_chars if total_chars > 0 else 1.0
        wer = 1.0 - (word_matches / total_words) if total_words > 0 else 1.0


    return {
        'cer': cer if cer is not None else 1.0,
        'wer': wer if wer is not None else 1.0
    }

def log_metrics_to_wandb(wandb_run, metrics_dict, epoch, global_step, commit=True, prefix="val"):
    if wandb_run is None:
        return

    log_payload = {}
    for key, value in metrics_dict.items():
        if not key.startswith(f"{prefix}/"):
            log_payload[f"{prefix}/{key}"] = value
        else:
            log_payload[key] = value
    if f"{prefix}/epoch" not in log_payload:
        log_payload[f"{prefix}/epoch_context"] = epoch 
    if f"{prefix}/global_step" not in log_payload:
        log_payload[f"{prefix}/global_step_context"] = global_step

    try:
        wandb_run.log(log_payload, step=global_step, commit=commit)
    except Exception as e:
        logger.error(f"Failed to log metrics to WandB: {e}")