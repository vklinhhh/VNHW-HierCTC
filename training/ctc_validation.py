# training/ctc_validation.py
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import logging
# from torchmetrics.text import CharErrorRate, WordErrorRate
import editdistance 

logger = logging.getLogger(__name__)

def compute_ctc_validation_metrics(model, val_loader, device, ctc_decoder):
    model.eval()
    total_val_loss = 0.0
    batch_count = 0

    all_preds_strings = []
    all_gts_strings = [] 

    logged_samples_count = 0
    max_samples_to_log = 5
    ctc_loss_fn = nn.CTCLoss(blank=ctc_decoder.blank_idx, reduction='sum', zero_infinity=True)

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None: continue

            try:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device) # Padded char indices [B, MaxLabelLen]
                label_lengths = batch['label_lengths'].to(device) # Actual lengths [B]
                texts_gt = batch['texts'] # Original text strings list[str]
                current_batch_size = pixel_values.size(0)
            except Exception as move_e:
                logger.error(f"Error moving validation batch {batch_idx} to device: {move_e}", exc_info=True)
                continue

            try:
                outputs = model(pixel_values=pixel_values)
                logits = outputs.get('logits') # Shape: [B, T, C]

                if logits is None:
                    logger.warning(f"Logits missing for batch {batch_idx}. Skipping.")
                    continue

                log_probs = logits.log_softmax(2).permute(1, 0, 2) # T, B, C
                time_steps = log_probs.size(0)
                input_lengths = torch.full((current_batch_size,), time_steps, dtype=torch.long, device='cpu')
                labels_cpu = labels.cpu()
                label_lengths_cpu = label_lengths.cpu()

                # Clamp lengths
                input_lengths_clamped = torch.clamp(input_lengths, min=0, max=time_steps)
                # First, clamp label_lengths to the actual padded dimension of labels_cpu
                label_lengths_clamped_intermediate = torch.clamp(label_lengths_cpu, min=0, max=labels_cpu.size(1))
                # Then, clamp against the model's output sequence length (time_steps)
                label_lengths_clamped = torch.clamp(label_lengths_clamped_intermediate, min=0, max=time_steps)
                # Filter out samples where label_length is 0 after clamping, as CTCLoss might error or give inf
                valid_mask = label_lengths_clamped > 0
                if torch.any(valid_mask):
                    active_log_probs = log_probs[:, valid_mask, :]
                    active_labels_cpu = labels_cpu[valid_mask]
                    active_input_lengths_clamped = input_lengths_clamped[valid_mask]
                    active_label_lengths_clamped = label_lengths_clamped[valid_mask]

                    if torch.isnan(active_log_probs).any() or torch.isinf(active_log_probs).any():
                        logger.warning(f"Validation batch {batch_idx}: Log_probs contained NaN/Inf. Skipping loss calculation for this batch.")
                        loss = torch.tensor(0.0)
                    else:
                        loss = ctc_loss_fn(active_log_probs.cpu(),
                                            active_labels_cpu, 
                                            active_input_lengths_clamped, 
                                            active_label_lengths_clamped)
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Validation CTC loss for batch {batch_idx} is NaN/Inf. Clamped Input Lens: {active_input_lengths_clamped.tolist()}, Clamped Label Lens: {active_label_lengths_clamped.tolist()}. Setting batch loss to 0.")
                            loss = torch.tensor(0.0)
                    total_val_loss += loss.item()
                else: #
                    loss = torch.tensor(0.0)
                decoded_preds = ctc_decoder(logits) # [B, T, C], returns list[str]
                all_preds_strings.extend(decoded_preds)
                all_gts_strings.extend(texts_gt)

                if batch_idx == 0:
                    num_to_log = min(max_samples_to_log - logged_samples_count, current_batch_size)
                    for i in range(num_to_log):
                        logger.info(f"--- Validation Sample {logged_samples_count + i} ---")
                        logger.info(f"  Ground Truth: '{texts_gt[i]}'")
                        logger.info(f"  Prediction  : '{decoded_preds[i]}'")
                        logger.info(f"---------------------------------")
                    logged_samples_count += num_to_log


                batch_count += 1 

            except Exception as batch_e:
                logger.error(f"Error processing validation batch {batch_idx}: {batch_e}", exc_info=True)
                logger.error(f"  Logits shape: {logits.shape if 'logits' in locals() else 'N/A'}")
                logger.error(f"  Labels shape: {labels.shape if 'labels' in locals() else 'N/A'}")
                logger.error(f"  Label lengths: {label_lengths.tolist() if 'label_lengths' in locals() else 'N/A'}")
                continue 

    model.train() 
    num_samples_for_loss_avg = len(all_gts_strings)
    if val_loader.dataset: num_samples_for_loss_avg = len(val_loader.dataset)

    avg_val_loss = total_val_loss / num_samples_for_loss_avg if num_samples_for_loss_avg > 0 else 0.0
    total_edit_distance_char = 0
    total_chars = 0
    total_edit_distance_word = 0
    total_words = 0

    if all_gts_strings: 
        for pred_str, gt_str in zip(all_preds_strings, all_gts_strings):
            try:
                total_edit_distance_char += editdistance.eval(pred_str, gt_str)
                total_chars += len(gt_str)
            except Exception as cer_e:
                logger.warning(f"Could not calculate char edit distance for GT='{gt_str}', Pred='{pred_str}': {cer_e}")

            try:
                pred_words = pred_str.split()
                gt_words = gt_str.split()
                total_edit_distance_word += editdistance.eval(pred_words, gt_words)
                total_words += len(gt_words)
            except Exception as wer_e:
                logger.warning(f"Could not calculate word edit distance for GT='{gt_str}', Pred='{pred_str}': {wer_e}")


    val_cer = total_edit_distance_char / total_chars if total_chars > 0 else 1.0
    val_wer = total_edit_distance_word / total_words if total_words > 0 else 1.0

    logger.info(f"Validation Results: Loss={avg_val_loss:.4f}, CER={val_cer:.4f}, WER={val_wer:.4f}")
    logger.info(f"(Based on {len(all_gts_strings)} samples for CER/WER, {num_samples_for_loss_avg} for loss estimation)")

    results = {
        'val_loss': avg_val_loss,
        'val_cer': val_cer,
        'val_wer': val_wer
    }
    return results