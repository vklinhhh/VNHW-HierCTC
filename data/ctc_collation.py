# data/ctc_collation.py
import torch
import logging

logger = logging.getLogger(__name__)
def ctc_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None

    try:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
    except Exception as e:
        logger.error(f"Error stacking pixel_values: {e}.")
        shapes = [item['pixel_values'].shape for item in batch]
        logger.error(f"Pixel value shapes in failing batch: {shapes}")
        return None 

    labels = [item['labels'] for item in batch]
    label_lengths = torch.tensor([len(lab) for lab in labels], dtype=torch.long)
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=0
    )
    texts = [item.get('text', '') for item in batch]

    collated_batch = {
        'pixel_values': pixel_values,
        'labels': padded_labels,
        'label_lengths': label_lengths,
        'texts': texts
    }

    if 'original_image_pil' in batch[0]:
        original_images_pil = [item.get('original_image_pil') for item in batch]

        if all(img is not None for img in original_images_pil):
                collated_batch['original_images_pil'] = original_images_pil
        else:
            logger.warning("batch error: some original_image_pil are None")


    return collated_batch