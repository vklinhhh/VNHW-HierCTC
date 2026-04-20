# VNHW-HierCTC

**Hierarchical CTC with Multi-Scale Feature Fusion for Vietnamese Handwritten Text Recognition: A Diacritic-Aware Approach**

This repository implements a Hierarchical CTC architecture with dynamic multi-scale feature fusion for recognizing Vietnamese handwritten text. The model is specifically designed to handle Vietnamese diacritical marks through a sequential CTC decoder。

Key components:
- Vision encoder based on `microsoft/trocr-base-handwritten`
- Dynamic multi-scale feature fusion
- Local feature enhancer for diacritical marks
- Visual diacritic attention module
- Character-diacritic compatibility matrix

---

## Requirements

- Python 3.9+
- CUDA 12.8 (for GPU training)
- NVIDIA driver >= 570

---

## Installation

```bash
git clone https://github.com/vklinhhh/VNHW-HierCTC.git
cd VNHW-HierCTC
python -m venv venv
source venv/bin/activate      
pip install -r requirements.txt
```

---

## Project Structure

```
VNHW-HierCTC/
├── data/
│   ├── ctc_ocr_dataset.py        # Dataset class with augmentation
│   └── ctc_collation.py          # Collate function for DataLoader
├── model/
│   ├── hierarchical_ctc_model.py # Main model architecture
│   └── ocr_wrappers/             # Wrappers for baseline models (VietOCR, EasyOCR, Tesseract)
├── training/
│   ├── ctc_trainer.py            # Training loop
│   └── ctc_validation.py         # Validation metrics
├── scripts/
│   ├── train_hierarchical_ctc.py # Training entry point
│   └── compare_ocr_models.py
├── utils/
│   ├── ctc_utils.py              # CTC decoder
│   ├── schedulers.py
│   └── compatibility_logging.py
├── test_data/                    # Sample test images
├── inference.py                  # Inference entry point
├── finetune.py                   # Fine-tuning entry point
└── requirements.txt
```

---

## Pretrained Model

```
ckpt/
└── best_model_hf/
    ├── config.json
    ├── pytorch_model.bin
    ├── model_info.json
    ├── preprocessor_config.json
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── vocab.json
    ├── merges.txt
    └── vocabularies/
        └── combined_char_vocab.json
```

---

## Inference

### Single image

```bash
python inference.py \
  --model_path ./ckpt/best_model_hf \
  --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \
  --image ./test_data/image.png
```

### Folder of images

```bash
python inference.py \
  --model_path ./ckpt/best_model_hf \
  --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \
  --folder ./test_data \
  --output results.csv \
  --confidence
```

### Folder with subfolders (recursive)

If images are organized in nested subfolders, add `--recursive`:

```
test_data/
├── batch_1/
│   ├── img1.png
│   └── img2.png
└── batch_2/
    ├── img3.png
    └── img4.png
```

```bash
python inference.py \
  --model_path ./ckpt/best_model_hf \
  --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \
  --folder ./test_data \
  --recursive \
  --output results.csv \
  --confidence
```

Without `--recursive`, only images directly inside `--folder` are processed. With `--recursive`, all images in all subfolders are included.

### All inference arguments

| Argument | Description | Default |
|---|---|---|
| `--model_path` | Path to model checkpoint directory | required |
| `--vocab_path` | Path to `combined_char_vocab.json` | required |
| `--image` | Path to a single image | — |
| `--folder` | Path to a folder of images | — |
| `--output` | Output CSV path (batch mode only) | auto-generated |
| `--device` | `cuda` or `cpu` | auto-detect |
| `--batch_size` | Batch size | `1` |
| `--confidence` | Include confidence scores in output | `False` |
| `--recursive` | Search images recursively in subfolders | `False` |
| `--extensions` | Image extensions to scan | `.jpg .jpeg .png .bmp .tiff` |

---

## Fine-tuning

### Preparing Fine-tuning Data

Fine-tuning data must follow a flat folder structure where **each image file has a corresponding `.txt` file with the same base name**.

```
finetuning_data/
├── sample_001.png
├── sample_001.txt
├── sample_002.jpg
├── sample_002.txt
├── sample_003.png
├── sample_003.txt
└── ...
```

**Rules:**

| Rule | Detail |
|---|---|
| Pairing | Image and label file must share the exact same base name |
| Label content | One line of Vietnamese text per file |
| Encoding | UTF-8 (critical for Vietnamese diacritics) |
| Image formats | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` |
| Image type | Each image should be a **single text line**, not a full page |

**Example label files:**

```
# sample_001.txt
xin chào

# sample_002.txt
trường đại học khoa học tự nhiên

# sample_003.txt
nghiên cứu nhận dạng chữ viết tay tiếng Việt
```

### Running Fine-tuning

```bash
python finetune.py \
  --model_path ./ckpt/best_model_hf \
  --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \
  --data_folder ./finetuning_data \
  --output_dir ./finetuned_model \
  --epochs 10 \
  --batch_size 8 \
  --learning_rate 1e-4
```

To fine-tune only the classification heads (faster, less data required):

```bash
python finetune.py \
  --model_path ./ckpt/best_model_hf \
  --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \
  --data_folder ./finetuning_data \
  --output_dir ./finetuned_model \
  --tune_classifiers_only \
  --epochs 5
```

### Fine-tuning Arguments

| Argument | Description | Default |
|---|---|---|
| `--model_path` | Path to pretrained model checkpoint | required |
| `--vocab_path` | Path to `combined_char_vocab.json` | required |
| `--data_folder` | Path to fine-tuning data folder | required |
| `--output_dir` | Where to save the fine-tuned model | required |
| `--epochs` | Number of training epochs | `10` |
| `--batch_size` | Batch size | `8` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--val_split` | Fraction of data used for validation | `0.1` |
| `--freeze_vision_encoder` | Freeze the vision encoder weights | `False` |
| `--freeze_fusion` | Freeze the fusion module | `False` |
| `--freeze_transformer` | Freeze the transformer encoder | `False` |
| `--tune_classifiers_only` | Only tune classification heads | `False` |
| `--num_transformer_layers_to_tune` | Number of top transformer layers to unfreeze | `2` |
| `--use_amp` | Use automatic mixed precision (GPU only) | `False` |
| `--num_workers` | DataLoader workers | `4` |
| `--seed` | Random seed | `42` |

After fine-tuning, the best model is saved to `<output_dir>/best_model_hf/` and can be used directly with `inference.py`.

---

## Training from Scratch

```bash
python scripts/train_hierarchical_ctc.py \
  --dataset_name vklinhhh/combined_vietnamese_cwl \
  --output_dir ./output/hierarchical_ctc \
  --epochs 30 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --use_dynamic_fusion \
  --use_feature_enhancer \
  --use_visual_diacritic_attention \
  --use_character_diacritic_compatibility \
  --use_amp
```