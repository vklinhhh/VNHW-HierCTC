import os
import sys
import argparse
import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
from data.ctc_ocr_dataset import CtcOcrDataset
from data.ctc_collation import ctc_collate_fn
from training.ctc_trainer import train_ctc_model
from utils.optimizers import create_optimizer
from utils.schedulers import CosineWarmupScheduler
from utils.ctc_utils import CTCDecoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('finetuning.log')
    ]
)
logger = logging.getLogger('Vietnamese-HTR-Finetuning')
class CustomDatasetLoader:
    """
        data_folder/
            page_001_line_001.png
            page_001_line_001.txt
            page_001_line_002.png
            page_001_line_002.txt
            ...
    """
    
    def __init__(
        self,
        data_folder: str,
        image_extensions: List[str] = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
        encoding: str = 'utf-8'
    ):
        self.data_folder = Path(data_folder)
        self.image_extensions = image_extensions
        self.encoding = encoding
        
        if not self.data_folder.exists():
            raise ValueError(f"Data folder not found: {data_folder}")
    
    def find_image_text_pairs(self) -> List[Dict]:
        pairs = []
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.data_folder.glob(f"*{ext}"))
            image_files.extend(self.data_folder.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} image files in {self.data_folder}")
        for image_path in tqdm(image_files, desc="Loading image-text pairs"):
            base_name = image_path.stem
            text_path = image_path.parent / f"{base_name}.txt"
            
            if text_path.exists():
                try:
                    with open(text_path, 'r', encoding=self.encoding) as f:
                        text_content = f.read().strip()
                    
                    if text_content:
                        pairs.append({
                            'image': str(image_path),
                            'label': text_content,
                            'image_id': base_name
                        })
                    else:
                        logger.warning(f"Empty text file: {text_path}")
                        
                except Exception as e:
                    logger.error(f"Error reading text file {text_path}: {e}")
            else:
                logger.warning(f"No text file found for image: {image_path}")
        
        logger.info(f"Successfully loaded {len(pairs)} image-text pairs")
        
        if len(pairs) == 0:
            raise ValueError(f"No valid image-text pairs found in {self.data_folder}")
        
        return pairs
    
    def create_huggingface_dataset(
        self,
        train_split: float = 0.9,
        val_split: float = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        pairs = self.find_image_text_pairs()
        dataset = Dataset.from_dict({
            'image': [p['image'] for p in pairs],
            'label': [p['label'] for p in pairs],
            'image_id': [p['image_id'] for p in pairs]
        })
        
        logger.info(f"Created HuggingFace dataset with {len(dataset)} samples")
        if val_split > 0:
            split_dataset = dataset.train_test_split(
                test_size=val_split,
                seed=seed
            )
            dataset_dict = DatasetDict({
                'train': split_dataset['train'],
                'validation': split_dataset['test']
            })
            logger.info(f"Split dataset: Train={len(dataset_dict['train'])}, Val={len(dataset_dict['validation'])}")
        else:
            dataset_dict = DatasetDict({
                'train': dataset
            })
            logger.info(f"Using all {len(dataset)} samples for training (no validation split)")
        
        return dataset_dict


def freeze_model_layers(
    model,
    freeze_vision_encoder: bool = True,
    freeze_fusion: bool = False,
    freeze_transformer: bool = False,
    num_transformer_layers_to_tune: int = 2,
    tune_classifiers_only: bool = False
):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if tune_classifiers_only:
        logger.info("Freezing ALL layers except classification heads")

        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, 'base_classifier'):
            for param in model.base_classifier.parameters():
                param.requires_grad = True
            logger.info("Base classifier: TRAINABLE")
        
        if hasattr(model, 'diacritic_classifier'):
            for param in model.diacritic_classifier.parameters():
                param.requires_grad = True
            logger.info("Diacritic classifier: TRAINABLE")
        
        if hasattr(model, 'final_classifier'):
            for param in model.final_classifier.parameters():
                param.requires_grad = True
            logger.info("Final classifier: TRAINABLE")

        if hasattr(model, 'diacritic_condition_proj') and model.diacritic_condition_proj:
            for param in model.diacritic_condition_proj.parameters():
                param.requires_grad = True
            logger.info("Diacritic conditioning: TRAINABLE")
    
    else:
        if freeze_vision_encoder:
            logger.info("Freezing: Vision Encoder")
            for param in model.vision_encoder.parameters():
                param.requires_grad = False
        else:
            logger.info("Vision Encoder: TRAINABLE")
        
        if freeze_fusion:
            logger.info("Freezing: Fusion Layers")
            if hasattr(model, 'fusion_projection'):
                for param in model.fusion_projection.parameters():
                    param.requires_grad = False
            if hasattr(model, 'dynamic_fusion') and model.dynamic_fusion:
                for param in model.dynamic_fusion.parameters():
                    param.requires_grad = False
        else:
            logger.info("Fusion Layers: TRAINABLE")
        if freeze_transformer:
            logger.info("Freezing: ALL Transformer Layers")
            for param in model.transformer_encoder.parameters():
                param.requires_grad = False
        elif num_transformer_layers_to_tune > 0 and hasattr(model, 'transformer_encoder'):
            num_layers = len(model.transformer_encoder.layers)
            num_to_freeze = max(0, num_layers - num_transformer_layers_to_tune)
            
            logger.info(f"Freezing: Bottom {num_to_freeze} transformer layers")
            logger.info(f"Top {num_transformer_layers_to_tune} transformer layers: TRAINABLE")
            
            for i, layer in enumerate(model.transformer_encoder.layers):
                if i < num_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            logger.info("ALL Transformer Layers: TRAINABLE")
    
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable before: {trainable_before:,} ({100*trainable_before/total_params:.2f}%)")
    logger.info(f"Trainable after:  {trainable_after:,} ({100*trainable_after/total_params:.2f}%)")
    logger.info(f"Frozen: {total_params - trainable_after:,} parameters")


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune Vietnamese HTR model on custom dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            # Fine-tune with custom data folder
            python finetune.py \\
                --model_path ./ckpt/best_model_hf \\
                --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \\
                --data_folder ./finetuning_data \\
                --output_dir ./finetuned_model \\
                --epochs 10 \\
                --batch_size 8 \\
                --learning_rate 1e-4
            
            # Fine-tune only classification heads 
            python finetune.py \\
                --model_path ./ckpt/best_model_hf \\
                --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \\
                --data_folder ./finetuning_data \\
                --tune_classifiers_only \\
                --epochs 5
        """
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to pre-trained model checkpoint'
    )
    parser.add_argument(
        '--vocab_path',
        type=str,
        required=True,
        help='Path to combined character vocabulary JSON file'
    )
    parser.add_argument(
        '--data_folder',
        type=str,
        required=True,
        help='Path to folder containing image-text pairs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./finetuned_model',
        help='Output directory for fine-tuned model (default: ./finetuned_model)'
    )
    
    parser.add_argument(
        '--val_split',
        type=float,
        default=0.1,
        help='Validation split ratio (default: 0.1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=11,
        help='Random seed'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Training batch size (default: 8)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='Weight decay (default: 0.01)'
    )
    
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=100,
        help='Number of warmup steps (default: 100)'
    )
    
    parser.add_argument(
        '--grad_accumulation',
        type=int,
        default=1,
        help='Gradient accumulation steps (default: 1)'
    )

    freeze_group = parser.add_argument_group('Layer Freezing Options')
    freeze_group.add_argument(
        '--tune_classifiers_only',
        action='store_true',
        help='Only fine-tune classification heads (fastest, recommended for small datasets)'
    )
    
    freeze_group.add_argument(
        '--freeze_vision_encoder',
        action='store_true',
        default=True,
        help='Freeze vision encoder (default: True)'
    )
    
    freeze_group.add_argument(
        '--unfreeze_vision_encoder',
        action='store_true',
        help='Unfreeze vision encoder (overrides --freeze_vision_encoder)'
    )
    
    freeze_group.add_argument(
        '--freeze_fusion',
        action='store_true',
        help='Freeze fusion layers'
    )
    
    freeze_group.add_argument(
        '--freeze_transformer',
        action='store_true',
        help='Freeze all transformer layers'
    )
    
    freeze_group.add_argument(
        '--num_transformer_layers_to_tune',
        type=int,
        default=12,
        help='Number of top transformer layers to keep trainable (default: 2)'
    )
    
    # Training options
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device (default: auto-detect)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    
    parser.add_argument(
        '--use_amp',
        action='store_true',
        help='Use automatic mixed precision training'
    )
    
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='Logging interval in steps (default: 10)'
    )
    
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=None,
        help='Evaluation interval in steps (default: end of epoch)'
    )
    
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=5,
        help='Early stopping patience (default: 5)'
    )

    parser.add_argument(
        '--image_extensions',
        type=str,
        nargs='+',
        default=['.png', '.jpg', '.jpeg'],
        help='Image file extensions (default: .png .jpg .jpeg)'
    )
    
    parser.add_argument(
        '--text_encoding',
        type=str,
        default='utf-8',
        help='Text file encoding (default: utf-8)'
    )
    
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Device: {device}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Data folder: {args.data_folder}")
    logger.info(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        with open(args.vocab_path, 'r', encoding='utf-8') as f:
            combined_vocab = json.load(f)
        
        char_to_idx = {char: idx for idx, char in enumerate(combined_vocab)}
        logger.info(f"Vocabulary loaded: {len(combined_vocab)} characters")
        vocab_output_path = os.path.join(args.output_dir, 'combined_char_vocab.json')
        with open(vocab_output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary saved to: {vocab_output_path}")
        
    except Exception as e:
        logger.error(f"Failed to load vocabulary: {e}")
        return 1

    try:
        dataset_loader = CustomDatasetLoader(
            data_folder=args.data_folder,
            image_extensions=args.image_extensions,
            encoding=args.text_encoding
        )
        
        dataset_dict = dataset_loader.create_huggingface_dataset(
            train_split=1.0 - args.val_split,
            val_split=args.val_split,
            seed=args.seed
        )
        
        logger.info(f"Dataset loaded successfully")
        logger.info(f"  - Training samples: {len(dataset_dict['train'])}")
        if 'validation' in dataset_dict:
            logger.info(f"  - Validation samples: {len(dataset_dict['validation'])}")
        
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {e}")
        return 1
    
    try:
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(
            args.model_path,
            combined_char_vocab=combined_vocab
        )
        
        model.to(device)
        processor = model.processor
        
        logger.info(f"Model loaded successfully")
        logger.info(f"  - Architecture: {model.config.vision_encoder_name}")
        logger.info(f"  - Fusion method: {model.config.feature_fusion_method}")
        logger.info(f"  - Dynamic fusion: {model.config.use_dynamic_fusion}")
        logger.info(f"  - Feature enhancer: {model.config.use_feature_enhancer}")
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return 1

    freeze_vision = args.freeze_vision_encoder and not args.unfreeze_vision_encoder
    
    freeze_model_layers(
        model=model,
        freeze_vision_encoder=freeze_vision,
        freeze_fusion=args.freeze_fusion,
        freeze_transformer=args.freeze_transformer,
        num_transformer_layers_to_tune=args.num_transformer_layers_to_tune,
        tune_classifiers_only=args.tune_classifiers_only
    )
    
    try:
        train_dataset = CtcOcrDataset(
            hf_dataset=dataset_dict['train'],
            processor=processor,
            char_to_idx_map=char_to_idx,
            unk_token='[UNK]',
            is_training=True
        )
        
        if 'validation' in dataset_dict:
            val_dataset = CtcOcrDataset(
                hf_dataset=dataset_dict['validation'],
                processor=processor,
                char_to_idx_map=char_to_idx,
                unk_token='[UNK]',
                is_training=False
            )
        else:
            logger.warning("No validation split provided, using 10% of training data")
            val_size = max(1, len(train_dataset) // 10)
            val_dataset = torch.utils.data.Subset(train_dataset, range(val_size))
        
        logger.info(f"Datasets prepared")
        logger.info(f"  - Training samples: {len(train_dataset)}")
        logger.info(f"  - Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        return 1
    
    
    try:
        optimizer = create_optimizer(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            discriminative_lr=False,
            encoder_lr_factor=0.1
        )
        
        total_steps = len(train_dataset) // args.batch_size * args.epochs
        
        lr_scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=args.warmup_steps,
            max_steps=total_steps,
            eta_min=args.learning_rate * 0.1
        )
        
        logger.info(f"  - Learning rate: {args.learning_rate}")
        logger.info(f"  - Weight decay: {args.weight_decay}")
        logger.info(f"  - Warmup steps: {args.warmup_steps}")
        logger.info(f"  - Total steps: {total_steps}")
        
    except Exception as e:
        logger.error(f"✗ Failed to setup optimizer: {e}")
        return 1

    try:
        train_ctc_model(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
            start_epoch=0,
            resumed_optimizer_steps=0,
            resumed_best_val_metric=float('inf'),
            best_metric_name='val_loss',
            project_name='',
            run_name=None,
            log_interval=args.log_interval,
            save_checkpoint_prefix='finetuned_checkpoint',
            use_amp=args.use_amp,
            scaler_state_to_load=None,
            grad_accumulation_steps=args.grad_accumulation,
            num_workers=args.num_workers,
            eval_steps=args.eval_steps,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_metric='val_loss',
            log_compatibility_matrix_interval=10000
        )
        
        logger.info(f"Fine-tuned model saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}", exc_info=True)
        return 1
    
    try:
        best_model_path = os.path.join(args.output_dir, 'best_model_hf')
        if os.path.exists(best_model_path):
            logger.info(f"Best model already saved at: {best_model_path}")
        else:
            final_model_path = os.path.join(args.output_dir, 'final_model')
            model.save_pretrained(final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")
        config_path = os.path.join(args.output_dir, 'finetuning_config.json')
        config = {
            'base_model': args.model_path,
            'data_folder': args.data_folder,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'freeze_vision_encoder': freeze_vision,
            'freeze_fusion': args.freeze_fusion,
            'freeze_transformer': args.freeze_transformer,
            'tune_classifiers_only': args.tune_classifiers_only,
            'num_training_samples': len(train_dataset),
            'num_validation_samples': len(val_dataset)
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())