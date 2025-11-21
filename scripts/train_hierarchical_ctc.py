# scripts/train_hierarchical_ctc.py
import os
import sys
import argparse
import torch
from datasets import load_dataset, DatasetDict
import logging
import math
import wandb  # Optional
import json
import numpy as np
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from model.hierarchical_ctc_model import (
    HierarchicalCtcMultiScaleOcrModel,
    HierarchicalCtcOcrConfig,
)
from data.ctc_ocr_dataset import CtcOcrDataset
from data.ctc_collation import ctc_collate_fn
from training.ctc_trainer import train_ctc_model
from utils.schedulers import CosineWarmupScheduler
# from utils.schedulers import CosineWarmupWithPlateauScheduler
from utils.optimizers import create_optimizer
from utils.ctc_utils import build_ctc_vocab, build_combined_vietnamese_charset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('main_training_hier_multiscale.log'),
    ],
)
logger = logging.getLogger('TrainHierarchicalMultiScaleScript')

BASE_CHAR_VOCAB_HIER = [
    '<blank>', '<unk>', 'a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'f', 'F',
    'j', 'J', 'w', 'W', 'z', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ',
    ',', '.', '?', '!', ':', ';', '-', '_', '(', ')', '[', ']', '{', '}', "'", '"', '/',
    '\\', '@', '#', '$', '%', '^', '&', '*', '+', '=', '<', '>', '|',
]
DIACRITIC_VOCAB_HIER = [
    'no_diacritic', 'acute', 'grave', 'hook', 'tilde', 'dot', 'circumflex',
    'breve', 'horn', 'stroke', 'circumflex_grave', 'circumflex_acute', 'circumflex_tilde',
    'circumflex_hook', 'circumflex_dot', 'breve_grave', 'breve_acute', 'breve_tilde',
    'breve_hook', 'breve_dot', 'horn_grave', 'horn_acute', 'horn_tilde', 'horn_hook', 'horn_dot',
]

def freeze_model_layers(model, num_transformer_layers_to_tune=3, tune_diacritic_enhancements=True):
    logger.info("Freezing: Vision Encoder")
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    if model.dynamic_fusion:
        logger.info("Freezing: Dynamic Fusion")
        for param in model.dynamic_fusion.parameters():
            param.requires_grad = False
    else: 
        if model.fusion_projection:
            logger.info("Freezing: Fusion Projection")
            for param in model.fusion_projection.parameters():
                param.requires_grad = False
        if model.pre_fusion_projections:
            logger.info("Freezing: Pre-Fusion Projections")
            for param in model.pre_fusion_projections.parameters():
                param.requires_grad = False
        if model.fusion_bilinear:
            logger.info("Freezing: Fusion Bilinear")
            for param in model.fusion_bilinear.parameters():
                param.requires_grad = False
        if model.post_static_fusion_norm: 
            logger.info("Freezing: Post Static Fusion Norm")
            for param in model.post_static_fusion_norm.parameters():
                param.requires_grad = False

    if model.feature_enhancer:
        logger.info("Freezing: Feature Enhancer")
        for param in model.feature_enhancer.parameters():
            param.requires_grad = False

    if model.pos_encoder:
        logger.info("Freezing: Positional Encoder")
        for param in model.pos_encoder.parameters():
            param.requires_grad = False

    total_transformer_layers = len(model.transformer_encoder.layers)
    num_layers_to_freeze = total_transformer_layers - num_transformer_layers_to_tune

    if num_layers_to_freeze > 0:
        logger.info(f"Freezing: First {num_layers_to_freeze} Transformer Encoder Layers")
        for i in range(num_layers_to_freeze):
            for param in model.transformer_encoder.layers[i].parameters():
                param.requires_grad = False
    
    logger.info(f"Tuning: Last {num_transformer_layers_to_tune} Transformer Encoder Layers")
    for i in range(num_layers_to_freeze, total_transformer_layers):
        for param in model.transformer_encoder.layers[i].parameters():
            param.requires_grad = True

    logger.info("Ensuring tuneable: Transformer final norm, Shared Layer, Classifiers")
    if model.transformer_encoder.norm:
        for param in model.transformer_encoder.norm.parameters():
            param.requires_grad = True
    for param in model.shared_layer.parameters(): # nn.Identity has no params
        param.requires_grad = True
    for param in model.base_classifier.parameters():
        param.requires_grad = True
    for param in model.diacritic_classifier.parameters():
        param.requires_grad = True
    if model.config.conditioning_method == 'concat_proj' and model.diacritic_condition_proj:
        for param in model.diacritic_condition_proj.parameters():
            param.requires_grad = True
    elif model.config.conditioning_method == 'gate' and model.diacritic_gate:
        for param in model.diacritic_gate.parameters():
            param.requires_grad = True
    for param in model.final_classifier.parameters():
        param.requires_grad = True

    if model.visual_diacritic_attention:
        for param in model.visual_diacritic_attention.parameters():
            param.requires_grad = tune_diacritic_enhancements
            
    if model.character_diacritic_compatibility:
        model.character_diacritic_compatibility.compatibility_matrix.requires_grad = True
        if hasattr(model.character_diacritic_compatibility, 'compatibility_predictor'):
            for param in model.character_diacritic_compatibility.compatibility_predictor.parameters():
                param.requires_grad = tune_diacritic_enhancements


    if model.few_shot_diacritic_adapter:
        for param in model.few_shot_diacritic_adapter.parameters():
            param.requires_grad = tune_diacritic_enhancements
            
    logger.info("--- Layer Freezing Complete ---")



def test_corrected_compatibility_matrix(model):
    if not hasattr(model, 'character_diacritic_compatibility') or model.character_diacritic_compatibility is None:
        logger.warning("No compatibility matrix found")
        return
    test_cases = [
        {'char': 'a', 'type': 'vowel', 'should_allow_acute': True},
        {'char': 'e', 'type': 'vowel', 'should_allow_acute': True},
        {'char': 'b', 'type': 'consonant', 'should_allow_acute': False},
        {'char': 'g', 'type': 'consonant', 'should_allow_acute': False},
        {'char': 'm', 'type': 'consonant', 'should_allow_acute': False},
    ]
    
    dummy_features = torch.randn(1, 1, model.config.shared_hidden_size).to(model.device)
    
    for test_case in test_cases:
        char = test_case['char']
        if char not in model.base_char_vocab:
            continue
            
        char_idx = model.base_char_vocab.index(char)
        base_logits = torch.zeros(1, 1, len(model.base_char_vocab)).to(model.device)
        base_logits[0, 0, char_idx] = 10.0
        compat_bias, _ = model.character_diacritic_compatibility(base_logits, dummy_features)
        if 'acute' in model.diacritic_vocab:
            acute_idx = model.diacritic_vocab.index('acute')
            acute_bias = compat_bias[0, 0, acute_idx].item()
            
            expected_positive = test_case['should_allow_acute']
            actual_positive = acute_bias > 0
            
            status = "ok" if (expected_positive == actual_positive) else "no"


def main():
    parser = argparse.ArgumentParser(description='Train Hierarchical Multi-Scale CTC Vietnamese OCR model')

    parser.add_argument('--dataset_name', type=str, default='cinnamon_ai', help='HF dataset (image, label, base_character, diacritic_type)')
    parser.add_argument('--vision_encoder', type=str, default='microsoft/trocr-base-handwritten', help='Vision encoder')
    parser.add_argument('--output_dir', type=str, default='outputs/hier_ctc_multiscale_model', help='Output directory')
    parser.add_argument('--combined_char_vocab_json', type=str, default=None, help='Path to JSON list of FINAL combined characters. If None, uses default generator.')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--load_weights_from', type=str, default=None, help="Load pre-trained vision encoder, other layers random.")

    parser.add_argument('--fusion_layers', type=str, default="-1,-4,-7", help='Comma-separated vision encoder layer indices to fuse ("-1,-4").')
    parser.add_argument('--fusion_method', type=str, default="concat_proj", choices=['concat_proj', 'add', 'bilinear', 'none'], help='Method to fuse features.')
    
    parser.add_argument('--use_dynamic_fusion', action='store_true', help='Use Dynamic Multi-Scale Fusion instead of static fusion method.')
    parser.add_argument('--use_feature_enhancer', action='store_true', help='Use Local Feature Enhancer for diacritical marks.')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help="Number of Transformer encoder layers after vision fusion.")
    parser.add_argument('--transformer_d_model', type=int, default=512, help="Dimension for Transformer layers (d_model). Should match fusion output if not projecting.")
    parser.add_argument('--transformer_nhead', type=int, default=8, help="Number of heads in Transformer MHA.")
    parser.add_argument('--transformer_dim_feedforward', type=int, default=2048, help="Dimension of Transformer FFN.")
    parser.add_argument('--transformer_dropout', type=float, default=0.1, help="Dropout for Transformer layers.")
    parser.add_argument('--pos_encoding_type', type=str, default="sinusoidal_1d", choices=["sinusoidal_1d", "learned_1d", "none"], help="Type of positional encoding.")
    parser.add_argument('--max_pos_enc_len', type=int, default=1024, help="Max sequence length for positional encoding (e.g., num_patches).")
    parser.add_argument('--shared_hidden_size',type=int, default=512, help='Hidden size after Transformer/RNN before branching.')
    parser.add_argument('--num_shared_layers', type=int, default=1)
    parser.add_argument('--conditioning_method', type=str, default='concat_proj', choices=['concat_proj', 'gate', 'none'])
    parser.add_argument('--classifier_dropout', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--early_stopping_metric', type=str, default='val_cer')

    parser.add_argument('--use_visual_diacritic_attention', action='store_true')
    parser.add_argument('--use_character_diacritic_compatibility', action='store_true')
    parser.add_argument('--use_few_shot_diacritic_adapter', action='store_true')
    parser.add_argument('--num_few_shot_prototypes', type=int, default=5)

    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=5000)
    parser.add_argument('--eval_steps', type=int, default=None)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16) 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--discriminative_lr', action='store_true')
    parser.add_argument('--encoder_lr_factor', type=float, default=0.1)
    parser.add_argument('--reset_scheduler_on_resume', action='store_true')
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--skip_final_eval', action='store_true')
    parser.add_argument('--test_dataset_name', type=str, default='vklinhhh/test_vietnamese_cwl')
    parser.add_argument('--test_dataset_split', type=str, default='train')
    # parser.add_argument('--ignore_scaler_state_on_resume', action='store_true')
    parser.add_argument('--log_compatibility_interval', type=int, default=5000, 
                    help='How often to log compatibility matrix (steps)')
    parser.add_argument('--freeze_except_last_n_transformer_layers', type=int, default=0, 
                        help='Number of final Transformer encoder layers to TUNE. 0 means tune all. Vision encoder and earlier custom layers will be frozen if > 0.')
    parser.add_argument('--freeze_diacritic_enhancements', action='store_true',
                        help='If freezing, also freeze diacritic enhancement modules (VDA, FSA). Compatibility matrix is usually still tuned.')
    parser.add_argument('--hierarchical_mode', type=str, default='enhanced_single',
                        choices=['parallel', 'sequential', 'multitask', 'enhanced_single'],
                        help='Hierarchical processing mode: '
                            'parallel (original), sequential (true hierarchy), '
                            'multitask (weighted multi-task), enhanced_single (apply enhancements to final only)')
    
    parser.add_argument('--multitask_final_weight', type=float, default=1.0,
                        help='Weight for final classifier loss in multitask mode')
    parser.add_argument('--multitask_base_weight', type=float, default=0.1,
                        help='Weight for base classifier loss in multitask mode')
    parser.add_argument('--multitask_diacritic_weight', type=float, default=0.1,
                        help='Weight for diacritic classifier loss in multitask mode')

    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit training set to this many samples (for fast experiments)')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Limit validation set to this many samples (for fast experiments)')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); logger.info(f"Device: {device}")
    run_name_to_use = args.wandb_run_name or os.path.basename(args.output_dir) or "hier_ctc_multiscale_run"
    try: fusion_layer_indices = [int(x.strip()) for x in args.fusion_layers.split(',')]
    except: logger.error(f"Invalid --fusion_layers. Using default [-1,-4]."); fusion_layer_indices = [-1,-4,-7]
    
    if args.combined_char_vocab_json and os.path.exists(args.combined_char_vocab_json):
        with open(args.combined_char_vocab_json, 'r', encoding='utf-8') as f:
            combined_char_list = json.load(f)
        if '<blank>' not in combined_char_list or combined_char_list[0] != '<blank>':
            if '<blank>' in combined_char_list:
                combined_char_list.remove('<blank>')
            combined_char_list.insert(0, '<blank>') 
        if '[UNK]' not in combined_char_list:
            combined_char_list.append('[UNK]')

        combined_vocab, combined_char_to_idx, _ = build_ctc_vocab(
            combined_char_list,
            add_blank=False,
            add_unk=False
        )
    else:
        generated_combined_chars = build_combined_vietnamese_charset()
        combined_vocab, combined_char_to_idx, _ = build_ctc_vocab(
            generated_combined_chars,
            add_blank=True,
            add_unk=True,
            unk_token='[UNK]'
        )

        os.makedirs(args.output_dir, exist_ok=True)
        vocab_save_path = os.path.join(args.output_dir, 'combined_char_vocab.json')
        with open(vocab_save_path, 'w', encoding='utf-8') as f:
            json.dump(combined_vocab, f, ensure_ascii=False, indent=4)

    try:
        logger.info(f'Loading dataset: {args.dataset_name}')
        hf_dataset = load_dataset(args.dataset_name)
        if 'validation' not in hf_dataset or 'train' not in hf_dataset:
            logger.warning(f'Splitting train set for validation.')
            if args.val_split <= 0: raise ValueError('--val_split required')
            split_dataset = hf_dataset['train'].train_test_split(test_size=args.val_split, seed=args.seed)
            hf_dataset = DatasetDict({'train': split_dataset['train'], 'validation': split_dataset['test']})
        train_hf_split = hf_dataset['train']
        val_hf_split = hf_dataset['validation']
        
        if args.max_train_samples is not None and len(train_hf_split) > args.max_train_samples:
            logger.info(f'Limiting train set from {len(train_hf_split)} to {args.max_train_samples} samples')
            train_hf_split = train_hf_split.select(range(args.max_train_samples))
        
        if args.max_val_samples is not None and len(val_hf_split) > args.max_val_samples:
            logger.info(f'Limiting validation set from {len(val_hf_split)} to {args.max_val_samples} samples')
            val_hf_split = val_hf_split.select(range(args.max_val_samples))
        
        logger.info(f'Train size: {len(train_hf_split)}, Val size: {len(val_hf_split)}')
    except Exception as dataset_load_e: logger.error(f'FATAL: Dataset load failed: {dataset_load_e}', exc_info=True); return 1

    multitask_loss_weights = [
        args.multitask_final_weight,
        args.multitask_base_weight,
        args.multitask_diacritic_weight
    ]
    try:
        logger.info("Initializing HierarchicalCtcMultiScaleOcrModel configuration...")
        model_config = HierarchicalCtcOcrConfig(
            vision_encoder_name=args.vision_encoder,
            base_char_vocab=BASE_CHAR_VOCAB_HIER,
            diacritic_vocab=DIACRITIC_VOCAB_HIER,
            combined_char_vocab=combined_vocab,
            vision_encoder_layer_indices=fusion_layer_indices,
            feature_fusion_method=args.fusion_method,
            hierarchical_mode=args.hierarchical_mode,
            multitask_loss_weights=multitask_loss_weights,
            use_dynamic_fusion=args.use_dynamic_fusion,
            use_feature_enhancer=args.use_feature_enhancer,
            num_transformer_encoder_layers=args.num_transformer_layers,
            transformer_d_model=args.transformer_d_model,
            transformer_nhead=args.transformer_nhead,
            transformer_dim_feedforward=args.transformer_dim_feedforward,
            transformer_dropout=args.transformer_dropout,
            positional_encoding_type=args.pos_encoding_type,
            max_seq_len_for_pos_enc=args.max_pos_enc_len,
            shared_hidden_size=args.shared_hidden_size,
            num_shared_layers=args.num_shared_layers,
            conditioning_method=args.conditioning_method,
            classifier_dropout=args.classifier_dropout,
            blank_idx=combined_char_to_idx['<blank>'],
            use_visual_diacritic_attention=args.use_visual_diacritic_attention,
            use_character_diacritic_compatibility=args.use_character_diacritic_compatibility,
            use_few_shot_diacritic_adapter=args.use_few_shot_diacritic_adapter,
            num_few_shot_prototypes=args.num_few_shot_prototypes
        )
        logger.info(f"Using hierarchical mode: {args.hierarchical_mode}")
        if args.hierarchical_mode == 'multitask':
            logger.info(f"Multitask loss weights: Final={args.multitask_final_weight}, "
                    f"Base={args.multitask_base_weight}, Diacritic={args.multitask_diacritic_weight}")
        
        model_load_path = args.load_weights_from if args.load_weights_from else args.vision_encoder
        init_kwargs = model_config.to_dict()
        model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(model_load_path, **init_kwargs)
        processor = model.processor

        if args.use_dynamic_fusion:
            logger.info("Using Dynamic Multi-Scale Fusion")
        else:
            logger.info(f"Using standard fusion method: {args.fusion_method}")
        if args.use_feature_enhancer:
            logger.info("Using Local Feature Enhancer for diacritical marks")

    except Exception as model_init_e:
        logger.error(f"FATAL: Model init failed: {model_init_e}", exc_info=True)
        return 1
    if args.freeze_except_last_n_transformer_layers > 0 :
        freeze_model_layers(
            model,
            num_transformer_layers_to_tune=args.freeze_except_last_n_transformer_layers,
            tune_diacritic_enhancements=not args.freeze_diacritic_enhancements 
        )
    try:
        train_dataset = CtcOcrDataset(train_hf_split, processor, combined_char_to_idx, unk_token='[UNK]', is_training=not args.no_augment)
        val_dataset = CtcOcrDataset(val_hf_split, processor, combined_char_to_idx, unk_token='[UNK]', is_training=False)
    except Exception as dataset_wrap_e: logger.error(f'FATAL: Dataset wrap failed: {dataset_wrap_e}', exc_info=True); return 1
    
    try:
        model.to(device); logger.info(f"Model on device: {device}")
        optimizer = create_optimizer(model, args.learning_rate, args.weight_decay, args.discriminative_lr, args.encoder_lr_factor)
        num_training_batches = math.ceil(len(train_dataset) / args.batch_size);
        total_steps_for_epoch = math.ceil(num_training_batches / args.grad_accumulation)
        total_steps = total_steps_for_epoch * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        logger.info(f"Scheduler Setup: Steps per epoch={total_steps_for_epoch}, Total Steps={total_steps}, Warmup={warmup_steps}")
        lr_scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
    except Exception as opt_sched_e: logger.error(f"FATAL: Opt/Sched failed: {opt_sched_e}", exc_info=True); return 1

    compat_matrix_in_optimizer = False
    if hasattr(model, 'character_diacritic_compatibility') and model.character_diacritic_compatibility is not None:
        target_param = model.character_diacritic_compatibility.compatibility_matrix
        for i, param_group in enumerate(optimizer.param_groups):
            if any(p is target_param for p in param_group['params']):
                logger.info(f"Compatibility matrix found in optimizer param_group {i} with initial lr={param_group['lr']}")
                compat_matrix_in_optimizer = True
                
                original_lr = param_group['lr']
                lr_multiplier = 25.0 
                param_group['lr'] = original_lr * lr_multiplier
                logger.info(f"Increased compatibility matrix learning rate in param_group {i} to {param_group['lr']} (original: {original_lr}, multiplier: {lr_multiplier})")

                original_wd = param_group.get('weight_decay', args.weight_decay)
                if original_wd > 0: 
                    param_group['weight_decay'] = 0.0 
                    logger.info(f"Set weight_decay for compatibility matrix param_group {i} to 0.0 (original: {original_wd})")
                break

    if hasattr(model, 'character_diacritic_compatibility') and model.character_diacritic_compatibility is not None and not compat_matrix_in_optimizer:
        logger.warning("Compatibility matrix parameter NOT found in any optimizer parameter group! It will not be updated.")

    start_epoch = 0; resumed_optimizer_steps = 0; higher_is_better = args.early_stopping_metric not in ['val_loss', 'val_cer', 'val_wer']; resumed_best_val_metric = -float('inf') if higher_is_better else float('inf'); scaler_state_to_load = None; checkpoint_to_load = args.resume_from_checkpoint
    if checkpoint_to_load is None and not args.load_weights_from:
        latest_checkpoint_path = os.path.join(args.output_dir, "checkpoints", "checkpoint_latest.pt")
        checkpoint_to_load = latest_checkpoint_path if os.path.isfile(latest_checkpoint_path) else None
        if checkpoint_to_load: logger.info(f"Found latest ckpt: {checkpoint_to_load}")
        else: logger.info("No checkpoint found to resume from.")

    if checkpoint_to_load and os.path.isfile(checkpoint_to_load):
        try:
            checkpoint = torch.load(checkpoint_to_load, map_location=device)

            if not args.load_weights_from and 'model_state_dict' in checkpoint:
                load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            load_optimizer_etc = False
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict']); logger.info("Optimizer loaded.")
                load_optimizer_etc = True
            else: logger.warning("Optimizer state missing in checkpoint.")

            if load_optimizer_etc and lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                if args.reset_scheduler_on_resume:
                    logger.warning("Resetting LR scheduler due to --reset_scheduler_on_resume flag.")
                    start_epoch_from_ckpt = checkpoint.get('epoch', -1) + 1
                    remaining_epochs = args.epochs - start_epoch_from_ckpt
                    if remaining_epochs > 0:
                        new_total_steps_for_scheduler = total_steps_for_epoch * remaining_epochs
                        new_warmup_steps = int(new_total_steps_for_scheduler * args.warmup_ratio)
                        lr_scheduler = CosineWarmupScheduler(optimizer, new_warmup_steps, new_total_steps_for_scheduler, last_epoch=-1)
                        logger.info(f"Reinitialized LR scheduler for {remaining_epochs} remaining epochs (Total Steps: {new_total_steps_for_scheduler}, Warmup: {new_warmup_steps}).")
                    else:
                        logger.warning("Cannot reset scheduler, no remaining epochs.")
                else:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict']); logger.info("Scheduler loaded.")
            elif load_optimizer_etc : logger.warning("LR Scheduler state not found or scheduler not used.")


            if load_optimizer_etc and args.use_amp and 'scaler_state_dict' in checkpoint:
                scaler_state_to_load = checkpoint['scaler_state_dict']; logger.info("Found AMP state for scaler.")
            elif load_optimizer_etc and args.use_amp: logger.warning("Scaler state not found.")

            start_epoch = checkpoint.get('epoch', -1) + 1
            resumed_optimizer_steps = checkpoint.get('step', 0)
            resumed_best_val_metric = checkpoint.get('best_val_metric', resumed_best_val_metric)
            logger.info(f"-> Resuming Epoch: {start_epoch} (Step: {resumed_optimizer_steps}), Best Metric: {resumed_best_val_metric:.4f}")

        except Exception as e:
            logger.error(f"ERROR loading checkpoint state: {e}", exc_info=True)
            start_epoch = 0; resumed_optimizer_steps = 0; resumed_best_val_metric = -float('inf') if higher_is_better else float('inf')
    else:
        if args.resume_from_checkpoint: logger.warning(f"Specified ckpt not found: {args.resume_from_checkpoint}.")
        logger.info("Starting fresh or from base weights.")

    wandb_run = None
    if args.wandb_project:
        try:
            wandb_config = vars(args).copy()
            wandb_config.update({
                "model_base_char_vocab_size": model.config.base_char_vocab_size,
                "model_diacritic_vocab_size": model.config.diacritic_vocab_size,
                "model_combined_char_vocab_size": model.config.combined_char_vocab_size,
                "model_type": model.config.model_type,
            })
            wandb_run = wandb.init(project=args.wandb_project, name=run_name_to_use, config=wandb_config, resume="allow")
            logger.info(f"Initialized WandB run: {wandb_run.name} (ID: {wandb_run.id})")
        except Exception as e: logger.error(f"Wandb init failed: {e}")

    trained_model = train_ctc_model(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir,
        start_epoch=start_epoch,
        resumed_optimizer_steps=resumed_optimizer_steps,
        resumed_best_val_metric=resumed_best_val_metric,
        best_metric_name=args.early_stopping_metric,
        project_name="",
        run_name=None,
        log_interval=args.log_interval,
        save_checkpoint_prefix='checkpoint',
        use_amp=args.use_amp,
        scaler_state_to_load=scaler_state_to_load, 
        grad_accumulation_steps=args.grad_accumulation,
        num_workers=args.num_workers,
        eval_steps=args.eval_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        log_compatibility_matrix_interval=args.log_compatibility_interval,
    )

    logger.info(f"Final model saved in {args.output_dir}")

    if not args.skip_final_eval and args.test_dataset_name:
        best_model_path = os.path.join(args.output_dir, "best_model_hf")
        if not os.path.isdir(best_model_path): best_model_path = args.output_dir
        eval_output_dir = os.path.join(args.output_dir, "final_evaluation_report")
        try:
            from scripts.evaluate_hierarchical_ctc import run_hierarchical_evaluation
            run_hierarchical_evaluation(
                model_path=best_model_path,
                dataset_name=args.test_dataset_name,
                output_dir=eval_output_dir,
                combined_char_vocab_path=os.path.join(args.output_dir, "combined_char_vocab.json"),
                dataset_split=args.test_dataset_split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device
            )
        except ImportError: logger.error("Could not import evaluate_hierarchical_ctc script.")
        except Exception as eval_e: logger.error(f"Final evaluation failed: {eval_e}", exc_info=True)

    return 0

if __name__ == "__main__":
    status = main()
    sys.exit(status)