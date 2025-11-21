import os
import sys
import argparse
import torch
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
from utils.ctc_utils import CTCDecoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('Vietnamese-HTR-Inference')


class VietnameseHTRInference:
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        device: Optional[str] = None,
        batch_size: int = 1
    ):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        self._load_vocabulary()
        self._load_model()
        
    def _load_vocabulary(self):
        try:
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                self.combined_vocab = json.load(f)
            self.idx_to_char = {i: char for i, char in enumerate(self.combined_vocab)}
            self.blank_idx = 0 
            self.decoder = CTCDecoder(
                idx_to_char_map=self.idx_to_char,
                blank_idx=self.blank_idx
            )
            
        except FileNotFoundError:
            logger.error(f"Vocabulary file not found: {self.vocab_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            raise
    
    def _load_model(self):
        try:
            self.model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(
                self.model_path,
                combined_char_vocab=self.combined_vocab
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            self.processor = self.model.processor
            has_vda = self.model.config.use_visual_diacritic_attention
            has_cdc = self.model.config.use_character_diacritic_compatibility
            has_fsa = self.model.config.use_few_shot_diacritic_adapter
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            
            return pixel_values.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def predict_single(
        self,
        image_path: str,
        return_confidence: bool = False
    ) -> Dict:
        try:
            pixel_values = self.preprocess_image(image_path)
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
            logits = outputs.get('logits')
            if logits is None:
                raise ValueError("Model output does not contain 'logits'")
            decoded_texts = self.decoder(logits)
            predicted_text = decoded_texts[0] if decoded_texts else ""
            
            result = {
                'image_path': image_path,
                'predicted_text': predicted_text,
                'success': True
            }
            
            if return_confidence:
                probs = torch.softmax(logits, dim=-1)
                max_probs = torch.max(probs, dim=-1)[0]
                avg_confidence = max_probs.mean().item()
                result['confidence'] = avg_confidence
            
            return result
            
        except Exception as e:
            logger.error(f"Error during inference for {image_path}: {e}")
            return {
                'image_path': image_path,
                'predicted_text': '',
                'success': False,
                'error': str(e)
            }
    
    def predict_folder(
        self,
        folder_path: str,
        output_csv: Optional[str] = None,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
        return_confidence: bool = False,
        recursive: bool = False
    ) -> List[Dict]:
        try:
            folder = Path(folder_path)
            
            if recursive:
                image_files = []
                for ext in image_extensions:
                    image_files.extend(folder.rglob(f"*{ext}"))
                    image_files.extend(folder.rglob(f"*{ext.upper()}"))
            else:
                image_files = []
                for ext in image_extensions:
                    image_files.extend(folder.glob(f"*{ext}"))
                    image_files.extend(folder.glob(f"*{ext.upper()}"))
            
            image_files = sorted(list(set(image_files)))
            
            if not image_files:
                logger.warning(f"No images found in {folder_path}")
                return []
            
            logger.info(f"Found {len(image_files)} images to process")
            results = []
            for image_path in tqdm(image_files, desc="Processing images"):
                result = self.predict_single(
                    str(image_path),
                    return_confidence=return_confidence
                )
                results.append(result)
            if output_csv:
                self._save_results_to_csv(results, output_csv)
            successful = sum(1 for r in results if r['success'])
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing folder {folder_path}: {e}")
            raise
    
    def _save_results_to_csv(self, results: List[Dict], output_path: str):
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Inference for Vietnamese Handwritten Text Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            # Process a single image
            python inference.py --model_path ./ckpt/best_model_hf --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json --image /Users/mac/Desktop/test.png
            
            # Process a folder of images
            python inference.py --model_path ./ckpt/best_model_hf --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json --folder ./test_images
            
            # Process folder with recursive search and save to CSV
            python inference.py --model_path ./ckpt/best_model_hf --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \\
                --folder ./test_images --recursive --output results.csv --confidence
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint directory'
    )
    
    parser.add_argument(
        '--vocab_path',
        type=str,
        required=True,
        help='Path to combined character vocabulary JSON file'
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image',
        type=str,
        help='Path to a single image file for inference'
    )
    
    input_group.add_argument(
        '--folder',
        type=str,
        help='Path to folder containing multiple images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path for batch processing (only used with --folder)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run inference on (default: auto-detect)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for processing (default: 1)'
    )
    
    parser.add_argument(
        '--confidence',
        action='store_true',
        help='Include confidence scores in output'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search for images recursively in subdirectories (only with --folder)'
    )
    
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
        help='Image file extensions to process (default: .jpg .jpeg .png .bmp .tiff)'
    )
    
    args = parser.parse_args()
    
    try:
        inference = VietnameseHTRInference(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            device=args.device,
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        return 1
    try:
        if args.image:
            logger.info(f"Processing single image: {args.image}")
            result = inference.predict_single(
                args.image,
                return_confidence=args.confidence
            )
            
            if result['success']:
                print("PREDICTION RESULT")
                print(f"Image: {result['image_path']}")
                print(f"Predicted Text: {result['predicted_text']}")
            else:
                print(f"\nPrediction failed: {result.get('error', 'Unknown error')}")
                return 1
                
        elif args.folder:
            output_csv = args.output
            if output_csv is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_csv = f"inference_results_{timestamp}.csv"
            
            results = inference.predict_folder(
                folder_path=args.folder,
                output_csv=output_csv,
                image_extensions=args.extensions,
                return_confidence=args.confidence,
                recursive=args.recursive
            )
            
        return 0
        
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())