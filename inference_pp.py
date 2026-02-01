# inference_conservative.py
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
from utils.vietnamese import (
    VietnameseConservativePostProcessor,
    create_conservative_postprocessor
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('Vietnamese-HTR-Inference-PP')


class VietnameseHTRInferencePP:
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        device: Optional[str] = None,
        # Post-processing options
        enable_postprocessing: bool = True,
        use_lm_for_ambiguous: bool = True,
        lm_type: str = 'ngram'  # 'ngram' or 'phobert'
    ):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.enable_postprocessing = enable_postprocessing
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        self._load_vocabulary()
        self._load_model()

        if enable_postprocessing:
            self.post_processor = create_conservative_postprocessor(
                use_lm=use_lm_for_ambiguous,
                lm_type=lm_type
            )
        else:
            self.post_processor = None
    
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
            
            logger.info(f"Vocabulary loaded: {len(self.combined_vocab)} characters")
            
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
            
            logger.info("Model loaded successfully")
            
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
        return_raw: bool = False,
        return_metadata: bool = False
    ) -> Dict:
        try:
            pixel_values = self.preprocess_image(image_path)
            
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
            
            logits = outputs.get('logits')
            if logits is None:
                raise ValueError("Model output does not contain 'logits'")
            
            decoded_texts = self.decoder(logits)
            raw_text = decoded_texts[0] if decoded_texts else ""
            
            result = {
                'image_path': image_path,
                'success': True
            }
            
            if self.enable_postprocessing and self.post_processor:
                corrected_text, metadata = self.post_processor.process(raw_text)
                result['predicted_text'] = corrected_text
                
                if return_raw:
                    result['raw_text'] = raw_text
                
                if return_metadata:
                    result['was_modified'] = metadata['was_modified']
                    result['corrections'] = metadata['corrections']
                
                if metadata['was_modified']:
                    logger.info(f"Corrections applied: {metadata['corrections']}")
            else:
                result['predicted_text'] = raw_text
                if return_raw:
                    result['raw_text'] = raw_text
            
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
        return_raw: bool = False,
        recursive: bool = False
    ) -> List[Dict]:
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
        
        logger.info(f"Processing {len(image_files)} images")
        
        results = []
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.predict_single(
                str(image_path),
                return_raw=return_raw,
                return_metadata=True
            )
            results.append(result)
        
        if output_csv:
            self._save_results(results, output_csv)
        successful = sum(1 for r in results if r['success'])
        corrected = sum(1 for r in results if r.get('was_modified', False))
        logger.info(f"Processed {len(results)} images: {successful} successful, {corrected} corrected")
        
        return results
    
    def _save_results(self, results: List[Dict], output_path: str):
        flat_results = []
        for r in results:
            flat_r = {
                'image_path': r['image_path'],
                'predicted_text': r['predicted_text'],
                'success': r['success']
            }
            
            if 'raw_text' in r:
                flat_r['raw_text'] = r['raw_text']
            
            if 'was_modified' in r:
                flat_r['was_modified'] = r['was_modified']
            
            if 'corrections' in r and r['corrections']:
                flat_r['corrections'] = json.dumps(r['corrections'], ensure_ascii=False)
            
            flat_results.append(flat_r)
        
        df = pd.DataFrame(flat_results)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Vietnamese HTR Inference with Conservative Post-Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inference with conservative post-processing
    python inference_conservative.py \\
        --model_path ./ckpt/best_model_hf \\
        --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \\
        --image ./test.png

    # Show raw OCR vs corrected
    python inference_conservative.py \\
        --model_path ./ckpt/best_model_hf \\
        --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \\
        --image ./test.png \\
        --show-raw

    # Disable post-processing (raw OCR only)
    python inference_conservative.py \\
        --model_path ./ckpt/best_model_hf \\
        --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \\
        --image ./test.png \\
        --no-postprocessing

    # Process folder
    python inference_conservative.py \\
        --model_path ./ckpt/best_model_hf \\
        --vocab_path ./ckpt/best_model_hf/vocabularies/combined_char_vocab.json \\
        --folder ./images \\
        --output results.csv
        """
    )

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Path to vocabulary JSON')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                             help='Path to single image')
    input_group.add_argument('--folder', type=str,
                             help='Path to folder containing images')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path')

    parser.add_argument('--no-postprocessing', action='store_true',
                        help='Disable post-processing (raw OCR only)')
    parser.add_argument('--no-lm', action='store_true',
                        help='Disable language model for ambiguous cases')
    parser.add_argument('--lm-type', type=str, default='ngram',
                        choices=['ngram', 'phobert'],
                        help='Language model type (ngram is faster, phobert is better)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'])
    parser.add_argument('--show-raw', action='store_true',
                        help='Show raw OCR output before correction')
    parser.add_argument('--recursive', action='store_true',
                        help='Search recursively in subdirectories')
    parser.add_argument('--extensions', type=str, nargs='+',
                        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                        help='Image file extensions')
    
    args = parser.parse_args()
    
    try:
        inference = VietnameseHTRInferencePP(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            device=args.device,
            enable_postprocessing=not args.no_postprocessing,
            use_lm_for_ambiguous=not args.no_lm,
            lm_type=args.lm_type
        )
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return 1
    
    try:
        if args.image:
            result = inference.predict_single(
                args.image,
                return_raw=args.show_raw,
                return_metadata=True
            )
            
            if result['success']:
                print("\n" + "=" * 60)
                print("PREDICTION RESULT")
                print("=" * 60)
                print(f"Image: {result['image_path']}")
                
                if args.show_raw and 'raw_text' in result:
                    print(f"Raw OCR:    {result['raw_text']}")
                    if result.get('was_modified'):
                        print(f"Corrected:  {result['predicted_text']}")
                        print(f"Changes:    {result.get('corrections', [])}")
                    else:
                        print(f"(No corrections needed)")
                else:
                    print(f"Predicted:  {result['predicted_text']}")
                
                print("=" * 60)
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
                return_raw=args.show_raw,
                recursive=args.recursive
            )
            
            print(f"\nResults saved to: {output_csv}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
