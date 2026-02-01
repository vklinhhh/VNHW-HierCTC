# utils/comparison_engine.py
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm

from model.ocr_wrappers import (
    BaseOCRModel,
    HierCTCModel,
    VietOCRModel,
    TesseractOCRModel,
    EasyOCRModel,
)
from utils.ocr_metrics import calculate_cer, calculate_wer, calculate_accuracy

logger = logging.getLogger(__name__)


class OCRComparisonEngine:    
    def __init__(
        self,
        our_model_path: Optional[str] = None,
        our_vocab_path: Optional[str] = None,
        device: Optional[str] = None,
        models_to_use: Optional[List[str]] = None
    ):
        self.models: Dict[str, BaseOCRModel] = {}
        
        if models_to_use is None:
            models_to_use = ['our', 'vietocr', 'tesseract', 'easyocr']
        
        if 'our' in models_to_use and our_model_path and our_vocab_path:
            self.models['our'] = HierCTCModel(our_model_path, our_vocab_path, device)
        
        if 'vietocr' in models_to_use:
            self.models['vietocr'] = VietOCRModel('vgg_seq2seq')
        
        if 'vietocr_transformer' in models_to_use:
            self.models['vietocr_transformer'] = VietOCRModel('vgg_transformer')
        
        if 'tesseract' in models_to_use:
            self.models['tesseract'] = TesseractOCRModel('vie')
        
        if 'easyocr' in models_to_use:
            self.models['easyocr'] = EasyOCRModel()
        
        logger.info(f"Initialized comparison engine with models: {list(self.models.keys())}")
    
    def load_all_models(self) -> Dict[str, bool]:
        results = {}
        for name, model in self.models.items():
            results[name] = model.load()
        return results
    
    def compare_single_image(
        self,
        image_path: str,
        ground_truth: Optional[str] = None
    ) -> Dict:
        results = {
            'image_path': image_path,
            'ground_truth': ground_truth,
            'predictions': {},
            'timing': {},
            'metrics': {}
        }
        
        for name, model in self.models.items():
            if not model.is_loaded:
                continue
            
            start_time = time.time()
            prediction = model.predict(image_path)
            inference_time = time.time() - start_time
            
            results['predictions'][name] = prediction
            results['timing'][name] = inference_time
            
            if ground_truth:
                cer = calculate_cer(prediction, ground_truth)
                wer = calculate_wer(prediction, ground_truth)
                accuracy = calculate_accuracy(prediction, ground_truth)
                
                results['metrics'][name] = {
                    'cer': cer,
                    'wer': wer,
                    'accuracy': accuracy
                }
        
        return results
    
    def compare_folder(
        self,
        folder_path: str,
        ground_truth_file: Optional[str] = None,
        image_extensions: Optional[List[str]] = None
    ) -> Tuple[List[Dict], Dict]:
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        folder = Path(folder_path)
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            logger.warning(f"No images found in {folder_path}")
            return [], {}

        ground_truth_map = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '\t' in line:
                        parts = line.split('\t', 1)
                        if len(parts) == 2:
                            ground_truth_map[parts[0]] = parts[1]
        
        logger.info(f"Processing {len(image_files)} images...")
        
        all_results = []
        
        for image_path in tqdm(image_files, desc="Comparing OCR models"):
            image_name = image_path.name
            gt = ground_truth_map.get(image_name)
            
            result = self.compare_single_image(str(image_path), gt)
            all_results.append(result)
        
        aggregate_metrics = self._calculate_aggregate_metrics(all_results)
        
        return all_results, aggregate_metrics
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        aggregate = {}
        
        model_names = set()
        for r in results:
            model_names.update(r.get('metrics', {}).keys())
        
        for model_name in model_names:
            cer_values = []
            wer_values = []
            accuracy_values = []
            timing_values = []
            
            for r in results:
                if model_name in r.get('metrics', {}):
                    metrics = r['metrics'][model_name]
                    cer_values.append(metrics['cer'])
                    wer_values.append(metrics['wer'])
                    accuracy_values.append(metrics['accuracy'])
                
                if model_name in r.get('timing', {}):
                    timing_values.append(r['timing'][model_name])
            
            if cer_values:
                aggregate[model_name] = {
                    'avg_cer': sum(cer_values) / len(cer_values),
                    'avg_wer': sum(wer_values) / len(wer_values),
                    'avg_accuracy': sum(accuracy_values) / len(accuracy_values),
                    'avg_time': sum(timing_values) / len(timing_values) if timing_values else 0,
                    'total_samples': len(cer_values)
                }
        
        return aggregate
