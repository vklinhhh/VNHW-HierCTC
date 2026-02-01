# model/ocr_wrappers/hierctc.py

import json
import time
import logging
from typing import Optional
from PIL import Image

from .base import BaseOCRModel

logger = logging.getLogger(__name__)


class HierCTCModel(BaseOCRModel):    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        device: Optional[str] = None
    ):
        super().__init__("VNHW-HierCTC (Ours)")
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.device = device
        self.model = None
        self.decoder = None
        self.processor = None
    
    def load(self) -> bool:
        try:
            import torch
            from model.hierarchical_ctc_model import HierarchicalCtcMultiScaleOcrModel
            from utils.ctc_utils import CTCDecoder
            
            start_time = time.time()

            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                combined_vocab = json.load(f)
            
            idx_to_char = {i: char for i, char in enumerate(combined_vocab)}
            
            self.model = HierarchicalCtcMultiScaleOcrModel.from_pretrained(
                self.model_path,
                combined_char_vocab=combined_vocab
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.processor = self.model.processor
            self.decoder = CTCDecoder(idx_to_char_map=idx_to_char, blank_idx=0)
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            logger.info(f"{self.name} loaded in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            return False
    
    def predict(self, image_path: str) -> str:
        if not self.is_loaded:
            return ""
        
        try:
            import torch
            
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
            
            logits = outputs.get('logits')
            decoded_texts = self.decoder(logits)
            
            return decoded_texts[0] if decoded_texts else ""
            
        except Exception as e:
            logger.error(f"Error in {self.name} prediction: {e}")
            return ""
