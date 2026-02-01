# model/ocr_wrappers/easyocr.py

import time
import logging

from .base import BaseOCRModel

logger = logging.getLogger(__name__)


class EasyOCRModel(BaseOCRModel):
    """EasyOCR with Vietnamese support."""
    
    def __init__(self):
        super().__init__("EasyOCR (vi)")
        self.reader = None
    
    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load(self) -> bool:
        try:
            import easyocr
            
            start_time = time.time()
            
            self.reader = easyocr.Reader(['vi'], gpu=self._has_cuda())
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            logger.info(f"{self.name} loaded in {self.load_time:.2f}s")
            return True
            
        except ImportError:
            logger.error("EasyOCR not installed. Install with: pip install easyocr")
            return False
        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            return False
    
    def predict(self, image_path: str) -> str:
        if not self.is_loaded:
            return ""
        
        try:
            results = self.reader.readtext(image_path, detail=0)
            text = ' '.join(results)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in {self.name} prediction: {e}")
            return ""
