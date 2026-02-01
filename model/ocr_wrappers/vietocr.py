# model/ocr_wrappers/vietocr.py

import time
import logging
from PIL import Image

from .base import BaseOCRModel

logger = logging.getLogger(__name__)


class VietOCRModel(BaseOCRModel):
    """VietOCR Transformer model by pbcquoc."""
    
    def __init__(self, model_type: str = 'vgg_seq2seq'):
        super().__init__(f"VietOCR ({model_type})")
        self.model_type = model_type
        self.detector = None
    
    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load(self) -> bool:
        try:
            from vietocr.tool.predictor import Predictor
            from vietocr.tool.config import Cfg
            
            start_time = time.time()

            config = Cfg.load_config_from_name(self.model_type)
            config['cnn']['pretrained'] = True
            config['device'] = 'cuda:0' if self._has_cuda() else 'cpu'
            
            self.detector = Predictor(config)
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            logger.info(f"{self.name} loaded in {self.load_time:.2f}s")
            return True
            
        except ImportError:
            logger.error("VietOCR not installed. Install with: pip install vietocr")
            return False
        except Exception as e:
            logger.error(f"Failed to load {self.name}: {e}")
            return False
    
    def predict(self, image_path: str) -> str:
        if not self.is_loaded:
            return ""
        
        try:
            image = Image.open(image_path).convert('RGB')
            text = self.detector.predict(image)
            return text
            
        except Exception as e:
            logger.error(f"Error in {self.name} prediction: {e}")
            return ""
