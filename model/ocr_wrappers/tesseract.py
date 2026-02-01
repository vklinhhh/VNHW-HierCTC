# model/ocr_wrappers/tesseract.py

import time
import logging
from PIL import Image

from .base import BaseOCRModel

logger = logging.getLogger(__name__)


class TesseractOCRModel(BaseOCRModel):
    """Tesseract OCR with Vietnamese language pack."""
    
    def __init__(self, lang: str = 'vie'):
        super().__init__(f"Tesseract ({lang})")
        self.lang = lang
    
    def load(self) -> bool:
        try:
            import pytesseract
            
            start_time = time.time()
            
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            
            langs = pytesseract.get_languages()
            if self.lang not in langs:
                logger.warning(f"Vietnamese language pack not found. Available: {langs}")
                logger.warning("Install with: sudo apt-get install tesseract-ocr-vie")
            
            self.load_time = time.time() - start_time
            self.is_loaded = True
            logger.info(f"{self.name} loaded in {self.load_time:.2f}s")
            return True
            
        except ImportError:
            logger.error("pytesseract not installed. Install with: pip install pytesseract")
            return False
        except Exception as e:
            logger.error(f"✗ Failed to load {self.name}: {e}")
            return False
    
    def predict(self, image_path: str) -> str:
        if not self.is_loaded:
            return ""
        
        if not self.is_loaded:
            return ""
        
        try:
            import pytesseract
            
            image = Image.open(image_path)
            
            # Use PSM 7 for single line text (better for handwritten lines)
            # PSM 6 for a single uniform block of text
            custom_config = f'--oem 3 --psm 7 -l {self.lang}'
            
            text = pytesseract.image_to_string(image, config=custom_config)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error in {self.name} prediction: {e}")
            return ""
