# model/ocr_wrappers/base.py
from abc import ABC, abstractmethod
from typing import Dict


class BaseOCRModel(ABC):
    """Abstract base class for OCR model wrappers."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_loaded = False
        self.load_time = 0
    
    @abstractmethod
    def load(self) -> bool:
        pass
    
    @abstractmethod
    def predict(self, image_path: str) -> str:
        pass
    
    def get_info(self) -> Dict:
        return {
            'name': self.name,
            'is_loaded': self.is_loaded,
            'load_time': self.load_time
        }
