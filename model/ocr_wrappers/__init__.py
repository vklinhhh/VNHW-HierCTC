# model/ocr_wrappers/__init__.py
"""
OCR model wrappers for comparison and benchmarking.
"""

from .base import BaseOCRModel
from .hierctc import HierCTCModel
from .vietocr import VietOCRModel
from .tesseract import TesseractOCRModel
from .easyocr import EasyOCRModel

__all__ = [
    'BaseOCRModel',
    'HierCTCModel',
    'VietOCRModel',
    'TesseractOCRModel',
    'EasyOCRModel',
]
