import torch
import torch.nn.functional as F
import logging
import unicodedata
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .segmenter import SimpleVietnameseSegmenter

logger = logging.getLogger(__name__)


@dataclass
class CorrectionCandidate:
    text: str
    score: float
    original_position: int = -1
    correction_type: str = "unknown"


class PhoBERTLanguageModel:
    SUPPORTED_MODELS = {
        'vinai/phobert-base': 'Fast, good quality',
        'vinai/phobert-base-v2': 'Improved version',
        'vinai/phobert-large': 'Best quality, slower',
    }
    
    def __init__(
        self,
        model_name: str = 'vinai/phobert-base-v2',
        device: Optional[str] = None,
        use_segmenter: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.use_segmenter = use_segmenter
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing PhoBERT language model: {model_name}")
        logger.info(f"Device: {self.device}")
        
        self._load_model(cache_dir)
        
        if use_segmenter:
            self.segmenter = SimpleVietnameseSegmenter()
        else:
            self.segmenter = None
        
        logger.info("PhoBERT language model initialized successfully")
    
    def _load_model(self, cache_dir: Optional[str] = None):
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            
            logger.info(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            
            logger.info(f"Loading model from {self.model_name}...")
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            self.mask_token = self.tokenizer.mask_token
            self.mask_token_id = self.tokenizer.mask_token_id
            
            logger.info(f"Model loaded. Mask token: {self.mask_token}")
            
        except ImportError:
            raise ImportError(
                "transformers library is required. Install with: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load PhoBERT model: {e}")
            raise
    
    def preprocess(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        
        if self.segmenter:
            text = self.segmenter.segment(text)
        
        return text
    
    def postprocess(self, text: str) -> str:
        if self.segmenter:
            return self.segmenter.unsegment(text)
        return text
    
    @torch.no_grad()
    def score_sentence(self, text: str) -> float:
        processed_text = self.preprocess(text)
        
        inputs = self.tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(self.device)
        
        input_ids = inputs['input_ids'][0]
        
        total_score = 0.0
        num_tokens = 0
        
        for i in range(1, len(input_ids) - 1):
            masked_input = input_ids.clone().unsqueeze(0)
            masked_input[0, i] = self.mask_token_id
            
            outputs = self.model(masked_input)
            logits = outputs.logits[0, i]
            
            log_prob = F.log_softmax(logits, dim=-1)[input_ids[i]]
            total_score += log_prob.item()
            num_tokens += 1
        
        return total_score / max(num_tokens, 1)
    
    @torch.no_grad()
    def get_word_probability(
        self,
        text: str,
        word_position: int
    ) -> Tuple[float, List[Tuple[str, float]]]:
        processed_text = self.preprocess(text)
        words = processed_text.split()
        
        if word_position < 0 or word_position >= len(words):
            return 0.0, []
        
        original_word = words[word_position]
        
        masked_words = words.copy()
        masked_words[word_position] = self.mask_token
        masked_text = ' '.join(masked_words)
        
        inputs = self.tokenizer(
            masked_text,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(self.device)
        
        input_ids = inputs['input_ids'][0]
        mask_positions = (input_ids == self.mask_token_id).nonzero(as_tuple=True)[0]
        
        if len(mask_positions) == 0:
            return 0.0, []
        
        mask_pos = mask_positions[0].item()
        
        outputs = self.model(input_ids.unsqueeze(0))
        logits = outputs.logits[0, mask_pos]
        probs = F.softmax(logits, dim=-1)
        
        original_tokens = self.tokenizer.encode(original_word, add_special_tokens=False)
        if original_tokens:
            original_prob = probs[original_tokens[0]].item()
        else:
            original_prob = 0.0
        
        top_k = 10
        top_probs, top_indices = torch.topk(probs, top_k)
        
        alternatives = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = self.tokenizer.decode([idx]).strip()
            if token and token != original_word:
                alternatives.append((token, prob))
        
        return original_prob, alternatives[:5]
    
    @torch.no_grad()
    def suggest_corrections(
        self,
        text: str,
        confidence_threshold: float = 0.1,
        max_suggestions: int = 3
    ) -> List[Dict]:
        suggestions = []
        
        processed_text = self.preprocess(text)
        words = processed_text.split()
        
        for i, word in enumerate(words):
            if not any(c.isalpha() for c in word):
                continue
            
            word_prob, alternatives = self.get_word_probability(text, i)
            
            for alt_word, alt_prob in alternatives:
                if alt_prob > word_prob + confidence_threshold:
                    suggestions.append({
                        'position': i,
                        'original': self.postprocess(word),
                        'suggestion': self.postprocess(alt_word),
                        'original_prob': word_prob,
                        'suggestion_prob': alt_prob,
                        'confidence': alt_prob - word_prob
                    })
                    break
        
        return suggestions
    
    @torch.no_grad()
    def fill_mask(
        self,
        text_with_mask: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        processed_text = self.preprocess(text_with_mask)
        
        processed_text = processed_text.replace('[MASK]', self.mask_token)
        processed_text = processed_text.replace('<mask>', self.mask_token)
        
        inputs = self.tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(self.device)
        
        input_ids = inputs['input_ids'][0]
        mask_positions = (input_ids == self.mask_token_id).nonzero(as_tuple=True)[0]
        
        if len(mask_positions) == 0:
            return [(text_with_mask, 1.0)]
        
        mask_pos = mask_positions[0].item()
        
        outputs = self.model(input_ids.unsqueeze(0))
        logits = outputs.logits[0, mask_pos]
        probs = F.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs, top_k)
        
        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = self.tokenizer.decode([idx]).strip()
            filled_text = processed_text.replace(self.mask_token, token, 1)
            filled_text = self.postprocess(filled_text)
            results.append((filled_text, prob))
        
        return results
    
    def correct_text(
        self,
        text: str,
        confidence_threshold: float = 0.15,
        max_corrections: int = 5
    ) -> Tuple[str, List[Dict]]:
        corrections_made = []
        
        suggestions = self.suggest_corrections(
            text,
            confidence_threshold=confidence_threshold,
            max_suggestions=1
        )
        
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        corrected_text = text
        applied_count = 0
        
        for suggestion in suggestions:
            if applied_count >= max_corrections:
                break
            
            original = suggestion['original']
            replacement = suggestion['suggestion']
            
            if original in corrected_text:
                corrected_text = corrected_text.replace(original, replacement, 1)
                corrections_made.append({
                    'original': original,
                    'corrected': replacement,
                    'confidence': suggestion['confidence'],
                    'type': 'lm_correction'
                })
                applied_count += 1
        
        return corrected_text, corrections_made


class VietnameseGPT2Model:
    def __init__(
        self,
        model_name: str = 'NlpHUST/gpt2-vietnamese',
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self._load_model(cache_dir)
    
    def _load_model(self, cache_dir: Optional[str] = None):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading Vietnamese GPT-2: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Vietnamese GPT-2 model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load Vietnamese GPT-2: {e}")
            raise
    
    @torch.no_grad()
    def score_sentence(self, text: str) -> float:
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(self.device)
        
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss.item()
        
        return -loss
    
    @torch.no_grad()
    def generate_continuation(
        self,
        text: str,
        max_new_tokens: int = 20,
        num_return_sequences: int = 3
    ) -> List[str]:
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        outputs = self.model.generate(
            inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        results = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            results.append(text)
        
        return results


class VietnameseLMCorrector:
    def __init__(
        self,
        model_type: str = 'phobert',
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ):
        self.model_type = model_type
        
        if model_type == 'phobert':
            model_name = model_name or 'vinai/phobert-base-v2'
            self.model = PhoBERTLanguageModel(
                model_name=model_name,
                device=device,
                **kwargs
            )
        elif model_type == 'gpt2':
            model_name = model_name or 'NlpHUST/gpt2-vietnamese'
            self.model = VietnameseGPT2Model(
                model_name=model_name,
                device=device,
                **kwargs
            )
        elif model_type == 'ngram':
            from .ngram_model import VietnameseNGramModel
            self.model = VietnameseNGramModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"VietnameseLMCorrector initialized with {model_type}")
    
    def correct(
        self,
        text: str,
        confidence_threshold: float = 0.15,
        max_corrections: int = 5
    ) -> Tuple[str, List[Dict]]:
        if self.model_type == 'phobert':
            return self.model.correct_text(
                text,
                confidence_threshold=confidence_threshold,
                max_corrections=max_corrections
            )
        elif self.model_type == 'gpt2':
            return text, []
        else:
            return text, []
    
    def score(self, text: str) -> float:
        return self.model.score_sentence(text)


def create_vietnamese_lm(
    model_type: str = 'phobert',
    device: Optional[str] = None
) -> VietnameseLMCorrector:
    return VietnameseLMCorrector(model_type=model_type, device=device)
