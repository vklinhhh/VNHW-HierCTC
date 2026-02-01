import math
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class VietnameseNGramModel:
    def __init__(self, n: int = 3, smoothing: str = 'laplace', alpha: float = 0.1):
        self.n = n
        self.smoothing = smoothing
        self.alpha = alpha
        
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        
        self.vocabulary = set()
        self.total_words = 0
        
        self.BOS = '<BOS>'
        self.EOS = '<EOS>'
        self.UNK = '<UNK>'
        
    def train(self, sentences: List[List[str]]):
        logger.info(f"Training {self.n}-gram model on {len(sentences)} sentences")
        
        for sentence in sentences:
            padded = [self.BOS] * (self.n - 1) + sentence + [self.EOS]
            
            for i in range(len(padded) - self.n + 1):
                context = tuple(padded[i:i + self.n - 1])
                word = padded[i + self.n - 1]
                
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
                self.vocabulary.add(word)
                self.total_words += 1
        
        for context in self.ngram_counts:
            for word in context:
                if word not in (self.BOS, self.EOS, self.UNK):
                    self.vocabulary.add(word)
        
        logger.info(f"Vocabulary size: {len(self.vocabulary)}")
        logger.info(f"Total n-grams: {sum(self.context_counts.values())}")
    
    def _get_probability(self, word: str, context: Tuple[str, ...]) -> float:
        if self.smoothing == 'laplace':
            return self._laplace_probability(word, context)
        elif self.smoothing == 'none':
            return self._mle_probability(word, context)
        else:
            return self._laplace_probability(word, context)
    
    def _mle_probability(self, word: str, context: Tuple[str, ...]) -> float:
        context_count = self.context_counts.get(context, 0)
        if context_count == 0:
            return 1.0 / len(self.vocabulary) if self.vocabulary else 0.0
        
        word_count = self.ngram_counts[context].get(word, 0)
        return word_count / context_count
    
    def _laplace_probability(self, word: str, context: Tuple[str, ...]) -> float:
        context_count = self.context_counts.get(context, 0)
        word_count = self.ngram_counts[context].get(word, 0)
        
        vocab_size = len(self.vocabulary) + 1
        
        return (word_count + self.alpha) / (context_count + self.alpha * vocab_size)
    
    def score_sentence(self, sentence: List[str]) -> float:
        padded = [self.BOS] * (self.n - 1) + sentence + [self.EOS]
        log_prob = 0.0
        
        for i in range(len(padded) - self.n + 1):
            context = tuple(padded[i:i + self.n - 1])
            word = padded[i + self.n - 1]
            
            prob = self._get_probability(word, context)
            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += math.log(1e-10)
        
        return log_prob
    
    def score_word_in_context(self, word: str, left_context: List[str], right_context: List[str] = None) -> float:
        padded_left = [self.BOS] * (self.n - 1) + left_context
        context = tuple(padded_left[-(self.n - 1):])
        
        prob = self._get_probability(word, context)
        
        if prob > 0:
            return math.log(prob)
        return math.log(1e-10)
    
    def get_most_likely_words(self, context: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        padded = [self.BOS] * (self.n - 1) + context
        context_tuple = tuple(padded[-(self.n - 1):])
        
        if context_tuple not in self.ngram_counts:
            word_counts = Counter()
            for ctx, counts in self.ngram_counts.items():
                word_counts.update(counts)
            
            total = sum(word_counts.values())
            results = [(word, count / total) for word, count in word_counts.most_common(top_k)]
        else:
            counts = self.ngram_counts[context_tuple]
            total = self.context_counts[context_tuple]
            
            results = [(word, count / total) for word, count in counts.most_common(top_k)]
        
        return results
    
    def perplexity(self, sentences: List[List[str]]) -> float:
        total_log_prob = 0.0
        total_words = 0
        
        for sentence in sentences:
            total_log_prob += self.score_sentence(sentence)
            total_words += len(sentence) + 1
        
        avg_log_prob = total_log_prob / total_words
        return math.exp(-avg_log_prob)
    
    def save(self, filepath: str):
        data = {
            'n': self.n,
            'smoothing': self.smoothing,
            'alpha': self.alpha,
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'vocabulary': list(self.vocabulary),
            'total_words': self.total_words
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'VietnameseNGramModel':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(n=data['n'], smoothing=data['smoothing'], alpha=data['alpha'])
        model.ngram_counts = defaultdict(Counter, {k: Counter(v) for k, v in data['ngram_counts'].items()})
        model.context_counts = defaultdict(int, data['context_counts'])
        model.vocabulary = set(data['vocabulary'])
        model.total_words = data['total_words']
        
        logger.info(f"Model loaded from {filepath}")
        return model


class ContextAwareCorrector:
    def __init__(
        self,
        language_model: Optional[VietnameseNGramModel] = None,
        dictionary: Optional[Set[str]] = None,
        lm_weight: float = 0.5,
        edit_weight: float = 0.5,
        max_edit_distance: int = 2
    ):
        self.lm = language_model
        self.dictionary = dictionary or set()
        self.lm_weight = lm_weight
        self.edit_weight = edit_weight
        self.max_edit_distance = max_edit_distance
    
    def _compute_edit_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._compute_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _get_candidates(self, word: str) -> List[str]:
        candidates = [word]
        
        word_lower = word.lower()
        
        for dict_word in self.dictionary:
            if abs(len(dict_word) - len(word_lower)) <= self.max_edit_distance:
                distance = self._compute_edit_distance(word_lower, dict_word)
                if distance <= self.max_edit_distance:
                    candidates.append(dict_word)
        
        return candidates
    
    def correct_word(
        self,
        word: str,
        left_context: List[str],
        right_context: List[str] = None
    ) -> Tuple[str, float]:
        candidates = self._get_candidates(word)
        
        if len(candidates) == 1:
            return word, 1.0
        
        best_candidate = word
        best_score = float('-inf')
        
        for candidate in candidates:
            edit_dist = self._compute_edit_distance(word.lower(), candidate.lower())
            edit_score = 1.0 - (edit_dist / max(len(word), len(candidate), 1))
            lm_score = 0.0
            if self.lm is not None:
                lm_score = self.lm.score_word_in_context(candidate.lower(), left_context)
                lm_score = 1.0 / (1.0 + math.exp(-lm_score / 10))
            
            combined_score = (self.edit_weight * edit_score + self.lm_weight * lm_score)
            
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = candidate
        
        if word[0].isupper() and best_candidate[0].islower():
            best_candidate = best_candidate[0].upper() + best_candidate[1:]
        
        return best_candidate, best_score
    
    def correct_sentence(self, words: List[str]) -> List[Tuple[str, float]]:
        results = []
        
        for i, word in enumerate(words):
            left_context = [w.lower() for w in words[:i]]
            right_context = [w.lower() for w in words[i+1:]]
            
            corrected, score = self.correct_word(word, left_context, right_context)
            results.append((corrected, score))
        
        return results


class BeamSearchDecoder:
    def __init__(
        self,
        language_model: VietnameseNGramModel,
        vocabulary: List[str],
        blank_idx: int = 0,
        beam_width: int = 10,
        lm_weight: float = 0.3,
        word_bonus: float = 0.1
    ):
        self.lm = language_model
        self.vocabulary = vocabulary
        self.blank_idx = blank_idx
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.word_bonus = word_bonus
        
        self.idx_to_char = {i: c for i, c in enumerate(vocabulary)}
    
    def decode(self, log_probs) -> str:
        T, V = log_probs.shape
        beams = [('', None, 0.0)]
        
        for t in range(T):
            new_beams = defaultdict(lambda: float('-inf'))
            
            for prefix, last_char, score in beams:
                for v in range(V):
                    char = self.idx_to_char.get(v, '')
                    char_prob = log_probs[t, v]
                    
                    if v == self.blank_idx:
                        key = (prefix, None)
                        new_score = score + char_prob
                        new_beams[key] = max(new_beams[key], new_score)
                    elif char == last_char:
                        key = (prefix, char)
                        new_score = score + char_prob
                        new_beams[key] = max(new_beams[key], new_score)
                    else:
                        new_prefix = prefix + char
                        
                        lm_score = 0.0
                        if char == ' ' and self.lm is not None:
                            words = prefix.split()
                            if words:
                                lm_score = self.lm.score_word_in_context(
                                    words[-1].lower(),
                                    [w.lower() for w in words[:-1]]
                                ) * self.lm_weight
                        
                        key = (new_prefix, char)
                        new_score = score + char_prob + lm_score
                        new_beams[key] = max(new_beams[key], new_score)
            
            beam_list = [(prefix, last_char, score) 
                        for (prefix, last_char), score in new_beams.items()]
            beam_list.sort(key=lambda x: x[2], reverse=True)
            beams = beam_list[:self.beam_width]
        
        if beams:
            return beams[0][0]
        return ''


def prepare_training_data_from_corpus(corpus_path: str) -> List[List[str]]:
    sentences = []
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    words = line.lower().split()
                    if words:
                        sentences.append(words)
    except Exception as e:
        logger.error(f"Error loading corpus: {e}")
    
    logger.info(f"Loaded {len(sentences)} sentences from {corpus_path}")
    return sentences


def create_sample_vietnamese_corpus() -> List[List[str]]:
    sample_sentences = [
        "chân chân quân tử vị kiến",
        "quy ai quy tài chi trì",
        "phạt kỳ điều mai tuấn bị",
        "nhục phẫn vị kiến quân tử",
        "chức như điều cơ quân bao",
        "thắng đội năm chờ nghĩ người",
        "ăn giờ năm mưa xót thầm",
        "cao sơn lưu thủy thực trì",
        "đau đớn nhẽ quân tâm thiếp ý",
        "xa xôi ai có thấu tình chẳng ai",
        "dương chi thủy bát lưu thức tân",
        "bí kỳ chi từ chỉ đến nơi",
        "hầm sâu yểm lệ bất dữ ngã",
        "thù thân hoài tài hoàn hạt nguyệt",
        "tôi yêu việt nam đất nước xinh đẹp",
        "học tập là con đường dẫn đến thành công",
        "mưa rơi trên phố chiều đông lạnh",
        "hoa nở rực rỡ khắp vườn xuân",
    ]
    
    return [s.split() for s in sample_sentences]
