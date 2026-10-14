# term_corrector.py
import re
import os
from typing import List, Set, Dict
from pypinyin import pinyin, Style
import Levenshtein

class TermCorrector:
    def __init__(self, config):
        self.config = config
        self.correction_count = 0
        self.standard_terms = self._load_terms_from_reference()
        self.chinese_terms = {term for term in self.standard_terms if self._is_chinese(term)}
        
        # Prepare pinyin of term references
        self.term_pinyin_cache = {
            term: self._normalize_pinyin(self._get_pinyin(term))
            for term in self.chinese_terms
        }
    
    def _load_terms_from_reference(self) -> Set[str]:
        terms = set()

        with open(self.config.term_reference_dir, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    terms.add(line)
        print(f"Loaded {len(terms)} terms.")
        return terms
    
    def _is_chinese(self, text: str) -> bool:
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    def _normalize_pinyin(self, pinyin_str: str) -> str:
        """Normalize a pinyin by removing one marks, spaces, and convert to lowercase"""
        pinyin_str = re.sub(r'[0-9]', '', pinyin_str)
        pinyin_str = pinyin_str.replace(' ', '').replace("'", '')
        pinyin_str = re.sub(r'[^a-zA-Z]', '', pinyin_str)
        return pinyin_str.lower()
    
    def _get_pinyin(self, text: str) -> str:
        """
            Normalize a pinyin by removing one marks, spaces, and convert to lowercase
        """
        try:
            result = pinyin(text, style=Style.NORMAL, heteronym=False)
            return ''.join([item[0] for item in result])
        except:
            print(f"Cannot recognize pinyin for this term: {text}")
            return text
    
    def _levenshtein_distance(self, s1: str, s2: str) -> float:
        """
            Calculate levenshtein distance between 2 pinyin terms
            (0-1, lower value represents more similiar)
        """
        if not s1 or not s2:
            return 1.0
        return Levenshtein.distance(s1, s2) / max(len(s1), len(s2))
    
    def correct_text(self, text: str) -> str:
        """
            Correct Chinese terminology in the text by sliding window.
        """
        if not text:
            return text
        
        corrected_text = text
        
        # Sort terms by word length
        sorted_terms = sorted(self.chinese_terms, key=len, reverse=True)
        
        # Record replaced term
        replaced_positions = set()
        corrections = []
        
        for term in sorted_terms:
            term_len = len(term)
            term_pinyin = self.term_pinyin_cache.get(term, '')
            
            if not term_pinyin:
                continue
            
            for i in range(len(corrected_text) - term_len + 1):
                if any(pos <= i < pos + term_len for pos in replaced_positions):
                    continue
                
                window = corrected_text[i:i + term_len]
                
                if not self._is_chinese(window):
                    continue
                
                window_pinyin = self._normalize_pinyin(self._get_pinyin(window))

                dist = self._levenshtein_distance(window_pinyin, term_pinyin)
                
                if dist < 0.1:
                    corrections.append((i, i + term_len, window, term))
                    replaced_positions.add(i)
                    self.correction_count += 1
                    print(f"       🔧 匹配 Match: '{window}' → '{term}' (距离 distance: {dist:.3f})")
        
        # Apply replacement
        for start, end, old, new in sorted(corrections, key=lambda x: x[0], reverse=True):
            corrected_text = corrected_text[:start] + new + corrected_text[end:]
        
        return corrected_text
    
    def correct_segments(self, segments: List[Dict]) -> List[Dict]:
        """
            Correct Chinese segments by pinyin distance matching
            Args:
                segments: a list of segments with transcript     
            Returns:
                List[Dict]: a list of corrected segments
        """
        print(f"\n🔧 Start Chinese term correction...")
        self.correction_count = 0
        
        for idx, seg in enumerate(segments):
            text = seg.get('text', '')
            if text and self._is_chinese(text):
                corrected = self.correct_text(text)
                if corrected != text:
                    seg['text'] = corrected
                    seg['was_corrected'] = True
        
        print(f"\n ✅ Correction completed. Corrected {self.correction_count} terms in total.")
        return segments