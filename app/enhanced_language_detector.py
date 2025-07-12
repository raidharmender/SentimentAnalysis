#!/usr/bin/env python3
"""
Enhanced Language Detection System

This module provides improved language detection for the sentiment analysis system,
with special handling for English files (M_ and F_ prefixes) and Chinese files.
"""

import os
import re
import whisper
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings


class EnhancedLanguageDetector:
    """Enhanced language detection with pattern-based and ML-based classification"""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the enhanced language detector
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        
        # Language patterns for file classification
        self.english_patterns = [
            r'^[MF]_\d+_\d+y\d+m_\d+\.wav$',  # M_1234_12y3m_1.wav or F_1234_12y3m_1.wav
            r'^[MF]_\d+_\d+y\d+m_\d+\.mp3$',  # MP3 versions
            r'^[MF]_\d+_\d+y\d+m_\d+\.m4a$',  # M4A versions
        ]
        
        self.chinese_patterns = [
            r'^OSR_cn_\d+_\d+_8k\.wav$',  # OSR_cn_000_0075_8k.wav
            r'^cn_\d+\.wav$',  # Simple Chinese pattern
            r'^chinese_.*\.wav$',  # chinese_ prefixed files
        ]
        
        # Language confidence thresholds
        self.confidence_thresholds = {
            'en': 0.3,  # Lower threshold for English (due to accents)
            'zh': 0.6,  # Higher threshold for Chinese
            'default': 0.5  # Default threshold for other languages
        }
        
        # Known language mappings for file patterns
        self.pattern_language_map = {
            'english': 'en',
            'chinese': 'zh',
            'mandarin': 'zh',
            'cantonese': 'zh'
        }
    
    def _load_model(self):
        """Load Whisper model with FP16 fix"""
        print(f"Loading Whisper model: {self.model_name}")
        try:
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Suppress FP16 warnings on CPU
            if device == "cpu":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
                    # Load model
                    self.model = whisper.load_model(self.model_name, device=device)
                    # Force FP32 precision on CPU to avoid FP16 errors
                    self.model = self.model.float()  # Convert to FP32
            else:
                # Load model for GPU
                self.model = whisper.load_model(self.model_name, device=device)
            
            print(f"Successfully loaded Whisper model: {self.model_name} on {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model {self.model_name}: {str(e)}")
    
    def detect_language_from_filename(self, file_path: str) -> Optional[str]:
        """
        Detect language based on filename patterns
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Language code or None if no pattern matches
        """
        filename = os.path.basename(file_path)
        
        # Check English patterns (M_ and F_ files)
        for pattern in self.english_patterns:
            if re.match(pattern, filename):
                return 'en'
        
        # Check Chinese patterns
        for pattern in self.chinese_patterns:
            if re.match(pattern, filename):
                return 'zh'
        
        # Check folder structure
        file_path_obj = Path(file_path)
        if 'chinese' in file_path_obj.parts:
            return 'zh'
        
        return None
    
    def detect_language_with_whisper(self, audio_path: str) -> Tuple[str, float]:
        """
        Detect language using Whisper with improved confidence handling
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        try:
            # Load audio and detect language
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Log mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect language
            _, probs = self.model.detect_language(mel)
            
            # Get top 3 language probabilities
            top_languages = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            detected_lang, confidence = top_languages[0]
            
            print(f"Whisper language detection: {top_languages}")
            
            return detected_lang, confidence
            
        except Exception as e:
            print(f"Warning: Whisper language detection failed: {str(e)}")
            return 'en', 0.0  # Default to English on failure
    
    def detect_language_enhanced(self, audio_path: str) -> Dict:
        """
        Enhanced language detection combining filename patterns and Whisper
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary with detection results
        """
        filename = os.path.basename(audio_path)
        print(f"🔍 Enhanced language detection for: {filename}")
        
        # Step 1: Check filename patterns
        pattern_language = self.detect_language_from_filename(audio_path)
        print(f"📁 Filename pattern detection: {pattern_language}")
        
        # Step 2: Use Whisper for ML-based detection
        whisper_language, whisper_confidence = self.detect_language_with_whisper(audio_path)
        print(f"🤖 Whisper detection: {whisper_language} (confidence: {whisper_confidence:.3f})")
        
        # Step 3: Combine results with intelligent logic
        final_language, confidence, method = self._combine_detection_results(
            pattern_language, whisper_language, whisper_confidence, audio_path
        )
        
        result = {
            "language": final_language,
            "confidence": confidence,
            "detection_method": method,
            "filename_pattern": pattern_language,
            "whisper_detection": whisper_language,
            "whisper_confidence": whisper_confidence,
            "filename": filename
        }
        
        print(f"✅ Final detection: {final_language} (confidence: {confidence:.3f}, method: {method})")
        return result
    
    def _combine_detection_results(self, pattern_lang: Optional[str], 
                                 whisper_lang: str, whisper_conf: float,
                                 audio_path: str) -> Tuple[str, float, str]:
        """
        Intelligently combine pattern-based and ML-based detection results
        
        Args:
            pattern_lang: Language detected from filename pattern
            whisper_lang: Language detected by Whisper
            whisper_conf: Whisper confidence score
            audio_path: Path to audio file
            
        Returns:
            Tuple of (final_language, confidence, detection_method)
        """
        # Case 1: Strong pattern match (English M_/F_ files or Chinese files)
        if pattern_lang:
            if pattern_lang == 'en':
                # For English files, be more lenient with Whisper results
                if whisper_lang == 'en' and whisper_conf > 0.2:
                    return 'en', max(whisper_conf, 0.8), 'pattern_whisper_agreement'
                elif whisper_lang in ['cy', 'ga', 'gd'] and whisper_conf < 0.7:
                    # Likely English with accent, correct to English
                    return 'en', 0.9, 'pattern_correction'
                else:
                    return 'en', 0.95, 'filename_pattern'
            
            elif pattern_lang == 'zh':
                # For Chinese files, require higher Whisper confidence
                if whisper_lang == 'zh' and whisper_conf > 0.4:
                    return 'zh', max(whisper_conf, 0.9), 'pattern_whisper_agreement'
                else:
                    return 'zh', 0.95, 'filename_pattern'
        
        # Case 2: No pattern match, rely on Whisper with confidence thresholds
        threshold = self.confidence_thresholds.get(whisper_lang, self.confidence_thresholds['default'])
        
        if whisper_conf > threshold:
            return whisper_lang, whisper_conf, 'whisper_high_confidence'
        
        # Case 3: Low confidence, apply fallback logic
        if whisper_lang in ['cy', 'ga', 'gd'] and whisper_conf < 0.6:
            # Likely English with accent
            return 'en', 0.7, 'accent_correction'
        
        # Default fallback
        return whisper_lang, whisper_conf, 'whisper_fallback'
    
    def validate_detection(self, audio_path: str, expected_language: str) -> Dict:
        """
        Validate language detection against expected language
        
        Args:
            audio_path: Path to the audio file
            expected_language: Expected language code
            
        Returns:
            Validation results
        """
        detection_result = self.detect_language_enhanced(audio_path)
        
        is_correct = detection_result["language"] == expected_language
        confidence = detection_result["confidence"]
        
        return {
            "file": os.path.basename(audio_path),
            "expected": expected_language,
            "detected": detection_result["language"],
            "correct": is_correct,
            "confidence": confidence,
            "method": detection_result["detection_method"],
            "details": detection_result
        }
    
    def batch_validate(self, file_list: List[Tuple[str, str]]) -> Dict:
        """
        Validate language detection on a batch of files
        
        Args:
            file_list: List of (file_path, expected_language) tuples
            
        Returns:
            Batch validation results
        """
        results = []
        correct_count = 0
        
        for file_path, expected_lang in file_list:
            result = self.validate_detection(file_path, expected_lang)
            results.append(result)
            if result["correct"]:
                correct_count += 1
        
        accuracy = correct_count / len(file_list) if file_list else 0
        
        return {
            "total_files": len(file_list),
            "correct_detections": correct_count,
            "accuracy": accuracy,
            "results": results
        }
    
    def get_detection_stats(self, results: List[Dict]) -> Dict:
        """
        Get statistics from detection results
        
        Args:
            results: List of detection results
            
        Returns:
            Statistics summary
        """
        if not results:
            return {"error": "No results to analyze"}
        
        # Handle both validation results and direct detection results
        if "detected" in results[0]:
            # Validation results format
            languages = [r["detected"] for r in results]
            confidences = [r["confidence"] for r in results]
            methods = [r["method"] for r in results]
        else:
            # Direct detection results format
            languages = [r["language"] for r in results]
            confidences = [r["confidence"] for r in results]
            methods = [r["detection_method"] for r in results]
        
        # Count language occurrences
        lang_counts = {}
        for lang in languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # Count method occurrences
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            "total_files": len(results),
            "languages_detected": lang_counts,
            "detection_methods": method_counts,
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences)
        } 