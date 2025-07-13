"""
Advanced speech-to-text transcription service using OpenAI Whisper.
Handles multi-language transcription with enhanced language detection.
"""

import os
import time
import warnings
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import certifi
import whisper
import torch
import numpy as np

from app.config import settings
from app.enhanced_language_detector import EnhancedLanguageDetector
from app.constants import (
    WhisperConfig, ErrorMessages, SuccessMessages, 
    ValidationConfig, ProcessingResult, AudioData
)


class TranscriptionService:
    """Advanced speech-to-text transcription with caching and optimization."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize transcription service with configurable model.
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium)
        """
        self.model_name = model_name or settings.get_whisper_model()
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize enhanced language detector
        self.language_detector = EnhancedLanguageDetector(self.model_name)
        
        # Load model
        self._load_model()
        
        # Processing configuration
        self._transcription_config = {
            "verbose": True,
            "language": None,
            "task": WhisperConfig.TASK_TRANSCRIBE
        }
    
    def _load_model(self) -> None:
        """Load Whisper model with error handling and optimization."""
        print(f"Loading Whisper model: {self.model_name}")
        try:
            os.environ['SSL_CERT_FILE'] = certifi.where()
            print(f"Using device: {self.device}")
            
            # Model loading with device-specific optimization
            if self.device == "cpu":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
                    self.model = whisper.load_model(self.model_name, device=self.device)
                    self.model = self.model.float()  # Force FP32 on CPU
            else:
                self.model = whisper.load_model(self.model_name, device=self.device)
            
            print(f"{SuccessMessages.MODEL_LOADED.format(self.model_name)} on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"{ErrorMessages.MODEL_LOAD_FAILED.format(str(e))}")
    
    @lru_cache(maxsize=100)
    def _get_cached_transcription_config(self, language: str, task: str) -> Dict:
        """Get cached transcription configuration."""
        return {
            "language": language,
            "task": task,
            "verbose": True
        }
    
    def transcribe_audio(self, audio_path: Union[str, Path], 
                        language: Optional[str] = None) -> ProcessingResult:
        """
        Transcribe audio file to text with enhanced language detection.
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, 'auto' for enhanced detection)
            
        Returns:
            Dictionary containing transcription results
            
        Raises:
            RuntimeError: If transcription fails
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing audio: {audio_path}")
        start_time = time.time()
        
        try:
            # Enhanced language detection for "auto" parameter
            language_detected = (
                self.language_detector.detect_language_enhanced(str(audio_path))["language"]
                if language == WhisperConfig.LANGUAGE_AUTO
                else language
            )
            
            if language == WhisperConfig.LANGUAGE_AUTO:
                detection_result = self.language_detector.detect_language_enhanced(str(audio_path))
                print(f"Enhanced auto-detected language: {detection_result['language']} "
                      f"(confidence: {detection_result['confidence']:.3f})")
            
            # Transcribe with device-specific handling
            result = self._transcribe_with_device_handling(
                audio_path, language_detected
            )
            
            return self._process_transcription_result(
                result, language_detected, time.time() - start_time
            )
            
        except Exception as e:
            raise RuntimeError(f"{ErrorMessages.TRANSCRIPTION_FAILED.format(str(e))}")
    
    def transcribe_audio_data(self, audio_data: AudioData, sample_rate: int, 
                            language: Optional[str] = None) -> ProcessingResult:
        """
        Transcribe audio data directly with optimization.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            language: Language code (optional, 'auto' for enhanced detection)
            
        Returns:
            Dictionary containing transcription results
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded")
        
        print("Transcribing audio data")
        start_time = time.time()
        
        try:
            # Optimize audio data format
            audio_data = self._optimize_audio_data(audio_data)
            
            # Enhanced language detection
            language_detected = (
                self._improved_language_detection(audio_data)
                if language in {WhisperConfig.LANGUAGE_AUTO, None}
                else language
            )
            
            print(f"Detected language: {language_detected}")
            
            # Transcribe with device-specific handling
            result = self._transcribe_with_device_handling(
                audio_data, language_detected
            )
            
            return self._process_transcription_result(
                result, language_detected, time.time() - start_time
            )
            
        except Exception as e:
            raise RuntimeError(f"{ErrorMessages.TRANSCRIPTION_FAILED.format(str(e))}")
    
    def _optimize_audio_data(self, audio_data: AudioData) -> AudioData:
        """Optimize audio data format for Whisper."""
        # Ensure float32 format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data
    
    def _transcribe_with_device_handling(self, audio_input: Union[str, Path, AudioData], 
                                       language: Optional[str]) -> Dict:
        """Handle transcription with device-specific optimizations."""
        config = self._get_cached_transcription_config(language or "auto", WhisperConfig.TASK_TRANSCRIBE)
        
        # Convert Path to string if needed
        if isinstance(audio_input, Path):
            audio_input = str(audio_input)
        
        if self.device == "cpu":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
                return self.model.transcribe(audio_input, **config)
        else:
            return self.model.transcribe(audio_input, **config)
    
    def _process_transcription_result(self, result: Dict, language: str, 
                                    processing_time: float) -> ProcessingResult:
        """Process and format transcription results."""
        # Extract results using dict comprehension
        extracted_data = {
            key: result.get(key, default_value)
            for key, default_value in [
                ("text", ""),
                ("segments", [])
            ]
        }
        
        transcript = extracted_data["text"].strip()
        # Use the specified language instead of Whisper's detection
        final_language = language
        segments = extracted_data["segments"]
        
        # Calculate confidence using vectorized operations
        confidence = self._calculate_confidence(segments)
        
        return {
            "transcript": transcript,
            "language": final_language,
            "confidence": confidence,
            "segments": segments,
            "processing_time": processing_time
        }
    
    def _calculate_confidence(self, segments: List[Dict]) -> float:
        """
        Calculate overall confidence score from segments using vectorized operations.
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Average confidence score
        """
        if not segments:
            return 0.0
        
        # Extract confidence scores using list comprehension
        confidence_scores = [
            segment.get("avg_logprob", 0.0) 
            for segment in segments 
            if segment.get("avg_logprob") is not None
        ]
        
        # Calculate weighted average based on segment duration
        if confidence_scores:
            weights = [
                segment.get("end", 0) - segment.get("start", 0)
                for segment in segments
                if segment.get("avg_logprob") is not None
            ]
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
                return sum(score * weight for score, weight in zip(confidence_scores, normalized_weights))
        
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", 
            "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", 
            "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", 
            "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", 
            "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", 
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", 
            "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", 
            "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", 
            "ba", "jw", "su"
        ]
    
    def detect_language(self, audio_path: Union[str, Path]) -> str:
        """
        Detect language of audio file using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detected language code
        """
        if not self.model:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # Use enhanced language detection
            detection_result = self.language_detector.detect_language_enhanced(str(audio_path))
            return detection_result["language"]
            
        except Exception as e:
            print(f"Warning: Language detection failed: {str(e)}")
            return "en"  # Default to English
    
    def _improved_language_detection(self, audio_data: AudioData) -> str:
        """
        Improved language detection for audio data with accent handling.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Detected language code
        """
        try:
            # Use Whisper's built-in language detection
            result = self._transcribe_with_device_handling(audio_data, None)
            detected_language = result.get("language", "en")
            
            # Apply accent correction logic
            if detected_language in ["cy", "ga", "gd"]:  # Welsh, Irish, Scottish Gaelic
                # Check if it might be English with accent
                confidence = result.get("language_probability", 0.0)
                if confidence < 0.8:  # Low confidence suggests possible accent
                    return "en"
            
            return detected_language
            
        except Exception as e:
            print(f"Warning: Language detection failed: {str(e)}")
            return "en"  # Default to English
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model is not None,
            "supported_languages": len(self.get_supported_languages()),
            "cuda_available": torch.cuda.is_available()
        }
    
    def batch_transcribe(self, audio_paths: List[Union[str, Path]], 
                        language: Optional[str] = None) -> List[ProcessingResult]:
        """
        Transcribe multiple audio files in batch.
        
        Args:
            audio_paths: List of audio file paths
            language: Language code for all files
            
        Returns:
            List of transcription results
        """
        return [
            {
                "file_path": str(path),
                "result": self.transcribe_audio(path, language)
            }
            for path in audio_paths
        ]
    
    def get_transcription_stats(self, result: ProcessingResult) -> Dict:
        """
        Get statistics from transcription result.
        
        Args:
            result: Transcription result
            
        Returns:
            Statistics dictionary
        """
        transcript = result.get("transcript", "")
        segments = result.get("segments", [])
        
        # Calculate statistics using list comprehensions
        word_count = len(transcript.split())
        segment_count = len(segments)
        
        # Duration statistics
        if segments:
            durations = [seg.get("end", 0) - seg.get("start", 0) for seg in segments]
            total_duration = sum(durations)
            avg_segment_duration = np.mean(durations)
        else:
            total_duration = avg_segment_duration = 0.0
        
        return {
            "word_count": word_count,
            "segment_count": segment_count,
            "total_duration": total_duration,
            "avg_segment_duration": avg_segment_duration,
            "confidence": result.get("confidence", 0.0),
            "processing_time": result.get("processing_time", 0.0)
        } 