import os
import certifi
import whisper
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import warnings
from app.config import settings
from app.enhanced_language_detector import EnhancedLanguageDetector


class TranscriptionService:
    """Handles speech-to-text transcription using OpenAI Whisper"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize transcription service
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large)
        """
        self.model_name = model_name or settings.WHISPER_MODEL
        self.model = None
        self._load_model()
        
        # Initialize enhanced language detector
        self.language_detector = EnhancedLanguageDetector(self.model_name)
    
    def _load_model(self):
        """Load Whisper model"""
        print(f"Loading Whisper model: {self.model_name}")
        try:
            os.environ['SSL_CERT_FILE'] = certifi.where()
            
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
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, 'auto' for improved detection, None for default)
            
        Returns:
            Dictionary containing transcription results
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        print(f"Transcribing audio: {audio_path}")
        start_time = time.time()
        
        try:
            # Handle language detection for "auto" parameter
            if language == "auto":
                # Use enhanced language detection
                detection_result = self.language_detector.detect_language_enhanced(audio_path)
                language_detected = detection_result["language"]
                print(f"Enhanced auto-detected language: {language_detected} (confidence: {detection_result['confidence']:.3f})")
            else:
                language_detected = language
            
            # Transcribe audio with warning suppression
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
                    result = self.model.transcribe(
                        audio_path,
                        language=language_detected,
                        verbose=True
                    )
            else:
                result = self.model.transcribe(
                    audio_path,
                    language=language_detected,
                    verbose=True
                )
            
            transcription_time = time.time() - start_time
            print(f"Transcription completed in {transcription_time:.2f}s")
            
            # Extract results
            transcript = result.get("text", "").strip()
            language_detected = result.get("language", language)
            segments = result.get("segments", [])
            
            # Calculate confidence score
            confidence = self._calculate_confidence(segments)
            
            return {
                "transcript": transcript,
                "language": language_detected,
                "confidence": confidence,
                "segments": segments,
                "processing_time": transcription_time
            }
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def transcribe_audio_data(self, audio_data: np.ndarray, sample_rate: int, 
                            language: str = None) -> Dict:
        """
        Transcribe audio data directly
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            language: Language code (optional, 'auto' for improved detection)
            
        Returns:
            Dictionary containing transcription results
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        print("Transcribing audio data")
        start_time = time.time()
        
        try:
            # Whisper expects audio in float32 format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Handle language detection
            if language == "auto" or language is None:
                # Use improved language detection for English accents
                language_detected = self._improved_language_detection(audio_data)
                print(f"Detected language: {language_detected}")
            else:
                language_detected = language
            
            # Transcribe audio data with warning suppression
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
                    result = self.model.transcribe(
                        audio_data,
                        language=language_detected,
                        verbose=True
                    )
            else:
                result = self.model.transcribe(
                    audio_data,
                    language=language_detected,
                    verbose=True
                )
            
            transcription_time = time.time() - start_time
            print(f"Transcription completed in {transcription_time:.2f}s")
            
            # Extract results
            transcript = result.get("text", "").strip()
            final_language = result.get("language", language_detected)
            segments = result.get("segments", [])
            
            # Calculate confidence score
            confidence = self._calculate_confidence(segments)
            
            return {
                "transcript": transcript,
                "language": final_language,
                "confidence": confidence,
                "segments": segments,
                "processing_time": transcription_time
            }
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def _calculate_confidence(self, segments: List[Dict]) -> float:
        """
        Calculate overall confidence score from segments
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Average confidence score
        """
        if not segments:
            return 0.0
        
        # Extract confidence scores from segments
        confidence_scores = []
        for segment in segments:
            if "avg_logprob" in segment:
                # Convert log probability to confidence (0-1)
                confidence = np.exp(segment["avg_logprob"])
                confidence_scores.append(confidence)
        
        if confidence_scores:
            return np.mean(confidence_scores)
        else:
            return 0.0
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes
        
        Returns:
            List of language codes
        """
        return whisper.tokenizer.LANGUAGES.keys()
    
    def detect_language(self, audio_path: str) -> str:
        """
        Detect language of audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Language code
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        try:
            # Load audio and detect language
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Log mel spectrogram
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect language
            _, probs = self.model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            
            return detected_lang
            
        except Exception as e:
            raise RuntimeError(f"Language detection failed: {str(e)}")
    
    def _improved_language_detection(self, audio_data: np.ndarray) -> str:
        """
        Improved language detection with better handling of English accents
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Language code
        """
        try:
            # First, try standard Whisper language detection
            audio = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            
            # Get top 3 language probabilities
            top_languages = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            detected_lang, confidence = top_languages[0]
            
            print(f"Language detection probabilities: {top_languages}")
            
            # English accent handling logic
            if detected_lang == "cy":  # Welsh detected
                # Check if it might be English with Welsh accent
                if confidence < 0.8:  # Lower confidence suggests possible misclassification
                    # Check second and third most likely languages
                    for lang, prob in top_languages[1:]:
                        if lang == "en" and prob > 0.3:  # English has significant probability
                            print(f"Correcting Welsh detection to English (confidence: {confidence:.3f} -> English: {prob:.3f})")
                            return "en"
            
            # Handle other potential English accent misclassifications
            english_accent_codes = ["cy", "ga", "gd", "en"]  # Welsh, Irish, Scottish, English
            if detected_lang in ["cy", "ga", "gd"] and confidence < 0.85:
                # Check if English is in top results
                for lang, prob in top_languages:
                    if lang == "en" and prob > 0.25:
                        print(f"Correcting {detected_lang} detection to English (confidence: {confidence:.3f} -> English: {prob:.3f})")
                        return "en"
            
            return detected_lang
            
        except Exception as e:
            print(f"Warning: Language detection failed, defaulting to English: {str(e)}")
            return "en"
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "model_name": self.model_name,
            "model_size": self.model.dims,
            "device": str(self.model.device),
            "status": "loaded"
        } 