import whisper
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from app.config import settings


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
    
    def _load_model(self):
        """Load Whisper model"""
        print(f"Loading Whisper model: {self.model_name}")
        try:
            self.model = whisper.load_model(self.model_name)
            print(f"Successfully loaded Whisper model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model {self.model_name}: {str(e)}")
    
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, will auto-detect if None)
            
        Returns:
            Dictionary containing transcription results
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded")
        
        print(f"Transcribing audio: {audio_path}")
        start_time = time.time()
        
        try:
            # Transcribe audio
            result = self.model.transcribe(
                audio_path,
                language=language,
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
            language: Language code (optional)
            
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
            
            # Transcribe audio data
            result = self.model.transcribe(
                audio_data,
                language=language,
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