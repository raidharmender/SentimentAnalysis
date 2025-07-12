"""
Audio processing module with advanced preprocessing capabilities.
Handles audio normalization, denoising, and feature extraction.
"""

import os
from functools import lru_cache
from typing import Tuple, Optional, Callable
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pydub import AudioSegment

from app.config import settings
from app.constants import (
    AudioConfig, ErrorMessages, SuccessMessages, 
    ValidationConfig, AudioData, ProcessingResult
)


class AudioProcessor:
    """Advanced audio preprocessing with configurable pipeline."""
    
    def __init__(self, target_sample_rate: Optional[int] = None, 
                 target_channels: Optional[int] = None):
        """
        Initialize audio processor with configuration.
        
        Args:
            target_sample_rate: Target sample rate (default from settings)
            target_channels: Target channels (default from settings)
        """
        self.target_sample_rate = target_sample_rate or settings.target_sample_rate
        self.target_channels = target_channels or settings.target_channels
        
        # Processing pipeline configuration
        self._pipeline_steps = {
            'normalize_sample_rate': self._normalize_sample_rate,
            'denoise': self._denoise_audio,
            'trim_silence': self._trim_silence,
            'normalize_amplitude': self._normalize_amplitude
        }
    
    @lru_cache(maxsize=128)
    def _get_audio_info(self, file_path: str) -> Tuple[float, int]:
        """Get cached audio information."""
        return librosa.get_duration(filename=file_path), librosa.get_samplerate(file_path)
    
    def load_audio(self, file_path: str) -> Tuple[AudioData, int]:
        """
        Load audio file with error handling and validation.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
            
        Raises:
            ValueError: If file cannot be loaded or is invalid
        """
        try:
            # Validate file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Load audio using librosa
            audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
            
            # Convert to mono if stereo using list comprehension
            audio = np.mean(audio, axis=0) if len(audio.shape) > 1 else audio
            
            # Validate audio data
            if not ValidationConfig.is_valid_audio_duration(len(audio) / sample_rate):
                raise ValueError("Audio duration is outside valid range")
            
            return audio, sample_rate
            
        except Exception as e:
            raise ValueError(f"{ErrorMessages.FILE_TOO_LARGE.format(file_path)}: {str(e)}")
    
    def _normalize_sample_rate(self, audio: AudioData, current_sample_rate: int) -> AudioData:
        """Resample audio to target sample rate using lambda for efficiency."""
        return (librosa.resample(audio, orig_sr=current_sample_rate, 
                                target_sr=self.target_sample_rate)
                if current_sample_rate != self.target_sample_rate else audio)
    
    def _denoise_audio(self, audio: AudioData, sample_rate: int) -> AudioData:
        """Remove noise using spectral gating with configurable parameters."""
        try:
            return nr.reduce_noise(
                y=audio, 
                sr=sample_rate,
                stationary=AudioConfig.NOISE_REDUCE_STATIONARY,
                prop_decrease=AudioConfig.NOISE_REDUCE_PROP_DECAY
            )
        except Exception as e:
            print(f"Warning: Noise reduction failed: {str(e)}")
            return audio
    
    def _trim_silence(self, audio: AudioData, threshold_db: float = AudioConfig.SILENCE_THRESHOLD) -> AudioData:
        """Trim silence using configurable threshold."""
        try:
            return librosa.effects.trim(audio, top_db=abs(threshold_db))[0]
        except Exception as e:
            print(f"Warning: Silence trimming failed: {str(e)}")
            return audio
    
    def _normalize_amplitude(self, audio: AudioData, target_db: float = -20.0) -> AudioData:
        """Normalize amplitude using vectorized operations."""
        try:
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 10 ** (target_db / 20)
                return audio * (target_rms / rms)
            return audio
        except Exception as e:
            print(f"Warning: Amplitude normalization failed: {str(e)}")
            return audio
    
    def process_audio(self, file_path: str, 
                     processing_steps: Optional[dict] = None) -> Tuple[AudioData, int]:
        """
        Complete audio preprocessing pipeline with configurable steps.
        
        Args:
            file_path: Path to audio file
            processing_steps: Dict of step_name: enabled for each processing step
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        print(f"Processing audio file: {file_path}")
        
        # Default processing steps
        steps = processing_steps or {
            'normalize_sample_rate': True,
            'denoise': True,
            'trim_silence': True,
            'normalize_amplitude': True
        }
        
        # Load audio
        audio, sample_rate = self.load_audio(file_path)
        duration = len(audio) / sample_rate
        print(f"Loaded audio: {duration:.2f}s, {sample_rate}Hz")
        
        # Apply processing pipeline using list comprehension
        processing_results = [
            (step_name, step_func(audio, sample_rate))
            for step_name, step_func in self._pipeline_steps.items()
            if steps.get(step_name, False)
        ]
        
        # Update audio after each step
        for step_name, processed_audio in processing_results:
            audio = processed_audio
            if step_name == 'normalize_sample_rate':
                sample_rate = self.target_sample_rate
            print(f"Applied {step_name}")
        
        return audio, sample_rate
    
    def save_processed_audio(self, audio: AudioData, sample_rate: int, 
                           output_path: str) -> str:
        """
        Save processed audio with automatic directory creation.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        try:
            # Create output directory using pathlib
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save audio
            sf.write(output_path, audio, sample_rate)
            print(f"{SuccessMessages.FILE_UPLOADED}: {output_path}")
            return output_path
            
        except Exception as e:
            raise ValueError(f"Error saving audio file: {str(e)}")
    
    def get_audio_duration(self, audio: AudioData, sample_rate: int) -> float:
        """Calculate audio duration using vectorized operations."""
        return len(audio) / sample_rate
    
    def extract_audio_features(self, audio: AudioData, sample_rate: int) -> dict:
        """
        Extract comprehensive audio features using vectorized operations.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary of audio features
        """
        features = {}
        
        # MFCC features using list comprehension
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, 
                                   n_mfcc=AudioConfig.MFCC_N_COEFFICIENTS)
        features.update({
            'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
            'mfcc_std': np.std(mfcc, axis=1).tolist(),
            'mfcc_delta_mean': np.mean(librosa.feature.delta(mfcc), axis=1).tolist(),
            'mfcc_delta_std': np.std(librosa.feature.delta(mfcc), axis=1).tolist()
        })
        
        # Spectral features using dict comprehension
        spectral_features = {
            'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=sample_rate),
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=sample_rate),
            'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(audio)
        }
        
        features.update({
            name: np.mean(feature).item() 
            for name, feature in spectral_features.items()
        })
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        features.update({
            'chroma_mean': np.mean(chroma, axis=1).tolist(),
            'chroma_std': np.std(chroma, axis=1).tolist()
        })
        
        # Harmonic features
        harmonic, percussive = librosa.effects.hpss(audio)
        features.update({
            'harmonic_ratio': np.sum(harmonic**2) / np.sum(audio**2),
            'percussive_ratio': np.sum(percussive**2) / np.sum(audio**2)
        })
        
        return features
    
    def batch_process(self, file_paths: list, 
                     processing_steps: Optional[dict] = None) -> list:
        """
        Process multiple audio files in batch.
        
        Args:
            file_paths: List of audio file paths
            processing_steps: Processing configuration
            
        Returns:
            List of processing results
        """
        return [
            {
                'file_path': file_path,
                'result': self.process_audio(file_path, processing_steps)
            }
            for file_path in file_paths
        ]
    
    @staticmethod
    def validate_audio_file(file_path: str) -> bool:
        """
        Validate audio file format and size.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            # Check file exists
            if not file_path.exists():
                return False
            
            # Check file size
            if file_path.stat().st_size > AudioConfig.MAX_FILE_SIZE:
                return False
            
            # Check file format
            if file_path.suffix.lower() not in AudioConfig.ALLOWED_FORMATS:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_processing_stats(self, audio: AudioData, sample_rate: int) -> dict:
        """
        Get comprehensive audio processing statistics.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary of audio statistics
        """
        duration = self.get_audio_duration(audio, sample_rate)
        
        return {
            'duration': duration,
            'sample_rate': sample_rate,
            'samples': len(audio),
            'rms': np.sqrt(np.mean(audio**2)),
            'peak': np.max(np.abs(audio)),
            'dynamic_range': np.max(audio) - np.min(audio),
            'zero_crossings': np.sum(np.diff(np.sign(audio)) != 0)
        } 