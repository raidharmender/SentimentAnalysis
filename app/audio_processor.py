import os
import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional
import noisereduce as nr
from pydub import AudioSegment
from app.config import settings


class AudioProcessor:
    """Handles audio preprocessing including normalization and denoising"""
    
    def __init__(self):
        self.target_sample_rate = settings.TARGET_SAMPLE_RATE
        self.target_channels = settings.TARGET_CHANNELS
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data and sample rate
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio using librosa
            audio, sample_rate = librosa.load(file_path, sr=None, mono=False)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            
            return audio, sample_rate
        except Exception as e:
            raise ValueError(f"Error loading audio file {file_path}: {str(e)}")
    
    def normalize_sample_rate(self, audio: np.ndarray, current_sample_rate: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio: Audio data
            current_sample_rate: Current sample rate
            
        Returns:
            Resampled audio data
        """
        if current_sample_rate != self.target_sample_rate:
            audio = librosa.resample(
                audio, 
                orig_sr=current_sample_rate, 
                target_sr=self.target_sample_rate
            )
        return audio
    
    def denoise_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Remove noise from audio using spectral gating
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            Denoised audio data
        """
        try:
            # Apply noise reduction
            denoised_audio = nr.reduce_noise(
                y=audio, 
                sr=sample_rate,
                stationary=False,
                prop_decrease=0.75
            )
            return denoised_audio
        except Exception as e:
            print(f"Warning: Noise reduction failed: {str(e)}")
            return audio
    
    def trim_silence(self, audio: np.ndarray, threshold_db: float = -40.0) -> np.ndarray:
        """
        Trim silence from beginning and end of audio
        
        Args:
            audio: Audio data
            threshold_db: Threshold in dB for silence detection
            
        Returns:
            Trimmed audio data
        """
        try:
            # Convert threshold to amplitude
            threshold = 10 ** (threshold_db / 20)
            
            # Trim silence
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=abs(threshold_db))
            return trimmed_audio
        except Exception as e:
            print(f"Warning: Silence trimming failed: {str(e)}")
            return audio
    
    def normalize_amplitude(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio amplitude to target dB level
        
        Args:
            audio: Audio data
            target_db: Target dB level
            
        Returns:
            Normalized audio data
        """
        try:
            # Calculate current RMS
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                # Calculate target RMS
                target_rms = 10 ** (target_db / 20)
                # Normalize
                normalized_audio = audio * (target_rms / rms)
                return normalized_audio
            return audio
        except Exception as e:
            print(f"Warning: Amplitude normalization failed: {str(e)}")
            return audio
    
    def process_audio(self, file_path: str, denoise: bool = True, 
                     trim_silence: bool = True, normalize_amplitude: bool = True) -> Tuple[np.ndarray, int]:
        """
        Complete audio preprocessing pipeline
        
        Args:
            file_path: Path to the audio file
            denoise: Whether to apply denoising
            trim_silence: Whether to trim silence
            normalize_amplitude: Whether to normalize amplitude
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        print(f"Processing audio file: {file_path}")
        
        # Load audio
        audio, sample_rate = self.load_audio(file_path)
        print(f"Loaded audio: {len(audio)/sample_rate:.2f}s, {sample_rate}Hz")
        
        # Normalize sample rate
        audio = self.normalize_sample_rate(audio, sample_rate)
        sample_rate = self.target_sample_rate
        print(f"Normalized sample rate to: {sample_rate}Hz")
        
        # Denoise
        if denoise:
            audio = self.denoise_audio(audio, sample_rate)
            print("Applied denoising")
        
        # Trim silence
        if trim_silence:
            original_length = len(audio)
            audio = self.trim_silence(audio)
            new_length = len(audio)
            print(f"Trimmed silence: {original_length/sample_rate:.2f}s -> {new_length/sample_rate:.2f}s")
        
        # Normalize amplitude
        if normalize_amplitude:
            audio = self.normalize_amplitude(audio)
            print("Normalized amplitude")
        
        return audio, sample_rate
    
    def save_processed_audio(self, audio: np.ndarray, sample_rate: int, 
                           output_path: str) -> str:
        """
        Save processed audio to file
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save audio
            sf.write(output_path, audio, sample_rate)
            print(f"Saved processed audio to: {output_path}")
            return output_path
        except Exception as e:
            raise ValueError(f"Error saving audio file: {str(e)}")
    
    def get_audio_duration(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Calculate audio duration in seconds
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            Duration in seconds
        """
        return len(audio) / sample_rate 