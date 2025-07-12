import librosa
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extract comprehensive audio features for emotion analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize audio feature extractor
        
        Args:
            config: Configuration dictionary for feature extraction
        """
        self.config = config or {
            "mfcc_coefficients": 13,
            "spectral_features": True,
            "prosodic_features": True,
            "chroma_features": True,
            "harmonic_features": True
        }
        
        logger.info("AudioFeatureExtractor initialized with config: %s", self.config)
    
    def extract_all_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Extract all audio features for emotion analysis
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary containing all extracted features
        """
        logger.info("Extracting audio features for %d samples at %d Hz", 
                   len(audio_data), sample_rate)
        
        features = {}
        
        try:
            # MFCC features (most important for emotion detection)
            features.update(self.extract_mfcc_features(audio_data, sample_rate))
            
            # Spectral features
            if self.config.get("spectral_features", True):
                features.update(self.extract_spectral_features(audio_data, sample_rate))
            
            # Prosodic features
            if self.config.get("prosodic_features", True):
                features.update(self.extract_prosodic_features(audio_data, sample_rate))
            
            # Chroma features (harmonic content)
            if self.config.get("chroma_features", True):
                features.update(self.extract_chroma_features(audio_data, sample_rate))
            
            # Harmonic features
            if self.config.get("harmonic_features", True):
                features.update(self.extract_harmonic_features(audio_data, sample_rate))
            
            logger.info("Successfully extracted %d feature categories", len(features))
            return features
            
        except Exception as e:
            logger.error("Error extracting audio features: %s", str(e))
            return self._get_default_features()
    
    def extract_mfcc_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Extract MFCC (Mel-Frequency Cepstral Coefficients) features
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary containing MFCC features
        """
        try:
            n_mfcc = self.config.get("mfcc_coefficients", 13)
            
            # Extract MFCC coefficients
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            return {
                "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
                "mfcc_std": np.std(mfcc, axis=1).tolist(),
                "mfcc_delta_mean": np.mean(mfcc_delta, axis=1).tolist(),
                "mfcc_delta_std": np.std(mfcc_delta, axis=1).tolist(),
                "mfcc_delta2_mean": np.mean(mfcc_delta2, axis=1).tolist(),
                "mfcc_delta2_std": np.std(mfcc_delta2, axis=1).tolist(),
                "mfcc_energy": np.mean(np.sum(mfcc**2, axis=0)).item()
            }
        except Exception as e:
            logger.error("Error extracting MFCC features: %s", str(e))
            return {}
    
    def extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Extract spectral features for emotion analysis
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary containing spectral features
        """
        try:
            # Spectral centroid (brightness of sound)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            
            # Spectral bandwidth (width of spectral band)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
            
            # Zero crossing rate (rate of sign changes)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Spectral contrast (contrast between peaks and valleys)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
            
            return {
                "spectral_centroid_mean": np.mean(spectral_centroids).item(),
                "spectral_centroid_std": np.std(spectral_centroids).item(),
                "spectral_rolloff_mean": np.mean(spectral_rolloff).item(),
                "spectral_rolloff_std": np.std(spectral_rolloff).item(),
                "spectral_bandwidth_mean": np.mean(spectral_bandwidth).item(),
                "spectral_bandwidth_std": np.std(spectral_bandwidth).item(),
                "zero_crossing_rate_mean": np.mean(zero_crossing_rate).item(),
                "zero_crossing_rate_std": np.std(zero_crossing_rate).item(),
                "spectral_contrast_mean": np.mean(spectral_contrast).item(),
                "spectral_contrast_std": np.std(spectral_contrast).item()
            }
        except Exception as e:
            logger.error("Error extracting spectral features: %s", str(e))
            return {}
    
    def extract_prosodic_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Extract prosodic features (pitch, energy, tempo) for emotion analysis
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary containing prosodic features
        """
        try:
            # Pitch (fundamental frequency) - important for emotion detection
            pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
            
            # Filter out low magnitude pitches
            pitch_threshold = np.percentile(magnitudes, 85)
            pitch_values = pitches[magnitudes > pitch_threshold]
            
            # Energy (RMS - Root Mean Square)
            energy = librosa.feature.rms(y=audio_data)
            
            # Tempo (beats per minute)
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            
            # Onset strength (sudden changes in amplitude)
            onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
            
            return {
                "pitch_mean": float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0,
                "pitch_std": float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0,
                "pitch_min": float(np.min(pitch_values)) if len(pitch_values) > 0 else 0.0,
                "pitch_max": float(np.max(pitch_values)) if len(pitch_values) > 0 else 0.0,
                "energy_mean": float(np.mean(energy)),
                "energy_std": float(np.std(energy)),
                "energy_max": float(np.max(energy)),
                "tempo": float(tempo),
                "onset_strength_mean": float(np.mean(onset_strength)),
                "onset_strength_std": float(np.std(onset_strength))
            }
        except Exception as e:
            logger.error("Error extracting prosodic features: %s", str(e))
            return {}
    
    def extract_chroma_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Extract chroma features (harmonic content) for emotion analysis
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary containing chroma features
        """
        try:
            # Chroma features (12-dimensional representation of harmonic content)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            
            # Chroma energy normalized
            chroma_cens = librosa.feature.chroma_cens(y=audio_data, sr=sample_rate)
            
            # Chroma energy distribution
            chroma_energy = np.sum(chroma, axis=0)
            
            return {
                "chroma_mean": np.mean(chroma, axis=1).tolist(),
                "chroma_std": np.std(chroma, axis=1).tolist(),
                "chroma_cens_mean": np.mean(chroma_cens, axis=1).tolist(),
                "chroma_cens_std": np.std(chroma_cens, axis=1).tolist(),
                "chroma_energy_mean": np.mean(chroma_energy).item(),
                "chroma_energy_std": np.std(chroma_energy).item()
            }
        except Exception as e:
            logger.error("Error extracting chroma features: %s", str(e))
            return {}
    
    def extract_harmonic_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Extract harmonic features for emotion analysis
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary containing harmonic features
        """
        try:
            # Harmonic-percussive source separation
            harmonic, percussive = librosa.effects.hpss(audio_data)
            
            # Harmonic ratio
            harmonic_ratio = np.sum(harmonic**2) / (np.sum(harmonic**2) + np.sum(percussive**2))
            
            # Harmonic energy
            harmonic_energy = np.sum(harmonic**2)
            percussive_energy = np.sum(percussive**2)
            
            return {
                "harmonic_ratio": harmonic_ratio.item(),
                "harmonic_energy": harmonic_energy.item(),
                "percussive_energy": percussive_energy.item(),
                "harmonic_percussive_ratio": (harmonic_energy / percussive_energy).item() if percussive_energy > 0 else 0.0
            }
        except Exception as e:
            logger.error("Error extracting harmonic features: %s", str(e))
            return {}
    
    def extract_emotion_relevant_features(self, audio_data: np.ndarray, 
                                        sample_rate: int) -> Dict:
        """
        Extract features specifically relevant for emotion detection
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary containing emotion-relevant features
        """
        try:
            # Get all features
            all_features = self.extract_all_features(audio_data, sample_rate)
            
            # Select features most relevant for emotion detection
            emotion_features = {
                # Pitch features (important for emotion)
                "pitch_mean": all_features.get("pitch_mean", 0.0),
                "pitch_std": all_features.get("pitch_std", 0.0),
                "pitch_range": all_features.get("pitch_max", 0.0) - all_features.get("pitch_min", 0.0),
                
                # Energy features (important for emotion intensity)
                "energy_mean": all_features.get("energy_mean", 0.0),
                "energy_std": all_features.get("energy_std", 0.0),
                "energy_max": all_features.get("energy_max", 0.0),
                
                # Tempo features (important for emotion type)
                "tempo": all_features.get("tempo", 0.0),
                
                # Spectral features (important for emotion quality)
                "spectral_centroid_mean": all_features.get("spectral_centroid_mean", 0.0),
                "spectral_rolloff_mean": all_features.get("spectral_rolloff_mean", 0.0),
                "zero_crossing_rate_mean": all_features.get("zero_crossing_rate_mean", 0.0),
                
                # MFCC features (important for overall emotion)
                "mfcc_energy": all_features.get("mfcc_energy", 0.0),
                
                # Harmonic features (important for voice quality)
                "harmonic_ratio": all_features.get("harmonic_ratio", 0.0)
            }
            
            return emotion_features
            
        except Exception as e:
            logger.error("Error extracting emotion-relevant features: %s", str(e))
            return self._get_default_emotion_features()
    
    def _get_default_features(self) -> Dict:
        """Get default features when extraction fails"""
        return {
            "mfcc_mean": [0.0] * 13,
            "mfcc_std": [0.0] * 13,
            "pitch_mean": 0.0,
            "energy_mean": 0.0,
            "tempo": 120.0,
            "spectral_centroid_mean": 0.0
        }
    
    def _get_default_emotion_features(self) -> Dict:
        """Get default emotion-relevant features when extraction fails"""
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_range": 0.0,
            "energy_mean": 0.0,
            "energy_std": 0.0,
            "energy_max": 0.0,
            "tempo": 120.0,
            "spectral_centroid_mean": 0.0,
            "spectral_rolloff_mean": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "mfcc_energy": 0.0,
            "harmonic_ratio": 0.5
        }
    
    def get_feature_summary(self, features: Dict) -> Dict:
        """
        Get a summary of extracted features for analysis
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary containing feature summary
        """
        summary = {
            "total_features": len(features),
            "feature_categories": list(features.keys()),
            "has_pitch": "pitch_mean" in features,
            "has_energy": "energy_mean" in features,
            "has_spectral": "spectral_centroid_mean" in features,
            "has_mfcc": "mfcc_mean" in features,
            "has_harmonic": "harmonic_ratio" in features
        }
        
        # Add some basic statistics
        numeric_features = {k: v for k, v in features.items() 
                          if isinstance(v, (int, float))}
        
        if numeric_features:
            summary["numeric_features_count"] = len(numeric_features)
            summary["feature_values_range"] = {
                "min": min(numeric_features.values()),
                "max": max(numeric_features.values()),
                "mean": np.mean(list(numeric_features.values())).item()
            }
        
        return summary 