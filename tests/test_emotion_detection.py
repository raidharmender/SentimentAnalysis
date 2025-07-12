import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.audio_feature_extractor import AudioFeatureExtractor
from app.emotion_detector import EmotionDetector


class TestAudioFeatureExtractor:
    """Test audio feature extraction functionality"""
    
    def setup_method(self):
        self.extractor = AudioFeatureExtractor()
    
    def test_extract_mfcc_features(self):
        """Test MFCC feature extraction"""
        # Create dummy audio data
        audio_data = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds
        sample_rate = 16000
        
        features = self.extractor.extract_mfcc_features(audio_data, sample_rate)
        
        # Check that MFCC features are extracted
        assert "mfcc_mean" in features
        assert "mfcc_std" in features
        assert "mfcc_delta_mean" in features
        assert "mfcc_energy" in features
        
        # Check dimensions
        assert len(features["mfcc_mean"]) == 13  # Default 13 coefficients
        assert len(features["mfcc_std"]) == 13
    
    def test_extract_prosodic_features(self):
        """Test prosodic feature extraction"""
        audio_data = np.random.randn(16000 * 5).astype(np.float32)
        sample_rate = 16000
        
        features = self.extractor.extract_prosodic_features(audio_data, sample_rate)
        
        # Check that prosodic features are extracted
        assert "pitch_mean" in features
        assert "energy_mean" in features
        assert "tempo" in features
        
        # Check data types
        assert isinstance(features["pitch_mean"], float)
        assert isinstance(features["energy_mean"], float)
        assert isinstance(features["tempo"], float)
    
    def test_extract_spectral_features(self):
        """Test spectral feature extraction"""
        audio_data = np.random.randn(16000 * 5).astype(np.float32)
        sample_rate = 16000
        
        features = self.extractor.extract_spectral_features(audio_data, sample_rate)
        
        # Check that spectral features are extracted
        assert "spectral_centroid_mean" in features
        assert "spectral_rolloff_mean" in features
        assert "zero_crossing_rate_mean" in features
        
        # Check data types
        assert isinstance(features["spectral_centroid_mean"], float)
        assert isinstance(features["spectral_rolloff_mean"], float)
        assert isinstance(features["zero_crossing_rate_mean"], float)
    
    def test_extract_all_features(self):
        """Test complete feature extraction"""
        audio_data = np.random.randn(16000 * 5).astype(np.float32)
        sample_rate = 16000
        
        features = self.extractor.extract_all_features(audio_data, sample_rate)
        
        # Check that multiple feature categories are extracted
        assert "mfcc_mean" in features
        assert "pitch_mean" in features
        assert "spectral_centroid_mean" in features
        assert "chroma_mean" in features
        assert "harmonic_ratio" in features
        
        # Check feature summary
        summary = self.extractor.get_feature_summary(features)
        assert summary["total_features"] > 0
        assert summary["has_pitch"] == True
        assert summary["has_energy"] == True
    
    def test_extract_emotion_relevant_features(self):
        """Test emotion-relevant feature extraction"""
        audio_data = np.random.randn(16000 * 5).astype(np.float32)
        sample_rate = 16000
        
        features = self.extractor.extract_emotion_relevant_features(audio_data, sample_rate)
        
        # Check that emotion-relevant features are extracted
        emotion_features = [
            "pitch_mean", "pitch_std", "pitch_range",
            "energy_mean", "energy_std", "energy_max",
            "tempo", "spectral_centroid_mean",
            "mfcc_energy", "harmonic_ratio"
        ]
        
        for feature in emotion_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))


class TestEmotionDetector:
    """Test emotion detection functionality"""
    
    def setup_method(self):
        self.detector = EmotionDetector()
    
    def test_emotion_categories(self):
        """Test emotion categories are properly defined"""
        categories = self.detector.EMOTION_CATEGORIES
        
        expected_emotions = ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]
        
        for emotion in expected_emotions:
            assert emotion in categories
            assert isinstance(categories[emotion], list)
            assert len(categories[emotion]) > 0
    
    def test_audio_emotion_patterns(self):
        """Test audio emotion patterns are loaded"""
        patterns = self.detector.audio_emotion_patterns
        
        expected_emotions = ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]
        
        for emotion in expected_emotions:
            assert emotion in patterns
            assert isinstance(patterns[emotion], dict)
            
            # Check that patterns have expected features
            expected_features = ["pitch_range", "energy_mean", "tempo", "spectral_centroid_mean"]
            for feature in expected_features:
                assert feature in patterns[emotion]
                assert isinstance(patterns[emotion][feature], tuple)
                assert len(patterns[emotion][feature]) == 2
    
    def test_text_emotion_keywords(self):
        """Test text emotion keywords are loaded"""
        keywords = self.detector.text_emotion_keywords
        
        expected_emotions = ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]
        
        for emotion in expected_emotions:
            assert emotion in keywords
            assert isinstance(keywords[emotion], list)
            assert len(keywords[emotion]) > 0
    
    def test_detect_text_emotion_joy(self):
        """Test text emotion detection for joy"""
        text = "I am so happy and excited about this wonderful news!"
        
        result = self.detector._detect_text_emotion(text)
        
        assert result["emotion"] == "joy"
        assert result["confidence"] > 0.0
        assert "probabilities" in result
        assert result["method"] == "keyword_based"
    
    def test_detect_text_emotion_anger(self):
        """Test text emotion detection for anger"""
        text = "I am so angry and furious about this terrible situation!"
        
        result = self.detector._detect_text_emotion(text)
        
        assert result["emotion"] == "anger"
        assert result["confidence"] > 0.0
    
    def test_detect_text_emotion_neutral(self):
        """Test text emotion detection for neutral text"""
        text = "The weather is okay today and everything is fine."
        
        result = self.detector._detect_text_emotion(text)
        
        assert result["emotion"] == "neutral"
        assert result["confidence"] > 0.0
    
    def test_detect_text_emotion_empty(self):
        """Test text emotion detection for empty text"""
        result = self.detector._detect_text_emotion("")
        
        assert result["emotion"] == "neutral"
        assert result["confidence"] == 0.0
    
    def test_detect_audio_emotion(self):
        """Test audio emotion detection"""
        # Mock features that would indicate joy
        features = {
            "pitch_range": 1.5,  # High pitch variation
            "energy_mean": 0.8,  # High energy
            "tempo": 150,  # Fast tempo
            "spectral_centroid_mean": 3000  # Bright sound
        }
        
        result = self.detector._detect_audio_emotion(features)
        
        assert "emotion" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["method"] == "rule_based"
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
    
    def test_fuse_emotions_weighted(self):
        """Test weighted emotion fusion"""
        audio_emotion = {
            "emotion": "joy",
            "confidence": 0.8,
            "probabilities": {"joy": 0.8, "neutral": 0.2}
        }
        
        text_emotion = {
            "emotion": "joy",
            "confidence": 0.6,
            "probabilities": {"joy": 0.6, "neutral": 0.4}
        }
        
        result = self.detector._weighted_fusion(audio_emotion, text_emotion)
        
        assert result["emotion"] == "joy"
        assert result["confidence"] > 0.0
        assert "probabilities" in result
        assert result["method"] == "weighted_fusion"
    
    def test_fuse_emotions_ensemble(self):
        """Test ensemble emotion fusion"""
        audio_emotion = {
            "emotion": "joy",
            "confidence": 0.8,
            "probabilities": {"joy": 0.8, "neutral": 0.2}
        }
        
        text_emotion = {
            "emotion": "joy",
            "confidence": 0.6,
            "probabilities": {"joy": 0.6, "neutral": 0.4}
        }
        
        result = self.detector._ensemble_fusion(audio_emotion, text_emotion)
        
        assert result["emotion"] == "joy"
        assert result["confidence"] > 0.0
        assert "probabilities" in result
        assert result["method"] == "ensemble_fusion"
    
    def test_detect_emotion_audio_only(self):
        """Test emotion detection with audio only"""
        audio_data = np.random.randn(16000 * 5).astype(np.float32)
        sample_rate = 16000
        
        result = self.detector.detect_emotion(audio_data, sample_rate)
        
        assert "primary_emotion" in result
        assert "confidence" in result
        assert "audio_emotion" in result
        assert "features" in result
        assert "fusion_method" in result
    
    def test_detect_emotion_text_only(self):
        """Test emotion detection with text only"""
        audio_data = np.random.randn(16000 * 5).astype(np.float32)
        sample_rate = 16000
        transcript = "I am so happy and excited about this wonderful news!"
        
        result = self.detector.detect_emotion(audio_data, sample_rate, transcript)
        
        assert "primary_emotion" in result
        assert "confidence" in result
        assert "text_emotion" in result
        assert "fusion_method" in result
    
    def test_detect_emotion_multi_modal(self):
        """Test multi-modal emotion detection"""
        audio_data = np.random.randn(16000 * 5).astype(np.float32)
        sample_rate = 16000
        transcript = "I am so happy and excited about this wonderful news!"
        
        result = self.detector.detect_emotion(audio_data, sample_rate, transcript)
        
        assert "primary_emotion" in result
        assert "confidence" in result
        assert "audio_emotion" in result
        assert "text_emotion" in result
        assert "fusion_method" in result
    
    def test_get_emotion_summary(self):
        """Test emotion summary generation"""
        emotion_result = {
            "primary_emotion": "joy",
            "confidence": 0.8,
            "audio_emotion": {"emotion": "joy", "confidence": 0.8},
            "text_emotion": {"emotion": "joy", "confidence": 0.6},
            "fusion_method": "weighted",
            "features": {"pitch_mean": 0.5, "energy_mean": 0.7}
        }
        
        summary = self.detector.get_emotion_summary(emotion_result)
        
        assert summary["primary_emotion"] == "joy"
        assert summary["confidence"] == 0.8
        assert summary["has_audio_features"] == True
        assert summary["has_text_features"] == True
        assert summary["fusion_method"] == "weighted"


class TestEmotionDetectionIntegration:
    """Test integration between feature extraction and emotion detection"""
    
    def setup_method(self):
        self.extractor = AudioFeatureExtractor()
        self.detector = EmotionDetector()
    
    def test_end_to_end_emotion_detection(self):
        """Test complete emotion detection pipeline"""
        # Create audio data that should indicate joy
        # High energy, fast tempo, bright spectral features
        audio_data = np.random.randn(16000 * 5).astype(np.float32) * 0.8  # High amplitude
        sample_rate = 16000
        transcript = "I am so happy and excited about this wonderful news!"
        
        # Extract features
        features = self.extractor.extract_emotion_relevant_features(audio_data, sample_rate)
        
        # Detect emotion
        result = self.detector.detect_emotion(audio_data, sample_rate, transcript)
        
        # Verify results
        assert "primary_emotion" in result
        assert "confidence" in result
        assert "features" in result
        assert "audio_emotion" in result
        assert "text_emotion" in result
        
        # Check that features were extracted
        assert len(result["features"]) > 0
        
        # Check that both audio and text emotions were detected
        assert result["audio_emotion"]["emotion"] in self.detector.EMOTION_CATEGORIES
        assert result["text_emotion"]["emotion"] in self.detector.EMOTION_CATEGORIES
    
    def test_emotion_detection_performance(self):
        """Test emotion detection performance with different audio lengths"""
        sample_rates = [8000, 16000, 22050, 44100]
        
        for sample_rate in sample_rates:
            # Create audio data
            duration = 3  # seconds
            audio_data = np.random.randn(sample_rate * duration).astype(np.float32)
            
            # Time the feature extraction
            import time
            start_time = time.time()
            
            features = self.extractor.extract_emotion_relevant_features(audio_data, sample_rate)
            
            extraction_time = time.time() - start_time
            
            # Verify features were extracted
            assert len(features) > 0
            assert extraction_time < 5.0  # Should complete within 5 seconds
            
            print(f"Sample rate {sample_rate}Hz: {extraction_time:.2f}s") 