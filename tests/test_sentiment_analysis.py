import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
from app.audio_processor import AudioProcessor
from app.transcription import TranscriptionService
from app.sentiment_analyzer import SentimentAnalyzer
from app.sentiment_service import SentimentAnalysisService


class TestAudioProcessor:
    """Test audio processing functionality"""
    
    def setup_method(self):
        self.processor = AudioProcessor()
    
    def test_normalize_sample_rate(self):
        """Test sample rate normalization"""
        # Create dummy audio data
        audio = np.random.randn(16000)  # 1 second at 16kHz
        current_sr = 8000
        
        # Test normalization
        normalized = self.processor.normalize_sample_rate(audio, current_sr)
        
        assert len(normalized) > len(audio)  # Should be longer after upsampling
        assert isinstance(normalized, np.ndarray)
    
    def test_denoise_audio(self):
        """Test audio denoising"""
        # Create dummy audio data
        audio = np.random.randn(16000)
        sample_rate = 16000
        
        # Test denoising
        denoised = self.processor.denoise_audio(audio, sample_rate)
        
        assert isinstance(denoised, np.ndarray)
        assert len(denoised) == len(audio)
    
    def test_trim_silence(self):
        """Test silence trimming"""
        # Create audio with silence at ends
        audio = np.concatenate([
            np.zeros(1000),  # Silence at start
            np.random.randn(14000),  # Audio content
            np.zeros(1000)   # Silence at end
        ])
        
        # Test trimming
        trimmed = self.processor.trim_silence(audio)
        
        assert isinstance(trimmed, np.ndarray)
        assert len(trimmed) <= len(audio)
    
    def test_normalize_amplitude(self):
        """Test amplitude normalization"""
        # Create audio with varying amplitude
        audio = np.random.randn(16000) * 0.1  # Low amplitude
        
        # Test normalization
        normalized = self.processor.normalize_amplitude(audio)
        
        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(audio)
    
    def test_get_audio_duration(self):
        """Test duration calculation"""
        audio = np.random.randn(16000)  # 1 second at 16kHz
        sample_rate = 16000
        
        duration = self.processor.get_audio_duration(audio, sample_rate)
        
        assert duration == 1.0


class TestTranscriptionService:
    """Test transcription functionality"""
    
    def setup_method(self):
        with patch('whisper.load_model'):
            self.service = TranscriptionService()
    
    def test_calculate_confidence(self):
        """Test confidence calculation"""
        # Mock segments with confidence scores
        segments = [
            {"avg_logprob": -0.5},
            {"avg_logprob": -0.3},
            {"avg_logprob": -0.7}
        ]
        
        confidence = self.service._calculate_confidence(segments)
        
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_get_supported_languages(self):
        """Test supported languages"""
        languages = self.service.get_supported_languages()
        # Accept dict_keys or list
        if not isinstance(languages, list):
            languages = list(languages)
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages  # English should be supported


class TestSentimentAnalyzer:
    """Test sentiment analysis functionality"""
    
    def setup_method(self):
        with patch('transformers.pipeline'):
            self.analyzer = SentimentAnalyzer()
    
    def test_analyze_sentiment_empty_text(self):
        """Test sentiment analysis with empty text"""
        result = self.analyzer.analyze_sentiment("")
        
        assert result["sentiment"] == "neutral"
        assert result["score"] == 0.0
        assert result["confidence"] == 0.0
    
    def test_analyze_sentiment_positive_text(self):
        """Test sentiment analysis with positive text"""
        result = self.analyzer.analyze_sentiment("I love this product! It's amazing!")
        
        assert "sentiment" in result
        assert "score" in result
        assert "confidence" in result
        assert "details" in result
    
    def test_analyze_sentiment_negative_text(self):
        """Test sentiment analysis with negative text"""
        result = self.analyzer.analyze_sentiment("This is terrible. I hate it.")
        
        assert "sentiment" in result
        assert "score" in result
        assert "confidence" in result
        assert "details" in result
    
    def test_aggregate_sentiment_results(self):
        """Test sentiment result aggregation"""
        details = {
            "vader": {"compound": 0.5},
            "textblob": {"polarity": 0.3}
        }
        
        result = self.analyzer._aggregate_sentiment_results(details)
        
        assert "sentiment" in result
        assert "score" in result
        assert "confidence" in result
    
    def test_analyze_segments(self):
        """Test segment analysis"""
        segments = [
            {"start": 0, "end": 5, "text": "Hello world"},
            {"start": 5, "end": 10, "text": "This is great"}
        ]
        
        results = self.analyzer.analyze_segments(segments)
        
        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert "sentiment" in result
            assert "score" in result
            assert "start" in result
            assert "end" in result
    
    def test_get_sentiment_summary(self):
        """Test sentiment summary generation"""
        segments = [
            {"sentiment": "positive", "score": 0.5},
            {"sentiment": "negative", "score": -0.3},
            {"sentiment": "positive", "score": 0.2}
        ]
        
        summary = self.analyzer.get_sentiment_summary(segments)
        
        assert "overall_sentiment" in summary
        assert "average_score" in summary
        assert "sentiment_distribution" in summary
        assert "total_segments" in summary


class TestSentimentAnalysisService:
    """Test the main sentiment analysis service"""
    
    def setup_method(self):
        with patch('app.audio_processor.AudioProcessor'), \
             patch('app.transcription.TranscriptionService'), \
             patch('app.sentiment_analyzer.SentimentAnalyzer'):
            self.service = SentimentAnalysisService()
    
    def test_system_status(self):
        """Test system status retrieval"""
        status = self.service.get_system_status()
        
        assert "audio_processor" in status
        assert "transcription" in status
        assert "sentiment_analysis" in status
        assert "storage" in status
    
    def test_get_sentiment_statistics_empty_db(self):
        """Test statistics with empty database"""
        mock_db = Mock()
        mock_db.query.return_value.count.return_value = 0
        
        stats = self.service.get_sentiment_statistics(mock_db)
        
        assert stats["total_analyses"] == 0
        assert stats["sentiment_distribution"]["positive"] == 0
        assert stats["sentiment_distribution"]["negative"] == 0
        assert stats["sentiment_distribution"]["neutral"] == 0


class TestIntegration:
    """Integration tests for the full pipeline (mocked)"""
    def test_full_pipeline_mock(self):
        from unittest.mock import patch, MagicMock
        service = SentimentAnalysisService()
        mock_db = MagicMock()
        # Patch process_audio, transcribe_audio_data, analyze_sentiment, and os.path.getsize
        with patch.object(service.audio_processor, 'process_audio', return_value=(np.zeros(16000), 16000)), \
             patch.object(service.transcription_service, 'transcribe_audio_data', return_value={"transcript": "hello", "language": "en", "confidence": 0.9, "segments": [], "processing_time": 0.1}), \
             patch.object(service.sentiment_analyzer, 'analyze_sentiment', return_value={"sentiment": "positive", "score": 0.8, "confidence": 0.9, "details": {}}), \
             patch('os.path.getsize', return_value=12345):
            result = service.analyze_audio_file("dummy_file.wav", mock_db)
            assert result["sentiment"]["overall_sentiment"] == "positive"


if __name__ == "__main__":
    pytest.main([__file__]) 