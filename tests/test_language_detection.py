import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.transcription import TranscriptionService


class TestLanguageDetection:
    """Test improved language detection functionality"""
    
    def setup_method(self):
        with patch('whisper.load_model'):
            self.service = TranscriptionService()
    
    def test_improved_language_detection_welsh_to_english(self):
        """Test that Welsh detection is corrected to English when appropriate"""
        # Mock audio data
        audio_data = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds of audio
        
        # Mock Whisper language detection to return Welsh with low confidence
        mock_probs = {
            "cy": 0.75,  # Welsh with 75% confidence
            "en": 0.35,  # English with 35% confidence
            "fr": 0.10   # French with 10% confidence
        }
        
        with patch.object(self.service.model, 'detect_language') as mock_detect, \
             patch('whisper.pad_or_trim') as mock_pad, \
             patch('whisper.log_mel_spectrogram') as mock_mel:
            
            # Mock the Whisper functions
            mock_pad.return_value = audio_data
            mock_mel.return_value = MagicMock()
            mock_detect.return_value = (None, mock_probs)
            
            # Test the improved detection
            detected_lang = self.service._improved_language_detection(audio_data)
            
            # Should correct Welsh to English since English has significant probability
            assert detected_lang == "en"
    
    def test_improved_language_detection_high_confidence_welsh(self):
        """Test that high confidence Welsh detection is not corrected"""
        # Mock audio data
        audio_data = np.random.randn(16000 * 10).astype(np.float32)
        
        # Mock Whisper language detection to return Welsh with high confidence
        mock_probs = {
            "cy": 0.95,  # Welsh with 95% confidence
            "en": 0.05,  # English with 5% confidence
            "fr": 0.01   # French with 1% confidence
        }
        
        with patch.object(self.service.model, 'detect_language') as mock_detect, \
             patch('whisper.pad_or_trim') as mock_pad, \
             patch('whisper.log_mel_spectrogram') as mock_mel:
            
            # Mock the Whisper functions
            mock_pad.return_value = audio_data
            mock_mel.return_value = MagicMock()
            mock_detect.return_value = (None, mock_probs)
            
            # Test the improved detection
            detected_lang = self.service._improved_language_detection(audio_data)
            
            # Should keep Welsh since confidence is high
            assert detected_lang == "cy"
    
    def test_improved_language_detection_irish_to_english(self):
        """Test that Irish detection is corrected to English when appropriate"""
        # Mock audio data
        audio_data = np.random.randn(16000 * 10).astype(np.float32)
        
        # Mock Whisper language detection to return Irish with low confidence
        mock_probs = {
            "ga": 0.70,  # Irish with 70% confidence
            "en": 0.30,  # English with 30% confidence
            "cy": 0.10   # Welsh with 10% confidence
        }
        
        with patch.object(self.service.model, 'detect_language') as mock_detect, \
             patch('whisper.pad_or_trim') as mock_pad, \
             patch('whisper.log_mel_spectrogram') as mock_mel:
            
            # Mock the Whisper functions
            mock_pad.return_value = audio_data
            mock_mel.return_value = MagicMock()
            mock_detect.return_value = (None, mock_probs)
            
            # Test the improved detection
            detected_lang = self.service._improved_language_detection(audio_data)
            
            # Should correct Irish to English since English has significant probability
            assert detected_lang == "en"
    
    def test_improved_language_detection_english_kept(self):
        """Test that English detection is kept as is"""
        # Mock audio data
        audio_data = np.random.randn(16000 * 10).astype(np.float32)
        
        # Mock Whisper language detection to return English
        mock_probs = {
            "en": 0.90,  # English with 90% confidence
            "fr": 0.05,  # French with 5% confidence
            "de": 0.05   # German with 5% confidence
        }
        
        with patch.object(self.service.model, 'detect_language') as mock_detect, \
             patch('whisper.pad_or_trim') as mock_pad, \
             patch('whisper.log_mel_spectrogram') as mock_mel:
            
            # Mock the Whisper functions
            mock_pad.return_value = audio_data
            mock_mel.return_value = MagicMock()
            mock_detect.return_value = (None, mock_probs)
            
            # Test the improved detection
            detected_lang = self.service._improved_language_detection(audio_data)
            
            # Should keep English
            assert detected_lang == "en"
    
    def test_improved_language_detection_fallback_to_english(self):
        """Test that detection failure falls back to English"""
        # Mock audio data
        audio_data = np.random.randn(16000 * 10).astype(np.float32)
        
        with patch.object(self.service.model, 'detect_language') as mock_detect, \
             patch('whisper.pad_or_trim') as mock_pad, \
             patch('whisper.log_mel_spectrogram') as mock_mel:
            
            # Mock the Whisper functions
            mock_pad.return_value = audio_data
            mock_mel.return_value = MagicMock()
            mock_detect.side_effect = Exception("Detection failed")
            
            # Test the improved detection
            detected_lang = self.service._improved_language_detection(audio_data)
            
            # Should fall back to English
            assert detected_lang == "en"
    
    def test_transcribe_with_auto_language_detection(self):
        """Test transcription with auto language detection"""
        # Mock audio data
        audio_data = np.random.randn(16000 * 10).astype(np.float32)
        sample_rate = 16000
        
        # Mock the improved language detection
        with patch.object(self.service, '_improved_language_detection') as mock_detect:
            mock_detect.return_value = "en"
            
            # Mock the transcribe method
            mock_result = {
                "text": "Hello, this is a test transcription.",
                "language": "en",
                "segments": [
                    {"avg_logprob": -0.5},
                    {"avg_logprob": -0.3}
                ]
            }
            
            with patch.object(self.service.model, 'transcribe') as mock_transcribe:
                mock_transcribe.return_value = mock_result
                
                # Test transcription with auto detection
                result = self.service.transcribe_audio_data(audio_data, sample_rate, "auto")
                
                # Verify improved detection was called
                mock_detect.assert_called_once()
                
                # Verify transcription was called with detected language
                mock_transcribe.assert_called_once_with(
                    audio_data,
                    language="en",
                    verbose=True
                )
                
                # Verify result
                assert result["language"] == "en"
                assert result["transcript"] == "Hello, this is a test transcription."
    
    def test_transcribe_with_specific_language(self):
        """Test transcription with specific language"""
        # Mock audio data
        audio_data = np.random.randn(16000 * 10).astype(np.float32)
        sample_rate = 16000
        
        # Mock the transcribe method
        mock_result = {
            "text": "Bonjour, ceci est un test.",
            "language": "fr",
            "segments": [
                {"avg_logprob": -0.4},
                {"avg_logprob": -0.2}
            ]
        }
        
        with patch.object(self.service.model, 'transcribe') as mock_transcribe:
            mock_transcribe.return_value = mock_result
            
            # Test transcription with specific language
            result = self.service.transcribe_audio_data(audio_data, sample_rate, "fr")
            
            # Verify transcription was called with specified language
            mock_transcribe.assert_called_once_with(
                audio_data,
                language="fr",
                verbose=True
            )
            
            # Verify result
            assert result["language"] == "fr"
            assert result["transcript"] == "Bonjour, ceci est un test." 