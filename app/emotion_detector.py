import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from app.audio_feature_extractor import AudioFeatureExtractor

logger = logging.getLogger(__name__)


class EmotionDetector:
    """Detect emotions from audio and text features"""
    
    # Emotion categories based on research
    EMOTION_CATEGORIES = {
        "joy": ["happy", "excited", "enthusiastic", "cheerful", "delighted"],
        "anger": ["angry", "frustrated", "irritated", "annoyed", "furious"],
        "sadness": ["sad", "depressed", "melancholy", "grief", "sorrowful"],
        "fear": ["afraid", "anxious", "worried", "scared", "terrified"],
        "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned"],
        "disgust": ["disgusted", "repulsed", "revolted", "appalled"],
        "neutral": ["neutral", "calm", "indifferent", "balanced", "composed"]
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize emotion detector
        
        Args:
            config: Configuration dictionary for emotion detection
        """
        self.config = config or {
            "use_audio_features": True,
            "use_text_features": True,
            "fusion_method": "weighted",  # weighted, ensemble, voting
            "confidence_threshold": 0.6
        }
        
        self.feature_extractor = AudioFeatureExtractor()
        self._load_emotion_rules()
        
        logger.info("EmotionDetector initialized with config: %s", self.config)
    
    def _load_emotion_rules(self):
        """Load rule-based emotion detection patterns"""
        # Audio feature patterns for different emotions
        self.audio_emotion_patterns = {
            "joy": {
                "pitch_range": (0.8, 2.0),  # Higher pitch variation
                "energy_mean": (0.6, 1.0),  # Higher energy
                "tempo": (120, 180),  # Faster tempo
                "spectral_centroid_mean": (2000, 4000)  # Brighter sound
            },
            "anger": {
                "pitch_range": (1.0, 2.5),  # High pitch variation
                "energy_mean": (0.7, 1.0),  # High energy
                "tempo": (140, 200),  # Fast tempo
                "spectral_centroid_mean": (2500, 5000)  # Bright, harsh sound
            },
            "sadness": {
                "pitch_range": (0.2, 0.8),  # Low pitch variation
                "energy_mean": (0.2, 0.5),  # Low energy
                "tempo": (60, 100),  # Slow tempo
                "spectral_centroid_mean": (1000, 2500)  # Darker sound
            },
            "fear": {
                "pitch_range": (0.5, 1.5),  # Moderate pitch variation
                "energy_mean": (0.3, 0.6),  # Moderate energy
                "tempo": (80, 120),  # Moderate tempo
                "spectral_centroid_mean": (1500, 3000)  # Moderate brightness
            },
            "surprise": {
                "pitch_range": (1.2, 2.0),  # High pitch variation
                "energy_mean": (0.6, 0.9),  # High energy
                "tempo": (100, 160),  # Fast tempo
                "spectral_centroid_mean": (2000, 4000)  # Bright sound
            },
            "disgust": {
                "pitch_range": (0.3, 1.0),  # Low pitch variation
                "energy_mean": (0.4, 0.7),  # Moderate energy
                "tempo": (70, 110),  # Slow tempo
                "spectral_centroid_mean": (1200, 2800)  # Darker sound
            },
            "neutral": {
                "pitch_range": (0.4, 1.2),  # Moderate pitch variation
                "energy_mean": (0.4, 0.7),  # Moderate energy
                "tempo": (80, 130),  # Moderate tempo
                "spectral_centroid_mean": (1500, 3200)  # Moderate brightness
            }
        }
        
        # Text emotion keywords
        self.text_emotion_keywords = {
            "joy": ["happy", "great", "wonderful", "excellent", "fantastic", "amazing", "love", "enjoy"],
            "anger": ["angry", "furious", "mad", "hate", "terrible", "awful", "horrible", "disgusting"],
            "sadness": ["sad", "depressed", "unhappy", "miserable", "terrible", "awful", "disappointed"],
            "fear": ["afraid", "scared", "worried", "anxious", "nervous", "terrified", "frightened"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "unexpected", "wow"],
            "disgust": ["disgusting", "revolting", "gross", "nasty", "awful", "terrible"],
            "neutral": ["okay", "fine", "alright", "normal", "usual", "standard"]
        }
    
    def detect_emotion(self, audio_data: np.ndarray, sample_rate: int, 
                      transcript: str = "") -> Dict:
        """
        Multi-modal emotion detection from audio and text
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            transcript: Transcribed text (optional)
            
        Returns:
            Dictionary containing emotion detection results
        """
        logger.info("Starting emotion detection for %d samples", len(audio_data))
        
        try:
            results = {
                "primary_emotion": "neutral",
                "confidence": 0.0,
                "probabilities": {},
                "audio_emotion": {},
                "text_emotion": {},
                "features": {},
                "fusion_method": self.config["fusion_method"]
            }
            
            # Extract audio features
            if self.config.get("use_audio_features", True):
                audio_features = self.feature_extractor.extract_emotion_relevant_features(
                    audio_data, sample_rate
                )
                results["features"] = audio_features
                
                # Audio-based emotion detection
                audio_emotion = self._detect_audio_emotion(audio_features)
                results["audio_emotion"] = audio_emotion
            
            # Text-based emotion detection
            if self.config.get("use_text_features", True) and transcript:
                text_emotion = self._detect_text_emotion(transcript)
                results["text_emotion"] = text_emotion
            
            # Fuse results
            if results["audio_emotion"] and results["text_emotion"]:
                final_emotion = self._fuse_emotions(
                    results["audio_emotion"], 
                    results["text_emotion"]
                )
            elif results["audio_emotion"]:
                final_emotion = results["audio_emotion"]
            elif results["text_emotion"]:
                final_emotion = results["text_emotion"]
            else:
                final_emotion = {"emotion": "neutral", "confidence": 0.5}
            
            results["primary_emotion"] = final_emotion["emotion"]
            results["confidence"] = final_emotion["confidence"]
            results["probabilities"] = final_emotion.get("probabilities", {})
            
            logger.info("Emotion detection completed: %s (confidence: %.2f)", 
                       results["primary_emotion"], results["confidence"])
            
            return results
            
        except Exception as e:
            logger.error("Error in emotion detection: %s", str(e))
            return {
                "primary_emotion": "neutral",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _detect_audio_emotion(self, features: Dict) -> Dict:
        """
        Detect emotion based on audio features
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            Dictionary containing audio emotion detection results
        """
        try:
            emotion_scores = {}
            
            for emotion, patterns in self.audio_emotion_patterns.items():
                score = 0.0
                total_patterns = len(patterns)
                
                for feature_name, (min_val, max_val) in patterns.items():
                    if feature_name in features:
                        feature_value = features[feature_name]
                        
                        # Calculate how well the feature matches the pattern
                        if min_val <= feature_value <= max_val:
                            # Perfect match
                            score += 1.0
                        else:
                            # Calculate distance from ideal range
                            if feature_value < min_val:
                                distance = (min_val - feature_value) / min_val
                            else:
                                distance = (feature_value - max_val) / max_val
                            
                            # Score decreases with distance
                            score += max(0, 1 - distance)
                
                # Normalize score
                emotion_scores[emotion] = score / total_patterns
            
            # Find best emotion
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[best_emotion]
            
            return {
                "emotion": best_emotion,
                "confidence": confidence,
                "probabilities": emotion_scores,
                "method": "rule_based"
            }
            
        except Exception as e:
            logger.error("Error in audio emotion detection: %s", str(e))
            return {"emotion": "neutral", "confidence": 0.0}
    
    def _detect_text_emotion(self, transcript: str) -> Dict:
        """
        Detect emotion based on text content
        
        Args:
            transcript: Transcribed text
            
        Returns:
            Dictionary containing text emotion detection results
        """
        try:
            if not transcript or not transcript.strip():
                return {"emotion": "neutral", "confidence": 0.0}
            
            # Convert to lowercase for matching
            text_lower = transcript.lower()
            words = text_lower.split()
            
            emotion_scores = {emotion: 0.0 for emotion in self.EMOTION_CATEGORIES.keys()}
            
            # Count emotion keywords
            for emotion, keywords in self.text_emotion_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        emotion_scores[emotion] += 1.0
            
            # Normalize scores by text length
            if len(words) > 0:
                for emotion in emotion_scores:
                    emotion_scores[emotion] /= len(words)
            
            # Find best emotion
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = min(1.0, emotion_scores[best_emotion] * 10)  # Scale confidence
            
            # If no clear emotion, default to neutral
            if confidence < 0.1:
                best_emotion = "neutral"
                confidence = 0.5
            
            return {
                "emotion": best_emotion,
                "confidence": confidence,
                "probabilities": emotion_scores,
                "method": "keyword_based"
            }
            
        except Exception as e:
            logger.error("Error in text emotion detection: %s", str(e))
            return {"emotion": "neutral", "confidence": 0.0}
    
    def _fuse_emotions(self, audio_emotion: Dict, text_emotion: Dict) -> Dict:
        """
        Fuse audio and text emotion predictions
        
        Args:
            audio_emotion: Audio-based emotion detection results
            text_emotion: Text-based emotion detection results
            
        Returns:
            Dictionary containing fused emotion results
        """
        try:
            fusion_method = self.config.get("fusion_method", "weighted")
            
            if fusion_method == "weighted":
                return self._weighted_fusion(audio_emotion, text_emotion)
            elif fusion_method == "ensemble":
                return self._ensemble_fusion(audio_emotion, text_emotion)
            elif fusion_method == "voting":
                return self._voting_fusion(audio_emotion, text_emotion)
            else:
                return self._weighted_fusion(audio_emotion, text_emotion)
                
        except Exception as e:
            logger.error("Error in emotion fusion: %s", str(e))
            return {"emotion": "neutral", "confidence": 0.0}
    
    def _weighted_fusion(self, audio_emotion: Dict, text_emotion: Dict) -> Dict:
        """Weighted combination of audio and text emotions"""
        # Weight based on confidence
        audio_weight = audio_emotion.get("confidence", 0.0)
        text_weight = text_emotion.get("confidence", 0.0)
        
        # Normalize weights
        total_weight = audio_weight + text_weight
        if total_weight == 0:
            return {"emotion": "neutral", "confidence": 0.0}
        
        audio_weight /= total_weight
        text_weight /= total_weight
        
        # Combine probabilities
        combined_emotions = {}
        for emotion in self.EMOTION_CATEGORIES.keys():
            audio_prob = audio_emotion.get("probabilities", {}).get(emotion, 0.0)
            text_prob = text_emotion.get("probabilities", {}).get(emotion, 0.0)
            
            combined_emotions[emotion] = (
                audio_prob * audio_weight + text_prob * text_weight
            )
        
        # Find best emotion
        best_emotion = max(combined_emotions, key=combined_emotions.get)
        confidence = combined_emotions[best_emotion]
        
        return {
            "emotion": best_emotion,
            "confidence": confidence,
            "probabilities": combined_emotions,
            "method": "weighted_fusion"
        }
    
    def _ensemble_fusion(self, audio_emotion: Dict, text_emotion: Dict) -> Dict:
        """Ensemble method for emotion fusion"""
        # Simple ensemble: average the probabilities
        combined_emotions = {}
        
        for emotion in self.EMOTION_CATEGORIES.keys():
            audio_prob = audio_emotion.get("probabilities", {}).get(emotion, 0.0)
            text_prob = text_emotion.get("probabilities", {}).get(emotion, 0.0)
            
            combined_emotions[emotion] = (audio_prob + text_prob) / 2
        
        best_emotion = max(combined_emotions, key=combined_emotions.get)
        confidence = combined_emotions[best_emotion]
        
        return {
            "emotion": best_emotion,
            "confidence": confidence,
            "probabilities": combined_emotions,
            "method": "ensemble_fusion"
        }
    
    def _voting_fusion(self, audio_emotion: Dict, text_emotion: Dict) -> Dict:
        """Voting method for emotion fusion"""
        # Simple voting: choose the emotion with higher confidence
        audio_confidence = audio_emotion.get("confidence", 0.0)
        text_confidence = text_emotion.get("confidence", 0.0)
        
        if audio_confidence > text_confidence:
            return audio_emotion
        else:
            return text_emotion
    
    def get_emotion_summary(self, emotion_result: Dict) -> Dict:
        """
        Get a summary of emotion detection results
        
        Args:
            emotion_result: Emotion detection results
            
        Returns:
            Dictionary containing emotion summary
        """
        summary = {
            "primary_emotion": emotion_result.get("primary_emotion", "neutral"),
            "confidence": emotion_result.get("confidence", 0.0),
            "has_audio_features": "audio_emotion" in emotion_result,
            "has_text_features": "text_emotion" in emotion_result,
            "fusion_method": emotion_result.get("fusion_method", "unknown")
        }
        
        # Add emotion probabilities if available
        if "probabilities" in emotion_result:
            summary["emotion_probabilities"] = emotion_result["probabilities"]
        
        # Add feature summary if available
        if "features" in emotion_result:
            summary["feature_summary"] = self.feature_extractor.get_feature_summary(
                emotion_result["features"]
            )
        
        return summary 