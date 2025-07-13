"""
Advanced sentiment analysis module with multi-language support.
Handles sentiment analysis using multiple tools and languages.
"""

import time
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

import torch
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer as NLTKSentimentIntensityAnalyzer
import cntext
from snownlp import SnowNLP
import malaya

from app.config import settings
from app.constants import (
    SentimentConfig, SentimentLabel, SentimentTools, 
    ErrorMessages, SuccessMessages, ValidationConfig,
    SentimentResult, ProcessingResult
)


class SentimentModel(Enum):
    """Enumeration of available sentiment models."""
    VADER = "vader"
    TEXTBLOB = "textblob"
    HUGGINGFACE = "huggingface"
    SNOWNLP = "snownlp"
    CNTEXT = "cntext"
    MALAYA = "malaya"


@dataclass
class SentimentScore:
    """Data class for sentiment scores."""
    label: SentimentLabel
    score: float
    confidence: float
    model: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "label": self.label.value,
            "score": self.score,
            "confidence": self.confidence,
            "model": self.model
        }


class SentimentAnalyzer:
    """Advanced sentiment analysis using multiple models with caching."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize sentiment analyzer with configurable model.
        
        Args:
            model_name: HuggingFace model name for sentiment analysis
        """
        self.model_name = model_name or settings.sentiment_model
        self.classifier = None
        self.tokenizer = None
        self.model = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self._load_model()
        
        # Model mapping for different analysis types
        self._model_handlers = {
            SentimentModel.VADER: self._analyze_vader,
            SentimentModel.TEXTBLOB: self._analyze_textblob,
            SentimentModel.HUGGINGFACE: self._analyze_huggingface
        }
    
    def _load_model(self) -> None:
        """Load sentiment analysis model with error handling."""
        print(f"Loading sentiment model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print(f"{SuccessMessages.MODEL_LOADED.format(self.model_name)}")
            
        except Exception as e:
            print(f"Warning: {ErrorMessages.MODEL_LOAD_FAILED.format(str(e))}")
            print("Falling back to VADER and TextBlob")
            self.classifier = None
    
    @lru_cache(maxsize=1000)
    def _cached_analyze(self, text: str, model_type: str) -> Optional[Dict]:
        """Cached sentiment analysis for performance."""
        if model_type == SentimentModel.VADER.value:
            return self._analyze_vader(text)
        elif model_type == SentimentModel.TEXTBLOB.value:
            return self._analyze_textblob(text)
        elif model_type == SentimentModel.HUGGINGFACE.value:
            return self._analyze_huggingface(text)
        return None
    
    def _analyze_vader(self, text: str) -> Optional[Dict]:
        """Analyze sentiment using VADER."""
        try:
            return self.vader_analyzer.polarity_scores(text)
        except Exception as e:
            print(f"Warning: VADER analysis failed: {str(e)}")
            return None
    
    def _analyze_textblob(self, text: str) -> Optional[Dict]:
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            return {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
        except Exception as e:
            print(f"Warning: TextBlob analysis failed: {str(e)}")
            return None
    
    def _analyze_huggingface(self, text: str) -> Optional[Dict]:
        """Analyze sentiment using HuggingFace model."""
        if not self.classifier:
            return None
        
        try:
            return self.classifier(text)
        except Exception as e:
            print(f"Warning: HuggingFace analysis failed: {str(e)}")
            return None
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment using multiple models with validation.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not ValidationConfig.is_valid_text(text):
            return self._get_empty_result()
        
        print(f"Analyzing sentiment for text: {text[:100]}...")
        start_time = time.time()
        
        # Analyze with all available models using dict comprehension
        details = {
            model.value: self._cached_analyze(text, model.value)
            for model in SentimentModel
            if model != SentimentModel.SNOWNLP and model != SentimentModel.CNTEXT and model != SentimentModel.MALAYA
        }
        
        # Aggregate results
        aggregated = self._aggregate_sentiment_results(details)
        
        result = {
            "text": text,
            "details": details,
            **aggregated,
            "processing_time": time.time() - start_time
        }
        
        print(f"Sentiment analysis completed in {result['processing_time']:.2f}s")
        return result
    
    def _get_empty_result(self) -> SentimentResult:
        """Get empty sentiment result for invalid input."""
        return {
            "sentiment": SentimentLabel.NEUTRAL.value,
            "score": 0.0,
            "confidence": 0.0,
            "details": {model.value: None for model in SentimentModel}
        }
    
    def _aggregate_sentiment_results(self, details: Dict) -> Dict:
        """
        Aggregate results from multiple sentiment analysis models.
        
        Args:
            details: Dictionary containing results from different models
            
        Returns:
            Aggregated sentiment results
        """
        # Extract scores and labels using list comprehensions
        sentiment_data = []
        
        # Process HuggingFace results
        if hf_result := details.get(SentimentModel.HUGGINGFACE.value):
            if isinstance(hf_result, list) and hf_result:
                result = hf_result[0]
                label = result.get("label", "").lower()
                score = result.get("score", 0.0)
                
                sentiment_data.append(self._map_label_to_sentiment(label, score))
        
        # Process VADER results
        if vader_result := details.get(SentimentModel.VADER.value):
            compound = vader_result.get("compound", 0.0)
            sentiment_data.append(self._map_compound_to_sentiment(compound))
        
        # Process TextBlob results
        if textblob_result := details.get(SentimentModel.TEXTBLOB.value):
            polarity = textblob_result.get("polarity", 0.0)
            sentiment_data.append(self._map_polarity_to_sentiment(polarity))
        
        return self._calculate_final_sentiment(sentiment_data)
    
    def _map_label_to_sentiment(self, label: str, score: float) -> SentimentScore:
        """Map HuggingFace label to sentiment score."""
        if "positive" in label:
            return SentimentScore(SentimentLabel.POSITIVE, score, 0.8, "huggingface")
        elif "negative" in label:
            return SentimentScore(SentimentLabel.NEGATIVE, -score, 0.8, "huggingface")
        else:
            return SentimentScore(SentimentLabel.NEUTRAL, 0.0, 0.8, "huggingface")
    
    def _map_compound_to_sentiment(self, compound: float) -> SentimentScore:
        """Map VADER compound score to sentiment."""
        if compound >= SentimentConfig.POSITIVE_THRESHOLD:
            return SentimentScore(SentimentLabel.POSITIVE, compound, 0.8, "vader")
        elif compound <= SentimentConfig.NEGATIVE_THRESHOLD:
            return SentimentScore(SentimentLabel.NEGATIVE, compound, 0.8, "vader")
        else:
            return SentimentScore(SentimentLabel.NEUTRAL, compound, 0.8, "vader")
    
    def _map_polarity_to_sentiment(self, polarity: float) -> SentimentScore:
        """Map TextBlob polarity to sentiment."""
        if polarity > SentimentConfig.NEUTRAL_THRESHOLD:
            return SentimentScore(SentimentLabel.POSITIVE, polarity, 0.7, "textblob")
        elif polarity < -SentimentConfig.NEUTRAL_THRESHOLD:
            return SentimentScore(SentimentLabel.NEGATIVE, polarity, 0.7, "textblob")
        else:
            return SentimentScore(SentimentLabel.NEUTRAL, polarity, 0.7, "textblob")
    
    def _calculate_final_sentiment(self, sentiment_data: List[SentimentScore]) -> Dict:
        """Calculate final sentiment from multiple model results."""
        if not sentiment_data:
            return {
                "sentiment": SentimentLabel.NEUTRAL.value,
                "score": 0.0,
                "confidence": 0.0
            }
        
        # Calculate weighted average score
        scores = [s.score for s in sentiment_data]
        avg_score = np.mean(scores)
        
        # Determine sentiment label
        if avg_score > SentimentConfig.POSITIVE_THRESHOLD:
            sentiment = SentimentLabel.POSITIVE
        elif avg_score < SentimentConfig.NEGATIVE_THRESHOLD:
            sentiment = SentimentLabel.NEGATIVE
        else:
            sentiment = SentimentLabel.NEUTRAL
        
        # Calculate confidence based on model agreement
        unique_labels = len(set(s.label for s in sentiment_data))
        confidence = {
            1: SentimentConfig.CONFIDENCE_HIGH,
            2: SentimentConfig.CONFIDENCE_MEDIUM,
            3: SentimentConfig.CONFIDENCE_LOW
        }.get(unique_labels, SentimentConfig.CONFIDENCE_LOW)
        
        return {
            "sentiment": sentiment.value,
            "score": avg_score,
            "confidence": confidence
        }
    
    def analyze_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for multiple text segments.
        
        Args:
            segments: List of text segments with timing information
            
        Returns:
            List of segments with sentiment analysis
        """
        return [
            {
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": text,
                **self.analyze_sentiment(text)
            }
            for segment in segments
            if (text := segment.get("text", "").strip())
        ]
    
    def get_sentiment_summary(self, segments: List[Dict]) -> Dict:
        """
        Generate summary statistics for sentiment analysis.
        
        Args:
            segments: List of segments with sentiment analysis
            
        Returns:
            Summary statistics
        """
        if not segments:
            return self._get_empty_summary()
        
        # Extract scores and sentiments using list comprehensions
        scores = [seg["score"] for seg in segments]
        sentiments = [seg["sentiment"] for seg in segments]
        
        # Calculate overall sentiment
        avg_score = np.mean(scores)
        overall_sentiment = self._map_score_to_sentiment(avg_score)
        
        # Calculate sentiment distribution using dict comprehension
        sentiment_distribution = {
            label.value: sentiments.count(label.value)
            for label in SentimentLabel
        }
        
        return {
            "overall_sentiment": overall_sentiment.value,
            "average_score": avg_score,
            "sentiment_distribution": sentiment_distribution,
            "total_segments": len(segments),
            "score_range": {"min": min(scores), "max": max(scores)}
        }
    
    def _get_empty_summary(self) -> Dict:
        """Get empty summary for no segments."""
        return {
            "overall_sentiment": SentimentLabel.NEUTRAL.value,
            "average_score": 0.0,
            "sentiment_distribution": {label.value: 0 for label in SentimentLabel},
            "total_segments": 0
        }
    
    def _map_score_to_sentiment(self, score: float) -> SentimentLabel:
        """Map score to sentiment label."""
        if score > SentimentConfig.POSITIVE_THRESHOLD:
            return SentimentLabel.POSITIVE
        elif score < SentimentConfig.NEGATIVE_THRESHOLD:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "huggingface_model": self.model_name,
            "huggingface_loaded": self.classifier is not None,
            "vader_available": True,
            "textblob_available": True,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }


class SentimentAnalyzerMultiTool:
    """Multi-language sentiment analysis using specialized tools."""
    
    def __init__(self):
        """Initialize multi-tool sentiment analyzer."""
        # English tools
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Malay tool
        self.malaya_model = malaya.sentiment.huggingface()
        
        # Tool mapping using dict comprehension
        self._tool_handlers = {
            "en": self._analyze_english,
            "zh": self._analyze_mandarin,
            "ms": self._analyze_malay
        }
    
    def analyze(self, text: str, language: str) -> SentimentResult:
        """
        Analyze sentiment using language-specific tools.
        
        Args:
            text: Text to analyze
            language: Language code
            
        Returns:
            Sentiment analysis result
        """
        handler = self._tool_handlers.get(language, self._analyze_english)
        return handler(text)
    
    def _analyze_english(self, text: str) -> SentimentResult:
        """Analyze English text using VADER with context-aware adjustments."""
        scores = self.vader_analyzer.polarity_scores(text)
        compound = scores.get("compound", 0.0)
        
        # Context-aware adjustments for customer service calls
        adjusted_compound = self._adjust_for_customer_service_context(text, compound)
        
        # Map to sentiment using lambda
        sentiment_mapper = lambda score: (
            SentimentLabel.POSITIVE if score >= SentimentConfig.POSITIVE_THRESHOLD
            else SentimentLabel.NEGATIVE if score <= SentimentConfig.NEGATIVE_THRESHOLD
            else SentimentLabel.NEUTRAL
        )
        
        return {
            'tool': SentimentTools.get_tool_for_language("en"),
            'sentiment': sentiment_mapper(adjusted_compound).value,
            'score': adjusted_compound,
            'confidence': 0.8,
            'details': {**scores, 'original_compound': compound, 'adjusted_compound': adjusted_compound}
        }
    
    def _adjust_for_customer_service_context(self, text: str, compound: float) -> float:
        """Adjust sentiment score based on customer service context."""
        text_lower = text.lower()
        
        # Customer service context indicators
        cs_indicators = [
            "customer care", "customer service", "may i speak with", "calling to check",
            "follow-up", "how is your experience", "thank you for your feedback"
        ]
        
        # Check if this is a customer service call
        is_customer_service = any(indicator in text_lower for indicator in cs_indicators)
        
        if is_customer_service:
            # For customer service calls, be more conservative with positive sentiment
            # Reduce the compound score slightly to avoid over-classification
            if compound > 0.5:
                # High positive scores get reduced more
                adjusted = compound * 0.7
            elif compound > 0.2:
                # Moderate positive scores get reduced slightly
                adjusted = compound * 0.85
            else:
                # Low scores remain mostly unchanged
                adjusted = compound
        else:
            adjusted = compound
        
        return adjusted
    
    def _analyze_mandarin(self, text: str) -> SentimentResult:
        """Analyze Mandarin text using snownlp and cntext."""
        # SnowNLP analysis
        snlp = SnowNLP(text)
        snownlp_sentiment = snlp.sentiments
        
        # Convert snownlp score (0-1) to compound score (-1 to 1)
        compound = (snownlp_sentiment - SentimentConfig.SNOWNLP_NEUTRAL_THRESHOLD) * 2
        
        # Cntext analysis
        try:
            # Load HuLiu dictionary for Chinese sentiment analysis
            huliu_dict = cntext.load_pkl_dict('HuLiu.pkl')
            cntext_result = cntext.sentiment(text, huliu_dict)
        except Exception as e:
            self.logger.warning(f"Cntext analysis failed: {str(e)}")
            cntext_result = None
        
        # Map to sentiment
        sentiment_mapper = lambda score: (
            SentimentLabel.POSITIVE if score >= SentimentConfig.POSITIVE_THRESHOLD
            else SentimentLabel.NEGATIVE if score <= SentimentConfig.NEGATIVE_THRESHOLD
            else SentimentLabel.NEUTRAL
        )
        
        return {
            'tool': 'snownlp+cntext',
            'sentiment': sentiment_mapper(compound).value,
            'score': compound,
            'confidence': 0.8,
            'details': {
                'snownlp_sentiment': snownlp_sentiment,
                'cntext': cntext_result,
                'cnsenti_sentiment': 'positive' if snownlp_sentiment > 0.5 else 'negative' if snownlp_sentiment < 0.5 else 'neutral',
                'cnsenti_emotion': 'joy' if snownlp_sentiment > 0.7 else 'sadness' if snownlp_sentiment < 0.3 else 'neutral'
            }
        }
    
    def _analyze_malay(self, text: str) -> SentimentResult:
        """Analyze Malay text using malaya."""
        result = self.malaya_model.predict([text])
        malay_sentiment = result[0]
        
        # Map malaya result to score
        score_mapper = {
            "positive": SentimentConfig.MALAY_POSITIVE_SCORE,
            "negative": SentimentConfig.MALAY_NEGATIVE_SCORE,
            "neutral": 0.0
        }
        
        score = score_mapper.get(malay_sentiment, 0.0)
        sentiment = SentimentLabel.POSITIVE if score > 0 else SentimentLabel.NEGATIVE if score < 0 else SentimentLabel.NEUTRAL
        
        return {
            'tool': SentimentTools.get_tool_for_language("ms"),
            'sentiment': sentiment.value,
            'score': score,
            'confidence': 0.8,
            'details': {'malaya_result': malay_sentiment}
        }
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Fallback method for compatibility."""
        return self._analyze_english(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models."""
        return {
            "multi_tool_analyzer": True,
            "vader_available": True,
            "snownlp_available": True,
            "cntext_available": True,
            "malaya_available": True,
            "supported_languages": list(self._tool_handlers.keys())
        }
    
    def get_sentiment_summary(self, segments: List[Dict]) -> Dict[str, Any]:
        """Generate sentiment summary from segments."""
        if not segments:
            return self._get_empty_summary()
        
        # Extract sentiment data from segments that have been analyzed
        sentiment_data = []
        for seg in segments:
            if isinstance(seg, dict):
                # Check if segment has sentiment analysis results
                if "sentiment" in seg and isinstance(seg["sentiment"], dict):
                    # Segment already has sentiment analysis
                    sentiment_data.append((
                        seg["sentiment"].get("score", 0.0),
                        seg["sentiment"].get("sentiment", "neutral")
                    ))
                elif "text" in seg and seg["text"].strip():
                    # Analyze text for sentiment
                    text = seg["text"].strip()
                    if text:
                        sentiment_result = self._analyze_english(text)
                        sentiment_data.append((
                            sentiment_result.get("score", 0.0),
                            sentiment_result.get("sentiment", "neutral")
                        ))
        
        if not sentiment_data:
            return self._get_empty_summary()
        
        scores, sentiments = zip(*sentiment_data)
        avg_score = np.mean(scores)
        
        # Map to sentiment
        sentiment_mapper = lambda score: (
            SentimentLabel.POSITIVE if score > SentimentConfig.POSITIVE_THRESHOLD
            else SentimentLabel.NEGATIVE if score < SentimentConfig.NEGATIVE_THRESHOLD
            else SentimentLabel.NEUTRAL
        )
        
        # Count sentiment distribution
        sentiment_distribution = {label.value: 0 for label in SentimentLabel}
        for sentiment in sentiments:
            if sentiment in sentiment_distribution:
                sentiment_distribution[sentiment] += 1
        
        return {
            "overall_sentiment": sentiment_mapper(avg_score).value,
            "average_score": avg_score,
            "sentiment_distribution": sentiment_distribution,
            "total_segments": len(segments),
            "score_range": {"min": min(scores), "max": max(scores)}
        }
    
    def _get_empty_summary(self) -> Dict[str, Any]:
        """Get empty summary."""
        return {
            "overall_sentiment": SentimentLabel.NEUTRAL.value,
            "average_score": 0.0,
            "sentiment_distribution": {label.value: 0 for label in SentimentLabel},
            "total_segments": 0
        }
    
    def analyze_segments(self, segments: List[Dict]) -> List[Dict]:
        """Analyze sentiment for segments."""
        return [
            {
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": text,
                "sentiment": self._analyze_english(text)["sentiment"],
                "score": self._analyze_english(text)["score"],
                "confidence": 0.8,
                "details": self._analyze_english(text)["details"]
            }
            for segment in segments
            if (text := segment.get("text", "").strip())
        ]


def _demo() -> None:
    """Demonstration of multi-language sentiment analysis."""
    analyzer = SentimentAnalyzerMultiTool()
    
    test_cases = [
        ("我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心", 'zh'),
        ("Saya sangat gembira hari ini!", 'ms'),
        ("I'm absolutely chuffed to bits!", 'en')
    ]
    
    for text, lang in test_cases:
        print(f'\n{lang.upper()}:')
        print(analyzer.analyze(text, lang))


if __name__ == '__main__':
    _demo() 