import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple, Optional
import numpy as np
import time
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.config import settings
import os
import re
from typing import Dict, Any

# English
from nltk.sentiment import SentimentIntensityAnalyzer
# Mandarin
from cnsenti import Sentiment as CNSentiSentiment, Emotion as CNSentiEmotion
from cntext import CnText
# Malay
import malaya


class SentimentAnalyzer:
    """Handles sentiment analysis using multiple models"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: Hugging Face model name for sentiment analysis
        """
        self.model_name = model_name or settings.SENTIMENT_MODEL
        self.classifier = None
        self.tokenizer = None
        self.model = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self._load_model()
    
    def _load_model(self):
        """Load sentiment analysis model"""
        print(f"Loading sentiment model: {self.model_name}")
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print(f"Successfully loaded sentiment model: {self.model_name}")
            
        except Exception as e:
            print(f"Warning: Failed to load Hugging Face model: {str(e)}")
            print("Falling back to VADER and TextBlob")
            self.classifier = None
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of text using multiple models
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        if not text or not text.strip():
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "details": {
                    "huggingface": None,
                    "vader": None,
                    "textblob": None
                }
            }
        
        print(f"Analyzing sentiment for text: {text[:100]}...")
        start_time = time.time()
        
        results = {
            "text": text,
            "details": {}
        }
        
        # Hugging Face model
        if self.classifier:
            try:
                hf_result = self.classifier(text)
                results["details"]["huggingface"] = hf_result
            except Exception as e:
                print(f"Warning: Hugging Face analysis failed: {str(e)}")
                results["details"]["huggingface"] = None
        
        # VADER sentiment analysis
        try:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            results["details"]["vader"] = vader_scores
        except Exception as e:
            print(f"Warning: VADER analysis failed: {str(e)}")
            results["details"]["vader"] = None
        
        # TextBlob sentiment analysis
        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            results["details"]["textblob"] = {
                "polarity": textblob_polarity,
                "subjectivity": textblob_subjectivity
            }
        except Exception as e:
            print(f"Warning: TextBlob analysis failed: {str(e)}")
            results["details"]["textblob"] = None
        
        # Aggregate results
        aggregated = self._aggregate_sentiment_results(results["details"])
        results.update(aggregated)
        
        analysis_time = time.time() - start_time
        results["processing_time"] = analysis_time
        
        print(f"Sentiment analysis completed in {analysis_time:.2f}s")
        return results
    
    def _aggregate_sentiment_results(self, details: Dict) -> Dict:
        """
        Aggregate results from multiple sentiment analysis models
        
        Args:
            details: Dictionary containing results from different models
            
        Returns:
            Aggregated sentiment results
        """
        scores = []
        labels = []
        
        # Process Hugging Face results
        if details.get("huggingface"):
            hf_result = details["huggingface"]
            if isinstance(hf_result, list) and len(hf_result) > 0:
                result = hf_result[0]
                label = result.get("label", "").lower()
                score = result.get("score", 0.0)
                
                # Map labels to standard format
                if "positive" in label:
                    labels.append("positive")
                    scores.append(score)
                elif "negative" in label:
                    labels.append("negative")
                    scores.append(-score)
                elif "neutral" in label:
                    labels.append("neutral")
                    scores.append(0.0)
        
        # Process VADER results
        if details.get("vader"):
            vader_scores = details["vader"]
            compound = vader_scores.get("compound", 0.0)
            
            if compound >= 0.05:
                labels.append("positive")
                scores.append(compound)
            elif compound <= -0.05:
                labels.append("negative")
                scores.append(compound)
            else:
                labels.append("neutral")
                scores.append(compound)
        
        # Process TextBlob results
        if details.get("textblob"):
            textblob_result = details["textblob"]
            polarity = textblob_result.get("polarity", 0.0)
            
            if polarity > 0.1:
                labels.append("positive")
                scores.append(polarity)
            elif polarity < -0.1:
                labels.append("negative")
                scores.append(polarity)
            else:
                labels.append("neutral")
                scores.append(polarity)
        
        # Determine final sentiment
        if not scores:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0
            }
        
        # Calculate weighted average
        avg_score = np.mean(scores)
        
        # Determine sentiment label
        if avg_score > 0.1:
            sentiment = "positive"
        elif avg_score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        # Calculate confidence based on agreement
        if len(set(labels)) == 1:
            confidence = 0.9  # All models agree
        elif len(set(labels)) == 2:
            confidence = 0.6  # Some disagreement
        else:
            confidence = 0.3  # High disagreement
        
        return {
            "sentiment": sentiment,
            "score": avg_score,
            "confidence": confidence
        }
    
    def analyze_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for multiple text segments
        
        Args:
            segments: List of text segments with timing information
            
        Returns:
            List of segments with sentiment analysis
        """
        results = []
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if text:
                sentiment_result = self.analyze_sentiment(text)
                
                # Add segment info to result
                segment_result = {
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": text,
                    "sentiment": sentiment_result["sentiment"],
                    "score": sentiment_result["score"],
                    "confidence": sentiment_result["confidence"],
                    "details": sentiment_result["details"]
                }
                results.append(segment_result)
        
        return results
    
    def get_sentiment_summary(self, segments: List[Dict]) -> Dict:
        """
        Generate summary statistics for sentiment analysis of segments
        
        Args:
            segments: List of segments with sentiment analysis
            
        Returns:
            Summary statistics
        """
        if not segments:
            return {
                "overall_sentiment": "neutral",
                "average_score": 0.0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "total_segments": 0
            }
        
        scores = [seg["score"] for seg in segments]
        sentiments = [seg["sentiment"] for seg in segments]
        
        # Calculate overall sentiment
        avg_score = np.mean(scores)
        if avg_score > 0.1:
            overall_sentiment = "positive"
        elif avg_score < -0.1:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        # Calculate sentiment distribution
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for sentiment in sentiments:
            sentiment_counts[sentiment] += 1
        
        return {
            "overall_sentiment": overall_sentiment,
            "average_score": avg_score,
            "sentiment_distribution": sentiment_counts,
            "total_segments": len(segments),
            "score_range": {
                "min": min(scores),
                "max": max(scores)
            }
        }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded models
        
        Returns:
            Dictionary with model information
        """
        return {
            "huggingface_model": self.model_name,
            "huggingface_loaded": self.classifier is not None,
            "vader_available": True,
            "textblob_available": True,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        } 


class SentimentAnalyzerMultiTool:
    def __init__(self):
        # English
        self.sia = SentimentIntensityAnalyzer()
        # Mandarin
        self.cnsenti_sentiment = CNSentiSentiment()
        self.cnsenti_emotion = CNSentiEmotion()
        self.cntext = CnText()
        # Malay
        self.malaya_model = malaya.sentiment.huggingface()
    
    def analyze(self, text: str, language: str) -> Dict[str, Any]:
        if language == 'en':
            return self.analyze_english(text)
        elif language == 'zh':
            return self.analyze_mandarin(text)
        elif language == 'ms':
            return self.analyze_malay(text)
        else:
            return {'error': f'Unsupported language: {language}'}
    
    def analyze_english(self, text: str) -> Dict[str, Any]:
        score = self.sia.polarity_scores(text)
        return {'tool': 'VADER', 'sentiment': score}
    
    def analyze_mandarin(self, text: str) -> Dict[str, Any]:
        senti_result = self.cnsenti_sentiment.sentiment_count(text)
        emotion_result = self.cnsenti_emotion.emotion_count(text)
        cntext_result = self.cntext.sentiment(text)
        return {
            'tool': 'cnsenti+cntext',
            'cnsenti_sentiment': senti_result,
            'cnsenti_emotion': emotion_result,
            'cntext': cntext_result
        }
    
    def analyze_malay(self, text: str) -> Dict[str, Any]:
        result = self.malaya_model.predict([text])
        return {'tool': 'malaya', 'sentiment': result[0]}

# Example usage
def _demo():
    analyzer = SentimentAnalyzerMultiTool()
    print('Mandarin:')
    print(analyzer.analyze("我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心", 'zh'))
    print('Malay:')
    print(analyzer.analyze("Saya sangat gembira hari ini!", 'ms'))
    print('English:')
    print(analyzer.analyze("I'm absolutely chuffed to bits!", 'en'))

if __name__ == '__main__':
    _demo() 