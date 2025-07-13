"""
Constants and configuration values for the Sentiment Analysis System.
Centralized management of all hardcoded values and magic numbers.
"""

from enum import Enum
from typing import Dict, List, Tuple

# Audio Processing Constants
class AudioConfig:
    """Audio processing configuration constants."""
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_FORMATS = {".wav", ".mp3", ".flac", ".m4a", ".opus"}
    SILENCE_THRESHOLD = -40  # dB
    MIN_SILENCE_DURATION = 0.5  # seconds
    NOISE_REDUCE_STATIONARY = True
    NOISE_REDUCE_PROP_DECAY = 0.95

# Whisper Model Constants
class WhisperConfig:
    """Whisper transcription configuration."""
    DEFAULT_MODEL = "base"
    AVAILABLE_MODELS = {"tiny", "base", "small", "medium", "large"}
    LANGUAGE_AUTO = "auto"
    TASK_TRANSCRIBE = "transcribe"
    TASK_TRANSLATE = "translate"

# Sentiment Analysis Constants
class SentimentConfig:
    """Sentiment analysis thresholds and configurations."""
    POSITIVE_THRESHOLD = 0.3  # Very conservative threshold for positive classification
    NEGATIVE_THRESHOLD = -0.3  # Very conservative threshold for negative classification
    NEUTRAL_THRESHOLD = 0.1
    CONFIDENCE_HIGH = 0.9
    CONFIDENCE_MEDIUM = 0.6
    CONFIDENCE_LOW = 0.3
    SNOWNLP_NEUTRAL_THRESHOLD = 0.5
    MALAY_POSITIVE_SCORE = 0.5
    MALAY_NEGATIVE_SCORE = -0.5

# Emotion Detection Constants
class EmotionConfig:
    """Emotion detection configuration."""
    EMOTIONS = {
        "joy": ["happy", "excited", "enthusiastic", "joyful", "cheerful"],
        "anger": ["angry", "frustrated", "irritated", "mad", "furious"],
        "sadness": ["sad", "depressed", "melancholy", "sorrowful", "grief"],
        "fear": ["afraid", "anxious", "worried", "scared", "terrified"],
        "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned"],
        "disgust": ["disgusted", "repulsed", "revolted", "appalled", "sickened"],
        "neutral": ["calm", "indifferent", "balanced", "composed", "serene"]
    }
    
    AUDIO_FEATURE_WEIGHTS = {
        "mfcc": 0.3,
        "prosodic": 0.25,
        "spectral": 0.2,
        "chroma": 0.15,
        "harmonic": 0.1
    }
    
    TEXT_WEIGHT = 0.4
    AUDIO_WEIGHT = 0.6

# Language Detection Constants
class LanguageConfig:
    """Language detection and mapping constants."""
    LANGUAGE_CODES = {
        "english": "en",
        "mandarin": "zh",
        "malay": "ms",
        "auto": "auto"
    }
    
    FILENAME_PATTERNS = {
        "english": ["M_", "F_"],
        "mandarin": ["chinese/"]
    }
    
    WHISPER_CONFIDENCE_THRESHOLD = 0.8
    ACCENT_CORRECTION_THRESHOLD = 0.6

# API Constants
class APIConfig:
    """API configuration and response constants."""
    DEFAULT_PORT = 8000
    DASHBOARD_PORT = 8501
    HEALTH_ENDPOINT = "/health"
    STATUS_ENDPOINT = "/status"
    ANALYZE_ENDPOINT = "/analyze"
    TEXT_ENDPOINT = "/analyze/text"
    STATISTICS_ENDPOINT = "/statistics"
    LANGUAGES_ENDPOINT = "/languages"
    
    RESPONSE_KEYS = {
        "analysis_id": "analysis_id",
        "filename": "filename",
        "duration": "duration",
        "transcription": "transcription",
        "sentiment": "sentiment",
        "summary": "summary",
        "segments": "segments",
        "processing_time": "processing_time"
    }

# Database Constants
class DatabaseConfig:
    """Database configuration and table constants."""
    DEFAULT_DB_URL = "sqlite:///./sentiment_analysis.db"
    POSTGRES_DB_URL = "postgresql://user:password@localhost/sentiment_db"
    
    TABLE_NAMES = {
        "audio_analysis": "audio_analysis",
        "speaker_segment": "speaker_segment"
    }
    
    DEFAULT_LIMIT = 100
    DEFAULT_OFFSET = 0

# File Path Constants
class PathConfig:
    """File and directory path constants."""
    UPLOAD_DIR = "uploads"
    PROCESSED_DIR = "processed"
    DOCS_DIR = "docs"
    TESTS_DIR = "tests"
    
    @classmethod
    def get_processed_filename(cls, original_filename: str) -> str:
        """Generate processed filename from original."""
        name, ext = original_filename.rsplit('.', 1)
        return f"{name}_processed.{ext}"

# Sentiment Tool Mapping
class SentimentTools:
    """Mapping of languages to sentiment analysis tools."""
    TOOL_MAPPING = {
        "en": "vader",
        "zh": "snownlp+cntext",
        "ms": "malaya",
        "default": "vader"
    }
    
    @classmethod
    def get_tool_for_language(cls, language: str) -> str:
        """Get the appropriate sentiment tool for a given language."""
        return cls.TOOL_MAPPING.get(language, cls.TOOL_MAPPING["default"])

# Error Messages
class ErrorMessages:
    """Centralized error message constants."""
    FILE_TOO_LARGE = "File size exceeds maximum allowed size of {}MB"
    UNSUPPORTED_FORMAT = "Unsupported file format. Allowed formats: {}"
    TRANSCRIPTION_FAILED = "Transcription failed: {}"
    SENTIMENT_ANALYSIS_FAILED = "Sentiment analysis failed: {}"
    EMOTION_DETECTION_FAILED = "Emotion detection failed: {}"
    LANGUAGE_NOT_SUPPORTED = "Language '{}' is not supported"
    MODEL_LOAD_FAILED = "Failed to load model: {}"
    DATABASE_ERROR = "Database operation failed: {}"

# Success Messages
class SuccessMessages:
    """Centralized success message constants."""
    ANALYSIS_COMPLETED = "Analysis completed successfully in {:.2f}s"
    FILE_UPLOADED = "File uploaded successfully"
    MODEL_LOADED = "Successfully loaded model: {}"
    DATABASE_CONNECTED = "Database connected successfully"

# Logging Constants
class LogConfig:
    """Logging configuration constants."""
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL = "INFO"
    LOG_FILE = "sentiment_analysis.log"
    
    LOGGERS = {
        "audio_processor": "audio_processor",
        "transcription": "transcription",
        "sentiment": "sentiment_analysis",
        "emotion": "emotion_detection",
        "api": "api",
        "dashboard": "dashboard"
    }

# Feature Extraction Constants
class FeatureConfig:
    """Audio feature extraction configuration."""
    MFCC_N_COEFFICIENTS = 13
    MFCC_DELTA_ORDER = 2
    SPECTRAL_ROLLOFF_PERCENTILE = 0.85
    CHROMA_N_CHROMA = 12
    HARMONIC_FILTER_SIZE = 31
    
    FEATURE_NAMES = [
        "mfcc_mean", "mfcc_std", "mfcc_delta_mean", "mfcc_delta_std",
        "spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
        "zero_crossing_rate", "chroma_mean", "chroma_std",
        "harmonic_ratio", "percussive_ratio"
    ]

# Validation Constants
class ValidationConfig:
    """Input validation constants."""
    MIN_TEXT_LENGTH = 1
    MAX_TEXT_LENGTH = 10000
    MIN_AUDIO_DURATION = 0.1  # seconds
    MAX_AUDIO_DURATION = 3600  # 1 hour
    
    @classmethod
    def is_valid_text(cls, text: str) -> bool:
        """Validate text input."""
        return (isinstance(text, str) and 
                cls.MIN_TEXT_LENGTH <= len(text.strip()) <= cls.MAX_TEXT_LENGTH)
    
    @classmethod
    def is_valid_audio_duration(cls, duration: float) -> bool:
        """Validate audio duration."""
        return cls.MIN_AUDIO_DURATION <= duration <= cls.MAX_AUDIO_DURATION

# Performance Constants
class PerformanceConfig:
    """Performance and optimization constants."""
    BATCH_SIZE = 32
    MAX_WORKERS = 4
    CACHE_TTL = 3600  # 1 hour
    REQUEST_TIMEOUT = 30  # seconds
    
    @classmethod
    def get_optimal_batch_size(cls, available_memory: int) -> int:
        """Calculate optimal batch size based on available memory."""
        return min(cls.BATCH_SIZE, max(1, available_memory // (1024 * 1024 * 100)))

# Enum for Processing States
class ProcessingState(Enum):
    """Enumeration of processing states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Enum for Sentiment Labels
class SentimentLabel(Enum):
    """Enumeration of sentiment labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

# Enum for Emotion Labels
class EmotionLabel(Enum):
    """Enumeration of emotion labels."""
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"

# Type Aliases for better code readability
AudioData = List[float]
AudioFeatures = Dict[str, float]
SentimentResult = Dict[str, any]
EmotionResult = Dict[str, any]
ProcessingResult = Dict[str, any] 