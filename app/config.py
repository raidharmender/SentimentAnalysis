"""
Application configuration using Pydantic settings.
Centralized configuration management with environment variable support.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

from app.constants import (
    AudioConfig, WhisperConfig, SentimentConfig, 
    APIConfig, DatabaseConfig, PathConfig, 
    ValidationConfig, PerformanceConfig
)


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    api_v1_str: str = Field(default="/api/v1", description="API version string")
    project_name: str = Field(default="Sentiment Analysis System", description="Project name")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=APIConfig.DEFAULT_PORT, description="API server port")
    dashboard_port: int = Field(default=APIConfig.DASHBOARD_PORT, description="Dashboard port")
    
    # Database Configuration
    database_url: str = Field(
        default=DatabaseConfig.DEFAULT_DB_URL, 
        description="Database connection URL"
    )
    
    # Audio Processing Configuration
    target_sample_rate: int = Field(
        default=AudioConfig.TARGET_SAMPLE_RATE, 
        description="Target audio sample rate"
    )
    target_channels: int = Field(
        default=AudioConfig.TARGET_CHANNELS, 
        description="Target audio channels (1=mono, 2=stereo)"
    )
    max_file_size: int = Field(
        default=AudioConfig.MAX_FILE_SIZE, 
        description="Maximum allowed file size in bytes"
    )
    allowed_audio_formats: List[str] = Field(
        default=list(AudioConfig.ALLOWED_FORMATS), 
        description="Allowed audio file formats"
    )
    
    # Whisper Configuration
    whisper_model: str = Field(
        default=WhisperConfig.DEFAULT_MODEL, 
        description="Whisper model size"
    )
    whisper_language: str = Field(
        default=WhisperConfig.LANGUAGE_AUTO, 
        description="Whisper language setting"
    )
    
    # Sentiment Analysis Configuration
    sentiment_model: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest", 
        description="HuggingFace sentiment model"
    )
    positive_threshold: float = Field(
        default=SentimentConfig.POSITIVE_THRESHOLD, 
        description="Positive sentiment threshold"
    )
    negative_threshold: float = Field(
        default=SentimentConfig.NEGATIVE_THRESHOLD, 
        description="Negative sentiment threshold"
    )
    
    # Storage Configuration
    upload_dir: str = Field(
        default=PathConfig.UPLOAD_DIR, 
        description="Upload directory path"
    )
    processed_dir: str = Field(
        default=PathConfig.PROCESSED_DIR, 
        description="Processed files directory path"
    )
    
    # Performance Configuration
    batch_size: int = Field(
        default=PerformanceConfig.BATCH_SIZE, 
        description="Processing batch size"
    )
    max_workers: int = Field(
        default=PerformanceConfig.MAX_WORKERS, 
        description="Maximum worker threads"
    )
    request_timeout: int = Field(
        default=PerformanceConfig.REQUEST_TIMEOUT, 
        description="Request timeout in seconds"
    )
    
    # Validation Configuration
    min_text_length: int = Field(
        default=ValidationConfig.MIN_TEXT_LENGTH, 
        description="Minimum text length for analysis"
    )
    max_text_length: int = Field(
        default=ValidationConfig.MAX_TEXT_LENGTH, 
        description="Maximum text length for analysis"
    )
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug or os.getenv("ENVIRONMENT", "development") == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.is_development
    
    def get_database_url(self) -> str:
        """Get database URL with environment override."""
        return os.getenv("DATABASE_URL", self.database_url)
    
    def get_whisper_model(self) -> str:
        """Get Whisper model with validation."""
        model = os.getenv("WHISPER_MODEL", self.whisper_model)
        if model not in WhisperConfig.AVAILABLE_MODELS:
            raise ValueError(f"Invalid Whisper model: {model}")
        return model
    
    def get_allowed_formats(self) -> set:
        """Get allowed audio formats as a set."""
        return set(self.allowed_audio_formats)
    
    def validate_config(self) -> None:
        """Validate configuration settings."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Invalid port number")
        
        if self.max_file_size <= 0:
            raise ValueError("Max file size must be positive")
        
        if self.target_sample_rate <= 0:
            raise ValueError("Target sample rate must be positive")
        
        if self.target_channels not in {1, 2}:
            raise ValueError("Target channels must be 1 (mono) or 2 (stereo)")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate_config()
except ValueError as e:
    print(f"Configuration error: {e}")
    raise 