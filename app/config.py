import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Sentiment Analysis System"
    
    # Database
    DATABASE_URL: str = "sqlite:///./sentiment_analysis.db"
    
    # Audio Processing
    TARGET_SAMPLE_RATE: int = 16000
    TARGET_CHANNELS: int = 1  # Mono
    
    # Whisper Model
    WHISPER_MODEL: str = "base"  # Options: tiny, base, small, medium, large
    
    # Sentiment Analysis
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # File Upload
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_AUDIO_FORMATS: list = [".wav", ".mp3", ".flac", ".m4a"]
    
    # Storage
    UPLOAD_DIR: str = "uploads"
    PROCESSED_DIR: str = "processed"
    
    class Config:
        env_file = ".env"


settings = Settings() 