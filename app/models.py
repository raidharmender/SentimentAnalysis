from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class AudioAnalysis(Base):
    """Database model for storing audio analysis results"""
    
    __tablename__ = "audio_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    duration = Column(Float, nullable=True)
    
    # Transcription
    transcript = Column(Text, nullable=True)
    transcription_confidence = Column(Float, nullable=True)
    
    # Sentiment Analysis
    sentiment_label = Column(String(50), nullable=True)  # positive, negative, neutral
    sentiment_score = Column(Float, nullable=True)
    sentiment_details = Column(JSON, nullable=True)  # Detailed sentiment breakdown
    
    # Processing metadata
    processing_time = Column(Float, nullable=True)  # Total processing time in seconds
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<AudioAnalysis(id={self.id}, filename='{self.filename}', sentiment='{self.sentiment_label}')>"


class SpeakerSegment(Base):
    """Database model for storing speaker-specific segments"""
    
    __tablename__ = "speaker_segments"
    
    id = Column(Integer, primary_key=True, index=True)
    audio_analysis_id = Column(Integer, nullable=False)
    speaker_id = Column(String(50), nullable=True)  # Speaker identifier if available
    
    # Segment details
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    transcript = Column(Text, nullable=True)
    
    # Sentiment for this segment
    sentiment_label = Column(String(50), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    sentiment_details = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<SpeakerSegment(id={self.id}, speaker='{self.speaker_id}', sentiment='{self.sentiment_label}')>" 