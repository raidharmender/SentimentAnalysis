import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import uvicorn

from app.database import get_db, create_tables
from app.sentiment_service import SentimentAnalysisService
from app.config import settings

# Create FastAPI app with enhanced OpenAPI documentation
app = FastAPI(
    title=settings.project_name,
    description="""
# 🎤 Sentiment Analysis System API

A comprehensive API for analyzing customer sentiment from audio files using Machine Learning and Natural Language Processing.

## Features

- **Audio Processing**: Upload and analyze WAV, MP3, FLAC, and M4A files
- **Speech-to-Text**: Powered by OpenAI Whisper for accurate transcription
- **Sentiment Analysis**: Multi-model approach using Hugging Face transformers, VADER, and TextBlob
- **Real-time Analysis**: Get results immediately after processing
- **Statistics**: View overall sentiment trends and distributions

## Quick Start

1. **Upload Audio**: Use the `/analyze` endpoint to upload and analyze audio files
2. **Text Analysis**: Use `/analyze/text` for direct text sentiment analysis
3. **View Results**: Retrieve analysis results using `/analyses/{id}`
4. **Get Statistics**: Check overall statistics with `/statistics`

## Authentication

Currently, this API does not require authentication. For production use, consider implementing proper authentication.

## Rate Limiting

Be mindful of processing time for large audio files. The system processes files sequentially.
    """,
    version="1.0.0",
    contact={
        "name": "Sentiment Analysis Team",
        "email": "team@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check and system status endpoints"
        },
        {
            "name": "analysis",
            "description": "Audio and text sentiment analysis endpoints"
        },
        {
            "name": "results",
            "description": "Retrieve and manage analysis results"
        },
        {
            "name": "statistics",
            "description": "System statistics and analytics"
        }
    ]
)

# Initialize service
sentiment_service = SentimentAnalysisService()

# Pydantic models for API responses
class AnalysisResponse(BaseModel):
    """Response model for audio analysis results"""
    analysis_id: int = Field(..., description="Unique analysis identifier")
    filename: str = Field(..., description="Original filename")
    duration: float = Field(..., description="Audio duration in seconds")
    transcription: dict = Field(..., description="Speech-to-text results")
    sentiment: dict = Field(..., description="Sentiment analysis results")
    summary: dict = Field(..., description="Analysis summary")
    segments: List[dict] = Field(..., description="Time-based sentiment segments")
    processing_time: float = Field(..., description="Total processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": 1,
                "filename": "customer_call.wav",
                "duration": 45.2,
                "transcription": {
                    "text": "Hello, I'm calling about my recent order...",
                    "language": "en",
                    "confidence": 0.92
                },
                "sentiment": {
                    "overall_sentiment": "positive",
                    "score": 0.65,
                    "confidence": 0.85
                },
                "summary": {
                    "overall_sentiment": "positive",
                    "average_score": 0.65,
                    "total_segments": 13
                },
                "segments": [],
                "processing_time": 3.45
            }
        }

class AnalysisListResponse(BaseModel):
    """Response model for paginated analysis list"""
    analyses: List[dict] = Field(..., description="List of analysis results")
    total: int = Field(..., description="Total number of analyses")
    limit: int = Field(..., description="Number of results per page")
    offset: int = Field(..., description="Number of results skipped")
    
    class Config:
        json_schema_extra = {
            "example": {
                "analyses": [
                    {
                        "analysis_id": 1,
                        "filename": "call1.wav",
                        "sentiment": "positive",
                        "duration": 30.5
                    }
                ],
                "total": 1,
                "limit": 10,
                "offset": 0
            }
        }

class StatisticsResponse(BaseModel):
    """Response model for system statistics"""
    total_analyses: int = Field(..., description="Total number of analyses performed")
    sentiment_distribution: dict = Field(..., description="Distribution of sentiment labels")
    average_sentiment_score: float = Field(..., description="Average sentiment score across all analyses")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_analyses": 25,
                "sentiment_distribution": {
                    "positive": 15,
                    "negative": 5,
                    "neutral": 5
                },
                "average_sentiment_score": 0.45
            }
        }

class SystemStatusResponse(BaseModel):
    """Response model for system status information"""
    audio_processor: dict = Field(..., description="Audio processing configuration")
    transcription: dict = Field(..., description="Speech-to-text model status")
    sentiment_analysis: dict = Field(..., description="Sentiment analysis model status")
    storage: dict = Field(..., description="Storage configuration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_processor": {
                    "target_sample_rate": 16000,
                    "target_channels": 1
                },
                "transcription": {
                    "model_name": "base",
                    "status": "loaded"
                },
                "sentiment_analysis": {
                    "huggingface_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "vader_available": True,
                    "textblob_available": True
                },
                "storage": {
                    "upload_dir": "uploads",
                    "processed_dir": "processed"
                }
            }
        }


@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    create_tables()
    print("Database tables created successfully")


@app.get("/", response_model=dict, tags=["health"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sentiment Analysis System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "/status"
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["analysis"])
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file to analyze (WAV, MP3, FLAC, M4A)"),
    save_processed_audio: bool = Query(True, description="Save processed audio file"),
    language: Optional[str] = Query(None, description="Language code (e.g., 'en', 'cy', 'auto'). If 'auto', will attempt to detect language with improved English accent handling"),
    db: Session = Depends(get_db)
):
    """
    Analyze sentiment from uploaded audio file
    
    Supports WAV, MP3, FLAC, and M4A formats
    """
    # Validate file format
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in settings.ALLOWED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed formats: {settings.ALLOWED_AUDIO_FORMATS}"
        )
    
    # Validate file size
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
        )
    
    try:
        # Save uploaded file
        upload_path = os.path.join(settings.upload_dir, file.filename)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze audio
        result = sentiment_service.analyze_audio_file(
            upload_path, db, save_processed_audio, language
        )
        
        return result
        
    except Exception as e:
        # Clean up uploaded file on error
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyses", response_model=AnalysisListResponse, tags=["results"])
async def get_analyses(
    limit: int = Query(100, ge=1, le=1000, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: Session = Depends(get_db)
):
    """Get list of all analyses with pagination"""
    try:
        analyses = sentiment_service.get_all_analyses(db, limit, offset)
        total = db.query(sentiment_service.models.AudioAnalysis).count()
        
        return {
            "analyses": analyses,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyses/{analysis_id}", response_model=dict, tags=["results"])
async def get_analysis_by_id(
    analysis_id: int = Path(..., description="Analysis ID to retrieve"),
    db: Session = Depends(get_db)
):
    """Get detailed analysis results by ID"""
    try:
        analysis = sentiment_service.get_analysis_by_id(db, analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics", response_model=StatisticsResponse, tags=["statistics"])
async def get_statistics(db: Session = Depends(get_db)):
    """Get overall sentiment statistics"""
    try:
        stats = sentiment_service.get_sentiment_statistics(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=SystemStatusResponse, tags=["health"])
async def get_system_status():
    """Get system status and model information"""
    try:
        status = sentiment_service.get_system_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/analyses/{analysis_id}", tags=["results"])
async def delete_analysis(
    analysis_id: int = Path(..., description="Analysis ID to delete"),
    db: Session = Depends(get_db)
):
    """Delete analysis by ID"""
    try:
        from app.models import AudioAnalysis, SpeakerSegment
        
        # Delete segments first
        db.query(SpeakerSegment).filter(
            SpeakerSegment.audio_analysis_id == analysis_id
        ).delete()
        
        # Delete main analysis
        analysis = db.query(AudioAnalysis).filter(AudioAnalysis.id == analysis_id).first()
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        db.delete(analysis)
        db.commit()
        
        return {"message": "Analysis deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/text", tags=["analysis"])
async def analyze_text_sentiment(text_data: dict):
    """
    Analyze sentiment from text input
    
    Expected format: {"text": "your text here"}
    """
    try:
        text = text_data.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Analyze sentiment
        result = sentiment_service.sentiment_analyzer.analyze_sentiment(text)
        
        return {
            "text": text,
            "sentiment": {
                "overall_sentiment": result["sentiment"],
                "score": result["score"],
                "confidence": result["confidence"]
            },
            "details": result["details"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/languages", tags=["analysis"])
async def get_supported_languages():
    """Get list of supported languages for transcription"""
    try:
        languages = sentiment_service.transcription_service.get_supported_languages()
        
        # Common language mappings
        language_names = {
            "en": "English",
            "cy": "Welsh",
            "ga": "Irish",
            "gd": "Scottish Gaelic",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            "bn": "Bengali",
            "ur": "Urdu",
            "tr": "Turkish",
            "pl": "Polish",
            "sv": "Swedish",
            "da": "Danish",
            "no": "Norwegian",
            "fi": "Finnish",
            "hu": "Hungarian",
            "cs": "Czech",
            "sk": "Slovak",
            "ro": "Romanian",
            "bg": "Bulgarian",
            "hr": "Croatian",
            "sr": "Serbian",
            "sl": "Slovenian",
            "et": "Estonian",
            "lv": "Latvian",
            "lt": "Lithuanian",
            "mt": "Maltese",
            "el": "Greek",
            "he": "Hebrew",
            "th": "Thai",
            "vi": "Vietnamese",
            "id": "Indonesian",
            "ms": "Malay",
            "tl": "Tagalog",
            "fa": "Persian",
            "uk": "Ukrainian",
            "be": "Belarusian",
            "mk": "Macedonian",
            "sq": "Albanian",
            "bs": "Bosnian",
            "me": "Montenegrin",
            "is": "Icelandic",
            "fo": "Faroese",
            "gl": "Galician",
            "ca": "Catalan",
            "eu": "Basque",
            "ast": "Asturian",
            "oc": "Occitan",
            "co": "Corsican",
            "fur": "Friulian",
            "sc": "Sardinian",
            "vec": "Venetian",
            "lmo": "Lombard",
            "pms": "Piedmontese",
            "rm": "Romansh",
            "wa": "Walloon",
            "lb": "Luxembourgish",
            "als": "Alemannic",
            "bar": "Bavarian",
            "ksh": "Colognian",
            "pfl": "Palatine German",
            "swg": "Swabian",
            "gsw": "Swiss German",
            "nds": "Low German",
            "stq": "Saterland Frisian",
            "fy": "West Frisian",
            "li": "Limburgish",
            "zea": "Zeelandic",
            "af": "Afrikaans",
            "zu": "Zulu",
            "xh": "Xhosa",
            "st": "Southern Sotho",
            "tn": "Tswana",
            "ts": "Tsonga",
            "ss": "Swati",
            "ve": "Venda",
            "nr": "Southern Ndebele",
            "sn": "Shona",
            "ny": "Chichewa",
            "lg": "Ganda",
            "sw": "Swahili",
            "am": "Amharic",
            "om": "Oromo",
            "so": "Somali",
            "ti": "Tigrinya",
            "ha": "Hausa",
            "yo": "Yoruba",
            "ig": "Igbo",
            "ff": "Fula",
            "wo": "Wolof",
            "bm": "Bambara",
            "dy": "Dyula",
            "sg": "Sango",
            "ln": "Lingala",
            "kg": "Kongo",
            "rw": "Kinyarwanda",
            "rn": "Kirundi",
            "ak": "Akan",
            "tw": "Twi",
            "ee": "Ewe",
            "fon": "Fon",
            "yo": "Yoruba",
            "ig": "Igbo",
            "ff": "Fula",
            "wo": "Wolof",
            "bm": "Bambara",
            "dy": "Dyula",
            "sg": "Sango",
            "ln": "Lingala",
            "kg": "Kongo",
            "rw": "Kinyarwanda",
            "rn": "Kirundi",
            "ak": "Akan",
            "tw": "Twi",
            "ee": "Ewe",
            "fon": "Fon"
        }
        
        # Format response
        formatted_languages = []
        for lang_code in languages:
            formatted_languages.append({
                "code": lang_code,
                "name": language_names.get(lang_code, lang_code.title())
            })
        
        return {
            "languages": formatted_languages,
            "total": len(formatted_languages),
            "auto_detect": "auto"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sentiment-analysis"}


if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 