import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uvicorn

from app.database import get_db, create_tables
from app.sentiment_service import SentimentAnalysisService
from app.config import settings

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Audio sentiment analysis system using ML and NLP",
    version="1.0.0"
)

# Initialize service
sentiment_service = SentimentAnalysisService()

# Pydantic models for API responses
class AnalysisResponse(BaseModel):
    analysis_id: int
    filename: str
    duration: float
    transcription: dict
    sentiment: dict
    summary: dict
    segments: List[dict]
    processing_time: float

class AnalysisListResponse(BaseModel):
    analyses: List[dict]
    total: int
    limit: int
    offset: int

class StatisticsResponse(BaseModel):
    total_analyses: int
    sentiment_distribution: dict
    average_sentiment_score: float

class SystemStatusResponse(BaseModel):
    audio_processor: dict
    transcription: dict
    sentiment_analysis: dict
    storage: dict


@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    create_tables()
    print("Database tables created successfully")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Sentiment Analysis System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    save_processed_audio: bool = Query(True, description="Save processed audio file"),
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
        upload_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Analyze audio
        result = sentiment_service.analyze_audio_file(
            upload_path, db, save_processed_audio
        )
        
        return result
        
    except Exception as e:
        # Clean up uploaded file on error
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyses", response_model=AnalysisListResponse)
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


@app.get("/analyses/{analysis_id}", response_model=dict)
async def get_analysis_by_id(
    analysis_id: int,
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


@app.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    """Get overall sentiment statistics"""
    try:
        stats = sentiment_service.get_sentiment_statistics(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status and model information"""
    try:
        status = sentiment_service.get_system_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: int,
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


@app.get("/health")
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