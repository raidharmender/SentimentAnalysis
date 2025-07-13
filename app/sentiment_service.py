import os
import time
import uuid
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from app.audio_processor import AudioProcessor
from app.transcription import TranscriptionService
from app.sentiment_analyzer import SentimentAnalyzerMultiTool
from app.models import AudioAnalysis, SpeakerSegment
from app.config import settings
from app.logging_config import get_sentiment_logger, get_audio_logger, get_database_logger


class SentimentAnalysisService:
    """Main service that orchestrates the entire sentiment analysis pipeline"""
    
    def __init__(self):
        """Initialize the sentiment analysis service"""
        self.audio_processor = AudioProcessor()
        self.transcription_service = TranscriptionService()
        self.sentiment_analyzer = SentimentAnalyzerMultiTool()
        
        # Initialize loggers
        self.logger = get_sentiment_logger()
        self.audio_logger = get_audio_logger()
        self.db_logger = get_database_logger()
        
        # Ensure directories exist
        os.makedirs(settings.upload_dir, exist_ok=True)
        os.makedirs(settings.processed_dir, exist_ok=True)
        
        self.logger.info("SentimentAnalysisService initialized successfully")
    
    def analyze_audio_file(self, file_path: str, db: Session, 
                          save_processed_audio: bool = True, language: str = None) -> Dict:
        """
        Complete pipeline for audio sentiment analysis
        
        Args:
            file_path: Path to the audio file
            db: Database session
            save_processed_audio: Whether to save processed audio file
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting analysis for: {file_path}")
            
            # Step 1: Audio Preprocessing
            self.logger.info("Step 1: Audio Preprocessing")
            self.audio_logger.info(f"Processing audio file: {file_path}")
            audio_data, sample_rate = self.audio_processor.process_audio(
                file_path,
                processing_steps={
                    'normalize_sample_rate': True,
                    'denoise': True,
                    'trim_silence': True,
                    'normalize_amplitude': True
                }
            )
            
            # Save processed audio if requested
            processed_audio_path = None
            if save_processed_audio:
                filename = os.path.basename(file_path)
                name, ext = os.path.splitext(filename)
                processed_filename = f"{name}_processed.wav"
                processed_audio_path = os.path.join(settings.processed_dir, processed_filename)
                self.audio_logger.info(f"Saving processed audio to: {processed_audio_path}")
                self.audio_processor.save_processed_audio(audio_data, sample_rate, processed_audio_path)
            
            # Step 2: Transcription
            self.logger.info("Step 2: Speech-to-Text Transcription")
            transcription_result = self.transcription_service.transcribe_audio_data(
                audio_data, sample_rate, language
            )
            
            # Step 3: Sentiment Analysis
            self.logger.info("Step 3: Sentiment Analysis")
            # Use detected language for sentiment analysis
            lang_code = transcription_result.get("language", "en")
            self.logger.info(f"Analyzing sentiment for language: {lang_code}")
            sentiment_result = self.sentiment_analyzer.analyze(
                transcription_result["transcript"], lang_code
            )
            
            # Step 4: Segment Analysis (if segments available)
            segment_results = []
            if transcription_result.get("segments"):
                self.logger.info("Step 4: Segment-level Analysis")
                self.logger.info(f"Analyzing {len(transcription_result['segments'])} segments")
                segment_results = self.sentiment_analyzer.analyze_segments(
                    transcription_result["segments"]
                )
            
            # Step 5: Generate Summary
            self.logger.info("Step 5: Generating Summary")
            summary = self.sentiment_analyzer.get_sentiment_summary(segment_results)
            
            # Step 6: Store Results
            self.logger.info("Step 6: Storing Results")
            analysis_record = self._store_analysis_results(
                db, file_path, processed_audio_path, transcription_result, 
                sentiment_result, summary, time.time() - start_time
            )
            
            # Store segment results
            if segment_results:
                self.logger.info(f"Storing {len(segment_results)} segment results")
                self._store_segment_results(db, analysis_record.id, segment_results)
            
            # Prepare final response
            response = {
                "analysis_id": analysis_record.id,
                "filename": os.path.basename(file_path),
                "duration": self.audio_processor.get_audio_duration(audio_data, sample_rate),
                "transcription": {
                    "text": transcription_result["transcript"],
                    "language": transcription_result["language"],
                    "confidence": transcription_result["confidence"],
                    "processing_time": transcription_result["processing_time"]
                },
                "sentiment": {
                    "overall_sentiment": sentiment_result["sentiment"],
                    "score": sentiment_result["score"],
                    "confidence": sentiment_result["confidence"],
                    "details": sentiment_result["details"]
                },
                "summary": summary,
                "segments": segment_results,
                "processing_time": time.time() - start_time
            }
            
            self.logger.info(f"Analysis completed successfully in {response['processing_time']:.2f}s")
            self.logger.info(f"Analysis ID: {analysis_record.id}, Sentiment: {sentiment_result['sentiment']}, Score: {sentiment_result['score']:.3f}")
            return response
            
        except Exception as e:
            self.logger.error(f"Analysis failed for file {file_path}: {str(e)}", exc_info=True)
            raise
    
    def _store_analysis_results(self, db: Session, file_path: str, 
                               processed_audio_path: str, transcription_result: Dict,
                               sentiment_result: Dict, summary: Dict, 
                               processing_time: float) -> AudioAnalysis:
        """
        Store analysis results in database
        
        Args:
            db: Database session
            file_path: Original file path
            processed_audio_path: Path to processed audio file
            transcription_result: Transcription results
            sentiment_result: Sentiment analysis results
            summary: Summary statistics
            processing_time: Total processing time
            
        Returns:
            AudioAnalysis record
        """
        self.db_logger.info(f"Storing analysis results for file: {os.path.basename(file_path)}")
        
        # Create analysis record
        analysis_record = AudioAnalysis(
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            duration=summary.get("total_segments", 0),  # Will be updated with actual duration
            
            # Transcription data
            transcript=transcription_result["transcript"],
            transcription_confidence=transcription_result["confidence"],
            
            # Sentiment data
            sentiment_label=sentiment_result["sentiment"],
            sentiment_score=sentiment_result["score"],
            sentiment_details=sentiment_result["details"],
            
            # Processing metadata
            processing_time=processing_time
        )
        
        try:
            db.add(analysis_record)
            db.commit()
            db.refresh(analysis_record)
            self.db_logger.info(f"Successfully stored analysis record with ID: {analysis_record.id}")
            return analysis_record
        except Exception as e:
            self.db_logger.error(f"Failed to store analysis results: {str(e)}", exc_info=True)
            db.rollback()
            raise
    
    def _store_segment_results(self, db: Session, analysis_id: int, 
                              segment_results: List[Dict]) -> None:
        """
        Store segment-level results in database
        
        Args:
            db: Database session
            analysis_id: ID of the main analysis record
            segment_results: List of segment analysis results
        """
        self.db_logger.info(f"Storing {len(segment_results)} segment results for analysis ID: {analysis_id}")
        
        try:
            for segment in segment_results:
                segment_record = SpeakerSegment(
                    audio_analysis_id=analysis_id,
                    start_time=segment["start"],
                    end_time=segment["end"],
                    transcript=segment["text"],
                    sentiment_label=segment["sentiment"],
                    sentiment_score=segment["score"],
                    sentiment_details=segment["details"]
                )
                db.add(segment_record)
            
            db.commit()
            self.db_logger.info(f"Successfully stored {len(segment_results)} segment records")
        except Exception as e:
            self.db_logger.error(f"Failed to store segment results: {str(e)}", exc_info=True)
            db.rollback()
            raise
    
    def get_analysis_by_id(self, db: Session, analysis_id: int) -> Optional[Dict]:
        """
        Retrieve analysis results by ID
        
        Args:
            db: Database session
            analysis_id: Analysis ID
            
        Returns:
            Analysis results or None
        """
        analysis = db.query(AudioAnalysis).filter(AudioAnalysis.id == analysis_id).first()
        if not analysis:
            return None
        
        # Get segment results
        segments = db.query(SpeakerSegment).filter(
            SpeakerSegment.audio_analysis_id == analysis_id
        ).all()
        
        return {
            "id": analysis.id,
            "filename": analysis.filename,
            "file_path": analysis.file_path,
            "file_size": analysis.file_size,
            "duration": analysis.duration,
            "transcript": analysis.transcript,
            "transcription_confidence": analysis.transcription_confidence,
            "sentiment_label": analysis.sentiment_label,
            "sentiment_score": analysis.sentiment_score,
            "sentiment_details": analysis.sentiment_details,
            "processing_time": analysis.processing_time,
            "created_at": analysis.created_at,
            "segments": [
                {
                    "id": seg.id,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "transcript": seg.transcript,
                    "sentiment_label": seg.sentiment_label,
                    "sentiment_score": seg.sentiment_score,
                    "sentiment_details": seg.sentiment_details
                }
                for seg in segments
            ]
        }
    
    def get_all_analyses(self, db: Session, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Get all analysis results with pagination
        
        Args:
            db: Database session
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of analysis results
        """
        analyses = db.query(AudioAnalysis).offset(offset).limit(limit).all()
        
        return [
            {
                "analysis_id": analysis.id,
                "filename": analysis.filename,
                "sentiment": analysis.sentiment_label,
                "duration": analysis.duration or 0,
                "processing_time": analysis.processing_time,
                "created_at": analysis.created_at.isoformat() if analysis.created_at else None
            }
            for analysis in analyses
        ]
    
    def get_sentiment_statistics(self, db: Session) -> Dict:
        """
        Get overall sentiment statistics
        
        Args:
            db: Database session
            
        Returns:
            Statistics summary
        """
        from sqlalchemy import func
        total_analyses = db.query(AudioAnalysis).count()
        
        if total_analyses == 0:
            return {
                "total_analyses": 0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "average_sentiment_score": 0.0
            }
        
        # Get sentiment distribution
        positive_count = db.query(AudioAnalysis).filter(
            AudioAnalysis.sentiment_label == "positive"
        ).count()
        
        negative_count = db.query(AudioAnalysis).filter(
            AudioAnalysis.sentiment_label == "negative"
        ).count()
        
        neutral_count = db.query(AudioAnalysis).filter(
            AudioAnalysis.sentiment_label == "neutral"
        ).count()
        
        # Calculate average sentiment score using SQL AVG
        avg_score = db.query(func.avg(AudioAnalysis.sentiment_score)).scalar() or 0.0
        
        return {
            "total_analyses": total_analyses,
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count
            },
            "average_sentiment_score": avg_score
        }
    
    def get_system_status(self) -> Dict:
        """
        Get system status and model information
        
        Returns:
            System status information
        """
        return {
            "audio_processor": {
                "target_sample_rate": self.audio_processor.target_sample_rate,
                "target_channels": self.audio_processor.target_channels
            },
            "transcription": self.transcription_service.get_model_info(),
            "sentiment_analysis": self.sentiment_analyzer.get_model_info(),
            "storage": {
                "upload_dir": settings.upload_dir,
                "processed_dir": settings.processed_dir
            }
        } 