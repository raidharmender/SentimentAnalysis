#!/usr/bin/env python3
"""
Example usage of the Sentiment Analysis System
This script demonstrates how to use the system programmatically
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.sentiment_service import SentimentAnalysisService
from app.database import get_db, create_tables


def create_sample_audio():
    """Create a sample audio file for testing"""
    # Generate a simple sine wave (440 Hz for 3 seconds)
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a simple tone
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # Add some noise
    noise = 0.05 * np.random.randn(len(audio))
    audio = audio + noise
    
    return audio, sample_rate


def example_basic_usage():
    """Example of basic usage"""
    print("🎤 Sentiment Analysis System - Basic Usage Example")
    print("=" * 50)
    
    # Create database tables
    create_tables()
    
    # Initialize service
    service = SentimentAnalysisService()
    
    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        audio, sample_rate = create_sample_audio()
        sf.write(tmp_file.name, audio, sample_rate)
        audio_path = tmp_file.name
    
    try:
        print(f"Created sample audio file: {audio_path}")
        
        # Get database session
        db = next(get_db())
        
        # Analyze the audio file
        print("\n🔍 Analyzing audio file...")
        result = service.analyze_audio_file(audio_path, db)
        
        # Display results
        print("\n📊 Analysis Results:")
        print(f"Analysis ID: {result['analysis_id']}")
        print(f"Filename: {result['filename']}")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        
        print(f"\n📝 Transcription:")
        print(f"Text: {result['transcription']['text']}")
        print(f"Language: {result['transcription']['language']}")
        print(f"Confidence: {result['transcription']['confidence']:.2%}")
        
        print(f"\n😊 Sentiment Analysis:")
        print(f"Overall Sentiment: {result['sentiment']['overall_sentiment'].title()}")
        print(f"Score: {result['sentiment']['score']:.3f}")
        print(f"Confidence: {result['sentiment']['confidence']:.2%}")
        
        print(f"\n📈 Summary:")
        summary = result['summary']
        print(f"Overall Sentiment: {summary['overall_sentiment'].title()}")
        print(f"Average Score: {summary['average_score']:.3f}")
        print(f"Total Segments: {summary['total_segments']}")
        
        # Display sentiment distribution
        dist = summary['sentiment_distribution']
        print(f"Sentiment Distribution:")
        for sentiment, count in dist.items():
            percentage = (count / summary['total_segments'] * 100) if summary['total_segments'] > 0 else 0
            print(f"  {sentiment.title()}: {count} ({percentage:.1f}%)")
        
        # Get system status
        print(f"\n🔧 System Status:")
        status = service.get_system_status()
        print(f"Audio Processor: {status['audio_processor']['target_sample_rate']} Hz")
        print(f"Whisper Model: {status['transcription']['model_name']}")
        print(f"Sentiment Model: {status['sentiment_analysis']['huggingface_model']}")
        
        # Get statistics
        print(f"\n📊 System Statistics:")
        stats = service.get_sentiment_statistics(db)
        print(f"Total Analyses: {stats['total_analyses']}")
        print(f"Average Sentiment Score: {stats['average_sentiment_score']:.3f}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        db.close()


def example_text_analysis():
    """Example of text-only sentiment analysis"""
    print("\n🎤 Text Sentiment Analysis Example")
    print("=" * 40)
    
    from app.sentiment_analyzer import SentimentAnalyzer
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Sample texts
    texts = [
        "I absolutely love this product! It's amazing and works perfectly!",
        "This is the worst experience I've ever had. Terrible service!",
        "The product is okay, nothing special but it works.",
        "I'm very satisfied with the customer support and the quality of the product.",
        "This is completely broken and useless. I want my money back!"
    ]
    
    print("Analyzing sample texts...")
    
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. Text: {text}")
        result = analyzer.analyze_sentiment(text)
        
        print(f"   Sentiment: {result['sentiment'].title()}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Confidence: {result['confidence']:.2%}")
        
        # Show model details
        if result['details'].get('vader'):
            vader = result['details']['vader']
            print(f"   VADER: compound={vader['compound']:.3f}, pos={vader['pos']:.3f}, neg={vader['neg']:.3f}")


def example_batch_analysis():
    """Example of batch analysis"""
    print("\n🎤 Batch Analysis Example")
    print("=" * 30)
    
    # Create database tables
    create_tables()
    
    # Initialize service
    service = SentimentAnalysisService()
    db = next(get_db())
    
    # Create multiple sample audio files
    audio_files = []
    for i in range(3):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            audio, sample_rate = create_sample_audio()
            sf.write(tmp_file.name, audio, sample_rate)
            audio_files.append(tmp_file.name)
    
    try:
        print(f"Created {len(audio_files)} sample audio files")
        
        # Analyze all files
        results = []
        for i, audio_path in enumerate(audio_files, 1):
            print(f"\nAnalyzing file {i}/{len(audio_files)}...")
            result = service.analyze_audio_file(audio_path, db)
            results.append(result)
        
        # Display batch results
        print(f"\n📊 Batch Analysis Results:")
        print(f"Total files analyzed: {len(results)}")
        
        sentiments = [r['sentiment']['overall_sentiment'] for r in results]
        scores = [r['sentiment']['score'] for r in results]
        
        print(f"Sentiments: {sentiments}")
        print(f"Average Score: {np.mean(scores):.3f}")
        print(f"Score Range: {min(scores):.3f} to {max(scores):.3f}")
        
        # Get all analyses from database
        all_analyses = service.get_all_analyses(db, limit=10)
        print(f"\nTotal analyses in database: {len(all_analyses)}")
        
    finally:
        # Clean up temporary files
        for audio_path in audio_files:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        db.close()


def main():
    """Run all examples"""
    print("🎤 Sentiment Analysis System - Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_text_analysis()
        example_batch_analysis()
        
        print("\n✅ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Start the API server: python main.py --mode api")
        print("2. Start the dashboard: python main.py --mode dashboard")
        print("3. Upload real audio files and analyze them!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        print("Make sure you have installed all dependencies:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main() 