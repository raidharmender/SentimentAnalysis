#!/usr/bin/env python3
"""
Debug Transcription Test

Simple script to test transcription on a single file and debug issues.
"""

import os
import sys
import traceback

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.transcription import TranscriptionService


def test_single_file():
    """Test transcription on a single file"""
    print("🔧 Initializing transcription service...")
    transcription_service = TranscriptionService()
    
    # Get first file from incoming folder
    incoming_folder = "incoming"
    wav_files = [f for f in os.listdir(incoming_folder) if f.endswith('.wav')]
    
    if not wav_files:
        print("❌ No WAV files found in incoming folder")
        return
    
    test_file = os.path.join(incoming_folder, wav_files[0])
    print(f"🎵 Testing file: {test_file}")
    print(f"📁 File size: {os.path.getsize(test_file) / (1024*1024):.1f} MB")
    
    try:
        print("🔍 Testing auto language detection...")
        result = transcription_service.transcribe_audio(test_file, language="auto")
        print("✅ Auto detection successful!")
        print(f"   Language: {result['language']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Transcript preview: {result['transcript'][:100]}...")
        
    except Exception as e:
        print(f"❌ Auto detection failed: {str(e)}")
        print("📋 Full traceback:")
        traceback.print_exc()
    
    try:
        print("\n🇬🇧 Testing English forced detection...")
        result = transcription_service.transcribe_audio(test_file, language="en")
        print("✅ English forced successful!")
        print(f"   Language: {result['language']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Transcript preview: {result['transcript'][:100]}...")
        
    except Exception as e:
        print(f"❌ English forced failed: {str(e)}")
        print("📋 Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    test_single_file() 