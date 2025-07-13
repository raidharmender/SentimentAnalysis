#!/usr/bin/env python3
"""
Language Detection Test with Incoming Files

This script tests the improved language detection system on audio files
from the incoming folder to verify the enhanced English accent handling.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.transcription import TranscriptionService
from app.audio_processor import AudioProcessor


def test_language_detection_on_file(file_path: str, transcription_service: TranscriptionService) -> Dict:
    """
    Test language detection on a single audio file
    
    Args:
        file_path: Path to the audio file
        transcription_service: Initialized transcription service
        
    Returns:
        Dictionary with test results
    """
    print(f"🎵 Testing: {os.path.basename(file_path)}")
    
    try:
        # Test 1: Auto-detect with improved language detection
        start_time = time.time()
        result_auto = transcription_service.transcribe_audio(file_path, language="auto")
        auto_time = time.time() - start_time
        
        # Test 2: Force English detection
        start_time = time.time()
        result_english = transcription_service.transcribe_audio(file_path, language="en")
        english_time = time.time() - start_time
        
        # Test 3: No language specification (default behavior)
        start_time = time.time()
        result_none = transcription_service.transcribe_audio(file_path, language=None)
        none_time = time.time() - start_time
        
        return {
            "filename": os.path.basename(file_path),
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 1),
            "auto_detection": {
                "language": result_auto["language"],
                "confidence": result_auto["confidence"],
                "processing_time": round(auto_time, 2)
            },
            "english_forced": {
                "language": result_english["language"],
                "confidence": result_english["confidence"],
                "processing_time": round(english_time, 2)
            },
            "no_specification": {
                "language": result_none["language"],
                "confidence": result_none["confidence"],
                "processing_time": round(none_time, 2)
            },
            "transcript_preview": result_auto["transcript"][:100] + "..." if result_auto["transcript"] else "No transcript"
        }
        
    except Exception as e:
        return {
            "filename": os.path.basename(file_path),
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 1),
            "error": str(e),
            "auto_detection": {"language": "error", "confidence": 0.0, "processing_time": 0.0},
            "english_forced": {"language": "error", "confidence": 0.0, "processing_time": 0.0},
            "no_specification": {"language": "error", "confidence": 0.0, "processing_time": 0.0},
            "transcript_preview": "Error occurred"
        }


def analyze_results(results: List[Dict]) -> Dict:
    """
    Analyze the test results
    
    Args:
        results: List of test results
        
    Returns:
        Analysis summary
    """
    successful_tests = [r for r in results if "error" not in r]
    failed_tests = [r for r in results if "error" in r]
    
    if not successful_tests:
        return {"error": "No successful tests to analyze"}
    
    # Language detection statistics
    auto_languages = [r["auto_detection"]["language"] for r in successful_tests]
    english_languages = [r["english_forced"]["language"] for r in successful_tests]
    none_languages = [r["no_specification"]["language"] for r in successful_tests]
    
    # Count language occurrences
    auto_lang_counts = {}
    for lang in auto_languages:
        auto_lang_counts[lang] = auto_lang_counts.get(lang, 0) + 1
    
    english_lang_counts = {}
    for lang in english_languages:
        english_lang_counts[lang] = english_lang_counts.get(lang, 0) + 1
    
    none_lang_counts = {}
    for lang in none_languages:
        none_lang_counts[lang] = none_lang_counts.get(lang, 0) + 1
    
    # Calculate average processing times
    auto_times = [r["auto_detection"]["processing_time"] for r in successful_tests]
    english_times = [r["english_forced"]["processing_time"] for r in successful_tests]
    none_times = [r["no_specification"]["processing_time"] for r in successful_tests]
    
    # Calculate average confidence scores
    auto_confidences = [r["auto_detection"]["confidence"] for r in successful_tests]
    english_confidences = [r["english_forced"]["confidence"] for r in successful_tests]
    none_confidences = [r["no_specification"]["confidence"] for r in successful_tests]
    
    return {
        "total_files": len(results),
        "successful_tests": len(successful_tests),
        "failed_tests": len(failed_tests),
        "success_rate": round(len(successful_tests) / len(results) * 100, 1),
        
        "language_detection_summary": {
            "auto_detection": {
                "languages_detected": auto_lang_counts,
                "avg_processing_time": round(sum(auto_times) / len(auto_times), 2),
                "avg_confidence": round(sum(auto_confidences) / len(auto_confidences), 3)
            },
            "english_forced": {
                "languages_detected": english_lang_counts,
                "avg_processing_time": round(sum(english_times) / len(english_times), 2),
                "avg_confidence": round(sum(english_confidences) / len(english_confidences), 3)
            },
            "no_specification": {
                "languages_detected": none_lang_counts,
                "avg_processing_time": round(sum(none_times) / len(none_times), 2),
                "avg_confidence": round(sum(none_confidences) / len(none_confidences), 3)
            }
        },
        
        "welsh_accent_analysis": {
            "auto_detected_as_welsh": auto_lang_counts.get("cy", 0),
            "english_forced_success": english_lang_counts.get("en", 0),
            "potential_corrections": auto_lang_counts.get("cy", 0) - english_lang_counts.get("cy", 0)
        }
    }


def main():
    """Main test function"""
    print("🌍 Language Detection Test with Incoming Files")
    print("=" * 60)
    print("Testing improved language detection on audio files from incoming folder")
    print()
    
    # Initialize services
    print("🔧 Initializing services...")
    transcription_service = TranscriptionService()
    audio_processor = AudioProcessor()
    
    # Get incoming folder path
    incoming_folder = Path("incoming")
    if not incoming_folder.exists():
        print(f"❌ Error: Incoming folder not found at {incoming_folder}")
        return
    
    # Get all WAV files
    wav_files = list(incoming_folder.glob("*.wav"))
    if not wav_files:
        print(f"❌ Error: No WAV files found in {incoming_folder}")
        return
    
    print(f"📁 Found {len(wav_files)} WAV files in incoming folder")
    print(f"🎯 Testing improved language detection...")
    print()
    
    # Test on a subset of files (first 10) to avoid long processing time
    test_files = wav_files[:10]
    print(f"🧪 Testing on first {len(test_files)} files...")
    print()
    
    results = []
    total_start_time = time.time()
    
    for i, file_path in enumerate(test_files, 1):
        print(f"📊 Progress: {i}/{len(test_files)}")
        result = test_language_detection_on_file(str(file_path), transcription_service)
        results.append(result)
        print(f"   ✅ {result['filename']}: {result['auto_detection']['language']} "
              f"(confidence: {result['auto_detection']['confidence']:.2f})")
        print()
    
    total_time = time.time() - total_start_time
    
    # Analyze results
    print("📈 Analyzing results...")
    analysis = analyze_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 LANGUAGE DETECTION TEST RESULTS")
    print("=" * 60)

    if "error" in analysis:
        print(f"❌ All files failed: {analysis['error']}")
        # Print the first error from results for debugging
        for r in results:
            if "error" in r:
                print(f"First error encountered in file {r['filename']}:")
                print(r['error'])
                break
        print("\nCheck your FFmpeg installation and audio file compatibility.")
        return

    print(f"📁 Total files tested: {analysis['total_files']}")
    print(f"✅ Successful tests: {analysis['successful_tests']}")
    print(f"❌ Failed tests: {analysis['failed_tests']}")
    print(f"📈 Success rate: {analysis['success_rate']}%")
    print(f"⏱️  Total processing time: {total_time:.1f}s")
    print()
    
    print("🌍 Language Detection Summary:")
    print("-" * 40)
    
    # Auto detection results
    auto_summary = analysis['language_detection_summary']['auto_detection']
    print(f"🔍 Auto Detection:")
    print(f"   Languages: {auto_summary['languages_detected']}")
    print(f"   Avg time: {auto_summary['avg_processing_time']}s")
    print(f"   Avg confidence: {auto_summary['avg_confidence']}")
    
    # English forced results
    english_summary = analysis['language_detection_summary']['english_forced']
    print(f"🇬🇧 English Forced:")
    print(f"   Languages: {english_summary['languages_detected']}")
    print(f"   Avg time: {english_summary['avg_processing_time']}s")
    print(f"   Avg confidence: {english_summary['avg_confidence']}")
    
    # No specification results
    none_summary = analysis['language_detection_summary']['no_specification']
    print(f"🤔 No Specification:")
    print(f"   Languages: {none_summary['languages_detected']}")
    print(f"   Avg time: {none_summary['avg_processing_time']}s")
    print(f"   Avg confidence: {none_summary['avg_confidence']}")
    
    print()
    
    # Welsh accent analysis
    welsh_analysis = analysis['welsh_accent_analysis']
    print("🏴󠁧󠁢󠁷󠁬󠁳󠁿 Welsh Accent Analysis:")
    print("-" * 40)
    print(f"   Auto-detected as Welsh: {welsh_analysis['auto_detected_as_welsh']}")
    print(f"   Successfully forced to English: {welsh_analysis['english_forced_success']}")
    print(f"   Potential corrections: {welsh_analysis['potential_corrections']}")
    
    print()
    
    # Detailed results for first few files
    print("📋 Detailed Results (First 5 files):")
    print("-" * 40)
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result['filename']} ({result['file_size_mb']}MB)")
        print(f"   Auto: {result['auto_detection']['language']} "
              f"(conf: {result['auto_detection']['confidence']:.2f}, "
              f"time: {result['auto_detection']['processing_time']}s)")
        print(f"   English: {result['english_forced']['language']} "
              f"(conf: {result['english_forced']['confidence']:.2f}, "
              f"time: {result['english_forced']['processing_time']}s)")
        print(f"   None: {result['no_specification']['language']} "
              f"(conf: {result['no_specification']['confidence']:.2f}, "
              f"time: {result['no_specification']['processing_time']}s)")
        print(f"   Preview: {result['transcript_preview']}")
        print()
    
    print("=" * 60)
    print("✅ Language detection test completed!")
    
    # Save detailed results to file
    import json
    output_file = "language_detection_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "summary": analysis,
            "detailed_results": results
        }, f, indent=2)
    
    print(f"💾 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main() 