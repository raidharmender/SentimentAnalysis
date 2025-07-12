#!/usr/bin/env python3
"""
Enhanced Language Detection Test

This script tests the enhanced language detection system on English files (M_ and F_ prefixes)
and Chinese files to validate the improved accuracy.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.enhanced_language_detector import EnhancedLanguageDetector


def get_test_files() -> List[Tuple[str, str]]:
    """
    Get test files with their expected languages
    
    Returns:
        List of (file_path, expected_language) tuples
    """
    test_files = []
    
    # English files (M_ and F_ prefixes)
    incoming_folder = Path("incoming")
    if incoming_folder.exists():
        for file_path in incoming_folder.glob("*.wav"):
            if file_path.name.startswith(('M_', 'F_')):
                test_files.append((str(file_path), 'en'))
    
    # Chinese files
    chinese_folder = Path("incoming/chinese")
    if chinese_folder.exists():
        for file_path in chinese_folder.glob("*.wav"):
            test_files.append((str(file_path), 'zh'))
    
    return test_files


def test_enhanced_detection():
    """Test the enhanced language detection system"""
    print("🔍 Enhanced Language Detection Test")
    print("=" * 60)
    print("Testing improved language detection for English and Chinese files")
    print()
    
    # Initialize enhanced language detector
    print("🔧 Initializing Enhanced Language Detector...")
    detector = EnhancedLanguageDetector()
    print()
    
    # Get test files
    test_files = get_test_files()
    if not test_files:
        print("❌ No test files found!")
        return
    
    print(f"📁 Found {len(test_files)} test files:")
    english_files = [f for f, lang in test_files if lang == 'en']
    chinese_files = [f for f, lang in test_files if lang == 'zh']
    print(f"   🇬🇧 English files: {len(english_files)}")
    print(f"   🇨🇳 Chinese files: {len(chinese_files)}")
    print()
    
    # Test on a subset for faster processing
    max_files_per_language = 5
    selected_files = []
    
    # Select English files
    for file_path in english_files[:max_files_per_language]:
        selected_files.append((file_path, 'en'))
    
    # Select Chinese files
    for file_path in chinese_files[:max_files_per_language]:
        selected_files.append((file_path, 'zh'))
    
    print(f"🧪 Testing on {len(selected_files)} files (max {max_files_per_language} per language)...")
    print()
    
    # Run batch validation
    start_time = time.time()
    validation_results = detector.batch_validate(selected_files)
    total_time = time.time() - start_time
    
    # Display results
    print("📊 VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total files tested: {validation_results['total_files']}")
    print(f"Correct detections: {validation_results['correct_detections']}")
    print(f"Accuracy: {validation_results['accuracy']:.1%}")
    print(f"Total processing time: {total_time:.2f}s")
    print()
    
    # Detailed results
    print("📋 DETAILED RESULTS")
    print("-" * 60)
    for result in validation_results['results']:
        status = "✅" if result['correct'] else "❌"
        print(f"{status} {result['file']}")
        print(f"   Expected: {result['expected']} | Detected: {result['detected']} | Confidence: {result['confidence']:.3f}")
        print(f"   Method: {result['method']}")
        print()
    
    # Get statistics
    stats = detector.get_detection_stats(validation_results['results'])
    print("📈 DETECTION STATISTICS")
    print("-" * 60)
    print(f"Languages detected: {stats['languages_detected']}")
    print(f"Detection methods: {stats['detection_methods']}")
    print(f"Average confidence: {stats['avg_confidence']:.3f}")
    print(f"Confidence range: {stats['min_confidence']:.3f} - {stats['max_confidence']:.3f}")
    print()
    
    # Language-specific analysis
    print("🌍 LANGUAGE-SPECIFIC ANALYSIS")
    print("-" * 60)
    
    english_results = [r for r in validation_results['results'] if r['expected'] == 'en']
    chinese_results = [r for r in validation_results['results'] if r['expected'] == 'zh']
    
    if english_results:
        english_correct = sum(1 for r in english_results if r['correct'])
        english_accuracy = english_correct / len(english_results)
        print(f"🇬🇧 English files: {english_correct}/{len(english_results)} correct ({english_accuracy:.1%})")
    
    if chinese_results:
        chinese_correct = sum(1 for r in chinese_results if r['correct'])
        chinese_accuracy = chinese_correct / len(chinese_results)
        print(f"🇨🇳 Chinese files: {chinese_correct}/{len(chinese_results)} correct ({chinese_accuracy:.1%})")
    
    print()
    
    # Summary
    if validation_results['accuracy'] >= 0.9:
        print("🎉 EXCELLENT RESULTS! Enhanced language detection is working well.")
    elif validation_results['accuracy'] >= 0.7:
        print("👍 GOOD RESULTS! Enhanced language detection shows improvement.")
    else:
        print("⚠️  NEEDS IMPROVEMENT! Consider adjusting detection parameters.")
    
    return validation_results


def test_individual_files():
    """Test individual files with detailed output"""
    print("\n🔬 INDIVIDUAL FILE TESTING")
    print("=" * 60)
    
    detector = EnhancedLanguageDetector()
    
    # Test a few specific files
    test_cases = [
        ("incoming/M_0101_15y2m_1.wav", "en"),
        ("incoming/F_0101_15y2m_1.wav", "en"),
        ("incoming/chinese/OSR_cn_000_0075_8k.wav", "zh"),
    ]
    
    for file_path, expected_lang in test_cases:
        if os.path.exists(file_path):
            print(f"\n🎵 Testing: {os.path.basename(file_path)}")
            print(f"Expected: {expected_lang}")
            
            try:
                result = detector.detect_language_enhanced(file_path)
                print(f"Result: {result}")
                
                is_correct = result['language'] == expected_lang
                status = "✅" if is_correct else "❌"
                print(f"{status} Detection {'correct' if is_correct else 'incorrect'}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        else:
            print(f"⚠️  File not found: {file_path}")


def main():
    """Main test function"""
    try:
        # Run comprehensive test
        results = test_enhanced_detection()
        
        # Run individual file tests
        test_individual_files()
        
        print("\n✅ Enhanced language detection test completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 