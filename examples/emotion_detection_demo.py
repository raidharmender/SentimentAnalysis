#!/usr/bin/env python3
"""
Emotion Detection Demonstration

This script demonstrates the emotion detection capabilities of the system
by analyzing different types of audio and text inputs.
"""

import os
import sys
import numpy as np
import time
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.audio_feature_extractor import AudioFeatureExtractor
from app.emotion_detector import EmotionDetector


def create_sample_audio(duration: float = 5.0, sample_rate: int = 16000, 
                       emotion_type: str = "neutral") -> np.ndarray:
    """
    Create sample audio data that simulates different emotions
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        emotion_type: Type of emotion to simulate
        
    Returns:
        Audio data as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    if emotion_type == "joy":
        # High energy, fast tempo, bright sound
        audio = 0.8 * np.sin(2 * np.pi * 440 * t) + 0.4 * np.sin(2 * np.pi * 880 * t)
        # Add some variation to simulate excitement
        audio += 0.2 * np.sin(2 * np.pi * 660 * t * (1 + 0.1 * np.sin(2 * np.pi * 2 * t)))
        
    elif emotion_type == "anger":
        # High energy, harsh sound
        audio = 0.9 * np.sin(2 * np.pi * 300 * t) + 0.6 * np.sin(2 * np.pi * 600 * t)
        # Add some noise to simulate harshness
        audio += 0.3 * np.random.randn(len(audio))
        
    elif emotion_type == "sadness":
        # Low energy, slow tempo, dark sound
        audio = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)
        # Add some low frequency components
        audio += 0.1 * np.sin(2 * np.pi * 100 * t)
        
    elif emotion_type == "fear":
        # Moderate energy, tense sound
        audio = 0.5 * np.sin(2 * np.pi * 350 * t) + 0.3 * np.sin(2 * np.pi * 700 * t)
        # Add some tremolo effect
        audio *= (1 + 0.1 * np.sin(2 * np.pi * 8 * t))
        
    else:  # neutral
        # Balanced sound
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio.astype(np.float32)


def demonstrate_audio_features():
    """Demonstrate audio feature extraction"""
    print("🎵 Audio Feature Extraction Demonstration")
    print("=" * 50)
    
    extractor = AudioFeatureExtractor()
    
    # Test with different emotions
    emotions = ["joy", "anger", "sadness", "fear", "neutral"]
    
    for emotion in emotions:
        print(f"\n📊 Analyzing {emotion.upper()} audio...")
        
        # Create sample audio
        audio_data = create_sample_audio(duration=5.0, emotion_type=emotion)
        sample_rate = 16000
        
        # Extract features
        start_time = time.time()
        features = extractor.extract_emotion_relevant_features(audio_data, sample_rate)
        extraction_time = time.time() - start_time
        
        # Display key features
        print(f"  ⏱️  Extraction time: {extraction_time:.2f}s")
        print(f"  🎵 Pitch mean: {features.get('pitch_mean', 0):.2f}")
        print(f"  ⚡ Energy mean: {features.get('energy_mean', 0):.2f}")
        print(f"  🥁 Tempo: {features.get('tempo', 0):.1f} BPM")
        print(f"  🌈 Spectral centroid: {features.get('spectral_centroid_mean', 0):.0f} Hz")
        print(f"  🎼 Harmonic ratio: {features.get('harmonic_ratio', 0):.2f}")


def demonstrate_text_emotion_detection():
    """Demonstrate text-based emotion detection"""
    print("\n📝 Text Emotion Detection Demonstration")
    print("=" * 50)
    
    detector = EmotionDetector()
    
    # Test texts for different emotions
    test_texts = {
        "joy": [
            "I am so happy and excited about this wonderful news!",
            "This is absolutely fantastic and amazing!",
            "I love this so much, it's incredible!"
        ],
        "anger": [
            "I am so angry and furious about this terrible situation!",
            "This is absolutely disgusting and horrible!",
            "I hate this so much, it's awful!"
        ],
        "sadness": [
            "I am so sad and depressed about this terrible news.",
            "This is absolutely awful and disappointing.",
            "I feel so miserable and unhappy about this."
        ],
        "fear": [
            "I am so afraid and worried about what might happen.",
            "This is absolutely terrifying and frightening.",
            "I am scared and anxious about the future."
        ],
        "surprise": [
            "I am so surprised and shocked by this unexpected news!",
            "This is absolutely amazing and astonishing!",
            "Wow, I can't believe this incredible revelation!"
        ],
        "neutral": [
            "The weather is okay today and everything is fine.",
            "This is a normal situation with standard procedures.",
            "Everything is working as expected and usual."
        ]
    }
    
    for emotion, texts in test_texts.items():
        print(f"\n🎭 Testing {emotion.upper()} detection:")
        
        for i, text in enumerate(texts, 1):
            result = detector._detect_text_emotion(text)
            print(f"  {i}. Text: {text[:50]}...")
            print(f"     Detected: {result['emotion']} (confidence: {result['confidence']:.2f})")


def demonstrate_audio_emotion_detection():
    """Demonstrate audio-based emotion detection"""
    print("\n🎵 Audio Emotion Detection Demonstration")
    print("=" * 50)
    
    detector = EmotionDetector()
    
    # Test with different emotions
    emotions = ["joy", "anger", "sadness", "fear", "neutral"]
    
    for emotion in emotions:
        print(f"\n🎭 Testing {emotion.upper()} audio detection:")
        
        # Create sample audio
        audio_data = create_sample_audio(duration=5.0, emotion_type=emotion)
        sample_rate = 16000
        
        # Detect emotion
        start_time = time.time()
        result = detector.detect_emotion(audio_data, sample_rate)
        detection_time = time.time() - start_time
        
        print(f"  ⏱️  Detection time: {detection_time:.2f}s")
        print(f"  🎯 Detected emotion: {result['primary_emotion']}")
        print(f"  🎯 Confidence: {result['confidence']:.2f}")
        print(f"  🎵 Audio emotion: {result['audio_emotion']['emotion']}")
        print(f"  📊 Method: {result['audio_emotion']['method']}")


def demonstrate_multi_modal_emotion_detection():
    """Demonstrate multi-modal emotion detection"""
    print("\n🎭 Multi-Modal Emotion Detection Demonstration")
    print("=" * 50)
    
    detector = EmotionDetector()
    
    # Test scenarios with matching audio and text
    test_scenarios = [
        {
            "name": "Happy Customer",
            "audio_emotion": "joy",
            "text": "I am so happy and excited about this wonderful service!",
            "expected": "joy"
        },
        {
            "name": "Angry Customer",
            "audio_emotion": "anger",
            "text": "I am so angry and furious about this terrible experience!",
            "expected": "anger"
        },
        {
            "name": "Sad Customer",
            "audio_emotion": "sadness",
            "text": "I am so disappointed and sad about this situation.",
            "expected": "sadness"
        },
        {
            "name": "Confused Customer",
            "audio_emotion": "fear",
            "text": "I am worried and anxious about what to do next.",
            "expected": "fear"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n👤 {scenario['name']}:")
        
        # Create audio
        audio_data = create_sample_audio(duration=5.0, emotion_type=scenario['audio_emotion'])
        sample_rate = 16000
        
        # Detect emotion
        result = detector.detect_emotion(audio_data, sample_rate, scenario['text'])
        
        print(f"  🎵 Audio emotion: {result['audio_emotion']['emotion']} "
              f"(confidence: {result['audio_emotion']['confidence']:.2f})")
        print(f"  📝 Text emotion: {result['text_emotion']['emotion']} "
              f"(confidence: {result['text_emotion']['confidence']:.2f})")
        print(f"  🎯 Final emotion: {result['primary_emotion']} "
              f"(confidence: {result['confidence']:.2f})")
        print(f"  🔗 Fusion method: {result['fusion_method']}")
        
        # Check if detection matches expectation
        if result['primary_emotion'] == scenario['expected']:
            print(f"  ✅ Correctly detected {scenario['expected']}")
        else:
            print(f"  ❌ Expected {scenario['expected']}, got {result['primary_emotion']}")


def demonstrate_feature_analysis():
    """Demonstrate detailed feature analysis"""
    print("\n🔬 Detailed Feature Analysis")
    print("=" * 50)
    
    extractor = AudioFeatureExtractor()
    detector = EmotionDetector()
    
    # Create audio with known characteristics
    audio_data = create_sample_audio(duration=10.0, emotion_type="joy")
    sample_rate = 16000
    
    # Extract all features
    print("📊 Extracting comprehensive audio features...")
    all_features = extractor.extract_all_features(audio_data, sample_rate)
    
    # Get feature summary
    summary = extractor.get_feature_summary(all_features)
    
    print(f"  📈 Total features extracted: {summary['total_features']}")
    print(f"  🎵 Has pitch features: {summary['has_pitch']}")
    print(f"  ⚡ Has energy features: {summary['has_energy']}")
    print(f"  🌈 Has spectral features: {summary['has_spectral']}")
    print(f"  🎼 Has MFCC features: {summary['has_mfcc']}")
    print(f"  🎶 Has harmonic features: {summary['has_harmonic']}")
    
    if 'feature_values_range' in summary:
        range_info = summary['feature_values_range']
        print(f"  📊 Feature value range: {range_info['min']:.2f} to {range_info['max']:.2f}")
        print(f"  📊 Average feature value: {range_info['mean']:.2f}")
    
    # Detect emotion and get summary
    emotion_result = detector.detect_emotion(audio_data, sample_rate)
    emotion_summary = detector.get_emotion_summary(emotion_result)
    
    print(f"\n🎭 Emotion Detection Summary:")
    print(f"  🎯 Primary emotion: {emotion_summary['primary_emotion']}")
    print(f"  🎯 Confidence: {emotion_summary['confidence']:.2f}")
    print(f"  🎵 Audio features available: {emotion_summary['has_audio_features']}")
    print(f"  📝 Text features available: {emotion_summary['has_text_features']}")
    print(f"  🔗 Fusion method: {emotion_summary['fusion_method']}")


def main():
    """Main demonstration function"""
    print("🎭 Emotion Detection System Demonstration")
    print("=" * 60)
    print("This demonstration shows the emotion detection capabilities")
    print("of our enhanced sentiment analysis system.")
    print()
    
    try:
        # Run demonstrations
        demonstrate_audio_features()
        demonstrate_text_emotion_detection()
        demonstrate_audio_emotion_detection()
        demonstrate_multi_modal_emotion_detection()
        demonstrate_feature_analysis()
        
        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("\n🎯 Key Benefits of Emotion Detection:")
        print("  • 7 emotion categories vs 3 sentiment levels")
        print("  • Multi-modal analysis (audio + text)")
        print("  • Real-time emotion tracking")
        print("  • Detailed audio feature analysis")
        print("  • Improved customer experience insights")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 