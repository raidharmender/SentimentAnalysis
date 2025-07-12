# Sentiment Analysis System

A comprehensive audio sentiment analysis system that processes audio files, performs speech-to-text transcription, and analyzes sentiment using multiple languages and tools.

## Features

- **Multi-language Support**: English, Mandarin, and Malay sentiment analysis
- **Advanced Audio Processing**: Noise reduction, normalization, and feature extraction
- **Speech-to-Text**: OpenAI Whisper integration with enhanced language detection
- **Emotion Detection**: Multi-modal emotion recognition from audio and text
- **REST API**: FastAPI backend with comprehensive endpoints
- **Interactive Dashboard**: Streamlit-based visualization and analysis
- **Multi-tool Sentiment Analysis**: VADER, TextBlob, HuggingFace, SnowNLP, Cntext, and Malaya

## Recent Refactoring Improvements

### 🚀 Pythonic Code Refactoring

The codebase has been comprehensively refactored to follow Python best practices and advanced concepts:

#### **Constants Management**
- **Centralized Constants**: All hardcoded values moved to `app/constants.py`
- **Organized Configuration**: Logical grouping of constants by functionality
- **Type Safety**: Strong typing with custom type aliases
- **Enum Usage**: Proper enumeration for states, labels, and models

```python
# Before: Hardcoded values scattered throughout code
sample_rate = 16000
max_file_size = 50 * 1024 * 1024
positive_threshold = 0.05

# After: Centralized constants
from app.constants import AudioConfig, SentimentConfig
sample_rate = AudioConfig.TARGET_SAMPLE_RATE
max_file_size = AudioConfig.MAX_FILE_SIZE
positive_threshold = SentimentConfig.POSITIVE_THRESHOLD
```

#### **Advanced Python Concepts**

**Lambda Functions & Functional Programming**
```python
# Lambda for sentiment mapping
sentiment_mapper = lambda score: (
    SentimentLabel.POSITIVE if score >= SentimentConfig.POSITIVE_THRESHOLD
    else SentimentLabel.NEGATIVE if score <= SentimentConfig.NEGATIVE_THRESHOLD
    else SentimentLabel.NEUTRAL
)

# Functional processing pipeline
processing_results = [
    (step_name, step_func(audio, sample_rate))
    for step_name, step_func in self._pipeline_steps.items()
    if steps.get(step_name, False)
]
```

**List & Dict Comprehensions**
```python
# List comprehension for feature extraction
confidence_scores = [
    segment.get("avg_logprob", 0.0) 
    for segment in segments 
    if segment.get("avg_logprob") is not None
]

# Dict comprehension for sentiment distribution
sentiment_distribution = {
    label.value: sentiments.count(label.value)
    for label in SentimentLabel
}
```

**Data Classes & Type Hints**
```python
@dataclass
class SentimentScore:
    """Data class for sentiment scores."""
    label: SentimentLabel
    score: float
    confidence: float
    model: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label.value,
            "score": self.score,
            "confidence": self.confidence,
            "model": self.model
        }
```

#### **Performance Optimizations**

**Caching with LRU Cache**
```python
@lru_cache(maxsize=1000)
def _cached_analyze(self, text: str, model_type: str) -> Optional[Dict]:
    """Cached sentiment analysis for performance."""
    return self._model_handlers[model_type](text)
```

**Vectorized Operations**
```python
# Vectorized audio processing
audio = np.mean(audio, axis=0) if len(audio.shape) > 1 else audio

# Vectorized confidence calculation
weights = [seg.get("end", 0) - seg.get("start", 0) for seg in segments]
normalized_weights = [w / total_weight for w in weights]
confidence = sum(score * weight for score, weight in zip(confidence_scores, normalized_weights))
```

**Batch Processing**
```python
def batch_process(self, file_paths: list, processing_steps: Optional[dict] = None) -> list:
    """Process multiple audio files in batch."""
    return [
        {
            'file_path': file_path,
            'result': self.process_audio(file_path, processing_steps)
        }
        for file_path in file_paths
    ]
```

#### **Error Handling & Validation**

**Comprehensive Validation**
```python
@classmethod
def is_valid_text(cls, text: str) -> bool:
    """Validate text input."""
    return (isinstance(text, str) and 
            cls.MIN_TEXT_LENGTH <= len(text.strip()) <= cls.MAX_TEXT_LENGTH)

@classmethod
def is_valid_audio_duration(cls, duration: float) -> bool:
    """Validate audio duration."""
    return cls.MIN_AUDIO_DURATION <= duration <= cls.MAX_AUDIO_DURATION
```

**Structured Error Messages**
```python
class ErrorMessages:
    """Centralized error message constants."""
    FILE_TOO_LARGE = "File size exceeds maximum allowed size of {}MB"
    TRANSCRIPTION_FAILED = "Transcription failed: {}"
    SENTIMENT_ANALYSIS_FAILED = "Sentiment analysis failed: {}"
```

#### **Configuration Management**

**Pydantic Settings with Validation**
```python
class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug or os.getenv("ENVIRONMENT", "development") == "development"
    
    def validate_config(self) -> None:
        """Validate configuration settings."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("Invalid port number")
```

#### **Modular Architecture**

**Component Separation**
- **Constants Module**: Centralized configuration management
- **Audio Processor**: Advanced audio preprocessing with pipeline
- **Sentiment Analyzer**: Multi-model sentiment analysis with caching
- **Transcription Service**: Optimized speech-to-text with device handling
- **API Layer**: RESTful endpoints with comprehensive error handling

**Pipeline Architecture**
```python
# Configurable processing pipeline
self._pipeline_steps = {
    'normalize_sample_rate': self._normalize_sample_rate,
    'denoise': self._denoise_audio,
    'trim_silence': self._trim_silence,
    'normalize_amplitude': self._normalize_amplitude
}
```

#### **Code Quality Improvements**

**Type Safety**
- Comprehensive type hints throughout the codebase
- Custom type aliases for better readability
- Enum usage for state management
- Dataclasses for structured data

**Documentation**
- Detailed docstrings with type information
- Inline comments for complex logic
- Comprehensive README with examples
- API documentation with OpenAPI/Swagger

**Testing**
- Unit tests for all refactored components
- Integration tests for end-to-end functionality
- Performance benchmarks for optimizations
- Error handling validation

### **Benefits of Refactoring**

1. **Maintainability**: Centralized constants and modular design
2. **Performance**: Caching, vectorization, and batch processing
3. **Readability**: Pythonic code with clear structure
4. **Type Safety**: Comprehensive type hints and validation
5. **Extensibility**: Easy to add new features and models
6. **Error Handling**: Structured error management
7. **Testing**: Improved testability with dependency injection

### **Advanced Features Added**

- **Smart Caching**: LRU cache for expensive operations
- **Batch Processing**: Efficient handling of multiple files
- **Device Optimization**: GPU/CPU-specific optimizations
- **Memory Management**: Efficient data structures and cleanup
- **Async Support**: Prepared for async/await patterns
- **Monitoring**: Performance metrics and logging

## System Architecture

### High-Level Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Audio Processor │───▶│  Transcription  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Sentiment      │◀───│  Multi-Language │◀───│  Text Output    │
│  Analysis       │    │  Detector       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Emotion        │    │  REST API       │    │  Dashboard      │
│  Detection      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Breakdown

#### **Audio Processing Pipeline**
- **Input Validation**: File format and size validation
- **Preprocessing**: Noise reduction, normalization, silence trimming
- **Feature Extraction**: MFCC, spectral, chroma, and harmonic features
- **Optimization**: Vectorized operations and batch processing

#### **Transcription Service**
- **Whisper Integration**: OpenAI Whisper with device optimization
- **Language Detection**: Enhanced detection with accent handling
- **Caching**: LRU cache for transcription configurations
- **Error Handling**: Comprehensive error management

#### **Sentiment Analysis**
- **Multi-Model**: VADER, TextBlob, HuggingFace, SnowNLP, Cntext, Malaya
- **Language-Specific**: Tool selection based on detected language
- **Caching**: Cached analysis results for performance
- **Aggregation**: Weighted averaging of multiple model results

#### **API Layer**
- **FastAPI**: Modern async web framework
- **Validation**: Pydantic models for request/response validation
- **Error Handling**: Structured error responses
- **Documentation**: Auto-generated OpenAPI documentation

## Installation

### Prerequisites

- Python 3.10 (recommended for compatibility)
- FFmpeg installed and configured
- Sufficient disk space for models

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd SentimentAnalysis_1
```

2. **Create virtual environment**
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -e .
```

4. **Download NLTK data**
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

5. **Create necessary directories**
```bash
mkdir -p uploads processed
```

## Usage

### Starting the Services

1. **Start the API server**
```bash
python -m app.api
```

2. **Start the dashboard**
```bash
streamlit run app/dashboard.py
```

### API Endpoints

#### **Analyze Audio File**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio_file.wav" \
  -F "language=auto"
```

#### **Analyze Text**
```bash
curl -X POST "http://localhost:8000/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am very happy today!", "language": "en"}'
```

#### **Get System Status**
```bash
curl "http://localhost:8000/status"
```

### Dashboard Features

- **File Upload**: Drag-and-drop audio file upload
- **Real-time Analysis**: Live processing with progress indicators
- **Multi-language Support**: Language selection and detection display
- **Visualization**: Charts and graphs for sentiment analysis
- **Export**: Download results in various formats

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Database
DATABASE_URL=sqlite:///./sentiment_analysis.db

# Whisper Model
WHISPER_MODEL=base

# Sentiment Model
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest

# File Upload
MAX_FILE_SIZE=52428800
UPLOAD_DIR=uploads
PROCESSED_DIR=processed
```

### Constants Configuration

All system constants are managed in `app/constants.py`:

```python
# Audio Processing
AudioConfig.TARGET_SAMPLE_RATE = 16000
AudioConfig.MAX_FILE_SIZE = 50 * 1024 * 1024

# Sentiment Analysis
SentimentConfig.POSITIVE_THRESHOLD = 0.05
SentimentConfig.NEGATIVE_THRESHOLD = -0.05

# Language Detection
LanguageConfig.WHISPER_CONFIDENCE_THRESHOLD = 0.8
```

## Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Audio processing tests
python -m pytest tests/test_audio_processor.py -v

# Sentiment analysis tests
python -m pytest tests/test_sentiment_analyzer.py -v

# API tests
python -m pytest tests/test_api.py -v
```

### Performance Testing
```bash
# Benchmark audio processing
python tests/benchmark_audio_processing.py

# Benchmark sentiment analysis
python tests/benchmark_sentiment_analysis.py
```

## Performance Optimizations

### **Caching Strategy**
- **LRU Cache**: For expensive operations (transcription, sentiment analysis)
- **Model Caching**: Pre-loaded models for faster inference
- **Configuration Caching**: Cached settings and parameters

### **Vectorization**
- **NumPy Operations**: Vectorized audio processing
- **Batch Processing**: Efficient handling of multiple files
- **Memory Optimization**: Efficient data structures

### **Device Optimization**
- **GPU Support**: CUDA acceleration for Whisper models
- **CPU Optimization**: FP32 precision for CPU inference
- **Memory Management**: Automatic cleanup and garbage collection

## Troubleshooting

### Common Issues

1. **FFmpeg Errors**
```bash
# Reinstall FFmpeg
brew reinstall ffmpeg

# Fix dylib dependencies
./scripts/fix_ffmpeg_dylibs.sh
```

2. **Model Loading Issues**
```bash
# Clear model cache
rm -rf ~/.cache/whisper

# Reinstall dependencies
pip install --force-reinstall torch torchaudio
```

3. **Memory Issues**
```bash
# Reduce batch size
export BATCH_SIZE=16

# Use smaller model
export WHISPER_MODEL=tiny
```

### Performance Tuning

1. **For Large Files**
   - Increase `MAX_FILE_SIZE` in constants
   - Use batch processing for multiple files
   - Enable GPU acceleration

2. **For High Throughput**
   - Use smaller Whisper models
   - Enable caching for repeated analysis
   - Implement async processing

3. **For Memory Constraints**
   - Reduce batch sizes
   - Use CPU-only mode
   - Implement streaming processing

## Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make changes following the refactored patterns**
   - Use constants from `app/constants.py`
   - Follow type hints and documentation standards
   - Add comprehensive tests

4. **Run tests and linting**
```bash
python -m pytest tests/ -v
python -m flake8 app/
python -m mypy app/
```

5. **Submit pull request**

### Code Standards

- **Type Hints**: All functions must have type hints
- **Documentation**: Comprehensive docstrings
- **Constants**: Use centralized constants
- **Error Handling**: Structured error management
- **Testing**: Unit tests for all new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Whisper for speech-to-text capabilities
- HuggingFace for sentiment analysis models
- VADER, TextBlob, SnowNLP, Cntext, and Malaya for multi-language sentiment analysis
- FastAPI and Streamlit for the web framework and dashboard 