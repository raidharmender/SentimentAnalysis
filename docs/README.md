# 🎤 Audio Sentiment Analysis System

A comprehensive Python 3.12 system for analyzing customer sentiment and emotions from audio files using Machine Learning and Natural Language Processing.

## 🚀 Features

- **Audio Processing**: Normalize sample rate (16kHz mono), denoise, and trim silence
- **Speech-to-Text**: Powered by OpenAI Whisper for accurate transcription
- **🌍 Enhanced Language Detection**: Improved detection for English accents with manual language selection
- **🎭 Emotion Detection**: Multi-modal emotion recognition (7 categories) using audio features and text analysis
- **Sentiment Analysis**: Multi-model approach using Hugging Face transformers, VADER, and TextBlob
- **REST API**: FastAPI-based API for easy integration
- **Web Dashboard**: Streamlit dashboard for visualization and analysis
- **Database Storage**: SQLite/PostgreSQL support for storing results
- **Segment Analysis**: Time-based sentiment analysis for detailed insights

## 📋 System Architecture

```
Audio File Upload → Preprocessing → Transcription → Emotion Detection → Sentiment Analysis → Storage & Dashboard
```

### Components:
1. **Ingestion**: Accept WAV/MP3/FLAC upload via REST API
2. **Preprocessing**: Normalize sample rate, denoise, trim silence
3. **Transcription**: OpenAI Whisper for speech-to-text conversion
4. **🎭 Emotion Detection**: Multi-modal analysis (audio features + text)
5. **Sentiment Analysis**: Multi-model NLP sentiment analysis
6. **Postprocessing**: Aggregate results per call/speaker
7. **Storage**: Database storage with dashboard visualization

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- FFmpeg (for audio processing)

### ⚠️ PyTorch Installation Required

**PyTorch is required for both Whisper and Hugging Face Transformers models.**

- Visit: https://pytorch.org/get-started/locally/
- Use the selector to get the correct install command for your Python version and hardware (CPU-only or with GPU support).
- Example (CPU-only):
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd SentimentAnalysis_1
```

2. **Create virtual environment**:
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
uv sync
```

4. **Install FFmpeg** (if not already installed):
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu**: `sudo apt install ffmpeg`
   - **Windows**: Download from https://ffmpeg.org/download.html

## 🚀 Quick Start

### 1. Start the System

```bash
# Start both API and dashboard
python main.py --mode both

# Or start separately
python main.py --mode api      # API only (port 8000)
python main.py --mode dashboard # Dashboard only (port 8501)
```

### 2. Access the System

- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **Health Check**: http://localhost:8000/health

### 3. Upload and Analyze Audio

#### Via Dashboard:
1. Open http://localhost:8501
2. Upload an audio file (WAV, MP3, FLAC, M4A)
3. Select language (or use auto-detect)
4. View results with emotion detection and sentiment analysis

#### Via API:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio_file.wav"
```

## 🎭 Emotion Detection

The system now includes advanced emotion detection capabilities:

### Emotion Categories
- **Joy** (happy, excited, enthusiastic)
- **Anger** (angry, frustrated, irritated)
- **Sadness** (sad, depressed, melancholy)
- **Fear** (afraid, anxious, worried)
- **Surprise** (surprised, shocked, amazed)
- **Disgust** (disgusted, repulsed, revolted)
- **Neutral** (calm, indifferent, balanced)

### Audio Features Extracted
- **MFCC Features**: 13 coefficients with delta and delta-delta
- **Prosodic Features**: Pitch, energy, tempo, onset strength
- **Spectral Features**: Centroid, rolloff, bandwidth, zero-crossing rate
- **Chroma Features**: Harmonic content analysis
- **Harmonic Features**: Harmonic-percussive separation

### Multi-Modal Analysis
- **Audio-based detection**: Rule-based patterns using audio features
- **Text-based detection**: Keyword analysis of transcribed text
- **Fusion methods**: Weighted, ensemble, and voting approaches

## 📊 API Usage

### Analyze Audio File
```bash
POST /analyze
Content-Type: multipart/form-data

Parameters:
- file: Audio file (WAV, MP3, FLAC, M4A)
- language: Language code (optional, 'auto' for improved detection)
- save_processed_audio: Boolean (default: true)
```

### Analyze Text Only
```bash
POST /analyze/text
Content-Type: application/json

{
  "text": "Your text to analyze"
}
```

### Get Analysis Results
```bash
GET /analyses/{analysis_id}
```

### Get Statistics
```bash
GET /statistics
```

### Get Supported Languages
```bash
curl -X GET "http://localhost:8000/languages"
```

## 🎯 Dashboard Features

### Upload and Analysis
- **File Upload**: Drag-and-drop or click to upload audio files
- **Language Selection**: Choose from 15+ languages or use auto-detect
- **Real-time Processing**: View processing progress and results

### Results Visualization
- **Emotion Detection**: 7 emotion categories with confidence scores
- **Sentiment Analysis**: Positive/negative/neutral with detailed breakdown
- **Audio Features**: Comprehensive feature analysis and visualization
- **Segment Analysis**: Time-based emotion and sentiment tracking

### Analytics
- **Overall Statistics**: System-wide sentiment and emotion trends
- **File Management**: View and manage uploaded files
- **Export Options**: Download results in various formats

## 🔧 Configuration

Edit `app/config.py` to customize settings:

```python
# Audio Processing
TARGET_SAMPLE_RATE = 16000  # Hz
TARGET_CHANNELS = 1  # Mono

# Whisper Model
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# Sentiment Analysis
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# File Upload
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".m4a"]
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
pytest tests/test_sentiment_analysis.py::TestAudioProcessor
pytest tests/test_sentiment_analysis.py::TestSentimentAnalyzer
```

## 📁 Project Structure

```
SentimentAnalysis_1/
├── app/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── models.py              # Database models
│   ├── database.py            # Database connection
│   ├── audio_processor.py     # Audio preprocessing
│   ├── transcription.py       # Speech-to-text service
│   ├── sentiment_analyzer.py  # Sentiment analysis
│   ├── sentiment_service.py   # Main orchestration service
│   ├── api.py                 # FastAPI endpoints
│   └── dashboard.py           # Streamlit dashboard
├── tests/
│   └── test_sentiment_analysis.py
├── uploads/                   # Uploaded audio files
├── processed/                 # Processed audio files
├── docs/                      # Documentation
│   ├── README.md              # Main documentation
│   ├── Audio_File_Storage.md  # Audio storage documentation
│   ├── API_Documentation.md   # API reference
│   └── SystemArchitecture.md  # System architecture
├── main.py                    # Main entry point
├── pyproject.toml            # uv project configuration
└── README.md
```

## 📁 Audio File Storage

For detailed information about audio file storage paths, processing flow, and management, see:
- **[Audio File Storage Documentation](Audio_File_Storage.md)** - Complete guide to file storage and management

## 🔍 Technical Details

### Audio Processing Pipeline
1. **Load Audio**: Support for WAV, MP3, FLAC, M4A formats
2. **Normalize Sample Rate**: Convert to 16kHz mono
3. **Denoise**: Remove background noise using spectral gating
4. **Trim Silence**: Remove leading/trailing silence
5. **Normalize Amplitude**: Standardize audio levels

### Transcription Pipeline
- **Model**: OpenAI Whisper (configurable size: tiny, base, small, medium, large)
- **Language Detection**: Automatic or manual specification
- **Confidence Scoring**: Based on model probabilities
- **Segment Extraction**: Time-aligned text segments

### Sentiment Analysis Pipeline
- **Hugging Face Transformers**: State-of-the-art transformer models
- **VADER**: Rule-based sentiment analysis
- **TextBlob**: Machine learning-based sentiment analysis
- **Ensemble Aggregation**: Combine results from multiple models
- **Confidence Scoring**: Based on model agreement

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg

# Copy application
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install uv && uv sync --frozen

# Expose ports
EXPOSE 8000 8501

# Run application
CMD ["python", "main.py", "--mode", "both"]
```

### Production Considerations
- Use PostgreSQL instead of SQLite for production
- Implement proper authentication and authorization
- Add rate limiting and request validation
- Use environment variables for sensitive configuration
- Implement proper logging and monitoring
- Consider using Redis for caching
- Set up proper backup strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the system status at `/status`

## 🔮 Future Enhancements

- [ ] Speaker diarization for multi-speaker audio
- [ ] Real-time streaming analysis
- [ ] Custom model training capabilities
- [ ] Advanced analytics and reporting
- [ ] Integration with CRM systems
- [ ] Mobile app support
- [ ] Cloud deployment templates

## 🌐 API Documentation

### Interactive Documentation
The API includes comprehensive interactive documentation powered by FastAPI and Swagger:

- **Swagger UI**: http://localhost:8000/docs - Interactive API explorer with try-it-out functionality
- **ReDoc**: http://localhost:8000/redoc - Alternative documentation view
- **OpenAPI JSON**: http://localhost:8000/openapi.json - Raw OpenAPI specification

### API Features
- **Organized Endpoints**: Grouped by functionality (health, analysis, results, statistics)
- **Detailed Descriptions**: Each endpoint includes comprehensive documentation
- **Request/Response Examples**: Pre-filled examples for easy testing
- **Parameter Validation**: Automatic validation with helpful error messages
- **Schema Documentation**: Complete data model documentation

### Quick API Test
```bash
# Health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status

# Text sentiment analysis
curl -X POST "http://localhost:8000/analyze/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

For detailed API documentation, see [API_Documentation.md](API_Documentation.md) and [curl_examples.md](curl_examples.md).

--- 