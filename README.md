# 🎤 Audio Sentiment Analysis System

A comprehensive Python 3.12 system for analyzing customer sentiment from audio files using Machine Learning and Natural Language Processing.

## 🚀 Features

- **Audio Processing**: Normalize sample rate (16kHz mono), denoise, and trim silence
- **Speech-to-Text**: Powered by OpenAI Whisper for accurate transcription
- **Sentiment Analysis**: Multi-model approach using Hugging Face transformers, VADER, and TextBlob
- **REST API**: FastAPI-based API for easy integration
- **Web Dashboard**: Streamlit dashboard for visualization and analysis
- **Database Storage**: SQLite/PostgreSQL support for storing results
- **Segment Analysis**: Time-based sentiment analysis for detailed insights

## 📋 System Architecture

```
Audio File Upload → Preprocessing → Transcription → Sentiment Analysis → Storage & Dashboard
```

### Components:
1. **Ingestion**: Accept WAV/MP3/FLAC upload via REST API
2. **Preprocessing**: Normalize sample rate, denoise, trim silence
3. **Transcription**: OpenAI Whisper for speech-to-text conversion
4. **Sentiment Analysis**: Multi-model NLP sentiment analysis
5. **Postprocessing**: Aggregate results per call/speaker
6. **Storage**: Database storage with dashboard visualization

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- FFmpeg (for audio processing)

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
pip install -r requirements.txt
```

4. **Install FFmpeg** (if not already installed):
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu**: `sudo apt install ffmpeg`
   - **Windows**: Download from https://ffmpeg.org/download.html

## 🚀 Quick Start

### Option 1: Run API Server Only
```bash
python main.py --mode api
```

### Option 2: Run Dashboard Only
```bash
python main.py --mode dashboard
```

### Option 3: Run Both API and Dashboard
```bash
python main.py --mode both
```

### Option 4: Run with Custom Ports
```bash
python main.py --mode both --api-port 8000 --dashboard-port 8501
```

## 📡 API Usage

### Start the API Server
```bash
python main.py --mode api
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### API Endpoints

#### 1. Analyze Audio
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_audio_file.wav"
```

#### 2. Get All Analyses
```bash
curl -X GET "http://localhost:8000/analyses?limit=10&offset=0"
```

#### 3. Get Analysis by ID
```bash
curl -X GET "http://localhost:8000/analyses/1"
```

#### 4. Get Statistics
```bash
curl -X GET "http://localhost:8000/statistics"
```

#### 5. Get System Status
```bash
curl -X GET "http://localhost:8000/status"
```

## 🎨 Dashboard Usage

### Start the Dashboard
```bash
python main.py --mode dashboard
```

The dashboard will be available at `http://localhost:8501`

### Dashboard Features:
- **Upload & Analyze**: Upload audio files and view analysis results
- **View Results**: Browse all analyses with filtering options
- **Statistics**: View sentiment distribution and trends
- **System Status**: Monitor system health and model status

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

## 📊 Example Output

### API Response
```json
{
  "analysis_id": 1,
  "filename": "customer_call.wav",
  "duration": 45.2,
  "transcription": {
    "text": "Hello, I'm calling about my recent order...",
    "language": "en",
    "confidence": 0.92,
    "processing_time": 3.45
  },
  "sentiment": {
    "overall_sentiment": "positive",
    "score": 0.65,
    "confidence": 0.85,
    "details": {
      "huggingface": [{"label": "positive", "score": 0.78}],
      "vader": {"compound": 0.65, "pos": 0.45, "neg": 0.12, "neu": 0.43},
      "textblob": {"polarity": 0.55, "subjectivity": 0.67}
    }
  },
  "summary": {
    "overall_sentiment": "positive",
    "average_score": 0.65,
    "sentiment_distribution": {"positive": 8, "negative": 2, "neutral": 3},
    "total_segments": 13
  },
  "segments": [...],
  "processing_time": 12.34
}
```

## 🏗️ Project Structure

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
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
└── README.md
```

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
RUN pip install -r requirements.txt

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

