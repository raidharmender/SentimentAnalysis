# Sentiment Analysis System

A comprehensive audio sentiment analysis system that processes audio files, performs speech-to-text transcription, and analyzes sentiment using multiple languages and tools.

---

## 🚀 What's New

- **Stricter Sentiment Thresholds**: The system now uses more conservative thresholds for positive/negative sentiment. See `app/constants.py` for `POSITIVE_THRESHOLD` and `NEGATIVE_THRESHOLD` (default: ±0.3).
- **Context-Aware Sentiment**: For customer service calls (detected by keywords), positive scores are reduced to avoid over-classification. See `app/sentiment_analyzer.py` for details.
- **Language Support in API**: Both `/analyze` and `/analyze/text` endpoints accept an optional `language` parameter (`en`, `zh`, `ms`, `auto`).
- **Swagger/OpenAPI Documentation**: Interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs) and [http://localhost:8000/redoc](http://localhost:8000/redoc).
- **.env File Format**: Environment variable files must NOT have inline comments. Each line should be `KEY=VALUE` only.

---

## Features

- **Multi-language Support**: English, Mandarin, and Malay sentiment analysis
- **Advanced Audio Processing**: Noise reduction, normalization, and feature extraction
- **Speech-to-Text**: OpenAI Whisper integration with enhanced language detection
- **Emotion Detection**: Multi-modal emotion recognition from audio and text
- **REST API**: FastAPI backend with comprehensive endpoints
- **Interactive Dashboard**: Streamlit-based visualization and analysis
- **Multi-tool Sentiment Analysis**: VADER, TextBlob, HuggingFace, SnowNLP, Cntext, and Malaya

---

## API Usage & Examples

### API Documentation
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Analyze Audio File
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -F "file=@audio_file.wav" \
  -F "language=auto"
```
- `language` is optional. Use `en`, `zh`, `ms`, or `auto` (default: auto-detect).

### Analyze Text
```bash
curl -X POST "http://localhost:8000/analyze/text" \
  -H "accept: application/json" \
  --data-urlencode "text=I am very happy today!" \
  --data-urlencode "language=en"
```
- `language` is optional. Use `en`, `zh`, or `ms` (default: en).

### Graceful Shutdown
```bash
curl -X POST "http://localhost:8000/shutdown" \
  -H "accept: application/json"
```
- Use this endpoint for controlled server shutdown instead of SIGTERM/SIGINT
- Performs cleanup operations (database connections, memory cleanup)
- Returns immediately with shutdown status, then exits process

### Example Sentiment Result
```json
{
  "sentiment": {
    "overall_sentiment": "neutral",
    "score": 0.12,
    "confidence": 0.8,
    "details": {
      "neg": 0.0,
      "neu": 0.9,
      "pos": 0.1,
      "compound": 0.12,
      "original_compound": 0.12,
      "adjusted_compound": 0.12
    }
  }
}
```
- `original_compound`: Raw VADER score
- `adjusted_compound`: Score after context-aware adjustment (for customer service calls)

---

## How Sentiment is Determined
- **Thresholds**: See `app/constants.py` for `POSITIVE_THRESHOLD` and `NEGATIVE_THRESHOLD` (default: ±0.3)
- **Context-Aware Logic**: For customer service calls (e.g., "customer care", "how is your experience"), positive scores are reduced before thresholding. This makes it harder to classify as "positive" unless the sentiment is very strong.
- **Neutral**: If the adjusted score is between the thresholds, the result is "neutral".

---

## Environment Variables

**Important:** Do NOT use inline comments in `.env` files. Each line must be `KEY=VALUE` only.

Example:
```env
HOST=0.0.0.0
PORT=8000
DEBUG=false
DATABASE_URL=sqlite:///./sentiment_analysis.db
WHISPER_MODEL=base
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
MAX_FILE_SIZE=52428800
UPLOAD_DIR=uploads
PROCESSED_DIR=processed
```

---

## Troubleshooting

### Address Already in Use
If you see `ERROR: [Errno 48] Address already in use`, stop any running server on the same port or use a different port in your `.env` file.

### FFmpeg Errors
See earlier troubleshooting section for FFmpeg issues.

### Model Loading Issues
See earlier troubleshooting section for model issues.

---

## Updating Sentiment Thresholds
To change how strict the sentiment classification is, edit these lines in `app/constants.py`:
```python
class SentimentConfig:
    POSITIVE_THRESHOLD = 0.3  # Increase for stricter positive
    NEGATIVE_THRESHOLD = -0.3  # Decrease for stricter negative
```
Restart the API after making changes.

---

## Interpreting Sentiment Results
- `overall_sentiment`: The final label (positive, negative, neutral)
- `score`: The (possibly adjusted) sentiment score
- `details`: Includes both the original and adjusted compound scores for transparency

---

## Database Schema

The system uses a SQLite database (`sentiment_analysis.db`) to store all analysis results and metadata.

### Tables

#### **audio_analyses**
Main table storing complete analysis results for each audio file:
- `id`: Unique identifier
- `filename`: Name of the audio file
- `file_path`: Path to the file
- `file_size`: Size in bytes
- `duration`: Duration in seconds
- `transcript`: Full transcription text
- `transcription_confidence`: Confidence score from transcription
- `sentiment_label`: Overall sentiment label (positive, negative, neutral)
- `sentiment_score`: Overall sentiment score (float)
- `sentiment_details`: JSON with detailed sentiment breakdown (VADER, TextBlob, etc.)
- `processing_time`: How long the analysis took
- `created_at`, `updated_at`: Timestamps

#### **speaker_segments**
Table storing segment-level analysis (time slices or speaker turns):
- `id`: Unique identifier
- `audio_analysis_id`: Foreign key to `audio_analyses`
- `speaker_id`: (Optional) Speaker label
- `start_time`, `end_time`: Segment timing
- `transcript`: Text for this segment
- `sentiment_label`: Sentiment for this segment
- `sentiment_score`: Score for this segment
- `sentiment_details`: JSON with detailed breakdown for this segment
- `created_at`: Timestamp

### Database Operations
- **Automatic Storage**: Every analysis is automatically stored in the database
- **Query Results**: Use `/analyses` endpoint to retrieve all analyses
- **Individual Results**: Use `/analyses/{id}` to get specific analysis details
- **Statistics**: Use `/statistics` endpoint for aggregated data

### Database Location
- **Default**: `./sentiment_analysis.db` (in project root)
- **Configurable**: Set `DATABASE_URL` in `.env` file
- **Backup**: Consider backing up this file for production use

---

## Examples

Comprehensive API usage examples are available in the `examples/` folder:

### 📁 Available Examples
- **`api_usage_examples.py`** - Python script with complete API client
- **`curl_examples.sh`** - Bash script with cURL commands
- **`nodejs_examples.js`** - Node.js script with axios
- **`README.md`** - Detailed documentation for all examples

### 🚀 Quick Start with Examples
```bash
# Start the API server
python main.py --mode api

# Run Python examples
python examples/api_usage_examples.py

# Run cURL examples
./examples/curl_examples.sh

# Run Node.js examples
node examples/nodejs_examples.js
```

### 📋 What Examples Cover
- Health checks and system status
- Text sentiment analysis (multiple languages)
- Audio file analysis and processing
- Database operations (list, get, delete)
- Statistics and analytics
- Graceful shutdown procedures

See `examples/README.md` for detailed documentation and troubleshooting.

---

## More
- See the rest of this README for architecture, setup, and advanced usage.
- For full API details, use the Swagger UI or ReDoc links above. 