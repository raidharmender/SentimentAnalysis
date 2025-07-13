# API Usage Examples

This folder contains comprehensive examples for using the Sentiment Analysis API with different programming languages and tools.

## 📁 Files Overview

### 1. `api_usage_examples.py`
**Python script** demonstrating all API endpoints using the `requests` library.

**Prerequisites:**
```bash
pip install requests
```

**Usage:**
```bash
python api_usage_examples.py
```

**Features:**
- Complete API client class (`SentimentAnalysisAPI`)
- All endpoint examples (health, status, text analysis, audio analysis, etc.)
- Error handling and connection validation
- Formatted output with JSON responses

### 2. `curl_examples.sh`
**Bash script** with cURL commands for all API endpoints.

**Prerequisites:**
```bash
# Make executable
chmod +x curl_examples.sh
```

**Usage:**
```bash
./curl_examples.sh
```

**Features:**
- Colored output for better readability
- Real-time API calls with live responses
- Safety measures (destructive operations commented out)
- Comprehensive endpoint coverage

### 3. `nodejs_examples.js`
**Node.js script** demonstrating API usage with `axios`.

**Prerequisites:**
```bash
npm install axios form-data
```

**Usage:**
```bash
node nodejs_examples.js
```

**Features:**
- Async/await pattern
- File upload handling
- Error handling and timeout configuration
- Modular API client class

## 🚀 Quick Start

1. **Start the API server:**
   ```bash
   python main.py --mode api
   ```

2. **Run any example:**
   ```bash
   # Python
   python examples/api_usage_examples.py
   
   # cURL
   ./examples/curl_examples.sh
   
   # Node.js
   node examples/nodejs_examples.js
   ```

## 📋 Endpoints Covered

All examples demonstrate these endpoints:

- `GET /health` - Health check
- `GET /status` - System status
- `GET /languages` - Supported languages
- `POST /analyze` - Audio file analysis
- `POST /analyze/text` - Text analysis
- `GET /analyses` - List all analyses
- `GET /analyses/{id}` - Get specific analysis
- `GET /statistics` - System statistics
- `DELETE /analyses/{id}` - Delete analysis
- `POST /shutdown` - Graceful shutdown

## 🔧 Customization

### Change API URL
Edit the `BASE_URL` or `base_url` variable in each file:
- Python: `base_url: str = "http://localhost:8000"`
- cURL: `BASE_URL="http://localhost:8000"`
- Node.js: `baseURL = 'http://localhost:8000'`

### Add Authentication
For production use, add authentication headers:
```python
# Python
headers = {'Authorization': 'Bearer your-token'}
response = requests.get(url, headers=headers)

# Node.js
this.client.defaults.headers.common['Authorization'] = 'Bearer your-token';
```

## ⚠️ Safety Notes

- **Graceful Shutdown**: The shutdown endpoint will terminate the server process
- **Delete Operations**: Delete examples are commented out for safety
- **File Uploads**: Audio analysis requires valid audio files (WAV, MP3, FLAC, M4A, OPUS)

## 🧪 Testing Different Scenarios

### Sentiment Analysis Examples
All scripts include examples for:
- **Positive sentiment**: "I am absolutely delighted with the service!"
- **Neutral sentiment**: "The product arrived on time and was as described."
- **Negative sentiment**: "I'm very disappointed with the quality."

### Language Support
- **English**: Default language
- **Chinese**: If supported by your installation
- **Malay**: If supported by your installation

### Audio Analysis
- Requires sample audio files in the `incoming/` directory
- Automatically skips if files are not available
- Demonstrates file upload and processing

## 📊 Expected Output

All examples provide:
- Formatted section headers
- JSON responses with proper indentation
- Error handling for connection issues
- Success/failure status messages

## 🔍 Troubleshooting

### Connection Errors
```
Error: Could not connect to the API server.
Make sure the server is running on http://localhost:8000
```
**Solution:** Start the API server with `python main.py --mode api`

### File Not Found
```
Sample audio file not found: incoming/Shoppee_Happy_1.opus
```
**Solution:** Place audio files in the `incoming/` directory or update the file path

### Module Errors
```
ModuleNotFoundError: No module named 'requests'
```
**Solution:** Install required dependencies:
```bash
pip install requests  # For Python
npm install axios form-data  # For Node.js
```

## 📝 Contributing

To add new examples:
1. Create a new file in the `examples/` directory
2. Follow the existing naming convention
3. Include comprehensive error handling
4. Add documentation in this README
5. Test with a running API server 