# 🚀 Deployment Guide

This guide covers different deployment options for the Sentiment Analysis System.

## 📋 Prerequisites

- Python 3.12+
- FFmpeg
- At least 4GB RAM (8GB+ recommended)
- 2GB+ free disk space for models

## 🏠 Local Development

### Quick Start
```bash
# Clone and setup
git clone <repository-url>
cd SentimentAnalysis_1

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Start the system
python main.py --mode both
```

### Manual Setup
```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p uploads processed data logs

# Initialize database
python -c "from app.database import create_tables; create_tables()"

# Start the system
python main.py --mode both
```

## 🐳 Docker Deployment

### Single Container
```bash
# Build and run
docker build -t sentiment-analysis .
docker run -p 8000:8000 -p 8501:8501 sentiment-analysis
```

### Multi-Container with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  sentiment-analysis:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    volumes:
      - ./uploads:/app/uploads
      - ./processed:/app/processed
      - ./data:/app/data
    environment:
      - DATABASE_URL=postgresql://postgres:${DB_PASSWORD}@db:5432/sentiment_analysis
      - REDIS_URL=redis://redis:6379
      - WHISPER_MODEL=base
      - SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
    depends_on:
      - db
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=sentiment_analysis
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - sentiment-analysis
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

## ☁️ Cloud Deployment

### AWS Deployment

#### EC2 Instance
```bash
# Launch EC2 instance (t3.large or larger)
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv ffmpeg git

# Clone repository
git clone <repository-url>
cd SentimentAnalysis_1

# Setup application
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/sentiment-analysis.service << EOF
[Unit]
Description=Sentiment Analysis System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/SentimentAnalysis_1
Environment=PATH=/home/ubuntu/SentimentAnalysis_1/.venv/bin
ExecStart=/home/ubuntu/SentimentAnalysis_1/.venv/bin/python main.py --mode both
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl enable sentiment-analysis
sudo systemctl start sentiment-analysis
```

#### ECS/Fargate
```yaml
# task-definition.json
{
  "family": "sentiment-analysis",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "sentiment-analysis",
      "image": "your-account.dkr.ecr.region.amazonaws.com/sentiment-analysis:latest",
      "portMappings": [
        {"containerPort": 8000, "protocol": "tcp"},
        {"containerPort": 8501, "protocol": "tcp"}
      ],
      "environment": [
        {"name": "DATABASE_URL", "value": "postgresql://user:pass@rds-endpoint:5432/db"},
        {"name": "WHISPER_MODEL", "value": "base"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sentiment-analysis",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Compute Engine
```bash
# Create instance
gcloud compute instances create sentiment-analysis \
  --zone=us-central1-a \
  --machine-type=e2-standard-4 \
  --image-family=debian-11 \
  --image-project=debian-cloud \
  --boot-disk-size=50GB

# Install dependencies
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv ffmpeg git

# Setup application (same as EC2)
```

#### Cloud Run
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/sentiment-analysis', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/sentiment-analysis']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'sentiment-analysis'
      - '--image'
      - 'gcr.io/$PROJECT_ID/sentiment-analysis'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--memory'
      - '4Gi'
      - '--cpu'
      - '2'
```

### Azure

#### Azure Container Instances
```bash
# Build and push to Azure Container Registry
az acr build --registry myregistry --image sentiment-analysis .

# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name sentiment-analysis \
  --image myregistry.azurecr.io/sentiment-analysis:latest \
  --dns-name-label sentiment-analysis \
  --ports 8000 8501 \
  --memory 4 \
  --cpu 2
```

## 🔧 Production Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/database

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Audio Processing
TARGET_SAMPLE_RATE=16000
TARGET_CHANNELS=1

# Models
WHISPER_MODEL=base
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest

# File Upload
MAX_FILE_SIZE=52428800
ALLOWED_AUDIO_FORMATS=.wav,.mp3,.flac,.m4a

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/sentiment-analysis.log
```

### Nginx Configuration
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api {
        server localhost:8000;
    }
    
    upstream dashboard {
        server localhost:8501;
    }
    
    server {
        listen 80;
        server_name your-domain.com;
        
        # API
        location /api/ {
            proxy_pass http://api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Dashboard
        location / {
            proxy_pass http://dashboard/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

### SSL/TLS Configuration
```bash
# Using Let's Encrypt
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com

# Or using Cloudflare
# Configure Cloudflare SSL/TLS to "Full (strict)"
```

## 📊 Monitoring and Logging

### Application Logging
```python
# Add to app/config.py
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/sentiment-analysis.log'),
        logging.StreamHandler()
    ]
)
```

### Health Checks
```bash
# API Health Check
curl http://localhost:8000/health

# System Status
curl http://localhost:8000/status
```

### Prometheus Metrics (Optional)
```python
# Add to requirements.txt
prometheus-client==0.17.1

# Add metrics endpoint
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
ANALYSIS_COUNTER = Counter('sentiment_analysis_total', 'Total analyses performed')
ANALYSIS_DURATION = Histogram('sentiment_analysis_duration_seconds', 'Analysis duration')

# Add to API
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## 🔒 Security Considerations

### Authentication
```python
# Add to requirements.txt
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4

# Implement JWT authentication
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
```

### Rate Limiting
```python
# Add to requirements.txt
slowapi==0.1.8

# Implement rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze_audio(request: Request, ...):
    # Your existing code
```

### Input Validation
```python
# Add file size and type validation
def validate_audio_file(file: UploadFile):
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    if not file.filename.lower().endswith(tuple(settings.ALLOWED_AUDIO_FORMATS)):
        raise HTTPException(status_code=400, detail="Invalid file type")
```

## 🚀 Scaling Considerations

### Horizontal Scaling
- Use load balancer (nginx, AWS ALB, GCP LB)
- Implement session management with Redis
- Use shared storage for uploaded files (S3, GCS, Azure Blob)
- Consider using message queues for async processing

### Vertical Scaling
- Increase CPU/memory for larger models
- Use GPU instances for faster processing
- Optimize database queries and indexing
- Implement caching strategies

### Database Scaling
```sql
-- Add indexes for better performance
CREATE INDEX idx_audio_analyses_created_at ON audio_analyses(created_at);
CREATE INDEX idx_audio_analyses_sentiment_label ON audio_analyses(sentiment_label);
CREATE INDEX idx_speaker_segments_analysis_id ON speaker_segments(audio_analysis_id);
```

## 🔄 Backup and Recovery

### Database Backup
```bash
# PostgreSQL backup
pg_dump sentiment_analysis > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump $DATABASE_URL > $BACKUP_DIR/backup_$DATE.sql
find $BACKUP_DIR -name "backup_*.sql" -mtime +7 -delete
```

### File Backup
```bash
# Backup uploaded and processed files
tar -czf backup_files_$(date +%Y%m%d_%H%M%S).tar.gz uploads/ processed/
```

## 📈 Performance Optimization

### Model Optimization
```python
# Use smaller models for faster processing
WHISPER_MODEL = "tiny"  # Instead of "base"
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Enable model caching
import torch
torch.hub.set_dir('/app/models')
```

### Caching
```python
# Add Redis caching
import redis
import json

redis_client = redis.Redis.from_url(settings.REDIS_URL)

def get_cached_result(analysis_id: int):
    cached = redis_client.get(f"analysis:{analysis_id}")
    return json.loads(cached) if cached else None

def cache_result(analysis_id: int, result: dict):
    redis_client.setex(f"analysis:{analysis_id}", 3600, json.dumps(result))
```

### Async Processing
```python
# Use Celery for background processing
from celery import Celery

celery_app = Celery('sentiment_analysis', broker=settings.REDIS_URL)

@celery_app.task
def analyze_audio_async(file_path: str):
    # Your analysis code here
    pass
```

## 🆘 Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce model size
   - Increase system memory
   - Use model quantization

2. **Slow Processing**
   - Use GPU instances
   - Implement caching
   - Optimize audio preprocessing

3. **Database Connection Issues**
   - Check connection string
   - Verify database is running
   - Check firewall settings

4. **Model Download Issues**
   - Check internet connection
   - Verify model names
   - Clear model cache

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py --mode api
```

### Performance Profiling
```python
# Add profiling
import cProfile
import pstats

def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        return result
    return wrapper
```

## 📞 Support

For deployment issues:
1. Check the logs: `docker-compose logs sentiment-analysis`
2. Verify system requirements
3. Test with the example script: `python examples/example_usage.py`
4. Check the API documentation: `http://localhost:8000/docs`
5. Review the system status: `http://localhost:8000/status` 