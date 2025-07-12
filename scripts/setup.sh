#!/bin/bash

# Sentiment Analysis System Setup Script
# This script automates the installation and setup process

set -e  # Exit on any error

echo "🎤 Setting up Sentiment Analysis System..."
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.12+ is installed
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
        print_success "Python 3.12 found"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ "$PYTHON_VERSION" == "3.12" ]]; then
            PYTHON_CMD="python3"
            print_success "Python 3.12 found"
        else
            print_error "Python 3.12+ is required. Found version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3.12+ is not installed"
        exit 1
    fi
}

# Check if FFmpeg is installed
check_ffmpeg() {
    print_status "Checking FFmpeg installation..."
    
    if command -v ffmpeg &> /dev/null; then
        print_success "FFmpeg found"
    else
        print_warning "FFmpeg not found. Installing..."
        
        # Detect OS and install FFmpeg
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if command -v brew &> /dev/null; then
                brew install ffmpeg
                print_success "FFmpeg installed via Homebrew"
            else
                print_error "Homebrew not found. Please install FFmpeg manually: https://ffmpeg.org/download.html"
                exit 1
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux
            if command -v apt-get &> /dev/null; then
                sudo apt-get update
                sudo apt-get install -y ffmpeg
                print_success "FFmpeg installed via apt"
            elif command -v yum &> /dev/null; then
                sudo yum install -y ffmpeg
                print_success "FFmpeg installed via yum"
            else
                print_error "Package manager not found. Please install FFmpeg manually: https://ffmpeg.org/download.html"
                exit 1
            fi
        else
            print_error "Unsupported OS. Please install FFmpeg manually: https://ffmpeg.org/download.html"
            exit 1
        fi
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf .venv
    fi
    
    $PYTHON_CMD -m venv .venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi
    
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p uploads processed data logs
    
    print_success "Directories created"
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    
    python -c "
from app.database import create_tables
create_tables()
print('Database tables created successfully')
"
    
    print_success "Database initialized"
}

# Download models (optional)
download_models() {
    print_status "Downloading ML models (this may take a while)..."
    
    # Download Whisper model
    python -c "
import whisper
print('Downloading Whisper model...')
whisper.load_model('base')
print('Whisper model downloaded')
"
    
    # Download sentiment model
    python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print('Downloading sentiment model...')
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
print('Sentiment model downloaded')
"
    
    print_success "Models downloaded"
}

# Create configuration file
create_config() {
    print_status "Creating configuration file..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Sentiment Analysis System Configuration

# Database
DATABASE_URL=sqlite:///./sentiment_analysis.db

# Audio Processing
TARGET_SAMPLE_RATE=16000
TARGET_CHANNELS=1

# Whisper Model
WHISPER_MODEL=base

# Sentiment Analysis
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest

# File Upload
MAX_FILE_SIZE=52428800
ALLOWED_AUDIO_FORMATS=.wav,.mp3,.flac,.m4a

# Storage
UPLOAD_DIR=uploads
PROCESSED_DIR=processed
EOF
        print_success "Configuration file created (.env)"
    else
        print_warning "Configuration file already exists (.env)"
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    if command -v pytest &> /dev/null; then
        pytest tests/ -v
        print_success "Tests completed"
    else
        print_warning "pytest not found. Skipping tests."
    fi
}

# Main setup function
main() {
    echo "Starting setup process..."
    echo ""
    
    check_python
    check_ffmpeg
    create_venv
    activate_venv
    install_dependencies
    create_directories
    create_config
    init_database
    
    # Ask user if they want to download models
    echo ""
    read -p "Do you want to download ML models now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_models
    else
        print_warning "Models will be downloaded automatically when first used"
    fi
    
    # Ask user if they want to run tests
    echo ""
    read -p "Do you want to run tests? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    echo ""
    echo "=========================================="
    print_success "Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source .venv/bin/activate  # Linux/macOS"
    echo "   .venv\\Scripts\\activate     # Windows"
    echo ""
    echo "2. Start the API server:"
    echo "   python main.py --mode api"
    echo ""
    echo "3. Start the dashboard:"
    echo "   python main.py --mode dashboard"
    echo ""
    echo "4. Or run both:"
    echo "   python main.py --mode both"
    echo ""
    echo "API Documentation: http://localhost:8000/docs"
    echo "Dashboard: http://localhost:8501"
    echo ""
}

# Run main function
main "$@" 