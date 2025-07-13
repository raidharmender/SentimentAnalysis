#!/usr/bin/env python3
"""
Setup script for logging configuration.
This script helps users configure logging for the Sentiment Analysis System.
"""

import os
import sys
import shutil
from pathlib import Path

def setup_logging():
    """Setup logging configuration for the application."""
    
    print("🎤 Sentiment Analysis System - Logging Setup")
    print("=" * 50)
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"✅ Created logs directory: {logs_dir.absolute()}")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("\n📝 Creating .env file from template...")
        if Path("logging_config.env.example").exists():
            shutil.copy("logging_config.env.example", ".env")
            print("✅ Created .env file from template")
        else:
            print("⚠️  Template file not found. Creating basic .env file...")
            create_basic_env_file()
    else:
        print("✅ .env file already exists")
    
    # Create log files
    log_files = [
        "logs/api.log",
        "logs/dashboard.log", 
        "logs/general.log",
        "logs/errors.log"
    ]
    
    for log_file in log_files:
        Path(log_file).touch(exist_ok=True)
        print(f"✅ Created log file: {log_file}")
    
    print("\n🎉 Logging setup completed!")
    print("\n📋 Log files will be created in the 'logs' directory:")
    print("   - api.log: API server logs")
    print("   - dashboard.log: Dashboard logs") 
    print("   - general.log: General application logs")
    print("   - errors.log: Error logs from all components")
    
    print("\n⚙️  Logging can be configured via environment variables:")
    print("   - API_LOG_LEVEL: Log level for API (DEBUG, INFO, WARNING, ERROR)")
    print("   - DASHBOARD_LOG_LEVEL: Log level for dashboard")
    print("   - GENERAL_LOG_LEVEL: Log level for general logs")
    print("   - LOG_DIR: Directory for log files")
    print("   - LOG_MAX_FILE_SIZE: Maximum log file size in bytes")
    print("   - LOG_BACKUP_COUNT: Number of backup log files")
    
    print("\n🚀 To start the application with logging:")
    print("   python main.py --mode api")
    print("   python main.py --mode dashboard")
    print("   python main.py --mode both")

def create_basic_env_file():
    """Create a basic .env file with logging configuration."""
    env_content = """# Sentiment Analysis System - Basic Logging Configuration

# Log Levels
API_LOG_LEVEL=INFO
DASHBOARD_LOG_LEVEL=INFO
GENERAL_LOG_LEVEL=INFO

# Logging Configuration
LOG_DIR=logs
LOG_MAX_FILE_SIZE=10485760
LOG_BACKUP_COUNT=5

# Development Mode
DEBUG=true
ENVIRONMENT=development
"""
    
    with open(".env", "w") as f:
        f.write(env_content)

def show_log_levels():
    """Show available log levels."""
    print("\n📊 Available Log Levels:")
    print("   DEBUG: Detailed information for debugging")
    print("   INFO: General information about program execution")
    print("   WARNING: Indicate a potential problem")
    print("   ERROR: A more serious problem")
    print("   CRITICAL: A critical problem that may prevent the program from running")

def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage: python setup_logging.py [--help]")
            print("\nOptions:")
            print("  --help    Show this help message")
            return
        elif sys.argv[1] == "--levels":
            show_log_levels()
            return
    
    setup_logging()

if __name__ == "__main__":
    main() 