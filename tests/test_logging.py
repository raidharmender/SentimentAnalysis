#!/usr/bin/env python3
"""
Test script for the logging system.
This script tests the logging configuration and functionality.
"""

import os
import sys
import time
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.logging_config import setup_logging_from_config, get_api_logger, get_dashboard_logger, get_sentiment_logger

def test_logging():
    """Test the logging system."""
    print("🧪 Testing Logging System")
    print("=" * 40)
    
    # Setup logging
    print("📝 Setting up logging...")
    setup_logging_from_config()
    
    # Get loggers
    api_logger = get_api_logger()
    dashboard_logger = get_dashboard_logger()
    sentiment_logger = get_sentiment_logger()
    
    print("✅ Logging setup completed")
    
    # Test different log levels
    print("\n📊 Testing log levels...")
    
    # API logger tests
    api_logger.debug("This is a DEBUG message from API")
    api_logger.info("This is an INFO message from API")
    api_logger.warning("This is a WARNING message from API")
    api_logger.error("This is an ERROR message from API")
    
    # Dashboard logger tests
    dashboard_logger.info("This is an INFO message from Dashboard")
    dashboard_logger.warning("This is a WARNING message from Dashboard")
    
    # Sentiment logger tests
    sentiment_logger.info("This is an INFO message from Sentiment Analysis")
    sentiment_logger.error("This is an ERROR message from Sentiment Analysis")
    
    print("✅ Log messages written")
    
    # Check if log files were created
    print("\n📁 Checking log files...")
    log_files = [
        "logs/api.log",
        "logs/dashboard.log",
        "logs/general.log",
        "logs/errors.log"
    ]
    
    for log_file in log_files:
        if Path(log_file).exists():
            size = Path(log_file).stat().st_size
            print(f"✅ {log_file} exists ({size} bytes)")
        else:
            print(f"❌ {log_file} not found")
    
    # Show sample log content
    print("\n📄 Sample log content:")
    for log_file in log_files:
        if Path(log_file).exists():
            print(f"\n--- {log_file} ---")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-3:]:  # Show last 3 lines
                    print(line.strip())
    
    print("\n🎉 Logging test completed!")

def test_log_rotation():
    """Test log rotation functionality."""
    print("\n🔄 Testing log rotation...")
    
    # Create a large log entry to test rotation
    api_logger = get_api_logger()
    
    large_message = "X" * 1000  # 1KB message
    
    for i in range(100):  # Write 100KB of logs
        api_logger.info(f"Test message {i}: {large_message}")
    
    print("✅ Log rotation test completed")

if __name__ == "__main__":
    test_logging()
    
    # Uncomment to test log rotation (creates large log files)
    # test_log_rotation() 