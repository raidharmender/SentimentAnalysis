#!/usr/bin/env python3
"""
Main entry point for the Sentiment Analysis System
"""

import uvicorn
import argparse
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.api import app
from app.database import create_tables
from app.config import settings
from app.logging_config import setup_logging_from_config, get_api_logger, get_dashboard_logger


def run_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Run the FastAPI server"""
    logger = get_api_logger()
    logger.info(f"Starting Sentiment Analysis API server on {host}:{port}")
    logger.info(f"API Documentation: http://{host}:{port}/docs")
    logger.info(f"Health Check: http://{host}:{port}/health")
    
    uvicorn.run(
        "app.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def run_dashboard(port: int = 8501):
    """Run the Streamlit dashboard"""
    import subprocess
    import sys
    
    logger = get_dashboard_logger()
    logger.info(f"Starting Streamlit dashboard on port {port}")
    logger.info(f"Dashboard URL: http://localhost:{port}")
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app/dashboard.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0"
    ])


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis System")
    parser.add_argument(
        "--mode",
        choices=["api", "dashboard", "both"],
        default="api",
        help="Run mode: api, dashboard, or both"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address for API server"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for API server"
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Port for Streamlit dashboard"
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload for API server"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    setup_logging_from_config()
    
    # Create database tables
    logger = get_api_logger()
    logger.info("Initializing database...")
    create_tables()
    logger.info("Database initialized successfully!")
    
    if args.mode == "api":
        run_api_server(args.host, args.api_port, not args.no_reload)
    elif args.mode == "dashboard":
        run_dashboard(args.dashboard_port)
    elif args.mode == "both":
        import threading
        import time
        
        # Start API server in a separate thread
        api_thread = threading.Thread(
            target=run_api_server,
            args=(args.host, args.api_port, not args.no_reload),
            daemon=True
        )
        api_thread.start()
        
        # Wait a moment for API to start
        time.sleep(2)
        
        # Start dashboard
        run_dashboard(args.dashboard_port)


if __name__ == "__main__":
    main() 