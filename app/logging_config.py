"""
Logging configuration for the Sentiment Analysis System.
Provides configurable logging with separate log files for API and dashboard components.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from app.config import settings


class LoggingConfig:
    """Configuration class for application logging."""
    
    # Default log levels
    DEFAULT_API_LOG_LEVEL = "INFO"
    DEFAULT_DASHBOARD_LOG_LEVEL = "INFO"
    DEFAULT_GENERAL_LOG_LEVEL = "INFO"
    
    # Log file paths
    LOG_DIR = "logs"
    API_LOG_FILE = "api.log"
    DASHBOARD_LOG_FILE = "dashboard.log"
    GENERAL_LOG_FILE = "general.log"
    ERROR_LOG_FILE = "errors.log"
    
    # Log format
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DETAILED_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    
    @classmethod
    def setup_logging(
        cls,
        api_log_level: Optional[str] = None,
        dashboard_log_level: Optional[str] = None,
        general_log_level: Optional[str] = None,
        log_dir: Optional[str] = None,
        enable_console: bool = True,
        enable_file: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ) -> None:
        """
        Setup logging configuration for the application.
        
        Args:
            api_log_level: Log level for API component
            dashboard_log_level: Log level for dashboard component
            general_log_level: Log level for general application logs
            log_dir: Directory to store log files
            enable_console: Whether to enable console logging
            enable_file: Whether to enable file logging
            max_file_size: Maximum size of log files before rotation
            backup_count: Number of backup log files to keep
        """
        # Create log directory
        log_dir = log_dir or cls.LOG_DIR
        Path(log_dir).mkdir(exist_ok=True)
        
        # Get log levels from environment or use defaults
        api_log_level = api_log_level or os.getenv("API_LOG_LEVEL", cls.DEFAULT_API_LOG_LEVEL)
        dashboard_log_level = dashboard_log_level or os.getenv("DASHBOARD_LOG_LEVEL", cls.DEFAULT_DASHBOARD_LOG_LEVEL)
        general_log_level = general_log_level or os.getenv("GENERAL_LOG_LEVEL", cls.DEFAULT_GENERAL_LOG_LEVEL)
        
        # Clear existing handlers
        logging.getLogger().handlers.clear()
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Create formatters
        console_formatter = logging.Formatter(cls.LOG_FORMAT)
        file_formatter = logging.Formatter(cls.DETAILED_LOG_FORMAT)
        
        # Setup console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # Setup file handlers
        if enable_file:
            # General application log
            general_handler = cls._create_rotating_file_handler(
                os.path.join(log_dir, cls.GENERAL_LOG_FILE),
                general_log_level,
                file_formatter,
                max_file_size,
                backup_count
            )
            root_logger.addHandler(general_handler)
            
            # Error log (all errors from all components)
            error_handler = cls._create_rotating_file_handler(
                os.path.join(log_dir, cls.ERROR_LOG_FILE),
                "ERROR",
                file_formatter,
                max_file_size,
                backup_count
            )
            root_logger.addHandler(error_handler)
        
        # Setup component-specific loggers
        cls._setup_api_logger(log_dir, api_log_level, file_formatter, enable_file, max_file_size, backup_count)
        cls._setup_dashboard_logger(log_dir, dashboard_log_level, file_formatter, enable_file, max_file_size, backup_count)
        
        # Log startup message
        logger = logging.getLogger("app.startup")
        logger.info("Logging system initialized successfully")
        logger.info(f"API log level: {api_log_level}")
        logger.info(f"Dashboard log level: {dashboard_log_level}")
        logger.info(f"General log level: {general_log_level}")
        logger.info(f"Log directory: {log_dir}")
    
    @classmethod
    def _create_rotating_file_handler(
        cls,
        file_path: str,
        level: str,
        formatter: logging.Formatter,
        max_file_size: int,
        backup_count: int
    ) -> logging.handlers.RotatingFileHandler:
        """Create a rotating file handler."""
        handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        handler.setLevel(getattr(logging, level.upper()))
        handler.setFormatter(formatter)
        return handler
    
    @classmethod
    def _setup_api_logger(
        cls,
        log_dir: str,
        log_level: str,
        formatter: logging.Formatter,
        enable_file: bool,
        max_file_size: int,
        backup_count: int
    ) -> None:
        """Setup API-specific logger."""
        api_logger = logging.getLogger("app.api")
        api_logger.setLevel(getattr(logging, log_level.upper()))
        
        if enable_file:
            api_handler = cls._create_rotating_file_handler(
                os.path.join(log_dir, cls.API_LOG_FILE),
                log_level,
                formatter,
                max_file_size,
                backup_count
            )
            api_logger.addHandler(api_handler)
    
    @classmethod
    def _setup_dashboard_logger(
        cls,
        log_dir: str,
        log_level: str,
        formatter: logging.Formatter,
        enable_file: bool,
        max_file_size: int,
        backup_count: int
    ) -> None:
        """Setup dashboard-specific logger."""
        dashboard_logger = logging.getLogger("app.dashboard")
        dashboard_logger.setLevel(getattr(logging, log_level.upper()))
        
        if enable_file:
            dashboard_handler = cls._create_rotating_file_handler(
                os.path.join(log_dir, cls.DASHBOARD_LOG_FILE),
                log_level,
                formatter,
                max_file_size,
                backup_count
            )
            dashboard_logger.addHandler(dashboard_handler)
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.
        
        Args:
            name: Logger name (e.g., 'app.api', 'app.dashboard', 'app.sentiment')
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)


def setup_logging_from_config() -> None:
    """Setup logging using configuration from settings."""
    # Clean environment variables by removing any inline comments
    def clean_env_var(var_name: str, default: str) -> str:
        value = os.getenv(var_name, default)
        if value and '#' in value:
            return value.split('#')[0].strip()
        return value
    
    LoggingConfig.setup_logging(
        api_log_level=os.getenv("API_LOG_LEVEL"),
        dashboard_log_level=os.getenv("DASHBOARD_LOG_LEVEL"),
        general_log_level=os.getenv("GENERAL_LOG_LEVEL"),
        log_dir=os.getenv("LOG_DIR"),
        enable_console=not settings.is_production,  # Disable console in production
        enable_file=True,
        max_file_size=int(clean_env_var("LOG_MAX_FILE_SIZE", "10485760")),  # 10MB default
        backup_count=int(clean_env_var("LOG_BACKUP_COUNT", "5"))
    )


# Convenience functions for getting loggers
def get_api_logger() -> logging.Logger:
    """Get the API logger."""
    return LoggingConfig.get_logger("app.api")


def get_dashboard_logger() -> logging.Logger:
    """Get the dashboard logger."""
    return LoggingConfig.get_logger("app.dashboard")


def get_sentiment_logger() -> logging.Logger:
    """Get the sentiment analysis logger."""
    return LoggingConfig.get_logger("app.sentiment")


def get_audio_logger() -> logging.Logger:
    """Get the audio processing logger."""
    return LoggingConfig.get_logger("app.audio")


def get_database_logger() -> logging.Logger:
    """Get the database logger."""
    return LoggingConfig.get_logger("app.database")


def get_transcription_logger() -> logging.Logger:
    """Get the transcription logger."""
    return LoggingConfig.get_logger("app.transcription") 