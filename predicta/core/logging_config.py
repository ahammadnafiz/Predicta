"""Logging configuration for Predicta."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import datetime
from .config import Config


class TorchErrorFilter(logging.Filter):
    """Filter to suppress PyTorch-related errors in Streamlit."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out torch-related error messages."""
        if "torch._C._get_custom_class_python_wrapper" in str(record.getMessage()):
            return False
        return True


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for Predicta.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("predicta")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt=Config.LOGGING_CONFIG["format"],
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(TorchErrorFilter())
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = f"predicta_{datetime.datetime.now().strftime('%Y%m%d')}.log"
    
    log_path = Config.get_log_file_path(log_file)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=Config.LOGGING_CONFIG["backup_count"]
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(TorchErrorFilter())
    logger.addHandler(file_handler)
    
    # Apply torch error filter to root logger as well
    root_logger = logging.getLogger()
    root_logger.addFilter(TorchErrorFilter())
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f"predicta.{name}")


# Module-level logger
logger = get_logger(__name__)
