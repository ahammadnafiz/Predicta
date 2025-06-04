"""Configuration module for Predicta."""

import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration class for Predicta application."""
    
    # Application settings
    APP_NAME = "Predicta"
    APP_VERSION = "1.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    ASSETS_DIR = BASE_DIR / "assets"
    TEMP_DIR = Path.home() / ".predicta" / "temp"
    LOGS_DIR = Path.home() / ".predicta" / "logs"
    MODELS_DIR = Path.home() / ".predicta" / "models"
    
    # Streamlit settings
    STREAMLIT_CONFIG = {
        "page_title": APP_NAME,
        "page_icon": "⚡",
        "initial_sidebar_state": "expanded",
    }
    
    # ML settings
    ML_CONFIG = {
        "random_state": 42,
        "test_size": 0.2,
        "cv_folds": 5,
        "max_features": 1000,
    }
    
    # Logging settings
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_rotation": "midnight",
        "backup_count": 7,
    }
    
    # File upload settings
    UPLOAD_CONFIG = {
        "max_file_size": 200,  # MB
        "allowed_extensions": [".csv", ".xlsx", ".xls"],
        "encoding_options": ["utf-8", "latin-1", "ISO-8859-1"],
    }
    
    # Image processing settings
    IMAGE_CONFIG = {
        "max_size": (1024, 1024),
        "allowed_formats": ["PNG", "JPEG", "JPG"],
        "quality": 95,
    }
    
    @classmethod
    def get_temp_file_path(cls, filename: str) -> Path:
        """Get temporary file path for a given filename."""
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        return cls.TEMP_DIR / filename
    
    @classmethod
    def get_log_file_path(cls, filename: str) -> Path:
        """Get log file path for a given filename."""
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.LOGS_DIR / filename
    
    @classmethod
    def get_model_file_path(cls, filename: str) -> Path:
        """Get model file path for a given filename."""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return cls.MODELS_DIR / filename
    
    @classmethod
    def get_asset_path(cls, filename: str) -> str:
        """Get asset file path for a given filename."""
        asset_path = cls.ASSETS_DIR / filename
        return str(asset_path)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not attr.startswith("_") and not callable(getattr(cls, attr))
        }
