# Predicta Package
"""
Predicta - Professional Machine Learning Platform

A comprehensive, modular machine learning platform providing tools for data exploration,
feature engineering, model training, and prediction with a user-friendly Streamlit interface.

Main Features:
- Data exploration and visualization
- Feature cleaning and preprocessing
- Feature engineering and transformation
- Feature selection and importance analysis
- Multiple ML algorithms (classification, regression, image processing)
- Hyperparameter optimization
- Model training and evaluation
- Code generation and display
- Professional logging and error handling

Example usage:
    from predicta import modules
    from predicta.app import PredictaApp
    
    # Create main application
    app = PredictaApp(data)
    app.run()
    
    # Or use individual modules
    explorer = modules.DataExplorer(data)
    explorer.explore_data()
"""

__version__ = "2.0.0"
__author__ = "Ahammad Nafiz"
__email__ = "ahammadnafiz@outlook.com"

# Core imports for package initialization
from .core.config import Config
from .core.logging_config import setup_logging

# Initialize logging
setup_logging()

# Import main modules for easy access
from . import core
from . import modules
from . import utils
from . import ui

# Package metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "Config",
    "setup_logging",
    "core",
    "modules", 
    "utils",
    "ui"
]
