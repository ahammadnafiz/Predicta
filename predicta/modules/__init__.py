"""
Predicta Modules

This package contains all the functional modules for the Predicta ML application.
"""

from .data_exploration import DataExplorer, DataOverview
from .feature_cleaning import DataImputer, OutlierDetector
from .feature_engineering import DataEncoder, DataTransformer
from .feature_selection import FeatureImportanceAnalyzer, BestParam
from .ml_models import ClassificationModel, RegressionModel, ImageModel, PredictAlgo, PredictImageAlgo
from .code_editor import CodeEditor

__all__ = [
    # Data Exploration
    'DataExplorer',
    'DataOverview',
    
    # Feature Cleaning
    'DataImputer',
    'OutlierDetector',
    
    # Feature Engineering
    'DataEncoder',
    'DataTransformer',
    
    # Feature Selection
    'FeatureImportanceAnalyzer',
    'BestParam',
    
    # ML Models
    'ClassificationModel',
    'RegressionModel',
    'ImageModel',
    'PredictAlgo',
    'PredictImageAlgo',
    
    # Code Editor
    'CodeEditor'
]
