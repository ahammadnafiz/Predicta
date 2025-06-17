"""
Pages Module

This module contains all the Streamlit pages for the Predicta application.
"""

from .prediction_app import PredictionApp
from .predicta_ai import DataApp

__all__ = ['PredictionApp', 'DataApp']