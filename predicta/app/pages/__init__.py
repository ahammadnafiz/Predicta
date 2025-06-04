"""
Pages Module

This module contains all the Streamlit pages for the Predicta application.
"""

from .prediction_app import PredictionApp
from .predicta_chat import PredictaChat
from .predicta_viz import PredictaViz

__all__ = ['PredictionApp', 'PredictaChat', 'PredictaViz']
