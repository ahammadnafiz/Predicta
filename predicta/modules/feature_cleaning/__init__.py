"""
Feature Cleaning Module

This module provides functionality for data cleaning operations including
handling missing data and outlier detection/treatment.
"""

from .missing_data import DataImputer
from .outlier import OutlierDetector

__all__ = ['DataImputer', 'OutlierDetector']
