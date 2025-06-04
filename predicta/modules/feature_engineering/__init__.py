"""
Feature Engineering Module

This module provides functionality for feature engineering operations including
encoding categorical variables and data transformations.
"""

from .encoding import DataEncoder
from .transform import DataTransformer

__all__ = ['DataEncoder', 'DataTransformer']
