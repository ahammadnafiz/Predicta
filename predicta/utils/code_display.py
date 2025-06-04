"""
Code Display Utility

This module provides functionality for displaying source code of methods in Streamlit applications.
"""

import inspect
import streamlit as st
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class ShowCode:
    """
    Utility class for displaying Python source code in Streamlit applications.
    
    This class allows users to view the source code of specific methods
    from target classes, which is useful for educational and debugging purposes.
    """
    
    def __init__(self):
        """Initialize ShowCode with no target class set."""
        self.target_class = None
        logger.debug("ShowCode instance initialized")
    
    def set_target_class(self, target_class):
        """
        Set the target class for code inspection.
        
        Args:
            target_class: The class whose methods will be inspected
        """
        try:
            self.target_class = target_class
            logger.debug(f"Target class set to: {target_class.__name__}")
        except Exception as e:
            logger.error(f"Error setting target class: {str(e)}")
            raise
    
    def _generate_code(self, func_name):
        """
        Generate source code for a specific function.
        
        Args:
            func_name (str): Name of the function to inspect
            
        Returns:
            str: Source code of the function
            
        Raises:
            ValueError: If target class is not set
            AttributeError: If function doesn't exist in target class
        """
        try:
            if self.target_class is None:
                raise ValueError("Target class not set. Use set_target_class() method.")
            
            func = getattr(self.target_class, func_name)
            source = inspect.getsource(func)
            
            logger.debug(f"Generated source code for function: {func_name}")
            return source
            
        except AttributeError as e:
            logger.error(f"Function '{func_name}' not found in target class: {str(e)}")
            raise AttributeError(f"Function '{func_name}' not found in target class")
        except Exception as e:
            logger.error(f"Error generating code for '{func_name}': {str(e)}")
            raise
    
    def _display_code(self, func_name):
        """
        Display source code for a specific function in Streamlit.
        
        Args:
            func_name (str): Name of the function to display
        """
        try:
            code = self._generate_code(func_name)
            st.code(code, language='python')
            logger.debug(f"Displayed code for function: {func_name}")
            
        except Exception as e:
            logger.error(f"Error displaying code for '{func_name}': {str(e)}")
            st.error(f"Error displaying code: {str(e)}")
    
    def display_multiple_functions(self, func_names):
        """
        Display source code for multiple functions.
        
        Args:
            func_names (list): List of function names to display
        """
        try:
            for func_name in func_names:
                st.subheader(f"Function: {func_name}")
                self._display_code(func_name)
                st.markdown("---")
                
            logger.debug(f"Displayed code for {len(func_names)} functions")
            
        except Exception as e:
            logger.error(f"Error displaying multiple functions: {str(e)}")
            st.error(f"Error displaying functions: {str(e)}")
    
    def get_available_methods(self):
        """
        Get list of available methods in the target class.
        
        Returns:
            list: List of method names in the target class
        """
        try:
            if self.target_class is None:
                raise ValueError("Target class not set. Use set_target_class() method.")
            
            methods = [method for method in dir(self.target_class) 
                      if callable(getattr(self.target_class, method)) and not method.startswith('__')]
            
            logger.debug(f"Found {len(methods)} available methods in target class")
            return methods
            
        except Exception as e:
            logger.error(f"Error getting available methods: {str(e)}")
            return []
