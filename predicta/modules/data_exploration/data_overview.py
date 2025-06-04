"""
Data Overview Module

This module provides functionality for displaying dataset overviews and basic information.
"""

import pandas as pd
import streamlit as st
from ...core.logging_config import get_logger
from ...utils.code_display import ShowCode

logger = get_logger(__name__)


class DataOverview:
    """
    Provides comprehensive overview functionality for datasets.
    
    This class handles displaying dataset information, basic statistics,
    and descriptive summaries in a user-friendly format.
    """
    
    def __init__(self, data):
        """
        Initialize DataOverview with dataset.
        
        Args:
            data (pd.DataFrame): The dataset to analyze
            
        Raises:
            ValueError: If data is not a pandas DataFrame
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")
            
            self.data = data
            self.view_code = ShowCode()
            self.view_code.set_target_class(DataOverview)
            
            logger.info(f"DataOverview initialized with dataset shape: {data.shape}")
            
        except Exception as e:
            logger.error(f"Error initializing DataOverview: {str(e)}")
            raise
    
    def _display_info(self):
        """
        Displays summary information about the dataset.
        
        Shows basic dataset metrics including row count, column count,
        data types, and non-null counts for each column.
        """
        try:
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Dataset Information</h1>", 
                       unsafe_allow_html=True)
            
            st.write(f"Rows: {self.data.shape[0]}")
            st.write(f"Columns: {self.data.shape[1]}")
            st.write("Info:")
            
            # Get the summary information of the DataFrame
            summary_info = self.data.dtypes.reset_index()
            summary_info.columns = ['Column', 'Dtype']
            summary_info['Non-Null Count'] = self.data.count().values
            
            # Create DataFrame from the summary information
            summary = pd.DataFrame(summary_info)
            st.write(summary)
            
            logger.info("Dataset information displayed successfully")
            
        except Exception as e:
            logger.error(f"Error displaying dataset info: {str(e)}")
            st.error(f"Error displaying dataset information: {str(e)}")
    
    def _describe(self):
        """
        Generates descriptive statistics for all columns.
        
        Returns:
            pd.DataFrame: Descriptive statistics for the dataset
        """
        try:
            # Generate descriptive statistics
            profile = self.data.describe(include='all')
            
            logger.info("Descriptive statistics generated successfully")
            return profile
            
        except Exception as e:
            logger.error(f"Error generating descriptive statistics: {str(e)}")
            st.error(f"Error generating descriptive statistics: {str(e)}")
            return pd.DataFrame()
    
    def data_overview(self):
        """
        Main method to display the complete dataset overview interface.
        
        Provides a user interface with options to view dataset information
        and descriptive statistics.
        """
        try:
            st.markdown("<h1 style='text-align: center; font-size: 30px;'>Dataset Overview</h1>", 
                       unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", 
                       unsafe_allow_html=True)
            
            st.dataframe(self.data, width=800)
            
            overview_option = st.sidebar.selectbox(
                "Select Overview Option",
                ["Dataset Information", "Describe"]
            )
            
            if overview_option == "Dataset Information":
                self._display_info()
                if st.checkbox('Show Code'):
                    self.view_code._display_code('_display_info')
                    
            elif overview_option == "Describe":
                st.markdown("<h1 style='text-align: center; font-size: 25px;'>Descriptive Statistics</h1>", 
                           unsafe_allow_html=True)
                descriptive_stats = self._describe()
                if not descriptive_stats.empty:
                    st.dataframe(descriptive_stats, width=800)
                if st.checkbox('Show Code'):
                    self.view_code._display_code('_describe')
            
            logger.info(f"Data overview completed for option: {overview_option}")
            
        except Exception as e:
            logger.error(f"Error in data overview: {str(e)}")
            st.error(f"Error displaying data overview: {str(e)}")


def create_data_overview(data):
    """
    Factory function to create DataOverview instance.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        
    Returns:
        DataOverview: Configured DataOverview instance
    """
    try:
        return DataOverview(data)
    except Exception as e:
        logger.error(f"Error creating DataOverview instance: {str(e)}")
        raise
