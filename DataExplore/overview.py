import pandas as pd
import streamlit as st

class DataOverview:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        self.data = data
    
    def _display_info(self):
        """
        Displays summary information about the dataset.
        """
        st.markdown("<h1 style='text-align: center; font-size: 25px;'>Dataset Information</h1>", unsafe_allow_html=True)
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
    
    def _describe(self):
        """Generates descriptive statistics for numerical columns."""
        #data = self.convert_to_float(self.data)

        # Generate descriptive statistics
        profile = self.data.describe(include='all')
        return profile

    def data_overview(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Dataset Overview</h1>", unsafe_allow_html=True)       
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)
        
        overview_option = st.sidebar.selectbox("Select Overview Option",[
                                                                        "Dataset Information",
                                                                        "Describe"
                                                                        ])
        if overview_option == "Dataset Information":
            self._display_info()
        elif overview_option == "Describe":
            st.markdown("<h1 style='text-align: center; font-size: 25px;'>Descriptive Statistics</h1>", unsafe_allow_html=True)
            st.dataframe(self._describe(), width=800)