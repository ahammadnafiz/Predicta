import pandas as pd
import numpy as np
import logging
import streamlit as st
import plotly.graph_objects as go
from show_code import ShowCode


class OutlierDetector:
    def __init__(self, data):
        self.data = data
        self.view_code = ShowCode()
        self.view_code.set_target_class(OutlierDetector)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        
        
    def outlier_detect_IQR(self, data, col, threshold=0.05):
        try:
            Q1 = np.percentile(data[col], 25)
            Q3 = np.percentile(data[col], 75)
            IQR = Q3 - Q1
            upper_fence = Q3 + 1.5 * IQR
            lower_fence = Q1 - 1.5 * IQR
            para = (upper_fence, lower_fence)
            outlier_index = data[(data[col] > upper_fence) | (data[col] < lower_fence)].index
            num_outliers = len(outlier_index)
            prop_outliers = num_outliers / len(data)
            st.write('Num of outlier detected:', num_outliers)
            st.write('Proportion of outlier detected:', prop_outliers)
            
            # Check if the proportion of outliers exceeds the threshold
            if prop_outliers > threshold:
                raise ValueError(f"Proportion of outliers ({prop_outliers:.2%}) exceeds the threshold ({threshold:.2%})")

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data[col],
                                    mode='markers',
                                    name='Data'))
            fig.add_trace(go.Scatter(x=outlier_index, y=data.loc[outlier_index, col],
                            mode='markers',
                            marker=dict(color='red', size=8),
                            name='Outliers'))

            fig.update_layout(title='Outlier Detection',
                            xaxis_title='Index',
                            yaxis_title=col)
            st.plotly_chart(fig)

            return outlier_index, para
        except Exception as e:
            st.info("An error occurred while processing your request. Please check your input data and try again.")
            # Log the error for debugging purposes
            self.logger.error("An error occurred while detecting outliers: %s", str(e))
            return [], ()

    def outlier_detect_mean_std(self, data, col, threshold=3):
        try:
            upper_fence = data[col].mean() + threshold * data[col].std()
            lower_fence = data[col].mean() - threshold * data[col].std()
            para = (upper_fence, lower_fence)
            outlier_index = data[(data[col] > upper_fence) | (data[col] < lower_fence)].index
            num_outliers = len(outlier_index)
            prop_outliers = num_outliers / len(data)
            st.write('Num of outlier detected:', num_outliers)
            st.write('Proportion of outlier detected:', prop_outliers)
            
            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data[col],
                                     mode='markers',
                                     name='Data'))
            fig.add_trace(go.Scatter(x=outlier_index, y=data.loc[outlier_index, col],
                            mode='markers',
                            marker=dict(color='red', size=8),
                            name='Outliers'))
            fig.update_layout(title='Outlier Detection',
                              xaxis_title='Index',
                              yaxis_title=col)
            st.plotly_chart(fig)
            
            
            return outlier_index, para
        except Exception as e:
            st.write("An error occurred while processing your request. Please check your input data and try again.")
            # Log the error for debugging purposes
            self.logger.error("An error occurred while detecting outliers: %s", str(e))
            return [], ()

    def outlier_detect_MAD(self, data, col, threshold=3.5):
        try:
            median = data[col].median()
            median_absolute_deviation = np.median([np.abs(y - median) for y in data[col]])
            modified_z_scores = pd.Series([0.6745 * (y - median) / median_absolute_deviation for y in data[col]])
            outlier_index = data.index[np.abs(modified_z_scores) > threshold]
            
            print('Num of outlier detected:', len(outlier_index))
            print('Proportion of outlier detected:', len(outlier_index) / len(data))
            
            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data[col],
                                    mode='markers',
                                    name='Data'))
            fig.add_trace(go.Scatter(x=outlier_index, y=data.loc[outlier_index, col],
                            mode='markers',
                            marker=dict(color='red', size=8),
                            name='Outliers'))

            fig.update_layout(title='Outlier Detection',
                            xaxis_title='Index',
                            yaxis_title=col)
            st.plotly_chart(fig)

            return outlier_index
        except Exception as e:
            st.write("An error occurred while processing your request. Please check your input data and try again.")
            # Log the error for debugging purposes
            self.logger.error("An error occurred while detecting outliers: %s", str(e))
            return [], ()

    def windsorization(self, data, col, para, strategy='both'):
        try:
            data_copy = data
            if strategy == 'both':
                data_copy.loc[data_copy[col] > para[0], col] = para[0]
                data_copy.loc[data_copy[col] < para[1], col] = para[1]
            elif strategy == 'top':
                data_copy.loc[data_copy[col] > para[0], col] = para[0]
            elif strategy == 'bottom':
                data_copy.loc[data_copy[col] < para[1], col] = para[1]
            return data_copy
        except Exception as e:
            self.logger.error("An error occurred while performing windsorization: %s", str(e))
            raise

    def drop_outlier(self, outlier_index):
        try:
            self.data = self.data.loc[~self.data.index.isin(outlier_index)]
            return self.data
        except Exception as e:
            self.logger.error("An error occurred while dropping outliers: %s", str(e))
            raise


    def outlier_detect(self):
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Detect and Impute Outliers</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h1>", unsafe_allow_html=True)
        st.dataframe(self.data, width=800)

        options = [
            "Detect Outliers using IQR",
            "Detect Outliers using Mean and Standard Deviation",
            "Detect Outliers using Median Absolute Deviation (MAD)",
            "Windsorize Outliers",
            "Drop Outliers"
        ]

        option = st.sidebar.selectbox("Select an Outlier Detection/Imputation Method", options)

        # Initialize session state for each option
        for opt in options:
            if f'{opt}_clicked' not in st.session_state:
                st.session_state[f'{opt}_clicked'] = False
            if f'{opt}_show_code' not in st.session_state:
                st.session_state[f'{opt}_show_code'] = False

        if option == "Detect Outliers using IQR":
            self._handle_detect_iqr()
        elif option == "Detect Outliers using Mean and Standard Deviation":
            self._handle_detect_mean_std()
        elif option == "Detect Outliers using Median Absolute Deviation (MAD)":
            self._handle_detect_mad()
        elif option == "Windsorize Outliers":
            self._handle_windsorize()
        elif option == "Drop Outliers":
            self._handle_drop_outliers()

        return self.data

    def _handle_option(self, option, function, **kwargs):
        st.markdown(f"<h1 style='text-align: center; font-size: 25px;'>{option}</h1>", unsafe_allow_html=True)
        
        if st.button("Execute"):
            st.session_state[f'{option}_clicked'] = True

        if st.session_state[f'{option}_clicked']:
            result = function(**kwargs)
            st.write(result)
            
            st.session_state[f'{option}_show_code'] = st.checkbox('Show Code', value=st.session_state[f'{option}_show_code'])
            
            if st.session_state[f'{option}_show_code']:
                self.view_code._display_code(function.__name__)

    def _handle_detect_iqr(self):
        threshold = st.number_input("Threshold", min_value=0.1, value=1.5)
        col = st.selectbox("Select a column", options=self.data.columns)
        self._handle_option("Detect Outliers using IQR", self.outlier_detect_IQR, data=self.data, col=col, threshold=threshold)

    def _handle_detect_mean_std(self):
        threshold = st.number_input("Threshold", min_value=0.1, value=3.0)
        col = st.selectbox("Select a column", options=self.data.columns)
        self._handle_option("Detect Outliers using Mean and Standard Deviation", self.outlier_detect_mean_std, data=self.data, col=col, threshold=threshold)

    def _handle_detect_mad(self):
        threshold = st.number_input("Threshold", min_value=0.1, value=3.5)
        col = st.selectbox("Select a column", options=self.data.columns)
        self._handle_option("Detect Outliers using Median Absolute Deviation (MAD)", self.outlier_detect_MAD, data=self.data, col=col, threshold=threshold)

    def _handle_windsorize(self):
        threshold = st.number_input("Threshold", min_value=0.1, value=1.5)
        col = st.selectbox("Select a column", options=self.data.columns)
        outlier_index_value, para = self.outlier_detect_IQR(data=self.data, col=col, threshold=threshold)
        strategy = st.selectbox("Windsorization Strategy", ['both', 'top', 'bottom'])
        self._handle_option("Windsorize Outliers", self.windsorization, data=self.data, col=col, para=para, strategy=strategy)

    def _handle_drop_outliers(self):
        threshold = st.number_input("Threshold", min_value=0.1, value=1.5)
        col = st.selectbox("Select a column", options=self.data.columns)
        outlier_index_value, para = self.outlier_detect_IQR(data=self.data, col=col, threshold=threshold)
        self._handle_option("Drop Outliers", self.drop_outlier, outlier_index=outlier_index_value)