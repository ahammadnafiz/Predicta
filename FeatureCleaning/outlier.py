import pandas as pd
import numpy as np
import logging
import streamlit as st
import plotly.graph_objects as go
from show_code import ShowCode
from sklearn.neighbors import LocalOutlierFactor


class OutlierDetector:
    def __init__(self, data):
        self.data = data
        self.view_code = ShowCode()
        self.view_code.set_target_class(OutlierDetector)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        
        
    def outlier_detect_IQR(self, data, col, threshold=0.05):
        """Detect outliers using the Interquartile Range (IQR) method."""
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
            st.info(f'Number of outliers detected: {num_outliers}')
            st.info(f'Proportion of outliers detected: {prop_outliers:.2%}')
            
            # Check if the proportion of outliers exceeds the threshold
            if prop_outliers > threshold:
                st.warning(f"Warning: Proportion of outliers ({prop_outliers:.2%}) exceeds the threshold ({threshold:.2%})")
                st.info("Consider adjusting your threshold or using a different outlier detection method")

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
        except KeyError as e:
            st.error(f"Column Error: Column '{col}' not found in the dataframe")
            st.info("Please select a valid column from your dataset")
            self.logger.error(f"Column not found: {str(e)}")
            return [], ()
        except ValueError as e:
            st.error(f"Value Error: {str(e)}")
            st.info("The IQR method requires numeric data. Make sure the selected column contains numeric values")
            self.logger.error(f"Value error during IQR outlier detection: {str(e)}")
            return [], ()
        except Exception as e:
            st.error(f"An unexpected error occurred during outlier detection: {str(e)}")
            st.info("Please check your data quality and try again")
            self.logger.error(f"Unexpected error during IQR outlier detection: {str(e)}")
            return [], ()

    def outlier_detect_mean_std(self, data, col, threshold=3):
        try:
            upper_fence = data[col].mean() + threshold * data[col].std()
            lower_fence = data[col].mean() - threshold * data[col].std()
            para = (upper_fence, lower_fence)
            outlier_index = data[(data[col] > upper_fence) | (data[col] < lower_fence)].index
            num_outliers = len(outlier_index)
            prop_outliers = num_outliers / len(data)
            st.info(f'Number of outliers detected: {num_outliers}')
            st.info(f'Proportion of outliers detected: {prop_outliers:.2%}')
            
            if prop_outliers > 0.1:  # If more than 10% are outliers
                st.warning(f"High proportion of outliers detected ({prop_outliers:.2%}). Consider using a different threshold or method.")
            
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
        except KeyError as e:
            st.error(f"Column Error: Column '{col}' not found in the dataframe")
            st.info("Please select a valid column from your dataset")
            self.logger.error(f"Column not found: {str(e)}")
            return [], ()
        except ValueError as e:
            st.error(f"Value Error: {str(e)}")
            st.info("Mean-Std method requires numeric data. Make sure the selected column contains numeric values")
            self.logger.error(f"Value error during Mean-Std outlier detection: {str(e)}")
            return [], ()
        except Exception as e:
            st.error(f"An unexpected error occurred during outlier detection: {str(e)}")
            st.info("Please check your data quality and try again")
            self.logger.error(f"Unexpected error during Mean-Std outlier detection: {str(e)}")
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

    def outlier_detect_LOF(self, data, cols, n_neighbors=20, contamination=0.1):
        """Detect outliers using the Local Outlier Factor (LOF) method.
        
        This method works well for multidimensional data by measuring the local deviation 
        of density of a point with respect to its neighbors.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The input data
        cols : list
            List of column names to use for outlier detection
        n_neighbors : int, default=20
            Number of neighbors to use by default for LOF calculations
        contamination : float, default=0.1
            The proportion of outliers in the data set            
        """
        try:
            # Check if columns exist in the dataframe
            for col in cols:
                if col not in data.columns:
                    raise KeyError(f"Column '{col}' not found in the dataframe")
            
            # Extract only the selected columns for LOF
            X = data[cols].copy()
            
            # Check if all values are numeric
            if not np.issubdtype(X.dtypes.values.all(), np.number):
                non_numeric_cols = [col for col in cols if not np.issubdtype(data[col].dtype, np.number)]
                raise ValueError(f"Non-numeric columns detected: {non_numeric_cols}. LOF requires numeric data.")
            
            # Handle NaN values
            if X.isnull().values.any():
                st.warning("The selected columns contain missing values, which will be replaced with column means for the LOF calculation.")
                X = X.fillna(X.mean())
            
            # Apply LOF
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            y_pred = lof.fit_predict(X)
            
            # LOF decision function: negative values represent outliers
            outlier_scores = -lof.negative_outlier_factor_
            
            # Get outlier indices (where y_pred is -1)
            outlier_index = data.index[y_pred == -1]
            
            num_outliers = len(outlier_index)
            prop_outliers = num_outliers / len(data)
            
            st.info(f'Number of outliers detected: {num_outliers}')
            st.info(f'Proportion of outliers detected: {prop_outliers:.2%}')
            
            # Create a 2D visualization of the outliers
            if len(cols) >= 2:
                # Choose first two columns for visualization
                fig = go.Figure()
                
                # Plot inliers
                inlier_idx = data.index[y_pred == 1]
                fig.add_trace(go.Scatter(
                    x=data.loc[inlier_idx, cols[0]],
                    y=data.loc[inlier_idx, cols[1]],
                    mode='markers',
                    name='Normal Data',
                    marker=dict(color='blue', size=6)
                ))
                
                # Plot outliers
                fig.add_trace(go.Scatter(
                    x=data.loc[outlier_index, cols[0]],
                    y=data.loc[outlier_index, cols[1]],
                    mode='markers',
                    name='Outliers',
                    marker=dict(color='red', size=8)
                ))
                
                fig.update_layout(
                    title='LOF Outlier Detection',
                    xaxis_title=cols[0],
                    yaxis_title=cols[1]
                )
                st.plotly_chart(fig)
                
                # Show outlier scores
                if st.checkbox("Show outlier scores"):
                    scores_df = pd.DataFrame({
                        'index': data.index,
                        'outlier_score': outlier_scores
                    }).sort_values('outlier_score', ascending=False)
                    st.write("Top outliers (higher score = more likely outlier):")
                    st.dataframe(scores_df.head(20))
            
            return outlier_index
            
        except KeyError as e:
            st.error(f"Column Error: {str(e)}")
            st.info("Please select valid columns from your dataset")
            self.logger.error(f"Column not found: {str(e)}")
            return []
        except ValueError as e:
            st.error(f"Value Error: {str(e)}")
            st.info("The LOF method requires numeric data. Make sure all selected columns contain numeric values")
            self.logger.error(f"Value error during LOF outlier detection: {str(e)}")
            return []
        except Exception as e:
            st.error(f"An unexpected error occurred during LOF outlier detection: {str(e)}")
            st.info("Please check your data quality and try again")
            self.logger.error(f"Unexpected error during LOF outlier detection: {str(e)}")
            return []

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
            "Detect Outliers using Local Outlier Factor (LOF)",
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
        elif option == "Detect Outliers using Local Outlier Factor (LOF)":
            self._handle_detect_lof()
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

    def _handle_detect_lof(self):
        n_neighbors = st.slider("Number of Neighbors", min_value=5, max_value=50, value=20)
        contamination = st.slider("Contamination (Expected proportion of outliers)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        
        numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("LOF requires at least 2 numeric columns for effective detection.")
            return
            
        cols = st.multiselect("Select columns for LOF detection", options=numeric_cols, 
                             default=numeric_cols[:min(2, len(numeric_cols))])
        
        if len(cols) < 2:
            st.warning("For best results, select at least 2 columns. LOF works better with multidimensional data.")
        
        if cols:
            self._handle_option("Detect Outliers using Local Outlier Factor (LOF)", 
                               self.outlier_detect_LOF, 
                               data=self.data, 
                               cols=cols, 
                               n_neighbors=n_neighbors, 
                               contamination=contamination)
        else:
            st.warning("Please select at least one column to proceed.")

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