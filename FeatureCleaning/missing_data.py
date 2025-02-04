import pandas as pd
import numpy as np
import logging
import streamlit as st
from sklearn.impute import SimpleImputer, KNNImputer
from show_code import ShowCode
import plotly.express as px
import plotly.figure_factory as ff

class DataImputer:
    def __init__(self, data):
        super().__init__()
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data.copy()
        self.original_data = data.copy()  # Keep original data for reference
        self.view_code = ShowCode()
        self.view_code.set_target_class(DataImputer)
        self._setup_logger()
        
    def _setup_logger(self):
        """Set up logging configuration"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())

    def auto_clean(self, threshold_missing=0.7, strategy='auto'):
        """
        Automatically clean the dataset using a comprehensive approach.
        
        Parameters:
        -----------
        threshold_missing : float
            Threshold for dropping columns with missing values (0 to 1)
        strategy : str
            'auto': automatically choose best strategy based on data
            'aggressive': maximum cleaning with data loss
            'conservative': minimal cleaning to preserve data
        """
        try:
            st.info("Starting automated cleaning process...")
            
            # 1. Initial analysis
            missing_stats = self.check_missing()
            total_rows = len(self.data)
            
            # 2. Drop columns with too many missing values
            cols_to_drop = missing_stats[missing_stats['proportion'] > threshold_missing].index.tolist()
            if cols_to_drop:
                self._drop_columns(cols_to_drop)
                st.info(f"Dropped {len(cols_to_drop)} columns with more than {threshold_missing*100}% missing values")
            
            # 3. Identify numeric and categorical columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            categorical_cols = self.data.select_dtypes(exclude=[np.number]).columns
            
            # 4. Handle numeric columns
            if len(numeric_cols) > 0:
                if strategy == 'aggressive':
                    self.impute_NA_with_simple_imputer(NA_col=numeric_cols, strategy='mean')
                elif strategy == 'conservative':
                    self.impute_NA_with_knn(NA_col=numeric_cols, n_neighbors=5)
                else:  # auto strategy
                    for col in numeric_cols:
                        missing_ratio = self.data[col].isnull().mean()
                        if missing_ratio < 0.1:
                            self.impute_NA_with_simple_imputer(NA_col=[col], strategy='mean')
                        else:
                            self.impute_NA_with_knn(NA_col=[col], n_neighbors=5)
            
            # 5. Handle categorical columns
            if len(categorical_cols) > 0:
                self.impute_NA_with_simple_imputer(NA_col=categorical_cols, strategy='most_frequent')
            
            # 6. Final cleanup
            remaining_nulls = self.data.isnull().sum().sum()
            if remaining_nulls > 0 and strategy == 'aggressive':
                self.drop_missing(axis=0)
            
            # 7. Generate report
            cleaning_report = {
                'original_rows': total_rows,
                'original_columns': len(self.original_data.columns),
                'final_rows': len(self.data),
                'final_columns': len(self.data.columns),
                'columns_dropped': len(cols_to_drop),
                'remaining_nulls': remaining_nulls
            }
            
            st.success("Automated cleaning completed!")
            st.write("Cleaning Report:", cleaning_report)
            
            return self.data
            
        except Exception as e:
            st.error(f"Error during automated cleaning: {str(e)}")
            self.logger.error(f"Auto-cleaning failed: {str(e)}")
            raise

    def check_missing(self):
        """Analyze missing values in the dataset."""
        
        # Calculate missing values statistics
        missing = pd.DataFrame({
            'Missing Values': self.data.isnull().sum(),
            'Percentage': (self.data.isnull().sum() / len(self.data) * 100).round(2)
        })
        missing = missing[missing['Missing Values'] > 0].sort_values('Percentage', ascending=False)
        
        if not missing.empty:
            st.write("Columns with missing values:")
            st.dataframe(missing)
            
            # Create missing values heatmap
            fig = px.imshow(
                self.data.isnull().T,
                labels=dict(x="Row Index", y="Columns", color="Missing"),
                title="Missing Values Heatmap",
                aspect="auto"
            )
            st.plotly_chart(fig)
        else:
            st.write("No missing values found in the dataset.")

    def _drop_columns(self, columns_to_drop):
        """Drops the specified columns from the DataFrame."""
        try:
            non_existent_cols = [col for col in columns_to_drop if col not in self.data.columns]
            if non_existent_cols:
                st.error(f"Columns not found: {', '.join(non_existent_cols)}")
                columns_to_drop = [col for col in columns_to_drop if col in self.data.columns]
            
            self.data = self.data.drop(columns_to_drop, axis=1)
            st.info(f"Dropped columns: {', '.join(columns_to_drop)}")
            return self.data
        except Exception as e:
            st.error(f"Error dropping columns: {str(e)}")
            raise

    def drop_missing(self, axis=0, threshold=None):
        """Drop rows or columns with missing values."""
        try:
            original_shape = self.data.shape
            if threshold:
                if axis == 0:
                    self.data = self.data.dropna(axis=0, thresh=threshold)
                else:
                    self.data = self.data.dropna(axis=1, thresh=threshold)
            else:
                self.data = self.data.dropna(axis=axis)
            
            if self.data.shape == original_shape:
                st.info("No rows/columns were dropped")
            else:
                st.info(f"Shape changed from {original_shape} to {self.data.shape}")
            return self.data
        except Exception as e:
            st.error(f"Error dropping missing values: {str(e)}")
            raise

    def add_var_denote_NA(self, NA_col=[]):
        """Create binary indicators for missing values."""
        try:
            for col in NA_col:
                if col in self.data.columns:
                    if self.data[col].isnull().sum() > 0:
                        self.data[f"{col}_is_missing"] = self.data[col].isnull().astype(int)
                    else:
                        st.info(f"No missing values in column: {col}")
            return self.data
        except Exception as e:
            st.error(f"Error creating missing indicators: {str(e)}")
            raise

    def impute_NA_with_arbitrary(self, impute_value, NA_col=[]):
        """Impute missing values with a specified value."""
        try:
            for col in NA_col:
                if col in self.data.columns:
                    missing_count = self.data[col].isnull().sum()
                    if missing_count > 0:
                        self.data[col].fillna(impute_value, inplace=True)
                        st.info(f"Imputed {missing_count} missing values in {col}")
            return self.data
        except Exception as e:
            st.error(f"Error during arbitrary imputation: {str(e)}")
            raise

    def impute_NA_with_avg(self, strategy='mean', NA_col=[]):
        """Impute missing values with mean, median, or mode."""
        try:
            for col in NA_col:
                if col in self.data.columns:
                    if self.data[col].isnull().sum() > 0:
                        if strategy == 'mean':
                            self.data[col].fillna(self.data[col].mean(), inplace=True)
                        elif strategy == 'median':
                            self.data[col].fillna(self.data[col].median(), inplace=True)
                        elif strategy == 'mode':
                            self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            return self.data
        except Exception as e:
            st.error(f"Error during average imputation: {str(e)}")
            raise

    def impute_NA_with_interpolation(self, method='linear', limit=None, limit_direction='forward', NA_col=[]):
        """Impute missing values using interpolation."""
        try:
            for col in NA_col:
                if col in self.data.columns:
                    if self.data[col].isnull().sum() > 0:
                        self.data[col] = self.data[col].interpolate(
                            method=method,
                            limit=limit,
                            limit_direction=limit_direction
                        )
            return self.data
        except Exception as e:
            st.error(f"Error during interpolation: {str(e)}")
            raise

    def impute_NA_with_knn(self, NA_col=[], n_neighbors=5):
        """Impute missing values using K-Nearest Neighbors."""
        try:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            for col in NA_col:
                if col in self.data.columns:
                    if self.data[col].isnull().sum() > 0:
                        self.data[col] = imputer.fit_transform(self.data[[col]])
            return self.data
        except Exception as e:
            st.error(f"Error during KNN imputation: {str(e)}")
            raise

    def impute_NA_with_simple_imputer(self, NA_col=[], strategy='mean'):
        """Impute missing values using SimpleImputer."""
        try:
            imputer = SimpleImputer(strategy=strategy)
            for col in NA_col:
                if col in self.data.columns:
                    if self.data[col].isnull().sum() > 0:
                        # Reshape the imputed values to 1D array before assignment
                        imputed_values = imputer.fit_transform(self.data[[col]]).ravel()
                        self.data[col] = imputed_values
                        st.info(f"Imputed missing values in column: {col}")
            return self.data
        except Exception as e:
            st.error(f"Error during simple imputation: {str(e)}")
            raise

    def _handle_check_missing_values(self):
        """Handle the Check Missing Values option in the Streamlit interface."""
        self._handle_option("Missing Values Analysis", self.check_missing)

    def _handle_drop_columns(self):
        """Handle the Drop Columns option in the Streamlit interface."""
        columns_to_drop = st.multiselect("Select Columns to Drop", self.data.columns)
        self._handle_option("Drop Columns", self._drop_columns, columns_to_drop=columns_to_drop)

    def _handle_drop_missing_values(self):
        """Handle the Drop Missing Values option in the Streamlit interface."""
        axis = st.radio("Drop rows or columns?", ["Rows", "Columns"])
        threshold = st.number_input("Minimum number of non-NA values required", min_value=0, value=0)
        axis = 0 if axis == "Rows" else 1
        self._handle_option("Drop Missing Values", self.drop_missing, axis=axis, threshold=threshold)

    def _handle_add_var_denote_na(self):
        """Handle the Add Variable to Denote NA option in the Streamlit interface."""
        selected_columns = st.multiselect("Select columns", self.data.columns)
        self._handle_option("Add Variable to Denote NA", self.add_var_denote_NA, NA_col=selected_columns)

    def _handle_impute_arbitrary(self):
        """Handle the Impute NA with Arbitrary Value option in the Streamlit interface."""
        impute_value = st.text_input("Enter Arbitrary Value")
        na_cols = st.multiselect("Select Columns", self.data.columns)
        if impute_value:
            try:
                impute_value = float(impute_value)
                self._handle_option("Impute NA with Arbitrary Value", 
                                  self.impute_NA_with_arbitrary, 
                                  impute_value=impute_value, 
                                  NA_col=na_cols)
            except ValueError:
                st.error("Please enter a valid numeric value")

    def _handle_impute_interpolation(self):
        """Handle the Impute NA with Interpolation option in the Streamlit interface."""
        na_cols = st.multiselect("Select Columns", self.data.columns)
        method = st.selectbox("Interpolation Method", ['linear', 'quadratic', 'cubic'])
        limit = st.number_input("Limit", min_value=0, value=None)
        limit_direction = st.selectbox("Limit Direction", ['forward', 'backward', 'both'])
        self._handle_option("Impute NA with Interpolation", 
                          self.impute_NA_with_interpolation,
                          method=method,
                          limit=limit,
                          limit_direction=limit_direction,
                          NA_col=na_cols)

    def _handle_impute_knn(self):
        """Handle the Impute NA with KNN option in the Streamlit interface."""
        n_neighbors = st.number_input("Number of Neighbors", min_value=1, value=5)
        selected_columns = st.multiselect("Select columns to impute", options=self.data.columns)
        self._handle_option("Impute NA with KNN", 
                          self.impute_NA_with_knn,
                          NA_col=selected_columns,
                          n_neighbors=n_neighbors)

    def _handle_impute_simple_imputer(self):
        """Handle the Impute NA with SimpleImputer option in the Streamlit interface."""
        na_cols = st.multiselect("Select Columns", self.data.columns)
        strategy = st.selectbox("Imputation Strategy", ["mean", "median", "most_frequent", "constant"])
        self._handle_option("Impute NA with SimpleImputer",
                          self.impute_NA_with_simple_imputer,
                          NA_col=na_cols,
                          strategy=strategy)

    def _handle_impute_average(self):
        """Handle the Impute NA with Average option in the Streamlit interface."""
        na_cols = st.multiselect("Select Columns", self.data.columns)
        strategy = st.selectbox("Strategy", ['mean', 'median', 'mode'])
        self._handle_option("Impute NA with Average",
                          self.impute_NA_with_avg,
                          strategy=strategy,
                          NA_col=na_cols)

    def _handle_auto_clean(self):
        """Handle the Auto Clean option in the Streamlit interface."""
        st.markdown("<h2 style='text-align: center; font-size: 25px;'>Auto Clean</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            threshold = st.slider(
                "Missing Value Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Columns with missing values above this threshold will be dropped"
            )
            
        with col2:
            strategy = st.selectbox(
                "Cleaning Strategy",
                ["auto", "aggressive", "conservative"],
                help="""
                auto: Automatically choose best strategy based on data
                aggressive: Maximum cleaning with potential data loss
                conservative: Minimal cleaning to preserve data
                """
            )
        
        if "auto_clean_clicked" not in st.session_state:
            st.session_state.auto_clean_clicked = False
            
        if st.button("Execute Auto Clean"):
            st.session_state.auto_clean_clicked = True
            
        if st.session_state.auto_clean_clicked:
            self.auto_clean(threshold_missing=threshold, strategy=strategy)
            
            if "auto_clean_show_code" not in st.session_state:
                st.session_state.auto_clean_show_code = False
                
            st.session_state.auto_clean_show_code = st.checkbox(
                'Show Code',
                value=st.session_state.auto_clean_show_code
            )
            
            if st.session_state.auto_clean_show_code:
                self.view_code._display_code('auto_clean')

        # Add comparison with original data
        if st.checkbox("Show Comparison with Original Data"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h3 style='text-align: center;'>Original Data</h3>", unsafe_allow_html=True)
                st.dataframe(self.original_data)
                st.write("Original Shape:", self.original_data.shape)
                st.write("Original Missing Values:", self.original_data.isnull().sum().sum())
                
            with col2:
                st.markdown("<h3 style='text-align: center;'>Cleaned Data</h3>", unsafe_allow_html=True)
                st.dataframe(self.data)
                st.write("New Shape:", self.data.shape)
                st.write("Remaining Missing Values:", self.data.isnull().sum().sum())

        # Add option to download cleaned data
        if st.button("Download Cleaned Data"):
            csv = self.data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

    def _handle_option(self, option_name, function, **kwargs):
        """Generic handler for imputation options."""
        st.markdown(f"<h2 style='text-align: center; font-size: 25px;'>{option_name}</h2>", unsafe_allow_html=True)
        
        if st.button("Execute"):
            st.session_state[f'{option_name}_clicked'] = True

        if st.session_state.get(f'{option_name}_clicked', False):
            result = function(**kwargs)
            st.write(result)
            
            show_code = st.checkbox(
                'Show Code',
                value=st.session_state.get(f'{option_name}_show_code', False)
            )
            st.session_state[f'{option_name}_show_code'] = show_code
            
            if show_code:
                self.view_code._display_code(function.__name__)

    def imputer(self):
        """Streamlit interface for data imputation."""
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>Data Imputation Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Show dataset info
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset Information</h2>", unsafe_allow_html=True)
        st.write(f"Shape: {self.data.shape}")
        st.write(f"Missing Values: {self.data.isnull().sum().sum()}")
        st.dataframe(self.data, width=800)

        # Options list including Auto Clean
        options = [
            "Auto Clean",
            "Check Missing Values",
            "Drop Columns",
            "Drop Missing Values",
            "Add Variable to Denote NA",
            "Impute NA with Arbitrary Value",
            "Impute NA with Interpolation",
            "Impute NA with KNN",
            "Impute NA with SimpleImputer",
            "Impute NA with Average"
        ]

        option = st.sidebar.selectbox("Select Imputation Method", options)

        # Handle selected option
        if option == "Auto Clean":
            self._handle_auto_clean()
        elif option == "Check Missing Values":
            self._handle_check_missing_values()
        elif option == "Drop Columns":
            self._handle_drop_columns()
        elif option == "Drop Missing Values":
            self._handle_drop_missing_values()
        elif option == "Add Variable to Denote NA":
            self._handle_add_var_denote_na()
        elif option == "Impute NA with Arbitrary Value":
            self._handle_impute_arbitrary()
        elif option == "Impute NA with Interpolation":
            self._handle_impute_interpolation()
        elif option == "Impute NA with KNN":
            self._handle_impute_knn()
        elif option == "Impute NA with SimpleImputer":
            self._handle_impute_simple_imputer()
        elif option == "Impute NA with Average":
            self._handle_impute_average()

        return self.data