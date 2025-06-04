import pandas as pd
import numpy as np
import logging
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer, KNNImputer
from ...utils.code_display import ShowCode
from ...core.logging_config import get_logger
from scipy import stats

class DataImputer:
    def __init__(self, data):
        super().__init__()
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data.copy()
        self.original_data = data.copy()  # Keep original data for reference
        self.view_code = ShowCode()
        self.view_code.set_target_class(DataImputer)
        self.logger = get_logger(__name__)
          # Add log container for collecting messages
        self.log_messages = []

    def log_info(self, message):
        """Add info message to log container and log it to the file system"""
        self.log_messages.append({"type": "info", "message": message})
        self.logger.info(message)
        
    def log_warning(self, message):
        """Add warning message to log container and log it to the file system"""
        self.log_messages.append({"type": "warning", "message": message})
        self.logger.warning(message)
        
    def log_error(self, message):
        """Add error message to log container and log it to the file system"""
        self.log_messages.append({"type": "error", "message": message})
        self.logger.error(message)
        
    def display_log_container(self):
        """Display all collected log messages in a container box"""
        if not self.log_messages:
            return
            
        with st.expander("📋 Log Messages", expanded=True):
            # Create separate sections for different message types
            info_messages = [msg for msg in self.log_messages if msg['type'] == 'info']
            warning_messages = [msg for msg in self.log_messages if msg['type'] == 'warning']
            error_messages = [msg for msg in self.log_messages if msg['type'] == 'error']
            
            # Display error messages first (most critical)
            if error_messages:
                st.markdown("### ❌ Errors")
                for msg in error_messages:
                    st.error(msg['message'])
            
            # Display warnings next
            if warning_messages:
                st.markdown("### ⚠️ Warnings")
                for msg in warning_messages:
                    st.warning(msg['message'])
            
            # Display info messages last
            if info_messages:
                st.markdown("### ℹ️ Information")
                for msg in info_messages:
                    st.info(msg['message'])
            
            # Provide option to clear log
            if st.button("Clear Log"):
                self.log_messages = []
                st.experimental_rerun()

    def auto_clean(self, threshold_missing=0.7, strategy='auto'):
    
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

    def check_missing(self, output_path=None):
        """Analyze missing values in the dataset."""
        try:
            missing_counts = self.data.isnull().sum()
            missing_proportions = self.data.isnull().mean()
            missing_types = self.data.dtypes
            
            result = pd.DataFrame({
                'total missing': missing_counts,
                'proportion': missing_proportions,
                'dtype': missing_types
            })
            
            visualize = st.checkbox('Visualize Missing Data', False)
            if visualize:
                # Add tabs for different visualization types
                missing_tabs = st.tabs(["Heatmap", "Bar Chart", "Matrix", "Correlation", "Patterns"])
                
                with missing_tabs[0]:  # Heatmap
                    missing_matrix = self.data.isnull().astype(int)  # Convert True/False to 1/0
                    fig = px.imshow(missing_matrix.T,
                                    labels=dict(x="Samples", y="Features", color="Missing"),
                                    color_continuous_scale=[[0, 'white'], [1, 'red']],
                                    title="Missing Values Heatmap")
                    
                    fig.update_layout(autosize=True, height=600)
                    st.plotly_chart(fig)
                
                with missing_tabs[1]:  # Bar Chart
                    if not missing_counts.empty:
                        missing_df = pd.DataFrame({
                            'Column': missing_counts.index,
                            'Missing Values': missing_counts.values,
                            'Percentage': (missing_proportions.values * 100).round(2)
                        }).sort_values('Missing Values', ascending=False)
                        
                        fig = px.bar(missing_df, 
                                    x='Column', 
                                    y='Missing Values',
                                    text='Percentage',
                                    labels={'Percentage': 'Missing %'},
                                    title='Missing Values by Column',
                                    color='Percentage',
                                    color_continuous_scale=px.colors.sequential.Viridis)
                                    
                        fig.update_traces(texttemplate='%{text}%', textposition='outside')
                        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                        st.plotly_chart(fig)
                
                with missing_tabs[2]:  # Matrix
                    # Create a missing values matrix using plotly
                    missing_matrix = self.data.isnull().astype(int)
                    
                    # Calculate the sample indices
                    num_samples = len(self.data)
                    if num_samples > 100:
                        # If too many samples, select at most 100 samples for visualization
                        step = max(1, num_samples // 100)
                        sample_indices = list(range(0, num_samples, step))
                        missing_matrix_viz = missing_matrix.iloc[sample_indices]
                    else:
                        missing_matrix_viz = missing_matrix
                    
                    # Create the heatmap
                    fig = px.imshow(
                        missing_matrix_viz.T,  # Transpose to show features as rows
                        labels=dict(x="Samples", y="Features", color="Missing"),
                        x=missing_matrix_viz.index,
                        y=missing_matrix_viz.columns,
                        color_continuous_scale=[[0, 'white'], [1, 'black']],
                        title="Missing Value Matrix Pattern"
                    )
                    
                    fig.update_layout(
                        height=600,
                        xaxis_showticklabels=False,  # Hide sample indices for cleaner visualization
                        yaxis=dict(autorange="reversed")  # Reverse y-axis to match missingno style
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("This visualization shows the pattern of missing values across samples. Each row is a feature, and black marks indicate missing values.")
                
                with missing_tabs[3]:  # Correlation of missingness
                    # Compute correlation between missingness of columns
                    missing_corr = self.data.isnull().corr()
                    fig = px.imshow(missing_corr,
                                    labels=dict(x="Features", y="Features", color="Correlation"),
                                    color_continuous_scale=px.colors.diverging.RdBu_r,
                                    title="Missing Value Correlation")
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig)
                    
                    st.info("High correlation suggests that missing values in one column may be related to missing values in another column.")
                
                with missing_tabs[4]:  # Missingness patterns
                    # Create missingness pattern visualization using plotly
                    st.markdown("### Missing Value Patterns")
                    
                    # Calculate correlation of missingness
                    missing_binary = self.data.isnull().astype(int)
                    
                    # Ensure we handle cases with all-null or no-null columns
                    # Filter columns that have some but not all missing values
                    valid_cols = [col for col in missing_binary.columns 
                                 if 0 < missing_binary[col].mean() < 1]
                    
                    if len(valid_cols) > 1:
                        # Calculate correlation matrix of missingness
                        missing_corr = missing_binary[valid_cols].corr()
                        
                        # Create a dendrogram-like clustering visualization
                        fig = px.imshow(
                            missing_corr,
                            color_continuous_scale='RdBu_r',
                            labels=dict(x="Features", y="Features", color="Correlation"),
                            title="Missing Value Pattern Correlation"
                        )
                        
                        # Update layout for better visualization
                        fig.update_layout(
                            height=600,
                            width=800,
                            xaxis={'side': 'bottom'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create co-occurrence patterns
                        st.markdown("### Missing Value Co-occurrence")
                        
                        # Calculate co-occurrence counts
                        missing_counts = {}
                        for i, col1 in enumerate(valid_cols):
                            for col2 in valid_cols[i+1:]:
                                pair = f"{col1} & {col2}"
                                both_missing = ((missing_binary[col1] == 1) & 
                                               (missing_binary[col2] == 1)).sum()
                                missing_counts[pair] = both_missing
                        
                        if missing_counts:
                            # Convert to dataframe for visualization
                            co_occur_df = pd.DataFrame({
                                'Variable Pair': list(missing_counts.keys()),
                                'Co-occurrence Count': list(missing_counts.values())
                            }).sort_values('Co-occurrence Count', ascending=False).head(15)
                            
                            # Create bar chart of co-occurrences
                            fig = px.bar(
                                co_occur_df,
                                x='Co-occurrence Count',
                                y='Variable Pair',
                                orientation='h',
                                title="Top Missing Value Co-occurrences",
                                color='Co-occurrence Count',
                                color_continuous_scale='Viridis'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough columns with missing values to analyze patterns.")
                    
                    st.info("These visualizations show relationships between missing values across different columns, helping identify if missingness in one variable is related to missingness in another.")
            if output_path:
                result.to_csv(f"{output_path}/missing_analysis.csv")
                self.logger.info(f"Missing value analysis saved to {output_path}/missing_analysis.csv")
            
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing missing values: {str(e)}")
            st.error(f"Error analyzing missing values: {str(e)}")
            raise

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
            
            numeric_cols = []
            for col in NA_col:
                if col in self.data.columns:
                    if self.data[col].isnull().sum() > 0:
                        if np.issubdtype(self.data[col].dtype, np.number):
                            numeric_cols.append(col)
                        else:
                            st.warning(f"Column {col} skipped: KNN imputation requires numeric data")
            
            if not numeric_cols:
                st.warning("No numeric columns with missing values selected.")
                return self.data
                
            # Store original data for comparison
            before_imputation = self.data[numeric_cols].copy()
            
            # Perform imputation on all selected numeric columns at once
            imputed_array = imputer.fit_transform(self.data[numeric_cols])
            
            # Update the original dataframe with imputed values
            for i, col in enumerate(numeric_cols):
                self.data[col] = imputed_array[:, i]
                st.info(f"Imputed missing values in column: {col} using KNN")
            
            # Add visual comparison
            if st.checkbox("Show before/after comparison"):
                self._show_imputation_comparison(numeric_cols)
            
            return self.data
        except Exception as e:
            st.error(f"Error during KNN imputation: {str(e)}")
            self.logger.error(f"Error during KNN imputation: {str(e)}")
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

    def impute_NA_with_grouped(self, group_cols, NA_col=[], method='mean'):
        """Impute missing values based on statistics from similar groups in data.
        
        Args:
            group_cols (list): Columns to group by
            NA_col (list): Columns with missing values to impute
            method (str): Statistic to use ('mean', 'median', 'mode')
        """
        try:
            if not group_cols:
                st.error("Please select at least one column to group by")
                return self.data
                
            # Verify group columns exist and have reasonable cardinality
            for col in group_cols:
                if col not in self.data.columns:
                    st.error(f"Group column '{col}' not found in data")
                    return self.data
                    
                # Check if group column has too many unique values
                n_unique = self.data[col].nunique()
                if n_unique > len(self.data) * 0.5:  # More than 50% unique values
                    st.warning(f"Column '{col}' has high cardinality ({n_unique} unique values). Grouping may be ineffective.")
                    
            # Validate columns to impute
            valid_na_cols = []
            for col in NA_col:
                if col in self.data.columns:
                    missing_count = self.data[col].isnull().sum()
                    if missing_count > 0:
                        if col in group_cols:
                            st.warning(f"Column '{col}' is used for grouping and cannot be imputed")
                        else:
                            valid_na_cols.append(col)
                    else:
                        st.info(f"No missing values in column: {col}")
                else:
                    st.warning(f"Column '{col}' not found in data")
                    
            if not valid_na_cols:
                st.warning("No valid columns with missing values to impute")
                return self.data
                
            # Store original data for comparison
            before_imputation = self.data[valid_na_cols].copy()
            
            # Group data and perform imputation
            for col in valid_na_cols:
                # Skip columns used for grouping
                if col in group_cols:
                    continue
                    
                # Get data type to determine imputation method
                dtype = self.data[col].dtype
                
                # Calculate aggregates by group
                if method == 'mean' and np.issubdtype(dtype, np.number):
                    group_values = self.data.groupby(group_cols)[col].transform('mean')
                elif method == 'median' and np.issubdtype(dtype, np.number):
                    group_values = self.data.groupby(group_cols)[col].transform('median')
                else:  # Use mode for non-numeric or if specified
                    # Mode requires special handling as it returns a Series
                    def get_mode(x):
                        mode_val = x.mode()
                        return mode_val.iloc[0] if not mode_val.empty else None
                    
                    group_values = self.data.groupby(group_cols)[col].transform(get_mode)
                
                # Fill missing values with group statistics
                missing_mask = self.data[col].isnull()
                missing_count = missing_mask.sum()
                
                # Apply the imputation
                self.data.loc[missing_mask, col] = group_values[missing_mask]
                
                # Check if any values are still missing (could happen if entire group had NaN)
                still_missing = self.data[col].isnull().sum()
                if still_missing > 0:
                    # Fall back to global statistic for any remaining NaNs
                    if method == 'mean' and np.issubdtype(dtype, np.number):
                        fallback_value = self.data[col].mean()
                    elif method == 'median' and np.issubdtype(dtype, np.number):
                        fallback_value = self.data[col].median()
                    else:
                        fallback_value = self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None
                    
                    self.data[col].fillna(fallback_value, inplace=True)
                    st.warning(f"Used global {method} for {still_missing} values in '{col}' where entire group had missing values")
                
                filled_count = missing_count - still_missing
                st.info(f"Imputed {filled_count} missing values in '{col}' using group {method}")
                
                # Show distribution of imputed values by group
                if st.checkbox(f"Show imputation results for {col}"):
                    # Create a figure showing distribution by group
                    if len(group_cols) == 1:  # Simple case with one grouping variable
                        group_col = group_cols[0]
                        fig = px.box(self.data, x=group_col, y=col, 
                                    title=f"Distribution of {col} by {group_col} after imputation")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show before-after comparison
                    self._show_imputation_comparison([col])
            
            return self.data
            
        except Exception as e:
            st.error(f"Error during grouped imputation: {str(e)}")
            self.logger.error(f"Grouped imputation failed: {str(e)}")
            raise

    def impute_NA_with_random_forest(self, NA_col=[], n_estimators=100, max_depth=None):
        """Impute missing values using Random Forest models.
        
        Args:
            NA_col (list): Columns with missing values to impute
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees, None for unlimited
        """
        try:
            # Import the random forest regressor
            from sklearn.ensemble import RandomForestRegressor
            
            # Identify numeric columns that have missing values
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            valid_na_cols = []
            
            for col in NA_col:
                if col in self.data.columns:
                    if self.data[col].isnull().sum() > 0:
                        if col in numeric_cols:
                            valid_na_cols.append(col)
                        else:
                            st.warning(f"Column '{col}' skipped: Random Forest imputation requires numeric data")
                    else:
                        st.info(f"No missing values in column: {col}")
                else:
                    st.warning(f"Column '{col}' not found in data")
            
            if not valid_na_cols:
                st.warning("No valid numeric columns with missing values to impute")
                return self.data
            
            # Make a copy of the original data for comparison
            before_imputation = self.data.copy()
            
            # For each column with missing values
            for col in valid_na_cols:
                # Create mask for missing values
                missing_mask = self.data[col].isnull()
                if missing_mask.sum() == 0:
                    continue
                    
                st.info(f"Imputing {missing_mask.sum()} missing values in '{col}' using Random Forest")
                
                # Create training and prediction sets
                X_train = self.data.loc[~missing_mask, numeric_cols].drop(columns=[col])
                y_train = self.data.loc[~missing_mask, col]
                X_predict = self.data.loc[missing_mask, numeric_cols].drop(columns=[col])
                
                # Check if we have enough features for training
                if X_train.shape[1] == 0:
                    st.warning(f"No numeric features available for training Random Forest for '{col}'")
                    continue
                
                # Check if there are any missing values in the feature set
                # If yes, we temporarily fill them with median
                for feature in X_train.columns:
                    if X_train[feature].isnull().sum() > 0:
                        X_train[feature].fillna(X_train[feature].median(), inplace=True)
                    if X_predict[feature].isnull().sum() > 0:
                        X_predict[feature].fillna(X_train[feature].median(), inplace=True)
                
                # Initialize and train random forest model
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                )
                
                # Train the model
                with st.spinner(f"Training Random Forest model for '{col}'..."):
                    rf.fit(X_train, y_train)
                
                # Predict missing values
                if not X_predict.empty:
                    predictions = rf.predict(X_predict)
                    self.data.loc[missing_mask, col] = predictions
                
                # Report feature importance if relevant
                if st.checkbox(f"Show feature importance for '{col}' imputation"):
                    importances = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Importance': rf.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        importances, 
                        x='Importance', 
                        y='Feature', 
                        orientation='h',
                        title=f"Feature Importance for '{col}' Imputation"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Show before/after comparison for all imputed columns
            if st.checkbox("Show before/after comparison"):
                self._show_imputation_comparison(valid_na_cols)
            
            return self.data
            
        except Exception as e:
            st.error(f"Error during Random Forest imputation: {str(e)}")
            self.logger.error(f"Random Forest imputation failed: {str(e)}")
            raise

    def impute_NA_with_matrix_factorization(self, NA_col=[], k=3, max_iter=30, regularization=0.1):
        """Impute missing values using matrix factorization techniques.
        
        Args:
            NA_col (list): Columns with missing values to impute
            k (int): Number of latent factors
            max_iter (int): Maximum number of iterations
            regularization (float): Regularization parameter
        """
        try:
            # Import necessary libraries
            from sklearn.impute import SimpleImputer
            
            # Make sure we're working with numeric data
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Filter columns to impute
            valid_na_cols = []
            for col in NA_col:
                if col in self.data.columns:
                    if col in numeric_cols and self.data[col].isnull().sum() > 0:
                        valid_na_cols.append(col)
                    elif col not in numeric_cols:
                        st.warning(f"Column '{col}' skipped: Matrix factorization requires numeric data")
                    elif self.data[col].isnull().sum() == 0:
                        st.info(f"No missing values in column: {col}")
                else:
                    st.warning(f"Column '{col}' not found in data")
            
            if not valid_na_cols:
                st.warning("No valid numeric columns with missing values to impute")
                return self.data
            
            # If no specific columns are provided, use all numeric columns with missing values
            if not NA_col:
                for col in numeric_cols:
                    if self.data[col].isnull().sum() > 0:
                        valid_na_cols.append(col)
                
                if not valid_na_cols:
                    st.warning("No numeric columns with missing values found")
                    return self.data
            
            # Create a copy of the data subset with numeric columns only
            X = self.data[valid_na_cols].values
            
            # Store the original data for visualization
            original_data = self.data.copy()
            
            # Create a mask for missing values
            missing_mask = np.isnan(X)
            
            # Initial simple imputation to get a starting point
            initial_imputer = SimpleImputer(strategy='mean')
            X_filled = initial_imputer.fit_transform(X)
            
            # Matrix factorization implementation
            with st.spinner('Performing matrix factorization...'):
                n_samples, n_features = X.shape
                
                # Initialize latent factors
                # U is the user-feature matrix (n_samples x k)
                # V is the item-feature matrix (n_features x k)
                np.random.seed(42)
                U = np.random.normal(scale=0.1, size=(n_samples, k))
                V = np.random.normal(scale=0.1, size=(n_features, k))
                
                # Stochastic Gradient Descent for matrix factorization
                for iteration in range(max_iter):
                    # Compute current reconstruction
                    X_pred = np.dot(U, V.T)
                    
                    # Compute error only on observed values
                    error = np.sum((X_filled - X_pred) ** 2 * (~missing_mask)) / np.sum(~missing_mask)
                    
                    # Update progress
                    if (iteration + 1) % 5 == 0:
                        progress_text = f"Iteration {iteration+1}/{max_iter}, Error: {error:.6f}"
                        st.text(progress_text)
                    
                    # Update latent factors
                    for i in range(n_samples):
                        for j in range(n_features):
                            if not missing_mask[i, j]:  # Only update based on observed values
                                # Compute error
                                eij = X_filled[i, j] - np.dot(U[i, :], V[j, :])
                                
                                # Update latent factors with regularization
                                for d in range(k):
                                    U[i, d] += 0.01 * (2 * eij * V[j, d] - regularization * U[i, d])
                                    V[j, d] += 0.01 * (2 * eij * U[i, d] - regularization * V[j, d])
                
                # Final prediction
                X_pred = np.dot(U, V.T)
                
                # Update the original data only for missing values
                for col_idx, col_name in enumerate(valid_na_cols):
                    for row_idx in range(n_samples):
                        if missing_mask[row_idx, col_idx]:
                            self.data.loc[self.data.index[row_idx], col_name] = X_pred[row_idx, col_idx]
            
            # Report completion
            st.success(f"Completed matrix factorization imputation for {len(valid_na_cols)} columns")
            
            # Visualize results
            with st.expander("Visualization of Matrix Factorization Results"):
                for col in valid_na_cols:
                    missing_count = original_data[col].isnull().sum()
                    st.write(f"Imputed {missing_count} values in column '{col}'")
                    
                    before_vs_after = pd.DataFrame({
                        'Original (with missing)': original_data[col],
                        'After Matrix Factorization': self.data[col]
                    })
                    
                    # Plot histograms for before and after
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=before_vs_after['Original (with missing)'].dropna(),
                        name='Original (non-missing values)',
                        opacity=0.7
                    ))
                    fig.add_trace(go.Histogram(
                        x=before_vs_after['After Matrix Factorization'],
                        name='After Imputation',
                        opacity=0.7
                    ))
                    fig.update_layout(
                        title=f"Distribution Before and After Matrix Factorization: {col}",
                        barmode='overlay',
                        xaxis_title=col,
                        yaxis_title='Count'
                    )
                    st.plotly_chart(fig)
            
            return self.data
            
        except Exception as e:
            st.error(f"Error during matrix factorization imputation: {str(e)}")
            self.logger.error(f"Matrix factorization imputation failed: {str(e)}")
            raise

    def _show_imputation_comparison(self, columns):
        """Show before-after comparison of imputed columns."""
        if not columns:
            return
            
        for col in columns:
            if col in self.data.columns and col in self.original_data.columns:
                # Create side-by-side plots
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Before Imputation: {col}**")
                    # Distribution plot before imputation
                    fig = px.histogram(
                        self.original_data, x=col,
                        title=f"Distribution before imputation",
                        labels={col: col},
                        opacity=0.7
                    )
                    fig.add_vline(x=self.original_data[col].mean(), line_dash="dash", line_color="red", 
                                annotation_text="Mean", annotation_position="top")
                    fig.add_vline(x=self.original_data[col].median(), line_dash="dash", line_color="green", 
                                annotation_text="Median", annotation_position="bottom")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show missing percentage
                    missing_pct = (self.original_data[col].isnull().sum() / len(self.original_data)) * 100
                    st.metric("Missing Values", f"{missing_pct:.2f}%")
                    
                with col2:
                    st.markdown(f"**After Imputation: {col}**")
                    # Distribution plot after imputation
                    fig = px.histogram(
                        self.data, x=col,
                        title=f"Distribution after imputation",
                        labels={col: col},
                        opacity=0.7
                    )
                    fig.add_vline(x=self.data[col].mean(), line_dash="dash", line_color="red", 
                                annotation_text="Mean", annotation_position="top")
                    fig.add_vline(x=self.data[col].median(), line_dash="dash", line_color="green", 
                                annotation_text="Median", annotation_position="bottom")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show missing percentage
                    missing_pct = (self.data[col].isnull().sum() / len(self.data)) * 100
                    st.metric("Missing Values", f"{missing_pct:.2f}%")
                
                # Summary statistics comparison
                with st.expander("View Summary Statistics"):
                    stats_before = self.original_data[col].describe()
                    stats_after = self.data[col].describe()
                    stats_df = pd.DataFrame({
                        'Before Imputation': stats_before,
                        'After Imputation': stats_after,
                        'Difference': stats_after - stats_before
                    })
                    st.dataframe(stats_df)

    def analyze_missingness_pattern(self):
        """Analyze whether missing data is MCAR, MAR, or MNAR to guide imputation strategy.
        
        This method helps determine if data is:
        - MCAR (Missing Completely At Random): Missing values occur randomly
        - MAR (Missing At Random): Missing values depend on observed data
        - MNAR (Missing Not At Random): Missing values depend on unobserved data
        """
        try:
            # Check if there are any missing values
            if self.data.isnull().sum().sum() == 0:
                st.warning("No missing values in the dataset to analyze")
                return
            
            st.markdown("## Missingness Pattern Analysis")
            st.markdown("""
            This analysis helps determine the mechanism of missingness in your data:
            
            - **MCAR (Missing Completely At Random)**: Missing values occur randomly
            - **MAR (Missing At Random)**: Missing values depend on observed data
            - **MNAR (Missing Not At Random)**: Missing values depend on unobserved data
            
            Understanding the type of missingness helps choose the appropriate imputation method.
            """)
            
            # Create tabs for different analyses
            pattern_tabs = st.tabs(["Little's MCAR Test", "Correlation Analysis", "Group Differences", "Visual Analysis"])
            
            with pattern_tabs[0]:
                st.markdown("### Little's MCAR Test")
                st.markdown("""
                Little's MCAR test assesses whether data is Missing Completely At Random.
                A significant p-value (p < 0.05) suggests the data is not MCAR.
                """)
                
                # Get numeric columns for Little's test
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
                missing_cols = [col for col in numeric_cols if self.data[col].isnull().any()]
                
                if len(missing_cols) < 2:
                    st.warning("Need at least 2 numeric columns with missing values for Little's MCAR test")
                else:
                    # Implement a simplified version of Little's MCAR test
                    # This is a simplified approximation of the test
                    try:
                        # Create binary indicators for missingness in each column
                        missing_indicators = pd.DataFrame()
                        for col in missing_cols:
                            missing_indicators[f"{col}_missing"] = self.data[col].isnull().astype(int)
                        
                        # Compute chi-square statistic from correlation matrix
                        corr_matrix = missing_indicators.corr()
                        chi_square = (len(self.data) - 1) * np.sum(np.tril(corr_matrix.values, k=-1)**2)
                        df = (len(missing_cols) * (len(missing_cols) - 1)) / 2
                        p_value = 1 - stats.chi2.cdf(chi_square, df) if df > 0 else 1.0
                        
                        # Display results
                        if p_value < 0.05:
                            st.error(f"Little's MCAR Test: Chi-square = {chi_square:.2f}, df = {df}, p-value = {p_value:.4f}")
                            st.markdown("""
                            **Result: Data is likely NOT Missing Completely At Random (MCAR)**
                            
                            This suggests the missing values may depend on other variables in your dataset (MAR) 
                            or on the missing values themselves (MNAR).
                            """)
                        else:
                            st.success(f"Little's MCAR Test: Chi-square = {chi_square:.2f}, df = {df}, p-value = {p_value:.4f}")
                            st.markdown("""
                            **Result: Data is likely Missing Completely At Random (MCAR)**
                            
                            This suggests the missingness does not depend on other variables, 
                            and simpler imputation methods may be appropriate.
                            """)
                    except Exception as e:
                        st.error(f"Error in Little's MCAR test: {str(e)}")
            
            with pattern_tabs[1]:
                st.markdown("### Correlation Analysis")
                st.markdown("""
                This analysis examines whether the missingness in one variable is correlated with the values of other variables.
                Significant correlations suggest the data is Missing At Random (MAR).
                """)
                
                # Create missing indicators
                missing_indicators = pd.DataFrame()
                for col in self.data.columns:
                    if self.data[col].isnull().any():
                        missing_indicators[f"{col}_missing"] = self.data[col].isnull().astype(int)
                
                if missing_indicators.empty:
                    st.info("No missing values to analyze")
                else:
                    # For each missing column, check correlation with other observed values
                    for miss_col in missing_indicators.columns:
                        orig_col = miss_col.replace("_missing", "")
                        
                        # Get numeric columns to test for correlation
                        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                        numeric_cols = [col for col in numeric_cols if col != orig_col]
                        
                        if not numeric_cols:
                            st.info(f"No numeric columns available to test correlations with {orig_col}")
                            continue
                        
                        # Calculate correlation between missingness and other variables
                        with st.expander(f"Correlations with missingness in '{orig_col}'"):
                            corr_results = []
                            for col in numeric_cols:
                                if not self.data[col].isnull().all():  # Skip if all values are null
                                    corr = self.data[col].corr(missing_indicators[miss_col], method='spearman')
                                    p_val = stats.spearmanr(self.data[col].fillna(self.data[col].median()), 
                                                           missing_indicators[miss_col])[1]
                                    corr_results.append({
                                        'Variable': col,
                                        'Correlation': corr,
                                        'P-value': p_val,
                                        'Significant': p_val < 0.05
                                    })
                            
                            if corr_results:
                                corr_df = pd.DataFrame(corr_results).sort_values('P-value')
                                st.dataframe(corr_df)
                                
                                # Create a bar chart for significant correlations
                                sig_corrs = corr_df[corr_df['Significant']]
                                if not sig_corrs.empty:
                                    st.markdown("**Significant correlations detected**")
                                    fig = px.bar(
                                        sig_corrs, 
                                        x='Variable', 
                                        y='Correlation',
                                        color='Correlation',
                                        color_continuous_scale=px.colors.diverging.RdBu_r,
                                        title=f"Variables significantly correlated with missingness in {orig_col}"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.markdown("""
                                    **Interpretation**: These correlations suggest the data might be 
                                    Missing At Random (MAR) - missingness depends on observed values.
                                    """)
                                else:
                                    st.info("No significant correlations found")
                                    st.markdown("""
                                    **Interpretation**: The lack of correlations suggests the data might be 
                                    Missing Completely At Random (MCAR) for this variable.
                                    """)
            
            with pattern_tabs[2]:
                st.markdown("### Group Differences Analysis")
                st.markdown("""
                This analysis examines whether there are significant differences in values when 
                grouped by missingness in another variable.
                """)
                
                # Select variables for analysis
                missing_cols = [col for col in self.data.columns if self.data[col].isnull().any()]
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    missing_col = st.selectbox("Select column with missing values", missing_cols, key="missing_col")
                with col2:
                    value_col = st.selectbox("Select numeric column to compare", numeric_cols, key="value_col")
                
                if missing_col and value_col:
                    # Create groups based on missingness
                    missing_mask = self.data[missing_col].isnull()
                    group1 = self.data.loc[~missing_mask, value_col].dropna()
                    group2 = self.data.loc[missing_mask, value_col].dropna()
                    
                    if len(group1) > 0 and len(group2) > 0:
                        # Perform t-test
                        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
                        
                        # Display results
                        st.write(f"T-test results: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
                        
                        # Interpretation
                        if p_val < 0.05:
                            st.error("Significant difference detected")
                            st.markdown(f"""
                            **Interpretation**: The values of '{value_col}' are significantly different between 
                            rows where '{missing_col}' is missing vs. non-missing. This suggests the data is likely 
                            Missing At Random (MAR).
                            """)
                        else:
                            st.success("No significant difference detected")
                            st.markdown(f"""
                            **Interpretation**: The values of '{value_col}' are not significantly different between 
                            rows where '{missing_col}' is missing vs. non-missing. This is consistent with 
                            Missing Completely At Random (MCAR).
                            """)
                        
                        # Visualization
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=group1,
                            name=f"{missing_col} not missing",
                            boxmean=True
                        ))
                        fig.add_trace(go.Box(
                            y=group2,
                            name=f"{missing_col} missing",
                            boxmean=True
                        ))
                        fig.update_layout(
                            title=f"Distribution of {value_col} by missingness in {missing_col}",
                            yaxis_title=value_col
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough data for comparison")
            
            with pattern_tabs[3]:
                st.markdown("### Visual Analysis")
                st.markdown("""
                Visualize patterns of missingness to gain insights about the missing data mechanism.
                """)
                
                # Create missing matrix with heatmap
                st.subheader("Missing Data Pattern")
                missing_matrix = self.data.isnull().astype(int)
                
                # Calculate and sort by pattern frequency
                pattern_freq = missing_matrix.value_counts().reset_index()
                if not pattern_freq.empty:
                    pattern_freq.columns = [*missing_matrix.columns, 'frequency']
                    pattern_freq = pattern_freq.sort_values('frequency', ascending=False).head(10)
                    
                    st.markdown("**Top Missing Data Patterns:**")
                    
                    # Format for better display
                    for i, col in enumerate(missing_matrix.columns):
                        pattern_freq[col] = pattern_freq[col].apply(lambda x: 'Missing' if x == 1 else 'Present')
                    
                    st.dataframe(pattern_freq)
                    
                    # Summary of findings
                    st.markdown("### Summary of Missingness Analysis")
                    
                    # Check for monotonicity
                    missing_counts = self.data.isnull().sum().sort_values(ascending=False)
                    if (missing_counts.diff() >= 0).all():
                        st.success("✅ **Monotone missingness pattern detected**")
                        st.markdown("""
                        A monotone pattern indicates that if a variable is missing for a subject, 
                        then all variables after it in the sequence are also missing. This can be 
                        handled efficiently with specialized imputation approaches.
                        """)
                    
                    # Calculate overall missingness proportion
                    total_missing = self.data.isnull().sum().sum()
                    total_elements = self.data.size
                    missing_proportion = total_missing / total_elements
                    
                    # Recommendations based on analysis
                    st.markdown("### Recommended Imputation Approaches")
                    
                    if missing_proportion < 0.05:
                        st.info("🔍 **Low overall missingness (< 5%)**")
                        st.markdown("""
                        With low missingness, simple approaches like mean/median imputation or row deletion 
                        may be sufficient without introducing significant bias.
                        """)
                    
                    # Add final recommendations based on all tabs' analyses
                    st.markdown("### Final Assessment")
                    st.markdown("""
                    Based on all analyses, your data missingness appears to be:
                    
                    1. **Check the 'Little's MCAR Test' tab** to see if data is Missing Completely At Random
                    2. **Check the 'Correlation Analysis' tab** to identify potential MAR patterns
                    3. **Check the 'Group Differences' tab** for further evidence of MAR
                    
                    #### Recommended Imputation Methods:
                    
                    - **If MCAR**: Simple imputation methods (mean, median, mode) are appropriate
                    - **If MAR**: Use more sophisticated methods like MICE, KNN, or Random Forest imputation
                    - **If MNAR**: Consider modeling the missingness explicitly or collecting additional data
                    
                    Remember that multiple imputation approaches may be needed for different variables.
                    """)
        
        except Exception as e:
            st.error(f"Error analyzing missingness patterns: {str(e)}")
            self.logger.error(f"Error analyzing missingness patterns: {str(e)}")
            raise

    def drop_duplicates(self, subset=None, keep='first', inplace=False):
        """Drop duplicate rows from the DataFrame.
        
        Args:
            subset (list, optional): Column names to consider for identifying duplicates. 
                                    Default is None, which uses all columns.
            keep (str, optional): 'first': Keep first occurrence, 'last': Keep last occurrence, 
                                 False: Drop all duplicates. Default is 'first'.
            inplace (bool, optional): If True, modifies the DataFrame in place. Default is False.
            
        Returns:
            DataFrame: DataFrame with duplicates removed.
        """
        try:
            original_shape = self.data.shape
            
            # If specified column subset is valid
            if subset:
                invalid_cols = [col for col in subset if col not in self.data.columns]
                if invalid_cols:
                    st.error(f"Invalid column(s) specified: {', '.join(invalid_cols)}")
                    return self.data
            
            # Store original data for comparison
            before_drop = self.data.copy()
            
            # Drop duplicates
            if inplace:
                self.data.drop_duplicates(subset=subset, keep=keep, inplace=True)
                result = self.data
            else:
                result = self.data.drop_duplicates(subset=subset, keep=keep, inplace=False)
                if not inplace:
                    self.data = result
            
            # Report results
            new_shape = self.data.shape
            rows_removed = original_shape[0] - new_shape[0]
            
            if rows_removed > 0:
                st.success(f"Removed {rows_removed} duplicate rows ({rows_removed/original_shape[0]:.2%} of data)")
                
                # Show before/after comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Before removing duplicates:**")
                    st.write(f"Shape: {original_shape}")
                    st.dataframe(before_drop.head(5))
                    
                with col2:
                    st.markdown("**After removing duplicates:**")
                    st.write(f"Shape: {new_shape}")
                    st.dataframe(self.data.head(5))
                
                # Visualize the duplicated values if subset is specified
                if subset and len(subset) > 0:
                    st.markdown("### Duplicate Values Distribution")
                    st.markdown(f"Showing distribution of values in columns used for duplicate detection: {', '.join(subset)}")
                    
                    for col in subset:
                        if col in self.data.columns:
                            # Count value frequencies before and after
                            before_counts = before_drop[col].value_counts().reset_index()
                            before_counts.columns = ['Value', 'Count (Before)']
                            
                            after_counts = self.data[col].value_counts().reset_index()
                            after_counts.columns = ['Value', 'Count (After)']
                            
                            # Merge the counts
                            merged_counts = pd.merge(before_counts, after_counts, on='Value', how='left')
                            merged_counts['Duplicates Removed'] = merged_counts['Count (Before)'] - merged_counts['Count (After)']
                            
                            # Sort by duplicates removed
                            merged_counts = merged_counts.sort_values('Duplicates Removed', ascending=False)
                            
                            # Show top values with duplicates
                            top_dupes = merged_counts[merged_counts['Duplicates Removed'] > 0].head(10)
                            
                            if not top_dupes.empty:
                                st.markdown(f"**Top duplicate values in '{col}':**")
                                st.dataframe(top_dupes)
                                
                                # Create visualization
                                fig = px.bar(top_dupes.head(5), 
                                           x='Value', 
                                           y=['Count (After)', 'Duplicates Removed'],
                                           title=f"Top duplicate values in '{col}'",
                                           barmode='stack')
                                
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No duplicate rows found.")
            
            return result
            
        except Exception as e:
            st.error(f"Error dropping duplicates: {str(e)}")
            self.logger.error(f"Error dropping duplicates: {str(e)}")
            raise

    def _handle_check_missing_values(self):
        """Handle the Check Missing Values option in the Streamlit interface."""
        self._handle_option("Check Missing Values", self.check_missing)

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
            "Check Missing Values",
            "Auto Clean",
            "Drop Duplicates",
            "Drop Columns",
            "Drop Missing Values",
            "Analyze Missingness Pattern",
            "Add Variable to Denote NA",
            "Impute NA with Arbitrary Value",
            "Impute NA with Interpolation",
            "Impute NA with KNN",
            "Impute NA with SimpleImputer",
            "Impute NA with Average",
            "Impute NA with Grouped",
            "Impute NA with Random Forest",
            "Impute NA with Matrix Factorization",
        ]

        option = st.sidebar.selectbox("Select Imputation Method", options)

        # Handle selected option
        if option == "Check Missing Values":
            self._handle_check_missing_values()
        elif option == "Auto Clean":
            self._handle_auto_clean()
        elif option == "Drop Duplicates":
            subset = st.multiselect("Select Columns to Consider for Duplicates", self.data.columns)
            keep = st.selectbox("Keep", ["first", "last", False])
            inplace = st.checkbox("Modify DataFrame in Place", value=False)
            if st.button("Execute Drop Duplicates"):
                self.drop_duplicates(subset=subset, keep=keep, inplace=inplace)
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
        elif option == "Impute NA with Grouped":
            group_cols = st.multiselect("Select Grouping Columns", self.data.columns)
            na_cols = st.multiselect("Select Columns to Impute", self.data.columns)
            method = st.selectbox("Imputation Method", ['mean', 'median', 'mode'])
            if st.button("Execute Grouped Imputation"):
                self.impute_NA_with_grouped(group_cols=group_cols, NA_col=na_cols, method=method)
        elif option == "Impute NA with Random Forest":
            na_cols = st.multiselect("Select Columns to Impute", self.data.columns)
            n_estimators = st.number_input("Number of Trees", min_value=10, value=100)
            max_depth = st.number_input("Max Depth (None for unlimited)", min_value=1, value=None)
            if st.button("Execute Random Forest Imputation"):
                self.impute_NA_with_random_forest(NA_col=na_cols, n_estimators=n_estimators, max_depth=max_depth)
        elif option == "Impute NA with Matrix Factorization":
            na_cols = st.multiselect("Select Columns to Impute", self.data.columns)
            k = st.number_input("Number of Latent Factors", min_value=1, value=3)
            max_iter = st.number_input("Maximum Iterations", min_value=1, value=30)
            regularization = st.number_input("Regularization Parameter", min_value=0.0, value=0.1)
            if st.button("Execute Matrix Factorization Imputation"):
                self.impute_NA_with_matrix_factorization(NA_col=na_cols, k=k, max_iter=max_iter, regularization=regularization)
        elif option == "Analyze Missingness Pattern":
            self.analyze_missingness_pattern()
        return self.data