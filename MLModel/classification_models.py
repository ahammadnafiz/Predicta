import numpy as np
import pandas as pd
import warnings
import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"predicta_ml_{datetime.datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ClassificationModels")

class ClassificationModel:
    def __init__(self, df):
        self.data = df
        logger.info(f"Dataset loaded with shape: {self.data.shape}")
        
        # Add log container for collecting messages
        self.log_messages = []
        
        # Perform initial data validation
        self.validate_dataset()
        
        # Convert data types if needed
        self.df = self.convert_to_float(self.data).select_dtypes(include='number')

    def log_info(self, message):
        """Add info message to log container and log it to the file system"""
        self.log_messages.append({"type": "info", "message": message})
        logger.info(message)
        
    def log_warning(self, message):
        """Add warning message to log container and log it to the file system"""
        self.log_messages.append({"type": "warning", "message": message})
        logger.warning(message)
        
    def log_error(self, message):
        """Add error message to log container and log it to the file system"""
        self.log_messages.append({"type": "error", "message": message})
        logger.error(message)

    def validate_dataset(self):
        """
        Perform initial validation of the dataset and log any issues found.
        """
        # Check if dataframe is empty
        if self.data.empty:
            message = "The provided dataset is empty."
            self.log_error(message)
            st.error(message)
            return False
            
        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            columns_with_missing = missing_values[missing_values > 0]
            message = f"Dataset contains missing values in {len(columns_with_missing)} columns"
            self.log_warning(message)
            
            # Display detailed information about missing values
            st.warning(message)
            st.info("Columns with missing values:")
            missing_df = pd.DataFrame({
                'Column': columns_with_missing.index,
                'Missing Values': columns_with_missing.values,
                'Percentage': (columns_with_missing.values / len(self.data) * 100).round(2)
            })
            st.table(missing_df)
            st.info("Consider handling missing values using the techniques in the FeatureCleaning section.")
            
        # Check for categorical/non-numeric columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            message = f"Dataset contains {len(categorical_cols)} categorical columns that may need encoding"
            self.log_info(message)
            st.info(message)
            st.info(f"Categorical columns: {', '.join(categorical_cols)}")
            st.info("Consider encoding categorical features using techniques in the FeatureEngineering section.")
        
        # Check for constant or nearly constant columns
        constant_cols = []
        nearly_constant_threshold = 0.95
        for col in self.data.columns:
            value_counts = self.data[col].value_counts(normalize=True)
            if value_counts.iloc[0] > nearly_constant_threshold:
                constant_cols.append((col, value_counts.iloc[0]))
        
        if constant_cols:
            message = f"Found {len(constant_cols)} columns with low variance (>95% same value)"
            self.log_warning(message)
            st.warning(message)
            for col, pct in constant_cols:
                st.info(f"Column '{col}' has {pct:.2%} of the same value")
            st.info("Consider removing or transforming low-variance features.")
            
        return True
        
    def check_data_for_model(self, features, target):
        """
        Validate data for model training and identify potential issues.
        Returns True if data is valid for modeling, False otherwise.
        """
        valid = True
        
        # Check if features exist in dataframe
        missing_features = [f for f in features if f not in self.data.columns]
        if missing_features:
            message = f"Selected features not found in dataset: {', '.join(missing_features)}"
            self.log_error(message)
            st.error(message)
            valid = False
        
        # Check if target exists in dataframe
        if target not in self.data.columns:
            message = f"Target variable '{target}' not found in dataset"
            self.log_error(message)
            st.error(message)
            valid = False
            
        if not valid:
            return False
            
        # Check for categorical features
        cat_features = [f for f in features if self.data[f].dtype == 'object' or self.data[f].dtype == 'category']
        if cat_features:
            message = f"Selected features contain categorical data: {', '.join(cat_features)}"
            self.log_warning(message)
            st.warning(message)
            st.info("Categorical features will be attempted to be converted to numeric. Consider encoding them properly using techniques in FeatureEngineering section.")
            
        # Check for missing values in selected features
        missing_in_features = self.data[features].isnull().sum()
        features_with_missing = missing_in_features[missing_in_features > 0]
        if not features_with_missing.empty:
            message = f"Selected features have missing values in {len(features_with_missing)} columns"
            self.log_warning(message)
            st.warning(message)
            for col, count in features_with_missing.items():
                pct = count / len(self.data) * 100
                st.info(f"Feature '{col}' has {count} missing values ({pct:.2f}%)")
            st.info("Missing values may cause issues. Consider handling them using techniques in the FeatureCleaning section.")
        
        # Check for missing values in target
        missing_in_target = self.data[target].isnull().sum()
        if missing_in_target > 0:
            message = f"Target variable '{target}' has {missing_in_target} missing values ({missing_in_target/len(self.data)*100:.2f}%)"
            self.log_error(message)
            st.error(message)
            st.info("Missing values in target variable will be excluded from training, which may bias your model.")
            valid = False
            
        # Check feature data types and ranges
        for feature in features:
            try:
                # Try to convert to numeric
                numeric_data = pd.to_numeric(self.data[feature], errors='coerce')
                if numeric_data.isnull().sum() > 0:
                    non_numeric = self.data[feature].iloc[numeric_data.isnull().idxmax()]
                    message = f"Feature '{feature}' contains non-numeric values like '{non_numeric}'"
                    self.log_warning(message)
                    st.warning(message)
                    st.info(f"Attempting to convert non-numeric values to NaN. You may want to encode these values properly.")
            except Exception as e:
                message = f"Error checking feature '{feature}': {str(e)}"
                self.log_error(message)
                st.error(message)
        
        # Check if target is suitable for classification
        if target:
            unique_values = self.data[target].nunique()
            if unique_values > 10:  # Arbitrary threshold for classification
                message = f"Target variable '{target}' has {unique_values} unique values, which may be too many for classification"
                self.log_warning(message)
                st.warning(message)
                st.info("Consider whether a regression model might be more appropriate.")
            elif unique_values <= 1:
                message = f"Target variable '{target}' has only {unique_values} unique value, making classification impossible"
                self.log_error(message)
                st.error(message)
                valid = False
                
        return valid

    def convert_to_float(self, data):
        """
        Convert integer and numeric object columns to float.
        Handle errors gracefully and report issues.
        """
        # Create a copy to avoid modifying the original data
        data = data.copy()
        conversion_issues = []
        
        # Convert integer columns to float
        int_cols = data.select_dtypes(include=['int']).columns
        for col in int_cols:
            try:
                data[col] = data[col].astype('float')
                self.log_info(f"Successfully converted integer column '{col}' to float")
            except Exception as e:
                message = f"Error converting integer column '{col}' to float: {str(e)}"
                self.log_error(message)
                conversion_issues.append((col, str(e)))
        
        # Convert object columns to float where possible
        obj_cols = data.select_dtypes(include=['object']).columns
        for col in obj_cols:
            try:
                # Save original column in case conversion fails completely
                original = data[col].copy()
                
                # Try to convert to numeric, with non-numeric values becoming NaN
                numeric_col = pd.to_numeric(data[col], errors='coerce')
                
                # Check how many values would be lost in conversion
                nan_count = numeric_col.isnull().sum()
                if nan_count > 0:
                    pct_lost = nan_count / len(numeric_col) * 100
                    if pct_lost > 50:  # If more than 50% would be lost, don't convert
                        message = f"Column '{col}' has {pct_lost:.2f}% non-numeric values. Keeping as categorical."
                        self.log_warning(message)
                        st.warning(message)
                        continue
                    else:
                        message = f"Column '{col}' converted to numeric with {nan_count} values ({pct_lost:.2f}%) becoming NaN"
                        self.log_warning(message)
                        st.warning(message)
                
                # Apply the conversion
                data[col] = numeric_col
                self.log_info(f"Successfully converted object column '{col}' to numeric")
                
            except Exception as e:
                message = f"Error converting object column '{col}': {str(e)}"
                self.log_error(message)
                conversion_issues.append((col, str(e)))
        
        # Report conversion issues if any
        if conversion_issues:
            st.warning(f"Encountered issues when converting {len(conversion_issues)} columns to numeric:")
            for col, error in conversion_issues:
                st.info(f"Column '{col}': {error}")
            st.info("These columns may not be used for modeling unless properly preprocessed.")
        
        return data

    def get_scaler_instance(self, scaler_name):

        if scaler_name == 'StandardScaler':
            return StandardScaler()
        elif scaler_name == 'MinMaxScaler':
            return MinMaxScaler()
        elif scaler_name == 'RobustScaler':
            return RobustScaler()
        elif scaler_name == 'Normalizer':
            return Normalizer()
        else:
            raise ValueError(f"Unsupported scaler method: {scaler_name}")
    
    def confusion_matrix(self, y_true, y_pred, labels=None):
        """
        Display a confusion matrix based on true and predicted labels.
        """
        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_pred)))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        st.write("Confusion Matrix:")
        st.table(cm_df)

        # Create figure and axes objects explicitly
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Plot heatmap on the specific axes
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        
        # Set labels and title using the axes object
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        # Display the figure using st.pyplot with the figure object
        st.pyplot(fig)

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

    def logistic_regression(self, C, max_iter):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
            
        # Validate data for modeling
        if not self.check_data_for_model(features, target):
            st.error("Data validation failed. Please fix the issues above before continuing.")
            return
            
        try:
            use_smote = st.checkbox("Use SMOTE")
            
            # Log the model configuration
            self.log_info(f"Training Logistic Regression with C={C}, max_iter={max_iter}, features={features}, target={target}")
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
            st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            if use_smote:
                try:
                    # Apply SMOTE to balance the class distribution
                    smote = SMOTE(random_state=random_state)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    st.info(f"Applied SMOTE: Training set resampled from {X_train.shape[0]} to {X_train_resampled.shape[0]} samples")
                    self.log_info(f"SMOTE applied successfully. New sample count: {X_train_resampled.shape[0]}")
                except Exception as e:
                    message = f"SMOTE resampling failed: {str(e)}"
                    self.log_error(message)
                    st.error(message)
                    st.info("Proceeding with original imbalanced data")
                    X_train_resampled, y_train_resampled = X_train, y_train
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            pipe = Pipeline([
                ('scaler', scaler),
                ('classifier', LogisticRegression(C=C, max_iter=max_iter))
            ])

            # Fit the model and catch convergence warnings
            with warnings.catch_warnings(record=True) as w:
                pipe.fit(X_train_resampled, y_train_resampled)
                for warning in w:
                    if "ConvergenceWarning" in str(warning.category):
                        st.warning(f"Model did not converge with max_iter={max_iter}. Consider increasing max_iter.")
                        self.log_warning(f"Convergence warning with max_iter={max_iter}")

            y_pred = pipe.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy:', accuracy)
            self.log_info(f"Model accuracy: {accuracy:.4f}")
            
            # Create classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write('Classification Report:')
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
            
            # Display confusion matrix
            self.confusion_matrix(y_test, y_pred)

            # Writing the coefficients and intercept
            coef = pipe.named_steps['classifier'].coef_
            intercept = pipe.named_steps['classifier'].intercept_
            st.write('Coefficients:', coef)
            st.write('Intercept:', intercept)

            # Save the trained model to a file
            model_file = "trained_model_logistic.pkl"
            joblib.dump(pipe, model_file)
            self.log_info(f"Model saved to {model_file}")

            # Provide a download button to the user
            st.download_button(
                label="Download Trained Model",
                data=open(model_file, 'rb').read(),
                file_name=model_file,
                mime="application/octet-stream"
            )
            
        except Exception as e:
            error_message = f"Error during Logistic Regression training: {str(e)}"
            self.log_error(error_message)
            st.error(error_message)
            st.info("Tip: Check that your data doesn't contain non-numeric values, or use encoding techniques from FeatureEngineering section.")

    def random_forest_classifier(self, n_estimators, max_depth, max_features):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
        
        if not target:
            st.warning("Please select a target variable.")
            return
            
        # Validate data for modeling
        if not self.check_data_for_model(features, target):
            st.error("Data validation failed. Please fix the issues above before continuing.")
            return
            
        try:
            use_smote = st.checkbox("Use SMOTE")
            
            # Log the model configuration
            self.log_info(f"Training Random Forest with n_estimators={n_estimators}, max_depth={max_depth}, max_features={max_features}, features={features}, target={target}")
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
            st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            if use_smote:
                try:
                    # Apply SMOTE to balance the class distribution
                    smote = SMOTE(random_state=random_state)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    st.info(f"Applied SMOTE: Training set resampled from {X_train.shape[0]} to {X_train_resampled.shape[0]} samples")
                    self.log_info(f"SMOTE applied successfully. New sample count: {X_train_resampled.shape[0]}")
                except Exception as e:
                    message = f"SMOTE resampling failed: {str(e)}"
                    self.log_error(message)
                    st.error(message)
                    st.info("Proceeding with original imbalanced data")
                    X_train_resampled, y_train_resampled = X_train, y_train
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            pipe = Pipeline([
                ('scaler', scaler),
                ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features))
            ])

            pipe.fit(X_train_resampled, y_train_resampled)

            y_pred = pipe.predict(X_test)

            # Display evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy:', accuracy)
            self.log_info(f"Model accuracy: {accuracy:.4f}")
            
            # Create classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write('Classification Report:')
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
            
            # Display confusion matrix
            self.confusion_matrix(y_test, y_pred)
                
            # Save the trained model to a file
            model_file = "trained_model_random_forest.pkl"
            joblib.dump(pipe, model_file)
            self.log_info(f"Model saved to {model_file}")

            # Provide a download button to the user
            st.download_button(
                label="Download Trained Model",
                data=open(model_file, 'rb').read(),
                file_name=model_file,
                mime="application/octet-stream"
            )
            
        except Exception as e:
            error_message = f"Error during Random Forest training: {str(e)}"
            self.log_error(error_message)
            st.error(error_message)
            st.info("Tip: Check that your data doesn't contain non-numeric values, or use encoding techniques from FeatureEngineering section.")

    def decision_tree_classifier(self, max_depth, min_samples_split):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
            
        if not target:
            st.warning("Please select a target variable.")
            return
            
        # Validate data for modeling
        if not self.check_data_for_model(features, target):
            st.error("Data validation failed. Please fix the issues above before continuing.")
            return
            
        try:
            use_smote = st.checkbox("Use SMOTE")
            
            # Log the model configuration
            self.log_info(f"Training Decision Tree with max_depth={max_depth}, min_samples_split={min_samples_split}, features={features}, target={target}")
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
            st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            if use_smote:
                try:
                    # Apply SMOTE to balance the class distribution
                    smote = SMOTE(random_state=random_state)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    st.info(f"Applied SMOTE: Training set resampled from {X_train.shape[0]} to {X_train_resampled.shape[0]} samples")
                    self.log_info(f"SMOTE applied successfully. New sample count: {X_train_resampled.shape[0]}")
                except Exception as e:
                    message = f"SMOTE resampling failed: {str(e)}"
                    self.log_error(message)
                    st.error(message)
                    st.info("Proceeding with original imbalanced data")
                    X_train_resampled, y_train_resampled = X_train, y_train
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            pipe = Pipeline([
                ('scaler', scaler),
                ('classifier', DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split))
            ])

            pipe.fit(X_train_resampled, y_train_resampled)

            y_pred = pipe.predict(X_test)

            # Display evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy:', accuracy)
            self.log_info(f"Model accuracy: {accuracy:.4f}")
            
            # Create classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write('Classification Report:')
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
            
            # Display confusion matrix
            self.confusion_matrix(y_test, y_pred)
                
            # Save the trained model to a file
            model_file = "trained_model_decision_tree_classifier.pkl"
            joblib.dump(pipe, model_file)
            self.log_info(f"Model saved to {model_file}")

            # Provide a download button to the user
            st.download_button(
                label="Download Trained Model",
                data=open(model_file, 'rb').read(),
                file_name=model_file,
                mime="application/octet-stream"
            )
            
        except Exception as e:
            error_message = f"Error during Decision Tree training: {str(e)}"
            self.log_error(error_message)
            st.error(error_message)
            st.info("Tip: Check that your data doesn't contain non-numeric values, or use encoding techniques from FeatureEngineering section.")

    def knn_classifier(self, n_neighbors, algorithm):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
            
        if not target:
            st.warning("Please select a target variable.")
            return
            
        # Validate data for modeling
        if not self.check_data_for_model(features, target):
            st.error("Data validation failed. Please fix the issues above before continuing.")
            return
            
        try:
            use_smote = st.checkbox("Use SMOTE")
            
            # Log the model configuration
            self.log_info(f"Training KNN Classifier with n_neighbors={n_neighbors}, algorithm={algorithm}, features={features}, target={target}")
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
            st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            if use_smote:
                try:
                    # Apply SMOTE to balance the class distribution
                    smote = SMOTE(random_state=random_state)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    st.info(f"Applied SMOTE: Training set resampled from {X_train.shape[0]} to {X_train_resampled.shape[0]} samples")
                    self.log_info(f"SMOTE applied successfully. New sample count: {X_train_resampled.shape[0]}")
                except Exception as e:
                    message = f"SMOTE resampling failed: {str(e)}"
                    self.log_error(message)
                    st.error(message)
                    st.info("Proceeding with original imbalanced data")
                    X_train_resampled, y_train_resampled = X_train, y_train
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            pipe = Pipeline([
                ('scaler', scaler),
                ('classifier', KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm))
            ])

            pipe.fit(X_train_resampled, y_train_resampled)

            y_pred = pipe.predict(X_test)
            
            # Display evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy:', accuracy)
            self.log_info(f"Model accuracy: {accuracy:.4f}")
            
            # Create classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write('Classification Report:')
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
            
            # Display confusion matrix
            self.confusion_matrix(y_test, y_pred)
                
            # Save the trained model to a file
            model_file = "trained_model_knn_classifier.pkl"
            joblib.dump(pipe, model_file)
            self.log_info(f"Model saved to {model_file}")

            # Provide a download button to the user
            st.download_button(
                label="Download Trained Model",
                data=open(model_file, 'rb').read(),
                file_name=model_file,
                mime="application/octet-stream"
            )
            
        except Exception as e:
            error_message = f"Error during KNN Classifier training: {str(e)}"
            self.log_error(error_message)
            st.error(error_message)
            st.info("Tip: KNN is sensitive to the scale of the data. Ensure proper scaling and check for outliers.")

    def support_vector_classifier(self, C, kernel):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
            
        if not target:
            st.warning("Please select a target variable.")
            return
            
        # Validate data for modeling
        if not self.check_data_for_model(features, target):
            st.error("Data validation failed. Please fix the issues above before continuing.")
            return
            
        try:
            use_smote = st.checkbox("Use SMOTE")
            
            # Log the model configuration
            self.log_info(f"Training SVC with C={C}, kernel={kernel}, features={features}, target={target}")
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
            st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            if use_smote:
                try:
                    # Apply SMOTE to balance the class distribution
                    smote = SMOTE(random_state=random_state)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    st.info(f"Applied SMOTE: Training set resampled from {X_train.shape[0]} to {X_train_resampled.shape[0]} samples")
                    self.log_info(f"SMOTE applied successfully. New sample count: {X_train_resampled.shape[0]}")
                except Exception as e:
                    message = f"SMOTE resampling failed: {str(e)}"
                    self.log_error(message)
                    st.error(message)
                    st.info("Proceeding with original imbalanced data")
                    X_train_resampled, y_train_resampled = X_train, y_train
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            # Warn about potential performance issues with large datasets
            if X_train_resampled.shape[0] > 10000:
                st.warning(f"Training SVM with {X_train_resampled.shape[0]} samples may be slow. Consider using a smaller dataset or a different algorithm.")
                self.log_warning(f"SVM training with large dataset: {X_train_resampled.shape[0]} samples")

            pipe = Pipeline([
                ('scaler', scaler),
                ('classifier', SVC(C=C, kernel=kernel, probability=True))
            ])

            # Track training time for large datasets
            start_time = datetime.datetime.now()
            pipe.fit(X_train_resampled, y_train_resampled)
            training_time = (datetime.datetime.now() - start_time).total_seconds()
            
            if training_time > 5:  # arbitrary threshold for "slow" training
                st.info(f"Model training took {training_time:.2f} seconds")
                self.log_info(f"SVM training completed in {training_time:.2f} seconds")

            y_pred = pipe.predict(X_test)

            # Display evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy:', accuracy)
            self.log_info(f"Model accuracy: {accuracy:.4f}")
            
            # Create classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write('Classification Report:')
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
            
            # Display confusion matrix
            self.confusion_matrix(y_test, y_pred)
                
            # Save the trained model to a file
            model_file = "trained_model_support_vector_classifier.pkl"
            joblib.dump(pipe, model_file)
            self.log_info(f"Model saved to {model_file}")

            # Provide a download button to the user
            st.download_button(
                label="Download Trained Model",
                data=open(model_file, 'rb').read(),
                file_name=model_file,
                mime="application/octet-stream"
            )
            
        except Exception as e:
            error_message = f"Error during SVM training: {str(e)}"
            self.log_error(error_message)
            st.error(error_message)
            st.info("Tip: SVMs are sensitive to the scale of the data and may struggle with non-numeric features. Ensure proper preprocessing.")

    def gradient_boosting_classifier(self, n_estimators, learning_rate, max_depth):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
            
        if not target:
            st.warning("Please select a target variable.")
            return
            
        # Validate data for modeling
        if not self.check_data_for_model(features, target):
            st.error("Data validation failed. Please fix the issues above before continuing.")
            return
            
        try:
            use_smote = st.checkbox("Use SMOTE")
            
            # Log the model configuration
            self.log_info(f"Training Gradient Boosting with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, features={features}, target={target}")
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
            st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            if use_smote:
                try:
                    # Apply SMOTE to balance the class distribution
                    smote = SMOTE(random_state=random_state)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    st.info(f"Applied SMOTE: Training set resampled from {X_train.shape[0]} to {X_train_resampled.shape[0]} samples")
                    self.log_info(f"SMOTE applied successfully. New sample count: {X_train_resampled.shape[0]}")
                except Exception as e:
                    message = f"SMOTE resampling failed: {str(e)}"
                    self.log_error(message)
                    st.error(message)
                    st.info("Proceeding with original imbalanced data")
                    X_train_resampled, y_train_resampled = X_train, y_train
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            pipe = Pipeline([
                ('scaler', scaler),
                ('classifier', GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth))
            ])

            # Track training time for large datasets
            start_time = datetime.datetime.now()
            pipe.fit(X_train_resampled, y_train_resampled)
            training_time = (datetime.datetime.now() - start_time).total_seconds()
            
            if training_time > 5:  # arbitrary threshold for "slow" training
                st.info(f"Model training took {training_time:.2f} seconds")
                self.log_info(f"Gradient Boosting training completed in {training_time:.2f} seconds")

            y_pred = pipe.predict(X_test)

            # Display evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy:', accuracy)
            self.log_info(f"Model accuracy: {accuracy:.4f}")
            
            # Create classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write('Classification Report:')
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
            
            # Display confusion matrix
            self.confusion_matrix(y_test, y_pred)
                
            # Save the trained model to a file
            model_file = "trained_model_gradient_boosting_classifier.pkl"
            joblib.dump(pipe, model_file)
            self.log_info(f"Model saved to {model_file}")

            # Provide a download button to the user
            st.download_button(
                label="Download Trained Model",
                data=open(model_file, 'rb').read(),
                file_name=model_file,
                mime="application/octet-stream"
            )
            
        except Exception as e:
            error_message = f"Error during Gradient Boosting training: {str(e)}"
            self.log_error(error_message)
            st.error(error_message)
            st.info("Tip: Gradient Boosting is sensitive to outliers and may struggle with imbalanced data. Consider using SMOTE or adjusting learning_rate.")

    def adaboost_classifier(self, n_estimators, learning_rate):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
            
        if not target:
            st.warning("Please select a target variable.")
            return
            
        # Validate data for modeling
        if not self.check_data_for_model(features, target):
            st.error("Data validation failed. Please fix the issues above before continuing.")
            return
            
        try:
            use_smote = st.checkbox("Use SMOTE")
            
            # Log the model configuration
            self.log_info(f"Training AdaBoost with n_estimators={n_estimators}, learning_rate={learning_rate}, features={features}, target={target}")
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
            st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            if use_smote:
                try:
                    # Apply SMOTE to balance the class distribution
                    smote = SMOTE(random_state=random_state)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    st.info(f"Applied SMOTE: Training set resampled from {X_train.shape[0]} to {X_train_resampled.shape[0]} samples")
                    self.log_info(f"SMOTE applied successfully. New sample count: {X_train_resampled.shape[0]}")
                except Exception as e:
                    message = f"SMOTE resampling failed: {str(e)}"
                    self.log_error(message)
                    st.error(message)
                    st.info("Proceeding with original imbalanced data")
                    X_train_resampled, y_train_resampled = X_train, y_train
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            pipe = Pipeline([
                ('scaler', scaler),
                ('classifier', AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate))
            ])

            # Track training time
            start_time = datetime.datetime.now()
            pipe.fit(X_train_resampled, y_train_resampled)
            training_time = (datetime.datetime.now() - start_time).total_seconds()
            
            if training_time > 5:  # arbitrary threshold for "slow" training
                st.info(f"Model training took {training_time:.2f} seconds")
                self.log_info(f"AdaBoost training completed in {training_time:.2f} seconds")

            y_pred = pipe.predict(X_test)

            # Display evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy:', accuracy)
            self.log_info(f"Model accuracy: {accuracy:.4f}")
            
            # Create classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write('Classification Report:')
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
            
            # Display confusion matrix
            self.confusion_matrix(y_test, y_pred)
                
            # Save the trained model to a file
            model_file = "trained_model_adaboost_classifier.pkl"
            joblib.dump(pipe, model_file)
            self.log_info(f"Model saved to {model_file}")

            # Provide a download button to the user
            st.download_button(
                label="Download Trained Model",
                data=open(model_file, 'rb').read(),
                file_name=model_file,
                mime="application/octet-stream"
            )
            
        except Exception as e:
            error_message = f"Error during AdaBoost training: {str(e)}"
            self.log_error(error_message)
            st.error(error_message)
            st.info("Tip: AdaBoost may struggle with noisy data. Consider preprocessing your data to remove outliers.")

    def xgboost_classifier(self, n_estimators, max_depth, learning_rate):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
            
        if not target:
            st.warning("Please select a target variable.")
            return
            
        # Validate data for modeling
        if not self.check_data_for_model(features, target):
            st.error("Data validation failed. Please fix the issues above before continuing.")
            return
            
        try:
            use_smote = st.checkbox("Use SMOTE")
            
            # Log the model configuration
            self.log_info(f"Training XGBoost with n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}, features={features}, target={target}")
            
            # Create train/test split with error handling for target conversion
            try:
                # Convert target to numeric if it's not already
                le = LabelEncoder()
                y = le.fit_transform(self.data[target].astype(str))
                
                # Ensure features are numeric
                X = self.data[features].copy()
                for col in X.columns:
                    if X[col].dtype == 'object':
                        st.info(f"Converting categorical feature '{col}' to numeric. Consider proper encoding.")
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                
                # Check for NaN values after conversion
                nan_cols = X.columns[X.isna().any()].tolist()
                if nan_cols:
                    st.warning(f"Features contain NaN values after conversion: {', '.join(nan_cols)}")
                    st.info("NaN values will be imputed or rows with NaN will be removed.")
                    # Fill NaN with column means
                    X = X.fillna(X.mean())
                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
                
            except Exception as e:
                message = f"Error preparing data for XGBoost: {str(e)}"
                self.log_error(message)
                st.error(message)
                return
            
            if use_smote:
                try:
                    # Apply SMOTE to balance the class distribution
                    smote = SMOTE(random_state=random_state)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    st.info(f"Applied SMOTE: Training set resampled from {X_train.shape[0]} to {X_train_resampled.shape[0]} samples")
                    self.log_info(f"SMOTE applied successfully. New sample count: {X_train_resampled.shape[0]}")
                except Exception as e:
                    message = f"SMOTE resampling failed: {str(e)}"
                    self.log_error(message)
                    st.error(message)
                    st.info("Proceeding with original imbalanced data")
                    X_train_resampled, y_train_resampled = X_train, y_train
            else:
                X_train_resampled, y_train_resampled = X_train, y_train

            # Create and fit scaler
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_test_scaled = scaler.transform(X_test)

            # Create and fit XGBoost classifier
            xgb_clf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            
            # Track training time
            start_time = datetime.datetime.now()
            
            # Fit the model
            xgb_clf.fit(X_train_scaled, y_train_resampled)
            
            training_time = (datetime.datetime.now() - start_time).total_seconds()
            if training_time > 5:  # arbitrary threshold for "slow" training
                st.info(f"Model training took {training_time:.2f} seconds")
                self.log_info(f"XGBoost training completed in {training_time:.2f} seconds")
            
            # Make predictions
            y_pred = xgb_clf.predict(X_test_scaled)
            
            # Transform predictions back to original labels
            y_test_original = le.inverse_transform(y_test)
            y_pred_original = le.inverse_transform(y_pred)

            # Display evaluation metrics
            accuracy = accuracy_score(y_test_original, y_pred_original)
            st.write('Accuracy:', accuracy)
            self.log_info(f"Model accuracy: {accuracy:.4f}")
            
            # Create classification report
            report = classification_report(y_test_original, y_pred_original, output_dict=True)
            st.write('Classification Report:')
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
            
            # Display confusion matrix
            self.confusion_matrix(y_test_original, y_pred_original)
                
            # Save the model and preprocessing objects
            model_data = {
                'model': xgb_clf,
                'scaler': scaler,
                'label_encoder': le,
                'feature_names': features
            }
            
            model_file = "trained_model_xgboost_classifier.pkl"
            joblib.dump(model_data, model_file)
            self.log_info(f"Model saved to {model_file}")

            # Provide download button
            with open(model_file, 'rb') as f:
                model_bytes = f.read()
                
            st.download_button(
                label="Download Trained Model",
                data=model_bytes,
                file_name=model_file,
                mime="application/octet-stream"
            )
            
        except Exception as e:
            error_message = f"Error during XGBoost training: {str(e)}"
            self.log_error(error_message)
            st.error(error_message)
            st.info("Tip: XGBoost may fail with categorical data or missing values. Ensure proper preprocessing.")

    def stacking_classifier(self):
        features = st.multiselect("Select Feature Columns", self.data.columns)
        target = st.selectbox("Select Target Variable", self.data.columns)

        scaler_name = st.selectbox("Select Scaler Method", ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer'])
        scaler = self.get_scaler_instance(scaler_name)
        test_size = st.slider("Test Size (proportion)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random State", value=42)

        if not features:
            st.warning("Please select at least one feature column.")
            return
                
        if not target:
            st.warning("Please select a target variable.")
            return
            
        # Validate data for modeling
        if not self.check_data_for_model(features, target):
            st.error("Data validation failed. Please fix the issues above before continuing.")
            return
            
        try:
            use_smote = st.checkbox("Use SMOTE")
            
            # Log the model configuration
            self.log_info(f"Training Stacking Classifier with features={features}, target={target}")
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(self.data[features], self.data[target], test_size=test_size, random_state=random_state)
            st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            if use_smote:
                try:
                    # Apply SMOTE to balance the class distribution
                    smote = SMOTE(random_state=random_state)
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                    st.info(f"Applied SMOTE: Training set resampled from {X_train.shape[0]} to {X_train_resampled.shape[0]} samples")
                    self.log_info(f"SMOTE applied successfully. New sample count: {X_train_resampled.shape[0]}")
                except Exception as e:
                    message = f"SMOTE resampling failed: {str(e)}"
                    self.log_error(message)
                    st.error(message)
                    st.info("Proceeding with original imbalanced data")
                    X_train_resampled, y_train_resampled = X_train, y_train
            else:
                X_train_resampled, y_train_resampled = X_train, y_train
                    
            # Define base estimators
            st.info("Using Decision Tree and Random Forest as base estimators")
            base_estimators = [
                ('decision_tree', DecisionTreeClassifier()),
                ('random_forest', RandomForestClassifier())
                # Add more base estimators as needed
            ]
            
            # Initialize StackingClassifier
            stacking_classifier = StackingClassifier(
                estimators=base_estimators,
                final_estimator=LogisticRegression()  # You can replace LogisticRegression() with any other final estimator
            )

            pipe = Pipeline([
                ('scaler', scaler),
                ('classifier', stacking_classifier)
            ])

            # Track training time for large datasets
            start_time = datetime.datetime.now()
            pipe.fit(X_train_resampled, y_train_resampled)
            training_time = (datetime.datetime.now() - start_time).total_seconds()
            
            if training_time > 10:  # higher threshold for stacking as it's expected to be slower
                st.info(f"Model training took {training_time:.2f} seconds")
                self.log_info(f"Stacking Classifier training completed in {training_time:.2f} seconds")

            y_pred = pipe.predict(X_test)

            # Display evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy:', accuracy)
            self.log_info(f"Model accuracy: {accuracy:.4f}")
            
            # Create classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write('Classification Report:')
            report_df = pd.DataFrame(report).transpose()
            st.table(report_df)
            
            # Display confusion matrix
            self.confusion_matrix(y_test, y_pred)
                
            # Save the trained model to a file
            model_file = "trained_model_stacking_classifier.pkl"
            joblib.dump(pipe, model_file)
            self.log_info(f"Model saved to {model_file}")

            # Provide a download button to the user
            st.download_button(
                label="Download Trained Model",
                data=open(model_file, 'rb').read(),
                file_name=model_file,
                mime="application/octet-stream"
            )
            
        except Exception as e:
            error_message = f"Error during Stacking Classifier training: {str(e)}"
            self.log_error(error_message)
            st.error(error_message)
            st.info("Tip: Stacking classifiers are complex and may be sensitive to data issues. Ensure your data is properly preprocessed.")