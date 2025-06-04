import logging

# Configure logging to ignore specific PyTorch errors
class TorchErrorFilter(logging.Filter):
    def filter(self, record):
        if "torch._C._get_custom_class_python_wrapper" in str(record.getMessage()):
            return False
        return True

# Apply the log filter to root logger
root_logger = logging.getLogger()
root_logger.addFilter(TorchErrorFilter())

# Use centralized patches from core module
from predicta.core.streamlit_patches import apply_streamlit_patches
apply_streamlit_patches()

# Suppress asyncio loop warnings
import asyncio
original_get_running_loop = asyncio.get_running_loop

def patched_get_running_loop():
    try:
        return original_get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Apply the patch
asyncio.get_running_loop = patched_get_running_loop

import pandas as pd
import streamlit as st
import logging
import joblib
from predicta.ui.theme.theme import Theme
from predicta.utils.assets import get_prediction_app_image, get_upload_image

class PredictionApp:
    def __init__(self):
        self.model = None
        self.preprocessed_data = None
        self.target_column = None
        self.theme = Theme()
        self.theme.init_styling()

    def show_hero_image(self):
        """Display the hero image."""
        st.image(get_prediction_app_image())
    
    def load_model(self, model_file):
        try:
            self.model = joblib.load(model_file)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            self.model = None

    def load_preprocessed_data(self, dataset_file):
        try:
            self.preprocessed_data = pd.read_csv(dataset_file)
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.warning("Please check the format of your CSV file and try again.")
            self.preprocessed_data = None

    def show_uploaded_data(self):
        st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 30px;'>Dataset</div>",
                unsafe_allow_html=True,
            )
        st.write(self.preprocessed_data)

    def select_features(self):
        st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 20px;'>Select Columns</div>",
                unsafe_allow_html=True,
            )
        feature_columns = st.multiselect("Select Feature Columns", self.preprocessed_data.columns)
        self.target_column = st.selectbox("Select Target Column", self.preprocessed_data.columns)
        return feature_columns

    def input_feature_values(self, feature_columns):
        st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 20px;'>Input Values for Selected Features</div>",
                unsafe_allow_html=True,
            )
        input_values = {}
        for feature in feature_columns:
            input_values[feature] = st.number_input(f"Enter value for {feature}", min_value=0.0, step=0.01)
        return input_values

    def make_prediction(self, input_data):
        try:
            if self.model is None:
                st.error("No model loaded. Please upload a valid model first.")
                return None
                
            prediction = self.model.predict(input_data)
            return prediction
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.warning("This could be due to incompatible feature formats or missing values. Please check your input data.")
            return None

    def display_prediction_result(self, prediction):
        if prediction is not None:
            st.subheader("Prediction Result")
            st.write(prediction)
            st.success("Prediction completed successfully!")
        else:
            st.error("Could not generate prediction. Please check the error messages above.")

    def run(self):
        self.show_hero_image()

        st.sidebar.title("Prediction App")
        st.sidebar.markdown("---")
        
        # Sidebar for uploading files
        st.sidebar.subheader("Upload Model")
        model_file = st.sidebar.file_uploader("Upload Trained Model", type=["pkl"])
        
        dataset_file = st.file_uploader("Upload Preprocessed Dataset (CSV)", type=["csv"])
        
        st.sidebar.divider()
        
        self.theme.contributor_info()

        try:
            if model_file and dataset_file:
                self.load_model(model_file)
                self.load_preprocessed_data(dataset_file)
                
                if self.preprocessed_data is not None:
                    self.show_uploaded_data()
                    
                    feature_columns = self.select_features()
                    
                    if feature_columns and self.target_column:
                        input_values = self.input_feature_values(feature_columns)
                        
                        if st.button("Make Prediction"):
                            input_data = pd.DataFrame([input_values])
                            prediction = self.make_prediction(input_data)
                            self.display_prediction_result(prediction)
                    else:
                        st.warning("Please select at least one feature column and a target column.")
            else:
                st.markdown(
                    "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset and model to make predictions.</div>",
                    unsafe_allow_html=True,
                )
                st.image(get_upload_image(), width=None)
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.warning("Please reload the page and try again. If the issue persists, check your data and model compatibility.")
            
        self.theme.show_footer()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Prediction App",
        page_icon="💫",
        initial_sidebar_state="expanded"
    )
    
    app = PredictionApp()
    app.run()