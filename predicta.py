import os
import uuid
import tempfile
import pandas as pd
import streamlit as st

# Suppress specific PyTorch-related errors in Streamlit
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

# Monkeypatch Streamlit's module path extraction to handle PyTorch modules
from streamlit.watcher import local_sources_watcher

original_get_module_paths = local_sources_watcher.get_module_paths

def patched_get_module_paths(module):
    try:
        return original_get_module_paths(module)
    except RuntimeError as e:
        if "torch._C._get_custom_class_python_wrapper" in str(e):
            return []
        raise e

# Apply the patch
local_sources_watcher.get_module_paths = patched_get_module_paths

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

from FeatureCleaning import missing_data, outlier
from FeatureEngineering import encoding, transform
from MLModel import predictmlalgo
from MLModel import predictimagealgo
from codeditor import PredictaCodeEditor
from DataExplore import explore, overview
from FeatureSelection import featureimportance, hyperparameter
from Theme import theme

class PredictaApp:
    """Main class for the Predicta application."""

    def __init__(self):
        self.df = None
        self.temp_dir = tempfile.gettempdir()

        # Check if a user_session exists in st.session_state
        if "user_session" not in st.session_state:
            # Generate a new user_session if it doesn't exist
            st.session_state.user_session = str(uuid.uuid4())

        self.user_session = st.session_state.user_session
        self.modified_df_path = os.path.join(self.temp_dir, f"modified_data_{self.user_session}.csv")
        self.load_modified_df()

        # Initialize custom styling
        theme.init_styling()

    def show_hero_image(self):
        """Display the hero image."""
        st.image("assets/Hero.png")
        

    def read_csv_with_encoding(self, uploaded_file, encodings=['utf-8', 'latin-1', 'ISO-8859-1']):
        """Try reading a CSV file with encoding until successful."""
        for encoding in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Error occurred while reading with encoding {encoding}: {e}")
                return None

        st.error(f"Failed to read CSV file with the specified encodings. Please check your file format.")
        return None

    def file_upload(self):
        file_type = st.sidebar.radio("Select file type", ["CSV", "Single Image"])
        
        if file_type == "CSV":
            if not os.path.exists(self.modified_df_path):
                uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
                if uploaded_file is not None:
                    try:
                        self.df = self.read_csv_with_encoding(uploaded_file)
                        if self.df is not None:
                            self.save_modified_df()
                            st.success("CSV file successfully loaded!")
                    except Exception as e:
                        st.error(f"Error loading the CSV file: {str(e)}")
            else:
                st.warning("A modified DataFrame already exists. Please clear the existing DataFrame before uploading a new one.")
        elif file_type == "Single Image":
            uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
            if uploaded_image is not None:
                try:
                    self.image_data = uploaded_image
                    st.success("Image uploaded successfully!")
                except Exception as e:
                    st.error(f"Error loading the image: {str(e)}")

    def handle_sidebar(self):
        """Handle the sidebar options."""
        st.sidebar.title("Predicta")
        st.sidebar.markdown("---")

        self.file_upload()

        st.sidebar.title("Tools")
        selected_option = st.sidebar.radio(
            "Select Option",
            [   
                "Dataset Overview",
                "Clean Data",
                "Detect Outlier",
                "Encoder",
                "Data Transformer",
                "Data Analysis",
                "Feature Importance Analyzer",
                "Best Parameter Selector",
                "PredictaCodeEditor",
                "Select ML Models",
                "Select Image Models",
                "Clear Modified DataSet",
            ],
        )
        if selected_option == "Data Analysis":
            self.handle_data_explore()
        elif selected_option == "Dataset Overview":
            self.overview_methods()
        elif selected_option == "Clean Data":
            self.clean_data()
        elif selected_option == "Detect Outlier":
            self.handle_detect_outlier()
        elif selected_option == 'Encoder':
            self.encode_data()
        elif selected_option == "PredictaCodeEditor":
            self.code_editor()
        elif selected_option == "Select ML Models":
            self.handle_select_ml_models()
        elif selected_option == "Clear Modified DataSet":
            self.clear_modified_df()
        elif selected_option == "Feature Importance Analyzer":
            self.feature_importance()
        elif selected_option == "Best Parameter Selector":
            self.find_parameter()
        elif selected_option == "Select Image Models":
            self.handle_select_image_models()
        elif selected_option == "Data Transformer":
            self.data_transformer()

        st.sidebar.divider()

        theme.contributor_info()

        st.sidebar.markdown("---")
        self.handle_about()
        self.handle_help()

    def handle_about(self):
        """Display information about the application."""
        st.sidebar.markdown("#### About")
        st.sidebar.info("Predicta is a powerful data analysis and machine learning tool designed to streamline your workflow and provide accurate predictions.")

    def handle_help(self):
        """Display help information."""
        st.sidebar.markdown("#### Help")
        st.sidebar.info("For any assistance or inquiries, please contact us at ahammadnafiz@outlook.com")

    def overview_methods(self):
        """Display DateSet information."""
        if self.df is not None:
            overview_data = overview.DataOverview(self.df)
            overview_data.data_overview()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to see Informations.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None)  
        theme.show_footer()
    
    def handle_data_explore(self):
        """Handle data exploration."""
        if self.df is not None:
            analysis = explore.DataAnalyzer(self.df)
            self.df = analysis.analyzer()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to Explore.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None)  
        theme.show_footer()

    def clean_data(self):
        """Handle missing data imputation."""
        if self.df is not None:
            impute = missing_data.DataImputer(self.df)
            self.df = impute.imputer()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to perform feature cleaning.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None)  
        theme.show_footer()

    def handle_detect_outlier(self):
        """Handle outlier detection."""
        if self.df is not None:
            out = outlier.OutlierDetector(self.df)
            self.df = out.outlier_detect()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to detect outlier.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None)  
        theme.show_footer()

    def encode_data(self):
        """Handle data encoding."""
        if self.df is not None:
            out = encoding.DataEncoder(self.df)
            self.df = out.encoder()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to encode data.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None)  
        theme.show_footer()
    
    def data_transformer(self):
        """Handle data transformation."""
        if self.df is not None:
            out = transform.DataTransformer(self.df)
            self.df = out.transformer()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to transform data.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None)  
        theme.show_footer()

    def feature_importance(self):
        """Handle feature importance analysis."""
        if self.df is not None:
            out = featureimportance.FeatureImportanceAnalyzer(self.df)
            out.analyze_features()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to find feature importance.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None)  
        theme.show_footer()
    
    def find_parameter(self):
        """Find best parameter."""
        if self.df is not None:
            out = hyperparameter.BestParam(self.df)
            out.select_hyper()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to find best parameters.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None)  
        theme.show_footer()

    def code_editor(self):
        """Launch the code editor."""
        editor = PredictaCodeEditor()
        editor.run_code_editor(self.df)
        self.save_modified_df()
        theme.show_footer()

    def handle_select_ml_models(self):
        """Handle selection of machine learning models."""
        if self.df is not None:
            try:
                model = predictmlalgo.PredictAlgo(self.df)
                model.algo()
                self.save_modified_df()
            except Exception as e:
                st.error(f"Error occurred while running ML models: {str(e)}")
                st.warning("Please check your dataset for compatibility with selected models.")
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to Perform Prediction.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None)
        theme.show_footer()

    def handle_select_image_models(self):
        """Handle selection of image processing models."""
        if hasattr(self, 'image_data') and self.image_data is not None:
            try:
                model = predictimagealgo.PredictImageAlgo(self.image_data)
                model.algo()
            except Exception as e:
                st.error(f"Error occurred while processing the image: {str(e)}")
                st.warning("The image processing failed. Please try with a different image or model.")
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload an image to perform image processing.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None)
        theme.show_footer()

    def save_modified_df(self):
        """Save the modified DataFrame to a CSV file."""
        if self.df is not None:
            try:
                self.df.to_csv(self.modified_df_path, index=False)
            except Exception as e:
                st.error(f"Error saving the dataset: {str(e)}")
                st.warning("Your changes might not be saved. Please check disk space or permissions.")

    def load_modified_df(self):
        """Load the modified DataFrame from a CSV file."""
        if os.path.exists(self.modified_df_path):
            try:
                self.df = pd.read_csv(self.modified_df_path)
            except Exception as e:
                st.error(f"Error loading the saved dataset: {str(e)}")
                self.df = None

    def clear_modified_df(self):
        """Clear the modified DataFrame."""
        if os.path.exists(self.modified_df_path):
            os.remove(self.modified_df_path)
            st.success("Modified DataFrame cleared successfully.")
        else:
            st.warning("No modified DataFrame exists.")
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px; '>First Upload a DataSet</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", width=None) 

        theme.show_footer()

    def run(self):
        """Run the Predicta application."""
        self.load_modified_df()
        self.show_hero_image()

        theme.display_styled_message('Please ensure uploaded datasets are cleared before exiting the application for security.')

        st.divider()

        self.handle_sidebar()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Predicta",
        page_icon="⚡",
        initial_sidebar_state="expanded"
    )

    app = PredictaApp()
    app.run()