"""Main Predicta application class."""

import os
import uuid
import tempfile
from pathlib import Path
from typing import Optional, List

import pandas as pd
import streamlit as st

from ..core.config import Config
from ..core.logging_config import get_logger
from ..modules.data_exploration import DataExplorer, DataOverview
from ..modules.feature_cleaning import DataImputer, OutlierDetector
from ..modules.feature_engineering import DataEncoder, DataTransformer
from ..modules.feature_selection import FeatureImportanceAnalyzer, BestParam
from ..modules.ml_models import PredictAlgo, PredictImageAlgo
from ..modules.code_editor import CodeEditor
from ..ui.theme import Theme

logger = get_logger(__name__)


class PredictaApp:
    """Main class for the Predicta application."""

    def __init__(self):
        """Initialize the Predicta application."""
        self.df: Optional[pd.DataFrame] = None
        self.image_data = None
        
        # Session management
        if "user_session" not in st.session_state:
            st.session_state.user_session = str(uuid.uuid4())
        
        self.user_session = st.session_state.user_session
        self.modified_df_path = Config.get_temp_file_path(f"modified_data_{self.user_session}.csv")
        
        # Load any existing modified data
        self.load_modified_df()
        
        # Initialize theme
        self.theme = Theme()
        self.theme.init_styling()
        
        logger.info(f"Initialized Predicta app for session: {self.user_session}")

    def show_hero_image(self) -> None:
        """Display the hero image."""
        hero_path = Config.ASSETS_DIR / "Hero.png"
        if hero_path.exists():
            st.image(str(hero_path))
        else:
            st.warning("Hero image not found")

    def read_csv_with_encoding(self, uploaded_file, encodings: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Try reading a CSV file with different encodings until successful.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            encodings: List of encodings to try
            
        Returns:
            Pandas DataFrame or None if reading fails
        """
        if encodings is None:
            encodings = Config.UPLOAD_CONFIG["encoding_options"]
        
        for encoding in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                logger.info(f"Successfully read CSV with encoding: {encoding}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading CSV with encoding {encoding}: {e}")
                st.error(f"Error occurred while reading with encoding {encoding}: {e}")
                return None

        error_msg = f"Failed to read CSV file with encodings: {encodings}"
        logger.error(error_msg)
        st.error(error_msg + " Please check your file format.")
        return None

    def file_upload(self) -> None:
        """Handle file upload functionality."""
        file_type = st.sidebar.radio("Select file type", ["CSV", "Single Image"])
        
        if file_type == "CSV":
            self._handle_csv_upload()
        elif file_type == "Single Image":
            self._handle_image_upload()

    def _handle_csv_upload(self) -> None:
        """Handle CSV file upload."""
        if not self.modified_df_path.exists():
            uploaded_file = st.file_uploader(
                "Upload CSV", 
                type=["csv"],
                help="Upload a CSV file for data analysis"
            )
            
            if uploaded_file is not None:
                try:
                    self.df = self.read_csv_with_encoding(uploaded_file)
                    if self.df is not None:
                        self.save_modified_df()
                        st.success("CSV file successfully loaded!")
                        logger.info(f"CSV file loaded: {uploaded_file.name}, Shape: {self.df.shape}")
                except Exception as e:
                    error_msg = f"Error loading the CSV file: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
        else:
            st.warning("A modified DataFrame already exists. Please clear the existing DataFrame before uploading a new one.")

    def _handle_image_upload(self) -> None:
        """Handle image file upload."""
        uploaded_image = st.file_uploader(
            "Upload Image", 
            type=["png", "jpg", "jpeg"],
            help="Upload an image for processing"
        )
        
        if uploaded_image is not None:
            try:
                self.image_data = uploaded_image
                st.success("Image uploaded successfully!")
                logger.info(f"Image uploaded: {uploaded_image.name}")
            except Exception as e:
                error_msg = f"Error loading the image: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)

    def handle_sidebar(self) -> None:
        """Handle the sidebar options."""
        st.sidebar.title("Predicta")
        st.sidebar.markdown("---")

        self.file_upload()

        st.sidebar.title("Tools")
        selected_option = st.sidebar.radio(
            "Select Option",
            [   
                "Dataset Overview",
                "Data Analysis",
                "Clean Data",
                "Detect Outlier",
                "Encoder",
                "Data Transformer",
                "Feature Importance Analyzer",
                "Best Parameter Selector",
                "Code Editor",
                "Select ML Models",
                "Select Image Models",
                "Clear Modified DataSet",
            ],
        )
        
        # Route to appropriate handler
        self._handle_tool_selection(selected_option)
        
        st.sidebar.divider()
        self.theme.contributor_info()
        st.sidebar.markdown("---")
        self._handle_about()
        self._handle_help()

    def _handle_tool_selection(self, option: str) -> None:
        """Route tool selection to appropriate handler."""
        handlers = {
            "Data Analysis": self.handle_data_explore,
            "Dataset Overview": self.overview_methods,
            "Clean Data": self.clean_data,
            "Detect Outlier": self.handle_detect_outlier,
            "Encoder": self.encode_data,
            "Code Editor": self.code_editor,
            "Select ML Models": self.handle_select_ml_models,
            "Clear Modified DataSet": self.clear_modified_df,
            "Feature Importance Analyzer": self.feature_importance,
            "Best Parameter Selector": self.find_parameter,
            "Select Image Models": self.handle_select_image_models,
            "Data Transformer": self.data_transformer,
        }

        handler = handlers.get(option)
        if handler:
            handler()
        else:
            st.error(f"Unknown option: {option}")

    def _handle_about(self) -> None:
        """Display information about the application."""
        st.sidebar.markdown("#### About")
        st.sidebar.info(
            "Predicta is a powerful data analysis and machine learning tool "
            "designed to streamline your workflow and provide accurate predictions."
        )

    def _handle_help(self) -> None:
        """Display help information."""
        st.sidebar.markdown("#### Help")
        st.sidebar.info(
            "For any assistance or inquiries, please contact us at "
            "ahammadnafiz@outlook.com"
        )

    def overview_methods(self) -> None:
        """Display dataset information."""
        if self.df is not None:
            # Use DataOverview for comprehensive dataset overview
            overview = DataOverview(self.df)
            overview.data_overview()
            self.save_modified_df()
        else:
            self._show_upload_prompt("Please upload a dataset to see information.")
        self.theme.show_footer()

    def handle_data_explore(self) -> None:
        """Handle data exploration."""
        if self.df is not None:
            explorer = DataExplorer(self.df)
            self.df = explorer.analyze()
            self.save_modified_df()
        else:
            self._show_upload_prompt("Please upload a dataset to explore.")
        self.theme.show_footer()

    def clean_data(self) -> None:
        """Handle missing data imputation."""
        if self.df is not None:
            # Use DataImputer for missing data handling
            imputer = DataImputer(self.df)
            # Call the main imputer interface
            self.df = imputer.imputer()
            self.save_modified_df()
        else:
            self._show_upload_prompt("Please upload a dataset to perform feature cleaning.")
        self.theme.show_footer()

    def handle_detect_outlier(self) -> None:
        """Handle outlier detection."""
        if self.df is not None:
            # Use OutlierDetector for outlier detection
            detector = OutlierDetector(self.df)
            self.df = detector.outlier_detect()
            self.save_modified_df()
        else:
            self._show_upload_prompt("Please upload a dataset to detect outliers.")
        self.theme.show_footer()

    def encode_data(self) -> None:
        """Handle data encoding."""
        if self.df is not None:
            # Use DataEncoder for data encoding
            encoder = DataEncoder(self.df)
            self.df = encoder.encoder()
            self.save_modified_df()
        else:
            self._show_upload_prompt("Please upload a dataset to encode data.")
        self.theme.show_footer()

    def data_transformer(self) -> None:
        """Handle data transformation."""
        if self.df is not None:
            transformer = DataTransformer(self.df)
            self.df = transformer.transformer()
            self.save_modified_df()
        else:
            self._show_upload_prompt("Please upload a dataset to transform data.")
        self.theme.show_footer()

    def feature_importance(self) -> None:
        """Handle feature importance analysis."""
        if self.df is not None:
            analyzer = FeatureImportanceAnalyzer(self.df)
            self.df = analyzer.analyze_features()
            self.save_modified_df()
        else:
            self._show_upload_prompt("Please upload a dataset to find feature importance.")
        self.theme.show_footer()

    def find_parameter(self) -> None:
        """Find best parameters."""
        if self.df is not None:
            # Use BestParam for hyperparameter tuning
            tuner = BestParam(self.df)
            self.df = tuner.select_hyper()
            self.save_modified_df()
        else:
            self._show_upload_prompt("Please upload a dataset to find best parameters.")

    def code_editor(self) -> None:
        """Launch the code editor."""
        try:
            editor = CodeEditor()
            editor.run_code_editor(self.df)
        except Exception as e:
            error_msg = f"Error launching code editor: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            st.info("Code editor functionality is currently unavailable. Please check your dependencies.")

    def handle_select_ml_models(self) -> None:
        """Handle selection of machine learning models."""
        if self.df is not None:
            try:
                # Use PredictAlgo for ML model interface
                ml_algo = PredictAlgo(self.df)
                ml_algo.algo()
            except Exception as e:
                error_msg = f"Error occurred while running ML models: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                st.warning("Please check your dataset for compatibility with selected models.")
        else:
            self._show_upload_prompt("Please upload a dataset to perform prediction.")

    def handle_select_image_models(self) -> None:
        """Handle selection of image processing models."""
        if hasattr(self, 'image_data') and self.image_data is not None:
            try:
                image_processor = PredictImageAlgo(self.image_data)
                image_processor.algo()
            except Exception as e:
                error_msg = f"Error occurred while processing the image: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                st.warning("The image processing failed. Please try with a different image or model.")
        else:
            self._show_upload_prompt("Please upload an image to perform image processing.")
        self.theme.show_footer()

    def save_modified_df(self) -> None:
        """Save the modified DataFrame to a CSV file."""
        if self.df is not None:
            try:
                self.df.to_csv(self.modified_df_path, index=False)
                logger.debug(f"DataFrame saved to: {self.modified_df_path}")
            except Exception as e:
                error_msg = f"Error saving the dataset: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                st.warning("Your changes might not be saved. Please check disk space or permissions.")

    def load_modified_df(self) -> None:
        """Load the modified DataFrame from a CSV file."""
        if self.modified_df_path.exists():
            try:
                self.df = pd.read_csv(self.modified_df_path)
                logger.debug(f"DataFrame loaded from: {self.modified_df_path}")
            except Exception as e:
                error_msg = f"Error loading the saved dataset: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                self.df = None

    def clear_modified_df(self) -> None:
        """Clear the modified DataFrame."""
        if self.modified_df_path.exists():
            try:
                self.modified_df_path.unlink()
                st.success("Modified DataFrame cleared successfully.")
                logger.info("Modified DataFrame cleared")
            except Exception as e:
                error_msg = f"Error clearing modified DataFrame: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
        else:
            st.warning("No modified DataFrame exists.")
            self._show_upload_prompt("First upload a dataset")
        
        self.theme.show_footer()

    def _show_upload_prompt(self, message: str) -> None:
        """Show upload prompt with image."""
        st.markdown(
            f"<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>{message}</div>",
            unsafe_allow_html=True,
        )
        upload_image_path = Config.ASSETS_DIR / "uploadfile.png"
        if upload_image_path.exists():
            st.image(str(upload_image_path), width=None)

    def run(self) -> None:
        """Run the Predicta application."""
        try:
            self.load_modified_df()
            self.show_hero_image()

            self.theme.display_styled_message(
                'Please ensure uploaded datasets are cleared before exiting the application for security.'
            )

            st.divider()
            self.handle_sidebar()
            
        except Exception as e:
            error_msg = f"Error running application: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
