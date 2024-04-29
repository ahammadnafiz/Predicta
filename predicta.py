import os
import pandas as pd
import streamlit as st

from FeatureCleaning import missing_data, outlier
from FeatureEngineering import encoding
from MLModel import predictmlalgo
from codeditor import PredictaCodeEditor
from DataExplore import explore
from FeatureSelection import featureimportance
from chat import ChatPredicta
import theme


class PredictaApp:
    """Main class for the Predicta application."""

    def __init__(self):
        self.df = None
        self.anthropi_api_key = None
        self.modified_df_path = "modified_data.csv"
        self.load_modified_df()

    def show_hero_image(self):
        """Display the hero image."""
        st.image("Hero.png")

    def show_footer(self):
        """Display the footer."""
        st.markdown("---")
        st.markdown("*copyright@infinitequants*")

        footer_content = """
        <div class="footer">
            Follow us: &nbsp;&nbsp;&nbsp;
            <a href="https://github.com/ahammadnafiz" target="_blank">GitHub</a> 🚀 |
            <a href="https://twitter.com/ahammadnafi_z" target="_blank">Twitter</a> 🐦
        </div>
        """
        st.markdown(footer_content, unsafe_allow_html=True)

    def read_csv_with_encoding(self, uploaded_file, encodings=['latin-1']):
        """
        Try reading a CSV file with multiple encodings until successful.
        """

        for encoding in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error occurred while reading with encoding {encoding}: {e}")
                return None
        
        print(f"Failed to read CSV file with the specified encodings.")
        return None

    def file_upload(self):
        """Handle file upload."""
        if not os.path.exists(self.modified_df_path):
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                self.df = self.read_csv_with_encoding(uploaded_file)
                self.save_modified_df()
        else:
            st.warning("A modified DataFrame already exists. Please clear the existing DataFrame before uploading a new one.")

    def handle_sidebar(self):
        """Handle the sidebar options."""
        st.sidebar.title("Predicta")
        st.sidebar.markdown("---")

        self.file_upload()
        
        with st.sidebar:
            self.anthropi_api_key = st.text_input(
                "Anthropic API Key", key="file_qa_api_key", type="password"
            )
            "[Get an Anthropic API key](https://console.anthropic.com/)"

        st.sidebar.title("Tools")
        selected_option = st.sidebar.radio(
            "Select Option",
            [
                "Data Explore",
                "Impute Missing Values",
                "Detect Outlier",
                "Encoder",
                "Feature Importance Analyzer",
                "Chat With Predicta",
                "PredictaCodeEditor",
                "Select ML Models",
                "Clear Modified DataFrame",
            ],
        )
        if selected_option == "Data Explore":
            self.handle_data_explore()
        elif selected_option == "Impute Missing Values":
            self.handle_impute_missing_values()
        elif selected_option == "Detect Outlier":
            self.handle_detect_outlier()
        elif selected_option == 'Encoder':
            self.encode_data()
        elif selected_option == "Chat With Predicta":
            self.handle_chat_with_predicta()
        elif selected_option == "PredictaCodeEditor":
            self.code_editor()
        elif selected_option == "Select ML Models":
            self.handle_select_ml_models()
        elif selected_option == "Clear Modified DataFrame":
            self.clear_modified_df()
        elif selected_option == "Feature Importance Analyzer":
            self.feature_importance()
            
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
        st.sidebar.info("For any assistance or inquiries, please contact us at support@predicta.com.")

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
            st.image("uploadfile.png", use_column_width=True)
        self.show_footer()
        
    def handle_impute_missing_values(self):
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
            st.image("uploadfile.png", use_column_width=True)
        self.show_footer()

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
            st.image("uploadfile.png", use_column_width=True)
        self.show_footer()
    
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
            st.image("uploadfile.png", use_column_width=True)
        self.show_footer()
    
    def feature_importance(self):
        """Handle feature importance analysis."""
        if self.df is not None:
            out = featureimportance.FeatureImportanceAnalyzer(self.df)
            self.df = out.analyze_features()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to find feature importance.</div>",
                unsafe_allow_html=True,
            )
            st.image("uploadfile.png", use_column_width=True)
        self.show_footer()

    def handle_chat_with_predicta(self):
        """Handle chat interaction with Predicta."""
        if self.df is not None:
            chat_page = ChatPredicta(self.df, self.anthropi_api_key)
            chat_page.chat_with_predicta()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to Chat.</div>",
                unsafe_allow_html=True,
            )
            st.image("uploadfile.png", use_column_width=True)
            self.show_footer()

    def code_editor(self):
        """Launch the code editor."""
        editor = PredictaCodeEditor()
        editor.run_code_editor(self.df)
        self.save_modified_df()
        self.show_footer()

    def handle_select_ml_models(self):
        """Handle selection of machine learning models."""
        if self.df is not None:
            model = predictmlalgo.PredictAlgo(self.df)
            model.algo()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to Perform Prediction.</div>",
                unsafe_allow_html=True,
            )
            st.image("uploadfile.png", use_column_width=True)
        self.show_footer()

    def save_modified_df(self):
        """Save the modified DataFrame to a CSV file."""
        if self.df is not None:
            self.df.to_csv(self.modified_df_path, index=False)

    def load_modified_df(self):
        """Load the modified DataFrame from a CSV file."""
        if os.path.exists(self.modified_df_path):
            self.df = pd.read_csv(self.modified_df_path)

    def clear_modified_df(self):
        """Clear the modified DataFrame."""
        if os.path.exists(self.modified_df_path):
            os.remove(self.modified_df_path)
            st.success("Modified DataFrame cleared successfully.")
        else:
            st.warning("No modified DataFrame exists.")
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to Perform Prediction.</div>",
                unsafe_allow_html=True,
            )
            st.image("uploadfile.png", use_column_width=True)
        self.show_footer()
    
    def run(self):
        """Run the Predicta application."""
        self.load_modified_df()  
        self.show_hero_image()
        self.handle_sidebar()
        
if __name__ == "__main__":
    st.set_page_config(
        page_title="Predicta",
        page_icon="⚡",
        initial_sidebar_state="expanded"
    )
    theme.footer()
    
    app = PredictaApp()
    app.run()
