import streamlit as st
import pandas as pd
from DataExplore import explore
from FeatureCleaning import missing_data, outlier
from FeatureEngineering import encoding
from chat import ChatPredicta
from MLModel import predictmlalgo
from codeditor import PredictaCodeEditor
import theme
import os


class PredictaApp:
    def __init__(self):
        self.df = None
        self.anthropi_api_key = None
        self.modified_df_path = "modified_data.csv"
        self.load_modified_df() 

    def show_hero_image(self):
        st.image("Hero.png")

    def show_footer(self):
        st.markdown("---")
        footer = "*copyright@infinitequants*"
        st.markdown(footer)

        footer_content = """
        <div class="footer">
            Follow us: &nbsp;&nbsp;&nbsp;
            <a href="https://github.com/ahammadnafiz" target="_blank">GitHub</a> 🚀 |
            <a href="https://twitter.com/ahammadnafi_z" target="_blank">Twitter</a> 🐦
        </div>
        """
        st.markdown(footer_content, unsafe_allow_html=True)
    
    def file_upload(self):
        if not os.path.exists(self.modified_df_path):
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file is not None:
                self.data = pd.read_csv(uploaded_file)
                self.df = self.data.copy(deep=True)
                self.save_modified_df()
        else:
            st.warning("A modified DataFrame already exists. Please clear the existing DataFrame before uploading a new one.")
            
    def handle_sidebar(self):
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
            
        st.sidebar.markdown("---")
        self.handle_about()
        self.handle_help()

    def handle_about(self):
        st.sidebar.markdown("#### About")
        st.sidebar.info("Predicta is a powerful data analysis and machine learning tool designed to streamline your workflow and provide accurate predictions.")

    def handle_help(self):
        st.sidebar.markdown("#### Help")
        st.sidebar.info("For any assistance or inquiries, please contact us at support@predicta.com.")

    def handle_data_explore(self):
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
        if self.df is not None:
            out = encoding.DataEncoder(self.df)
            self.df = out.encoder()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to detect outlier.</div>",
                unsafe_allow_html=True,
            )
            st.image("uploadfile.png", use_column_width=True)
        self.show_footer()

    def handle_chat_with_predicta(self):
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
        editor = PredictaCodeEditor()
        editor.run_code_editor(self.df)
        self.save_modified_df()
        self.show_footer()

    def handle_select_ml_models(self):
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
        if self.df is not None:
            self.df.to_csv(self.modified_df_path, index=False)

    def load_modified_df(self):
        if os.path.exists(self.modified_df_path):
            self.df = pd.read_csv(self.modified_df_path)

    def clear_modified_df(self):
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
