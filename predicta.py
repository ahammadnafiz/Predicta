import os
import pandas as pd
import streamlit as st
import uuid
import tempfile

from FeatureCleaning import missing_data, outlier
from FeatureEngineering import encoding, transform
from MLModel import predictmlalgo
from codeditor import PredictaCodeEditor
from DataExplore import explore, overview
from FeatureSelection import featureimportance, hyperparameter
from chat import ChatPredicta
from Theme import theme

class PredictaApp:
    """Main class for the Predicta application."""

    def __init__(self):
        self.df = None
        self.anthropi_api_key = None
        self.temp_dir = tempfile.gettempdir()

        # Check if a user_session exists in st.session_state
        if "user_session" not in st.session_state:
            # Generate a new user_session if it doesn't exist
            st.session_state.user_session = str(uuid.uuid4())

        self.user_session = st.session_state.user_session
        self.modified_df_path = os.path.join(self.temp_dir, f"modified_data_{self.user_session}.csv")
        self.load_modified_df()

    def show_hero_image(self):
        """Display the hero image."""
        st.image("assets/Hero.png")

    def show_footer(self):
        """Display the footer."""
        st.markdown("---")
        st.markdown("*copyright@ahammadnafiz*")

        footer_content = """
        <div class="footer">
            Follow me: &nbsp;&nbsp;&nbsp;
            <a href="https://github.com/ahammadnafiz" target="_blank">GitHub</a> 🚀 |
            <a href="https://twitter.com/ahammadnafi_z" target="_blank">Twitter</a> 🐦 |
            <a href="https://github.com/ahammadnafiz/Predicta/blob/main/LICENSE" target="_blank">License</a> 📜
        </div>
        """
        st.markdown(footer_content, unsafe_allow_html=True)

    def read_csv_with_encoding(self, uploaded_file, encodings=['latin-1']):
        """Try reading a CSV file with encoding until successful."""
        for encoding in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.info(f"Error occurred while reading with encoding {encoding}: {e}")
                return None

        st.info(f"Failed to read CSV file with the specified encodings.")
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

        # with st.sidebar:
        #     self.anthropi_api_key = st.text_input(
        #         "Anthropic API Key", key="file_qa_api_key", type="password"
        #     )
        #     "[Get an Anthropic API key](https://console.anthropic.com/)"

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
                # "Chat With Predicta",
                "PredictaCodeEditor",
                "Select ML Models",
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
        # elif selected_option == "Chat With Predicta":
        #     self.handle_chat_with_predicta()
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
        elif selected_option == "Data Transformer":
            self.data_transformer()

        # st.sidebar.divider()

        # self.contributor_info()

        st.sidebar.markdown("---")
        self.handle_about()
        self.handle_help()

    def contributor_info(self):
        nafiz_info = {
            "name": "Ahammad Nafiz",
            "role": "Curious Learner",
            "image_url": "https://avatars.githubusercontent.com/u/86776685?s=400&u=82112040d4a196f3d796c1aa4e7112d403c19450&v=4",
            "linkedin_url": "https://www.linkedin.com/in/ahammad-nafiz/",
            "github_url": "https://github.com/ahammadnafiz",
        }

        st.sidebar.write("#### 👨‍💻 Developed by:")
        st.sidebar.markdown(theme.contributor_card(
            **nafiz_info,
        ),
            unsafe_allow_html=True)

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
            options = overview.DataOverview(self.df)
            self.df = options.data_overview()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to see Informations.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", use_column_width=True)
        self.show_footer()
    
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
            st.image("assets/uploadfile.png", use_column_width=True)
        self.show_footer()

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
            st.image("assets/uploadfile.png", use_column_width=True)
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
            st.image("assets/uploadfile.png", use_column_width=True)
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
            st.image("assets/uploadfile.png", use_column_width=True)
        self.show_footer()
    
    def data_transformer(self):
        """Handle data encoding."""
        if self.df is not None:
            out = transform.DataTransformer(self.df)
            self.df = out.transformer()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to transform data.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", use_column_width=True)
        self.show_footer()

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
            st.image("assets/uploadfile.png", use_column_width=True)
        self.show_footer()
    
    def find_parameter(self):
        """Find best parameter"""
        if self.df is not None:
            out = hyperparameter.BestParam(self.df)
            out.select_hyper()
            self.save_modified_df()
        else:
            st.markdown(
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to find best parameters.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", use_column_width=True)
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
            st.image("assets/uploadfile.png", use_column_width=True)
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
            st.image("assets/uploadfile.png", use_column_width=True)
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
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px; '>First Upload a DataSet</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", use_column_width=True)

        self.show_footer()

    def run(self):
        """Run the Predicta application."""
        self.load_modified_df()
        self.show_hero_image()

        custom_css = """
        .my-message {
            text-align: center;
            margin: 20px auto;
            font-size: 17px;
            font-weight: bold;
            color: #333;
            background-color: #f8f8f8;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #ddd;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            """

        # Inject custom CSS into Streamlit once at the beginning
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

        # Display styled markdown message using custom CSS class
        st.markdown(
            "<div class='my-message'>Please ensure uploaded datasets are cleared before exiting the application for security.</div>",
            unsafe_allow_html=True,
        )

        st.divider()

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
