import streamlit as st
import pandas as pd
import joblib
from Theme import theme


class PredictionApp:
    def __init__(self):
        self.model = None
        self.preprocessed_data = None
        self.target_column = None

    def show_hero_image(self):
        """Display the hero image."""
        st.image("assets/Prediction app.png")
    
    def show_footer(self):
        """Display the footer."""
        st.markdown("---")
        st.markdown("*copyright@ahammadnafiz*")

        footer_content = """
        <div class="footer">
            Follow us: &nbsp;&nbsp;&nbsp;
            <a href="https://github.com/ahammadnafiz" target="_blank">GitHub</a> 🚀 |
            <a href="https://twitter.com/ahammadnafi_z" target="_blank">Twitter</a> 🐦|
            <a href="https://github.com/ahammadnafiz/Predicta/blob/main/LICENSE" target="_blank">License</a> 📜
        </div>
        """
        st.markdown(footer_content, unsafe_allow_html=True)
    
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
    
    def load_model(self, model_file):
        self.model = joblib.load(model_file)

    def load_preprocessed_data(self, dataset_file):
        self.preprocessed_data = pd.read_csv(dataset_file)

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
        prediction = self.model.predict(input_data)
        return prediction

    def display_prediction_result(self, prediction):
        st.subheader("Prediction Result")
        st.write(prediction)

    def run(self):
        self.show_hero_image()

        st.sidebar.title("Prediction App")
        st.sidebar.markdown("---")
        
        # Sidebar for uploading files
        st.sidebar.subheader("Upload Model")
        model_file = st.sidebar.file_uploader("Upload Trained Model", type=["pkl"])

        dataset_file = st.file_uploader("Upload Preprocessed Dataset (CSV)", type=["csv"])
        
        # st.sidebar.divider()
        
        # self.contributor_info()

        if model_file and dataset_file:
            self.load_model(model_file)
            self.load_preprocessed_data(dataset_file)
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
                "<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to Explore.</div>",
                unsafe_allow_html=True,
            )
            st.image("assets/uploadfile.png", use_column_width=True)
        self.show_footer()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Prediction App",
        page_icon="💫",
        initial_sidebar_state="expanded"
    )

    theme.footer()
    
    app = PredictionApp()
    app.run()

    # Import custom CSS
    st.markdown('<link rel="stylesheet" href="assets/styles.css">', unsafe_allow_html=True)