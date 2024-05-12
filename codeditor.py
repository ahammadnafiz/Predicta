import streamlit as st
from code_editor import code_editor
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import zscore

st.set_option('deprecation.showPyplotGlobalUse', False)

class PredictaCodeEditor:
    """
    A class to create a Predicta Code Editor in a Streamlit app.
    """

    def __init__(self):
        """
        Initialize the class and load the custom buttons and code editor settings.
        """
        self.load_buttons()
        self.load_code_editor_settings()

    def load_buttons(self):
        """
        Define custom buttons for the code editor.
        """
        self.custom_buttons = [
            {"name": "copy", "feather": "Copy", "hasText": True, "alwaysOn": True, "commands": ["copyAll"], "style": {"top": "0rem", "right": "0.4rem"}},
            {"name": "run", "feather": "Play", "primary": True, "hasText": True, "showWithIcon": True, "commands": ["submit"], "style": {"bottom": "0rem", "right": "0.4rem"}}
        ]

    def load_code_editor_settings(self):
        """
        Define the initial code and mode for the code editor.
        """
        self.demo_sample_python_code = '''# Start Writing Code!'''
        self.mode_list = ["python"]

    def run_code_editor(self, df):
        """ Run the Predicta Code Editor in a Streamlit app.
        """
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>", unsafe_allow_html=True)
        st.dataframe(df, width=800)
        st.markdown("<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 30px;'>Predicta Code Editor.</div>", unsafe_allow_html=True)
        st.divider()

        editor_choice = st.sidebar.selectbox("Choose Editor", ["Code Editor", "Binder"])
        
        if editor_choice == "Binder":
            html_code = """
            <iframe src="https://mybinder.org/?embed=true" height="800" style="width: 100%; border: none;"></iframe>
            """
            st.markdown(html_code, unsafe_allow_html=True)
        else:
            
            height = 800
            language = "python"
            
            height_type = st.sidebar.selectbox("Height Format:", ["CSS", "Max Lines", "Min-Max Lines"], index=0)
            if height_type == "CSS":
                height = st.sidebar.text_input("Height (CSS):", "600px")
            elif height_type == "Max Lines":
                height = st.slider("Max Lines:", 1, 40, 22)
            elif height_type == "Min-Max Lines":
                height = st.slider("Min-Max Lines:", 1, 40, (19, 22))

            st.markdown(
                """
                <style>
                .centered {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100%;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="centered">', unsafe_allow_html=True)
            response_dict = code_editor(self.demo_sample_python_code, height=height, lang=language, buttons=self.custom_buttons)
            if response_dict['type'] == "submit" and len(response_dict['text']) != 0:
                st.write('Output')
                buffer = io.StringIO()
                sys.stdout = buffer
                try:
                    exec(response_dict['text'])
                except Exception as e:
                    st.write(f"Error executing code: {e}")
                finally:
                    sys.stdout = sys.__stdout__
                    output = buffer.getvalue()
                    if output:
                        st.code(output, language=response_dict['lang'])
                    buffer.close()
            st.markdown('</div>', unsafe_allow_html=True)