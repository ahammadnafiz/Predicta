import streamlit as st
from code_editor import code_editor
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

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
        self.demo_sample_python_code = '''# Your dataframe is available as 'df'
# Start writing code to analyze or manipulate the data
print(df.head())
print(df.describe())

# Example: Create a simple plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
df.plot(kind='scatter', x=df.columns[0], y=df.columns[1])
plt.title('Scatter plot of first two columns')
plt.tight_layout()
st.pyplot(plt)
'''
        self.mode_list = ["python"]

    def run_code_editor(self, df):
        """ Run the Predicta Code Editor in a Streamlit app.
        """
        st.markdown("<h2 style='text-align: center; font-size: 20px;'>Dataset Preview</h2>", unsafe_allow_html=True)
        st.dataframe(df.head(), width=800)
        st.markdown("<div style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 30px;'>Predicta Code Editor</div>", unsafe_allow_html=True)
        st.divider()

        editor_choice = st.sidebar.selectbox("Choose Editor", ["Code Editor", "Jupyter"])
        
        if editor_choice == "Jupyter":
            url = "https://jupyter.org/try#jupyterlab"
            self.display_jupyter_iframe(url)
        else:
            self.display_code_editor(df)

    def display_jupyter_iframe(self, url):
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Embedded Website</title>
            <style>
                body, html {{
                    height: 100%;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background-color: #f5f5f5;
                }}
                .iframe-container {{
                    width: 100%;
                    max-width: 1200px;
                    height: 80vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    overflow: hidden;
                }}
                iframe {{
                    width: 100%;
                    height: 100%;
                    border: none;
                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                    border-radius: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="iframe-container">
                <iframe src="{url}"></iframe>
            </div>
        </body>
        </html>
        """
        st.markdown(html_code, unsafe_allow_html=True)

    def display_code_editor(self, df):
        height = self.get_editor_height()
        language = "python"
        
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
        if response_dict['type'] == "submit" and response_dict['text']:
            self.execute_code(response_dict, df)
        st.markdown('</div>', unsafe_allow_html=True)

    def get_editor_height(self):
        height_type = st.sidebar.selectbox("Height Format:", ["CSS", "Max Lines", "Min-Max Lines"], index=0)
        if height_type == "CSS":
            return st.sidebar.text_input("Height (CSS):", "600px")
        elif height_type == "Max Lines":
            return st.slider("Max Lines:", 1, 40, 22)
        elif height_type == "Min-Max Lines":
            return st.slider("Min-Max Lines:", 1, 40, (19, 22))

    def execute_code(self, response_dict, df):
        st.write('Output')
        buffer = io.StringIO()
        sys.stdout = buffer
        try:
            exec(response_dict['text'], {'df': df, 'st': st, 'plt': plt, 'np': np, 'pd': pd, 'go': go, 'sns': sns, 'zscore': zscore})
        except Exception as e:
            st.error(f"Error executing code: {e}")
        finally:
            sys.stdout = sys.__stdout__
            output = buffer.getvalue()
            if output:
                st.code(output, language=response_dict['lang'])
            buffer.close()