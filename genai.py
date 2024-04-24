import streamlit as st
import pandas as pd
import io

import google.generativeai as genai

genai.configure(api_key="AIzaSyBf21hIHZ-XRnqv754gLpNjX7HuFzmA4zY")
model = genai.GenerativeModel('gemini-pro')

class ChatPredicta:
    def __init__(self, df):
        self.df = df

    def chat_with_predicta(self):
        if self.df is not None:
            st.markdown("<h2 style='text-align: center; font-size: 25px;'>Ask a question about the data</h2>", unsafe_allow_html=True)
            question = st.text_input("", placeholder="Type your question here...", key='query')

            if question:
                # Convert dataframe to CSV format
                csv_data = self.df.to_csv(index=False)
                csv_file = io.StringIO(csv_data)
                final_csv = csv_file.read()

                # Create the user message with the CSV data and the question
                user_message = f"**CSV Data**:\n```\n{final_csv}\n```\n\n**Question**: {question}"

                response = model.generate_content(user_message)

                # Display the generated text as the answer
                st.write("🤖 **Predicta Response:**")
                st.write(response.text)

        

# Streamlit app
def main():
    st.set_page_config(page_title="Chat with Predicta", page_icon=":speech_balloon:")
    st.markdown("<h1 style='text-align: center; font-size: 30px;'>Chat with Predicta</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Options")
    file_uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if file_uploaded is not None:
        df = pd.read_csv(file_uploaded)
        st.dataframe(df)
        predicta = ChatPredicta(df)
        predicta.chat_with_predicta()
    else:
            st.markdown("<p style='text-align: center; margin-top: 20px; margin-bottom: 20px; font-size: 15px;'>Please upload a dataset to Chat With Predicta.</p>", unsafe_allow_html=True)
            st.image('uploadfile.png', use_column_width=True)
if __name__ == "__main__":
    main()
