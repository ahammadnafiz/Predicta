import streamlit as st

def footer():

    custom_css = """
    <style>
        .footer {
            text-align: center;
            font-size: 16px;
            color: #f9f7f3; /* Vibrant Red */
            margin-top: 30px;
        }

        .footer a {
            color: #ade8f4; /* Green */
            text-decoration: none;
            margin: 0 5px;
        }

        .footer a:hover {
            color: #FF7F50; /* Bright Orange */
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

