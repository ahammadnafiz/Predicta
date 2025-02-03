import streamlit as st

def init_styling():
        """Initialize custom styling for the application with dark theme."""
        st.markdown("""
            <style>
            /* Base Theme Colors */
            :root {
                --primary-color: #03a3fd;
                --background-color: #0e0e10;
                --secondary-bg: #1e1e1e;
                --text-color: #FAFAFA;
                --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            
            /* Global Styles */
            .stApp {
                background-color: var(--background-color);
                color: var(--text-color);
                font-family: 'sans-serif';
            }
            
            /* Modern Card Styling */
            .stCard {
                border-radius: 12px;
                box-shadow: var(--card-shadow);
                background: var(--secondary-bg);
                padding: 1.5rem;
                margin: 1rem 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            /* Header Styling */
            .main-header {
                background: linear-gradient(90deg, #03a3fd 0%, #0288d1 100%);
                color: var(--text-color);
                padding: 2rem;
                border-radius: 12px;
                margin-bottom: 2rem;
                text-align: center;
            }
            
            /* Navigation Menu Styling */
            .nav-item {
                padding: 0.75rem 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                transition: all 0.3s ease;
                background: var(--secondary-bg);
            }
            
            .nav-item:hover {
                background: rgba(3, 163, 253, 0.1);
            }
            
            /* Modern Button Styling */
            .stButton > button {
                border-radius: 8px;
                padding: 0.5rem 1rem;
                font-weight: 500;
                background-color: var(--primary-color) !important;
                color: var(--text-color) !important;
                border: none !important;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                background-color: #0288d1 !important;
                transform: translateY(-1px);
            }
            
            /* Footer Styling */
            .footer {
                background: var(--secondary-bg);
                padding: 2rem;
                border-radius: 12px;
                text-align: center;
                margin-top: 3rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .footer a {
                color: var(--primary-color);
                text-decoration: none;
                margin: 0 0.5rem;
                transition: color 0.3s ease;
            }
            
            .footer a:hover {
                color: #0288d1;
            }
            
            /* Status Messages */
            .status-message {
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
                text-align: center;
            }
            
            .success-message {
                background: rgba(3, 163, 253, 0.1);
                color: var(--text-color);
                border: 1px solid rgba(3, 163, 253, 0.3);
            }
            
            .warning-message {
                background: rgba(255, 171, 0, 0.1);
                color: var(--text-color);
                border: 1px solid rgba(255, 171, 0, 0.3);
            }

            /* File Uploader Styling */
            .uploadedFile {
                background-color: var(--secondary-bg) !important;
                color: var(--text-color) !important;
                border: 1px solid rgba(3, 163, 253, 0.3) !important;
            }

            /* Sidebar Styling */
            .css-1d391kg {
                background-color: var(--secondary-bg) !important;
            }

            /* Input Fields */
            .stTextInput > div > div {
                background-color: var(--secondary-bg) !important;
                color: var(--text-color) !important;
                border-color: rgba(255, 255, 255, 0.1) !important;
            }

            /* Radio Buttons */
            .stRadio > div {
                background-color: transparent !important;
                color: var(--text-color) !important;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background-color: var(--secondary-bg) !important;
                border-radius: 8px;
            }

            .stTabs [data-baseweb="tab"] {
                color: var(--text-color) !important;
            }

            /* DataFrames */
            .dataframe {
                background-color: var(--secondary-bg) !important;
                color: var(--text-color) !important;
            }

            .dataframe th {
                background-color: rgba(3, 163, 253, 0.1) !important;
                color: var(--text-color) !important;
            }

            /* Charts */
            .stPlot {
                background-color: var(--secondary-bg) !important;
            }
            </style>
        """, unsafe_allow_html=True)

def show_footer():
        """Display the footer."""
        st.markdown("""
            <div class='footer'>
                <p style='color: #888;'>© 2025 Predicta by Ahammad Nafiz</p>
                <div style='margin-top: 1rem;'>
                    <a href='https://github.com/ahammadnafiz' target='_blank'>GitHub</a> •
                    <a href='https://twitter.com/ahammadnafi_z' target='_blank'>Twitter</a> •
                    <a href='https://github.com/ahammadnafiz/Predicta/blob/main/LICENSE' target='_blank'>License</a>
                </div>
            </div>
        """, unsafe_allow_html=True)

def contributor_info():
        st.sidebar.markdown("""
            <div class='stCard'>
                <h4 style='color: #03a3fd;'>👨‍💻 Developer</h4>
                <div style='display: flex; align-items: center; gap: 1rem;'>
                    <img src='https://avatars.githubusercontent.com/u/86776685?s=400&u=82112040d4a196f3d796c1aa4e7112d403c19450&v=4' 
                         style='width: 60px; height: 60px; border-radius: 50%;'>
                    <div>
                        <h5 style='margin: 0; color: #FAFAFA;'>Ahammad Nafiz</h5>
                        <p style='margin: 0; color: #888;'>Curious Learner</p>
                    </div>
                </div>
                <div style='margin-top: 1rem;'>
                    <a href='https://github.com/ahammadnafiz' target='_blank' style='margin-right: 1rem;'>
                        <img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white' height='25'>
                    </a>
                    <a href='https://www.linkedin.com/in/ahammad-nafiz/' target='_blank'>
                        <img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' height='25'>
                    </a>
                </div>
            </div>
        """, unsafe_allow_html=True)