"""
Theme Module

This module provides styling and theming functionality for the Predicta application.
"""

import streamlit as st
from ...core.logging_config import get_logger

logger = get_logger(__name__)


class Theme:
    """
    Theme management class for Predicta application.
    
    Provides dark theme styling and UI customization for Streamlit components.
    """
    
    def __init__(self):
        """Initialize Theme instance."""
        logger.debug("Theme instance initialized")
    
    def init_styling(self):
        """Initialize custom styling for the application with dark theme."""
        try:
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
                    box-shadow: var(--card-shadow);
                }
                
                /* Sidebar Styling */
                .css-1d391kg {
                    background-color: var(--secondary-bg);
                }
                
                /* Button Styling */
                .stButton > button {
                    background: linear-gradient(90deg, #03a3fd 0%, #0288d1 100%);
                    color: var(--text-color);
                    border: none;
                    border-radius: 8px;
                    padding: 0.5rem 1rem;
                    font-weight: 600;
                    transition: transform 0.2s ease;
                }
                
                .stButton > button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(3, 163, 253, 0.3);
                }
                
                /* Input Styling */
                .stTextInput > div > div > input {
                    background-color: var(--secondary-bg);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 8px;
                    color: var(--text-color);
                }
                
                /* Selectbox Styling */
                .stSelectbox > div > div > select {
                    background-color: var(--secondary-bg);
                    color: var(--text-color);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                
                /* Metrics Styling */
                .metric-container {
                    background: var(--secondary-bg);
                    padding: 1rem;
                    border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    margin: 0.5rem 0;
                }
                
                /* Success/Info/Warning/Error Styling */
                .stSuccess {
                    background-color: rgba(72, 187, 120, 0.1);
                    border-left: 4px solid #48bb78;
                }
                
                .stInfo {
                    background-color: rgba(3, 163, 253, 0.1);
                    border-left: 4px solid #03a3fd;
                }
                
                .stWarning {
                    background-color: rgba(237, 137, 54, 0.1);
                    border-left: 4px solid #ed8936;
                }
                
                .stError {
                    background-color: rgba(245, 101, 101, 0.1);
                    border-left: 4px solid #f56565;
                }
                
                /* DataFrame Styling */
                .dataframe {
                    background-color: var(--secondary-bg);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                }
                
                /* Plotly Chart Background */
                .js-plotly-plot {
                    background-color: var(--secondary-bg);
                    border-radius: 8px;
                }
                
                /* Custom Message Styling */
                .my-message {
                    text-align: center;
                    margin-top: 3rem;
                    font-family: 'sans-serif';
                    font-size: 15px;
                    color: #ACF39D;
                    background: var(--secondary-bg);
                    padding: 1.5rem;
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                }
                
                /* Code Block Styling */
                .stCodeBlock {
                    background-color: #2d2d2d;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                }
                
                /* Tab Styling */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 8px;
                }
                
                .stTabs [data-baseweb="tab"] {
                    background-color: var(--secondary-bg);
                    border-radius: 8px;
                    color: var(--text-color);
                }
                
                .stTabs [aria-selected="true"] {
                    background: linear-gradient(90deg, #03a3fd 0%, #0288d1 100%);
                }
                
                /* Expander Styling */
                .streamlit-expanderHeader {
                    background-color: var(--secondary-bg);
                    border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }
                
                /* File Uploader Styling */
                .stFileUploader {
                    background-color: var(--secondary-bg);
                    border: 2px dashed rgba(255, 255, 255, 0.3);
                    border-radius: 8px;
                    padding: 2rem;
                    text-align: center;
                }
                
                /* Progress Bar Styling */
                .stProgress > div > div > div > div {
                    background: linear-gradient(90deg, #03a3fd 0%, #0288d1 100%);
                }
                
                /* Footer Styling */
                .footer {
                    background: var(--secondary-bg);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 2rem;
                    margin: 3rem 0 2rem 0;
                    text-align: center;
                    box-shadow: var(--card-shadow);
                }
                
                .footer p {
                    color: var(--text-color);
                    margin: 0 0 1rem 0;
                    font-size: 16px;
                    font-weight: 500;
                }
                
                .footer .footer-links {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 1.5rem;
                    flex-wrap: wrap;
                    margin-top: 1rem;
                }
                
                .footer .footer-links a {
                    color: var(--primary-color);
                    text-decoration: none;
                    font-weight: 600;
                    padding: 0.5rem 1rem;
                    border-radius: 6px;
                    transition: all 0.3s ease;
                    border: 1px solid rgba(3, 163, 253, 0.3);
                }
                
                .footer .footer-links a:hover {
                    background: rgba(3, 163, 253, 0.1);
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(3, 163, 253, 0.2);
                }
                
                .footer .separator {
                    color: rgba(255, 255, 255, 0.3);
                    font-weight: 300;
                }
                </style>
            """, unsafe_allow_html=True)
            
            logger.info("Theme styling applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying theme styling: {str(e)}")
            st.error(f"Error applying theme: {str(e)}")
    
    def display_styled_message(self, message_text):
        """
        Display a styled message with custom CSS.
        
        Args:
            message_text (str): The message to display
        """
        try:
            st.markdown(
                f"<div class='my-message'>{message_text}</div>",
                unsafe_allow_html=True,
            )
            logger.debug(f"Styled message displayed: {message_text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error displaying styled message: {str(e)}")
            st.error(f"Error displaying message: {str(e)}")
    
    def create_metric_card(self, title, value, delta=None):
        """
        Create a styled metric card.
        
        Args:
            title (str): The metric title
            value (str): The metric value
            delta (str, optional): The metric delta value
        """
        try:
            delta_html = f"<small style='color: #ACF39D;'>{delta}</small>" if delta else ""
            
            st.markdown(f"""
                <div class='metric-container'>
                    <h4 style='margin: 0; color: var(--text-color);'>{title}</h4>
                    <h2 style='margin: 0.5rem 0; color: var(--primary-color);'>{value}</h2>
                    {delta_html}
                </div>
            """, unsafe_allow_html=True)
            
            logger.debug(f"Metric card created: {title}")
            
        except Exception as e:
            logger.error(f"Error creating metric card: {str(e)}")
            st.error(f"Error creating metric card: {str(e)}")
    
    def show_footer(self):
        """Display the modern, centered footer."""
        try:
            st.markdown("""
                <div class='footer'>
                    <p>© 2025 Predicta by Ahammad Nafiz</p>
                    <div class='footer-links'>
                        <a href='https://github.com/ahammadnafiz/Predicta' target='_blank'>
                            GitHub
                        </a>
                        <span class='separator'>•</span>
                        <a href='https://twitter.com/ahammadnafi_z' target='_blank'>
                            Twitter
                        </a>
                        <span class='separator'>•</span>
                        <a href='https://github.com/ahammadnafiz/Predicta/blob/main/LICENSE' target='_blank'>
                            License
                        </a>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            logger.debug("Footer displayed")
            
        except Exception as e:
            logger.error(f"Error displaying footer: {str(e)}")
            st.error(f"Error displaying footer: {str(e)}")
    
    def contributor_info(self):
        """Display contributor information in sidebar."""
        try:
            st.sidebar.markdown("""
                <div class='stCard'>
                    <h4 style='color: #03a3fd;'>👨‍💻 Developer</h4>
                    <div style='display: flex; align-items: center; gap: 1rem;'>
                        <img src='https://avatars.githubusercontent.com/u/86776685?s=400&u=82112040d4a196f3d796c1aa4e7112d403c19450&v=4' 
                             style='width: 60px; height: 60px; border-radius: 50%;'>
                        <div>
                            <h5 style='margin: 0; color: #FAFAFA;'>Ahammad Nafiz</h5>
                            <p style='margin: 0; color: #888;'>Python | ML | AI</p>
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
            
            logger.debug("Contributor info displayed")
            
        except Exception as e:
            logger.error(f"Error displaying contributor info: {str(e)}")
            st.error(f"Error displaying contributor info: {str(e)}")


def create_theme():
    """
    Factory function to create Theme instance.
    
    Returns:
        Theme: New Theme instance
    """
    try:
        return Theme()
    except Exception as e:
        logger.error(f"Error creating Theme instance: {str(e)}")
        raise


# Legacy function for backward compatibility
def init_styling():
    """Legacy function for backward compatibility."""
    theme = create_theme()
    theme.init_styling()


def display_styled_message(message_text):
    """Legacy function for backward compatibility."""
    theme = create_theme()
    theme.display_styled_message(message_text)


def show_footer():
    """Legacy function for backward compatibility."""
    theme = create_theme()
    theme.show_footer()


def contributor_info():
    """Legacy function for backward compatibility."""
    theme = create_theme()
    theme.contributor_info()
