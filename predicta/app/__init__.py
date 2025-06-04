"""Main Streamlit application for Predicta."""

import streamlit as st

# Apply patches first
from ..core.streamlit_patches import apply_streamlit_patches
apply_streamlit_patches()

from ..core.config import Config
from ..core.logging_config import get_logger
from .predicta_app import PredictaApp

logger = get_logger(__name__)


def main():
    """Main entry point for the Streamlit application."""
    # Configure Streamlit page
    st.set_page_config(**Config.STREAMLIT_CONFIG)
    
    # Initialize and run the app
    app = PredictaApp()
    app.run()


if __name__ == "__main__":
    main()
