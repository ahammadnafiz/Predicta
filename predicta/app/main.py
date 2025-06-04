"""Main Streamlit application for Predicta."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

# Apply patches first
from predicta.core.streamlit_patches import apply_streamlit_patches
apply_streamlit_patches()

from predicta.core.config import Config
from predicta.core.logging_config import get_logger
from predicta.app.predicta_app import PredictaApp

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
