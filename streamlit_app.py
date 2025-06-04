"""
Streamlit Cloud entry point for Predicta application.
This file is used when deploying to Streamlit Cloud.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path for proper imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the main application
from predicta.app.main import main

if __name__ == "__main__":
    main()
