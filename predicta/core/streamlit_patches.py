"""Streamlit patches and utilities for Predicta."""

import asyncio
import logging
from typing import List, Any
from streamlit.watcher import local_sources_watcher


def apply_streamlit_patches() -> None:
    """Apply patches to handle PyTorch-related issues in Streamlit."""
    
    # Patch module path extraction
    original_get_module_paths = local_sources_watcher.get_module_paths
    
    def patched_get_module_paths(module: Any) -> List[str]:
        """Patched version that handles PyTorch modules gracefully."""
        try:
            return original_get_module_paths(module)
        except RuntimeError as e:
            if "torch._C._get_custom_class_python_wrapper" in str(e):
                return []
            raise e
    
    local_sources_watcher.get_module_paths = patched_get_module_paths
    
    # Patch asyncio event loop handling
    original_get_running_loop = asyncio.get_running_loop
    
    def patched_get_running_loop() -> asyncio.AbstractEventLoop:
        """Patched version that creates a new loop if none exists."""
        try:
            return original_get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    asyncio.get_running_loop = patched_get_running_loop


# Apply patches when module is imported
apply_streamlit_patches()
