"""
Assets Utility Module

This module provides utilities for managing asset paths in the Predicta application.
"""

import os
from pathlib import Path
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class AssetManager:
    """
    Asset management class for handling asset paths and resources.
    
    Provides centralized asset path management and resource loading.
    """
    
    def __init__(self):
        """Initialize AssetManager with base asset directory."""
        self.base_dir = Path(__file__).parent.parent / "assets"
        logger.debug(f"AssetManager initialized with base directory: {self.base_dir}")
    
    def get_asset_path(self, asset_name):
        """
        Get the full path to an asset file.
        
        Args:
            asset_name (str): Name of the asset file
            
        Returns:
            str: Full path to the asset file
        """
        try:
            asset_path = self.base_dir / asset_name
            if asset_path.exists():
                logger.debug(f"Asset found: {asset_path}")
                return str(asset_path)
            else:
                # Fallback to old assets directory for backward compatibility
                fallback_path = Path.cwd() / "assets" / asset_name
                if fallback_path.exists():
                    logger.warning(f"Using fallback asset path: {fallback_path}")
                    return str(fallback_path)
                else:
                    logger.error(f"Asset not found: {asset_name}")
                    return str(asset_path)  # Return expected path even if it doesn't exist
                    
        except Exception as e:
            logger.error(f"Error getting asset path for '{asset_name}': {str(e)}")
            return str(self.base_dir / asset_name)
    
    def get_hero_image(self):
        """Get path to the hero image."""
        return self.get_asset_path("Hero.png")
    
    def get_upload_image(self):
        """Get path to the upload file image."""
        return self.get_asset_path("uploadfile.png")
    
    def get_prediction_app_image(self):
        """Get path to the prediction app image."""
        return self.get_asset_path("Prediction app.png")
    
    def get_banner_image(self):
        """Get path to the banner image."""
        return self.get_asset_path("Predicta_banner.png")
    
    def get_profile_image(self):
        """Get path to the profile picture."""
        return self.get_asset_path("Profile Picture.png")
    
    def list_assets(self):
        """
        List all available assets.
        
        Returns:
            list: List of asset filenames
        """
        try:
            if self.base_dir.exists():
                assets = [f.name for f in self.base_dir.iterdir() if f.is_file()]
                logger.info(f"Found {len(assets)} assets")
                return assets
            else:
                logger.warning("Assets directory not found")
                return []
                
        except Exception as e:
            logger.error(f"Error listing assets: {str(e)}")
            return []


# Global asset manager instance
_asset_manager = None


def get_asset_manager():
    """
    Get the global AssetManager instance.
    
    Returns:
        AssetManager: Global AssetManager instance
    """
    global _asset_manager
    if _asset_manager is None:
        _asset_manager = AssetManager()
    return _asset_manager


def get_asset_path(asset_name):
    """
    Convenience function to get asset path.
    
    Args:
        asset_name (str): Name of the asset file
        
    Returns:
        str: Full path to the asset file
    """
    return get_asset_manager().get_asset_path(asset_name)


def get_hero_image():
    """Convenience function to get hero image path."""
    return get_asset_manager().get_hero_image()


def get_upload_image():
    """Convenience function to get upload image path."""
    return get_asset_manager().get_upload_image()


def get_prediction_app_image():
    """Convenience function to get prediction app image path."""
    return get_asset_manager().get_prediction_app_image()


def get_banner_image():
    """Convenience function to get banner image path."""
    return get_asset_manager().get_banner_image()


def get_profile_image():
    """Convenience function to get profile image path."""
    return get_asset_manager().get_profile_image()
