"""CLI module for Predicta."""

import argparse
import sys
import signal
import time
from pathlib import Path
from typing import List, Optional

from predicta.core.config import Config
from predicta.core.logging_config import setup_logging, get_logger


logger = get_logger(__name__)


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point for Predicta."""
    parser = argparse.ArgumentParser(
        description="Predicta - Advanced Data Analysis and Machine Learning Platform",
        prog="predicta"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the Predicta application")
    run_parser.add_argument(
        "--host", 
        default="localhost", 
        help="Host to run the application on"
    )
    run_parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port to run the application on"
    )
    run_parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Run in debug mode"
    )
    run_parser.add_argument(
        "--auto-open", 
        action="store_true", 
        help="Automatically open browser"
    )
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean temporary files")
    clean_parser.add_argument(
        "--all", 
        action="store_true", 
        help="Clean all cached data including models"
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return 0
    
    # Setup logging
    log_level = "DEBUG" if getattr(parsed_args, "debug", False) else "INFO"
    setup_logging(level=log_level)
    
    # Execute command
    try:
        if parsed_args.command == "run":
            return run_app(parsed_args)
        elif parsed_args.command == "version":
            return show_version()
        elif parsed_args.command == "config":
            return show_config()
        elif parsed_args.command == "clean":
            return clean_files(parsed_args)
        else:
            logger.error(f"Unknown command: {parsed_args.command}")
            return 1
    except Exception as e:
        logger.error(f"Error executing command '{parsed_args.command}': {e}")
        return 1


def run_app(args: argparse.Namespace) -> int:
    """Run the Predicta Streamlit application with graceful shutdown."""
    import subprocess
    
    # Path to the main Streamlit app
    app_path = Path(__file__).parent / "app" / "main.py"
    
    if not app_path.exists():
        logger.error(f"Main application file not found: {app_path}")
        return 1
    
    cmd = [
        "streamlit", "run", str(app_path),
        "--server.address", args.host,
        "--server.port", str(args.port),
        "--theme.base", "dark",
        "--server.headless", "true"  # Prevents automatic browser opening
    ]
    
    if args.debug:
        cmd.extend(["--logger.level", "debug"])
    
    # Only auto-open if explicitly requested
    if not args.auto_open:
        cmd.extend(["--browser.gatherUsageStats", "false"])
    
    logger.info(f"Starting Predicta application on {args.host}:{args.port}")
    
    if not args.auto_open:
        logger.info(f"Open your browser and navigate to: http://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop the application")
    
    process = None
    
    def signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("Received shutdown signal, stopping application...")
        if process:
            try:
                process.terminate()
                # Give it time to shut down gracefully
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate gracefully, forcing shutdown...")
                process.kill()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        process = subprocess.Popen(cmd)
        
        # Wait for the process to complete
        while True:
            try:
                return_code = process.poll()
                if return_code is not None:
                    # Process has ended
                    if return_code == 0:
                        logger.info("Application stopped normally")
                    else:
                        logger.error(f"Application exited with code {return_code}")
                    return return_code
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                # This should be handled by the signal handler, but just in case
                signal_handler(signal.SIGINT, None)
                
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        return 1


def show_version() -> int:
    """Show version information."""
    print(f"Predicta {Config.APP_VERSION}")
    return 0


def show_config() -> int:
    """Show configuration information."""
    config = Config.to_dict()
    
    print("Predicta Configuration:")
    print("=" * 50)
    
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    return 0


def clean_files(args: argparse.Namespace) -> int:
    """Clean temporary files."""
    import shutil
    
    dirs_to_clean = [Config.TEMP_DIR]
    
    if args.all:
        dirs_to_clean.extend([Config.LOGS_DIR, Config.MODELS_DIR])
    
    cleaned_count = 0
    
    for directory in dirs_to_clean:
        if directory.exists():
            try:
                shutil.rmtree(directory)
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleaned directory: {directory}")
                cleaned_count += 1
            except Exception as e:
                logger.error(f"Failed to clean directory {directory}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"Successfully cleaned {cleaned_count} directories")
    else:
        logger.info("No directories to clean")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())