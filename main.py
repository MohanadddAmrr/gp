#!/usr/bin/env python3
"""
TactiVision Pro - Main Entry Point
====================================

Unified entry point for the TactiVision Pro football analytics platform.
Provides command-line interface, setup wizard, and auto-detection of features.

Usage:
    python main.py                    # Launch interactive menu
    python main.py --mode dashboard   # Launch dashboard
    python main.py --mode process     # Process video
    python main.py --mode test        # Run tests
    python main.py --help             # Show all options

Author: TactiVision Pro Team
Version: 2.0.0
"""

import sys
import os
import argparse
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tactivision.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
VERSION = "2.0.0"
APP_NAME = "TactiVision Pro"
CONFIG_PATH = Path("config.yaml")
REQUIREMENTS_PATH = Path("requirements.txt")

# ASCII Art Logo
LOGO = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ████████╗ █████╗  ██████╗████████╗██╗██╗   ██╗██╗███████╗██╗   ║
║   ╚══██╔══╝██╔══██╗██╔════╝╚══██╔══╝██║██║   ██║██║██╔════╝██║   ║
║      ██║   ███████║██║        ██║   ██║██║   ██║██║█████╗  ██║   ║
║      ██║   ██╔══██║██║        ██║   ██║╚██╗ ██╔╝██║██╔══╝  ██║   ║
║      ██║   ██║  ██║╚██████╗   ██║   ██║ ╚████╔╝ ██║██║     ██║   ║
║      ╚═╝   ╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═══╝  ╚═╝╚═╝     ╚═╝   ║
║                                                                  ║
║              Professional Football Analytics Platform            ║
║                          Version 2.0.0                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


class Colors:
    """Terminal colors for output formatting."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(text: str, color: str = Colors.BLUE):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.ENDC}")


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print_colored(f"  {text}", Colors.HEADER + Colors.BOLD)
    print("=" * 70 + "\n")


def print_success(text: str):
    """Print success message."""
    print_colored(f"✓ {text}", Colors.GREEN)


def print_warning(text: str):
    """Print warning message."""
    print_colored(f"⚠ {text}", Colors.WARNING)


def print_error(text: str):
    """Print error message."""
    print_colored(f"✗ {text}", Colors.FAIL)


def print_info(text: str):
    """Print info message."""
    print_colored(f"ℹ {text}", Colors.CYAN)


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are installed."""
    dependencies = {
        'ultralytics': False,
        'opencv': False,
        'numpy': False,
        'pandas': False,
        'streamlit': False,
        'plotly': False,
        'sqlalchemy': False,
        'torch': False,
        'yaml': False,
    }
    
    try:
        import ultralytics
        dependencies['ultralytics'] = True
    except ImportError:
        pass
    
    try:
        import cv2
        dependencies['opencv'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    try:
        import pandas
        dependencies['pandas'] = True
    except ImportError:
        pass
    
    try:
        import streamlit
        dependencies['streamlit'] = True
    except ImportError:
        pass
    
    try:
        import plotly
        dependencies['plotly'] = True
    except ImportError:
        pass
    
    try:
        import sqlalchemy
        dependencies['sqlalchemy'] = True
    except ImportError:
        pass
    
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        import yaml
        dependencies['yaml'] = True
    except ImportError:
        pass
    
    return dependencies


def install_dependencies():
    """Install required dependencies."""
    print_header("Installing Dependencies")
    
    if not REQUIREMENTS_PATH.exists():
        print_error("requirements.txt not found!")
        return False
    
    try:
        print_info("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print_success("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def check_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"GPU detected: {gpu_name}")
            return True
        else:
            print_warning("No GPU detected. Processing will use CPU (slower).")
            return False
    except ImportError:
        print_warning("PyTorch not installed. Cannot check GPU.")
        return False


def load_config() -> Optional[Dict[str, Any]]:
    """Load configuration from config.yaml."""
    if not CONFIG_PATH.exists():
        print_warning("config.yaml not found. Using default settings.")
        return None
    
    try:
        import yaml
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        return None


def setup_wizard():
    """Run first-time setup wizard."""
    print_header("Welcome to TactiVision Pro Setup")
    
    print("This wizard will help you set up TactiVision Pro for the first time.\n")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    print_info("Checking dependencies...")
    deps = check_dependencies()
    
    missing = [name for name, installed in deps.items() if not installed]
    installed = [name for name, installed in deps.items() if installed]
    
    if installed:
        print_success(f"Found: {', '.join(installed)}")
    
    if missing:
        print_warning(f"Missing: {', '.join(missing)}")
        response = input("\nWould you like to install missing dependencies? (y/n): ")
        if response.lower() == 'y':
            if not install_dependencies():
                return False
    
    # Check GPU
    print_info("Checking GPU availability...")
    has_gpu = check_gpu()
    
    # Create necessary directories
    print_info("Creating directories...")
    directories = [
        "input_videos",
        "outputs",
        "models",
        "logs",
        "backups",
        "demo/demo_outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {directory}")
    
    # Download YOLO model if not present
    model_path = Path("yolov8n.pt")
    if not model_path.exists():
        print_info("Downloading YOLO model...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            print_success("YOLO model downloaded successfully!")
        except Exception as e:
            print_warning(f"Could not download YOLO model automatically: {e}")
            print_info("The model will be downloaded on first use.")
    
    # Initialize database
    print_info("Initializing database...")
    try:
        from services.database_manager import DatabaseManager
        db = DatabaseManager()
        db.initialize_database()
        print_success("Database initialized!")
    except Exception as e:
        print_warning(f"Could not initialize database: {e}")
    
    print_header("Setup Complete!")
    print("TactiVision Pro is ready to use.\n")
    print("Quick start:")
    print("  1. Place your match videos in the 'input_videos' folder")
    print("  2. Run: python main.py --mode dashboard")
    print("  3. Or process a video: python main.py --mode process --video input.mp4")
    
    return True


def launch_dashboard(config: Optional[Dict] = None):
    """Launch the Streamlit dashboard."""
    print_header("Launching Dashboard")
    
    dashboard_path = Path("demo/dashboard_final.py")
    if not dashboard_path.exists():
        # Fallback to basic dashboard
        dashboard_path = Path("demo/dashboard.py")
        if not dashboard_path.exists():
            print_error("Dashboard not found!")
            return False
    
    port = 8501
    if config and 'dashboard' in config:
        port = config['dashboard'].get('port', 8501)
    
    print_info(f"Starting dashboard on port {port}...")
    print_info("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port", str(port),
            "--server.headless", "true"
        ])
        return True
    except KeyboardInterrupt:
        print_info("\nDashboard stopped.")
        return True
    except Exception as e:
        print_error(f"Failed to launch dashboard: {e}")
        return False


def process_video(video_path: Path, model, db_manager: DatabaseManager = None,
                 enable_wearable: bool = False, enable_broadcast: bool = False,
                 enable_export: bool = False, enable_streaming: bool = False,
                 enable_social: bool = False, device: str = 'cpu'):

    
    if not Path(video_path).exists():
        print_error(f"Video file not found: {video_path}")
        return False
    
    print_info(f"Processing: {video_path}")
    
    if roster_path:
        print_info(f"Using roster: {roster_path}")
    
    if output_dir:
        print_info(f"Output directory: {output_dir}")
    
    try:
        # Import processing modules
        from demo.run_demo import process_video as run_processing
        
        # Run processing
        result = run_processing(video_path, roster_path, output_dir)
        
        if result:
            print_success("Video processing completed!")
            return True
        else:
            print_error("Video processing failed!")
            return False
            
    except Exception as e:
        print_error(f"Processing error: {e}")
        logger.exception("Video processing failed")
        return False


def run_tests(test_type: str = "all"):
    """Run the test suite."""
    print_header("Running Tests")
    
    test_files = {
        "all": ["tests/"],
        "ball": ["tests/test_ball_tracker.py"],
        "possession": ["tests/test_possession.py", "tests/test_possession_validation.py"],
        "unit": ["tests/test_suite.py"]
    }
    
    if test_type not in test_files:
        print_error(f"Unknown test type: {test_type}")
        print_info(f"Available types: {', '.join(test_files.keys())}")
        return False
    
    files = test_files[test_type]
    
    try:
        import pytest
        args = ["-v"] + files
        result = pytest.main(args)
        return result == 0
    except ImportError:
        print_error("pytest not installed. Run: pip install pytest")
        return False
    except Exception as e:
        print_error(f"Test execution failed: {e}")
        return False


def export_data(match_id: int, format_type: str, output_path: str):
    """Export match data."""
    print_header("Exporting Data")
    
    try:
        from services.database_manager import DatabaseManager
        db = DatabaseManager()
        
        if format_type == "json":
            export_path = db.export_match_to_json(match_id, output_path)
            print_success(f"Exported to: {export_path}")
            return True
        elif format_type == "csv":
            print_info("CSV export not yet implemented")
            return False
        else:
            print_error(f"Unknown format: {format_type}")
            return False
            
    except Exception as e:
        print_error(f"Export failed: {e}")
        return False


def show_status():
    """Show system status."""
    print_header("System Status")
    
    # Python version
    print_info(f"Python: {sys.version}")
    
    # Dependencies
    deps = check_dependencies()
    print("\nDependencies:")
    for name, installed in deps.items():
        status = "✓" if installed else "✗"
        color = Colors.GREEN if installed else Colors.FAIL
        print(f"  {color}{status} {name}{Colors.ENDC}")
    
    # GPU
    print("\nGPU:")
    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"  Available: {torch.cuda.get_device_name(0)}")
            print_info(f"  CUDA Version: {torch.version.cuda}")
        else:
            print_warning("  Not available (CPU only)")
    except ImportError:
        print_error("  PyTorch not installed")
    
    # Database
    print("\nDatabase:")
    try:
        from services.database_manager import DatabaseManager
        db = DatabaseManager()
        matches = db.get_all_matches()
        print_success(f"  Connected")
        print_info(f"  Matches: {len(matches)}")
    except Exception as e:
        print_error(f"  Error: {e}")
    
    # Disk space
    print("\nDisk Space:")
    import shutil
    total, used, free = shutil.disk_usage("/")
    print_info(f"  Free: {free // (2**30)} GB")
    print_info(f"  Used: {used // (2**30)} GB")
    print_info(f"  Total: {total // (2**30)} GB")


def interactive_menu():
    """Show interactive menu."""
    while True:
        print(LOGO)
        print("\nMain Menu:")
        print("  1. 🚀 Launch Dashboard")
        print("  2. 🎬 Process Video")
        print("  3. 🧪 Run Tests")
        print("  4. 📊 System Status")
        print("  5. ⚙️  Setup Wizard")
        print("  6. 📤 Export Data")
        print("  7. ❌ Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        config = load_config()
        
        if choice == "1":
            launch_dashboard(config)
        elif choice == "2":
            video_path = input("Enter video path: ").strip()
            roster_path = input("Enter roster path (optional): ").strip()
            roster_path = roster_path if roster_path else None
            process_video(video_path, roster_path, config=config)
        elif choice == "3":
            print("\nTest types: all, ball, possession, unit")
            test_type = input("Enter test type (default: all): ").strip() or "all"
            run_tests(test_type)
        elif choice == "4":
            show_status()
        elif choice == "5":
            setup_wizard()
        elif choice == "6":
            match_id = int(input("Enter match ID: ").strip())
            format_type = input("Enter format (json/csv): ").strip()
            output_path = input("Enter output path: ").strip()
            export_data(match_id, format_type, output_path)
        elif choice == "7":
            print_info("Goodbye!")
            break
        else:
            print_error("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")
        print("\n" * 3)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='main.py',
        description=f'{APP_NAME} v{VERSION} - Professional Football Analytics Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive menu
  python main.py

  # Launch dashboard
  python main.py --mode dashboard

  # Process a video
  python main.py --mode process --video match.mp4 --roster team.json

  # Run tests
  python main.py --mode test --test-type all

  # Check system status
  python main.py --mode status

  # Run setup wizard
  python main.py --mode setup
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {VERSION}'
    )
    
    parser.add_argument(
        '--mode',
        choices=['menu', 'dashboard', 'process', 'test', 'status', 'setup', 'export'],
        default='menu',
        help='Operation mode (default: menu)'
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to video file (for process mode)'
    )
    
    parser.add_argument(
        '--roster',
        type=str,
        help='Path to roster JSON file (for process mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (for process mode)'
    )
    
    parser.add_argument(
        '--test-type',
        choices=['all', 'ball', 'possession', 'unit'],
        default='all',
        help='Type of tests to run (default: all)'
    )
    
    parser.add_argument(
        '--match-id',
        type=int,
        help='Match ID (for export mode)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'pdf', 'xlsx'],
        default='json',
        help='Export format (default: json)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set debug mode
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print_info("Debug mode enabled")
    
    # Load configuration
    global CONFIG_PATH
    CONFIG_PATH = Path(args.config)
    config = load_config()
    
    # Execute based on mode
    if args.mode == 'menu':
        interactive_menu()
    elif args.mode == 'dashboard':
        launch_dashboard(config)
    elif args.mode == 'process':
        if not args.video:
            print_error("--video is required for process mode")
            sys.exit(1)
        success = process_video(args.video, args.roster, args.output, config)
        sys.exit(0 if success else 1)
    elif args.mode == 'test':
        success = run_tests(args.test_type)
        sys.exit(0 if success else 1)
    elif args.mode == 'status':
        show_status()
    elif args.mode == 'setup':
        setup_wizard()
    elif args.mode == 'export':
        if args.match_id is None:
            print_error("--match-id is required for export mode")
            sys.exit(1)
        success = export_data(args.match_id, args.format, f"export_{args.match_id}.{args.format}")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
