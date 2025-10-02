#!/usr/bin/env python3
"""
Launch script for LIDAR Editor application
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

# Set environment variables for Qt
os.environ['QT_QPA_PLATFORM'] = 'xcb'

if __name__ == "__main__":
    from main_gui import main
    main()

