"""
FastAPI app for Vercel deployment.
This file exports the FastAPI app from server.api.app.
"""

import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from main import app

# Export the app for Vercel
app = app