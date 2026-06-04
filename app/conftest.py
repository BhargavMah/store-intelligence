"""
conftest.py — pytest configuration for the store-intelligence app tests.

Sets up sys.path so that both app/ modules and the parent schema.py are importable
without needing to install the package.
"""
import os
import sys

# Add app/ directory to path (for database, ingestion, metrics, etc.)
APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(APP_DIR)

sys.path.insert(0, APP_DIR)
sys.path.insert(0, ROOT_DIR)
