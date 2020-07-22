"""Top-level package for SlipPY."""

__author__ = """Michael Watson"""
__email__ = 'mike.watson@sheffield.ac.uk'
__version__ = '0.1.0'

try:
    import cupy
    CUDA = True
except ImportError:
    CUDA = False

