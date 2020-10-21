import multiprocessing
"""Top-level package for SlipPY."""

__author__ = """Michael Watson"""
__email__ = 'mike.watson@sheffield.ac.uk'
__version__ = '0.1.0'

try:
    import cupy  # noqa: F401
    CUDA = True
except ImportError:
    CUDA = False

CORES = multiprocessing.cpu_count()


class OverRideCuda:
    def __init__(self):
        self.cuda = CUDA

    def __enter__(self):
        global CUDA
        CUDA = False

    def __exit__(self, err_type, value, traceback):
        global CUDA
        CUDA = self.cuda
