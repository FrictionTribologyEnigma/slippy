import multiprocessing
import os
"""Top-level package for SlipPY."""

__author__ = """Michael Watson"""
__email__ = 'mike.watson@sheffield.ac.uk'
__version__ = '0.2.0'

try:
    import cupy  # noqa: F401
    CUDA = True
    asnumpy = cupy.asnumpy
except ImportError:
    CUDA = False
    import numpy
    asnumpy = numpy.asarray

CORES = multiprocessing.cpu_count()
OUTPUT_DIR = os.getcwd()
ERROR_IF_MISSING_MODEL = True
ERROR_IF_MISSING_SUB_MODEL = True
ERROR_IN_DATA_CHECK = True


class OverRideCuda:
    def __init__(self):
        self.cuda = CUDA

    def __enter__(self):
        global CUDA
        CUDA = False

    def __exit__(self, err_type, value, traceback):
        global CUDA
        CUDA = self.cuda
