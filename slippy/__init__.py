import multiprocessing
import os
import numpy
"""Top-level package for SlipPY."""

__author__ = """Michael Watson"""
__email__ = 'mike.watson@sheffield.ac.uk'
__version__ = '0.6.0'

try:
    import cupy  # noqa: F401
    CUDA = True
    xp = cupy
except ImportError:
    CUDA = False
    xp = numpy


def asnumpy(obj):
    # decided by the type of the object rather than the CUDA flag: the flag can be toggled
    # mid-session (OverRideCuda) while arrays made under the other setting are still around
    if 'cupy' in type(obj).__module__:
        return cupy.asnumpy(obj)
    return numpy.asarray(obj)


CUBIC_EPS = 1e-7
CORES = multiprocessing.cpu_count()
OUTPUT_DIR = os.getcwd()
ERROR_IF_MISSING_MODEL = True
ERROR_IF_MISSING_SUB_MODEL = True
ERROR_IN_DATA_CHECK = True
dtype = 'float64'
material_names = []


class OverRideCuda:
    def __init__(self):
        self.cuda = CUDA

    def __enter__(self):
        global CUDA
        CUDA = False

    def __exit__(self, err_type, value, traceback):
        global CUDA
        CUDA = self.cuda
