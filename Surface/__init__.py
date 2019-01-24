"""
===========================================================
Surface generation and manipulation (slippy.surface)
===========================================================
.. currentmodule:: slippy.surface

This module contains functions and classes for reading surfaces from file, 
manipulating, generating and analysing surfaces.

The Surface class
=================

Each of the generation classes are subclasses of the main surface class this
class contains all the functionallity for analysing and displaying surfaces.

This class can be used with experimentally measured surfaces and contains 
functionallity for reading common file types including .csv, .txt and .al3d

.. autosummary::
   :toctree: generated/
   
   Surface


Generation classes
==================

Several generation calsses exist to help generate a wide variety of analytical 
or random surfaces.

.. autosummary::
   :toctree: generated/
   
   FlatSurface         -- Flat or sloping surface.
   RoundSurface        -- Round surfaces
   PyramidSurface      -- Square based pyramid surfaces
   RandomSurface       -- Surfaces based on transformations of random sequences
   DiscFreqSurface     -- Surfaces containing only specific frequency componenets
   ProbFreqSurface     -- Surfaces containing stocastic frequency components
   HurstFractalSurface -- A Hurst Fractal Surface

Functions
=========

.. autosummary::
   :toctree: generated/

   surface_like        -- Generate a random surface which is similar to the supplied surface
   alicona_read        -- Read alicona data files
"""


import numpy as np
import warnings

from .ACF import ACF
from .Surface import *
from .Geometric import *
from .Random import *
from .FFTBased import *
from .alicona import *

__all__ = [s for s in dir() if not s.startswith("_")]