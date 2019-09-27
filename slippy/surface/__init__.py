"""
===========================================================
Surface generation and manipulation (:mod:`slippy.surface`)
===========================================================

.. currentmodule:: slippy.surface

This package contains functions and classes for reading surfaces from file,
manipulating, generating and analysing surfaces.

The Surface class
=================

Each of the generation classes are subclasses of the main surface class this
class contains all the functionality for analysing and displaying surfaces.

This class can be used with experimentally measured surfaces and contains 
functionality for reading common file types including .csv, .txt and .al3d

.. autosummary::
   :toctree: generated
   
   Surface


Generation classes
==================

Several generation classes exist to help generate a wide variety of analytical
or random surfaces.

.. autosummary::
   :toctree: generated
   
   FlatSurface         -- Flat or sloping surface.
   RoundSurface        -- Round surfaces
   PyramidSurface      -- Square based pyramid surfaces
   RandomSurface       -- Surfaces based on transformations of random sequences
   DiscFreqSurface     -- Surfaces containing specific frequency components
   ProbFreqSurface     -- Surfaces containing stochastic frequency components
   HurstFractalSurface -- A Hurst Fractal Surface

Functions
=========

.. autosummary::
   :toctree: generated

   assurface             -- Make a surface object
   read_surface          -- Read a surface object from file
   alicona_read          -- Read alicona data files
   read_tst_file         -- Read a bruker umt tst file
   roughness             -- Find 2d roughness parameters
   subtract_polynomial   -- fit and subtract a n degree polynomial
   get_mat_vr            -- get the material or void volume ratio for a height
   get_height_of_mat_vr  -- get the height of particular material or void ratio
   get_summit_curvatures -- find curvatures of points on the surface
   find_summits          -- find peaks on the surface
   low_pass_filter       -- low pass FIR filter the surface
   surface_like          -- Generate a random surface 'like' another surface
   
"""


from .ACF_class import ACF
from .Surface_class import *
from .Geometric import *
from .Random import *
from .FFTBased import *
from .alicona import *
from .roughness_funcs import *
from .read_tst_file import *

__all__ = [s for s in dir() if not s.startswith("_")]
