"""
===========================================================
Surface generation and manipulation (:mod:`slippy.surface`)
===========================================================

.. currentmodule:: slippy.surface

This package contains functions and classes for reading surfaces from file,
manipulating, generating and analysing surfaces.

The Surface class for experimental surfaces
===========================================

The Surface class contains methods for reading surfaces from files including .mat, .txt, .csv and .al3d files. It also
contains methods for analysing surface rougness and other parameters.

.. autosummary::
   :toctree: generated
   
   Surface


Analytical Surfaces
===================

Surfaces which can be described by a mathematical function, these can be combined, rotated or shifted with no loss in
resolution.

.. autosummary::
   :toctree: generated
   
   FlatSurface         -- Flat or sloping surface.
   RoundSurface        -- Round surfaces
   PyramidSurface      -- Square based pyramid surfaces
   RandomSurface       -- Surfaces based on transformations of random sequences
   DiscFreqSurface     -- Surfaces containing specific frequency components
   HurstFractalSurface -- A Hurst Fractal Surface

Random Surfaces
===============

Surfaces based on transformations of random surfaces or probabilistic discriptions of the FFT.

.. autosummary::
   :toctree: generated

   RandomSurface     -- Surfaces from transformations of random sequences
   ProbFreqSurface   -- Surfaces based on a probablistic description of the FFT
   surface_like      -- random surfaces with similar properties to the input surface

Functions
=========

Functional interfaces for common tasks, these are all aliased by class methods, apart from surface_like.

.. autosummary::
   :toctree: generated

   assurface              -- Make a surface object
   read_surface           -- Read a surface object from file
   alicona_read           -- Read alicona data files
   read_tst_file          -- Read a bruker umt tst file
   roughness              -- Find 2d roughness parameters
   surface_like           -- Generate a random surface 'like' another surface
   find_summits           -- Get summit loactions
   get_height_of_mat_vr   -- Find the height of a specified material or void volume ratio
   get_mat_vr             -- Find the material or void volume ratio at a particular height
   get_summit_curvatures  -- Fine the summit curvatures
   low_pass_filter        -- Low pass filter a surface or surface profile
   subtract_polynomial    -- Fit and subtract an n degree polynomial from a surface profile
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
