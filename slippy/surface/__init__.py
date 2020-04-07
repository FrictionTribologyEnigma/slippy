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
contains methods for analysing surface roughness and other parameters.

.. autosummary::
   :toctree: generated/

   Surface


Analytical Surfaces
===================

Surfaces which can be described by a mathematical function, these can be combined, rotated or shifted with no loss in
resolution.

.. autosummary::
   :toctree: generated/

   FlatSurface         -- Flat or sloping surface.
   RoundSurface        -- Round surfaces
   PyramidSurface      -- Square based pyramid surfaces
   DiscFreqSurface     -- Surfaces containing specific frequency components
   HurstFractalSurface -- A Hurst Fractal Surface

Random Surfaces
===============

Surfaces based on transformations of random surfaces or probabilistic descriptions of the FFT.

.. autosummary::
    :toctree: generated/

   RandomPerezSurface     -- Surfaces with set height distribution and PSD found by the Perez method
   RandomFilterSurface    --Surfaces from transformations of random sequences
   ProbFreqSurface   -- Surfaces based on a probabilistic description of the FFT
   surface_like      -- random surfaces with similar properties to the input surface

Functions
=========

Functional interfaces for common tasks, these are all aliased by class methods in the surface class, apart from
surface_like.

.. autosummary::
   :toctree: generated/

   assurface              -- Make a surface object
   read_surface           -- Read a surface object from file
   alicona_read           -- Read alicona data files
   read_tst_file          -- Read a bruker umt tst file
   roughness              -- Find 2d roughness parameters
   get_height_of_mat_vr   -- Find the height of a specified material or void volume ratio
   get_mat_vr             -- Find the material or void volume ratio at a particular height
   subtract_polynomial    -- Fit and subtract an n degree polynomial from a surface profile
"""


from .ACF_class import ACF
from .Surface_class import Surface, assurface, read_surface
from .Geometric import FlatSurface, RoundSurface, PyramidSurface
from .Random import RandomFilterSurface, RandomPerezSurface
from .FFTBased import ProbFreqSurface, DiscFreqSurface, HurstFractalSurface
from .alicona import alicona_read
from .roughness_funcs import roughness, subtract_polynomial, get_mat_vr, get_height_of_mat_vr
from .read_tst_file import read_tst_file

__all__ = ['Surface', 'assurface', 'read_surface', 'ACF', 'FlatSurface', 'RoundSurface', 'PyramidSurface',
           'RandomFilterSurface', 'RandomPerezSurface', 'ProbFreqSurface', 'DiscFreqSurface', 'HurstFractalSurface',
           'alicona_read', 'roughness', 'subtract_polynomial', 'get_mat_vr', 'get_height_of_mat_vr',
           'read_tst_file']
