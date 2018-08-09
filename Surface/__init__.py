"""
=======================================
Surface generation and manipulation
=======================================
Surface
===========
    Base class containing methods for manipulation, plotting, analysis and 
    reading from file, all other 'Surface' classes inherit this class
    
Geometric
=========
    Classes for creating descrete surfaces with simply defined shapes:
        FlatSurface           -- Flat or sloping surface.
        RoundSurface          -- Spherical surface, different raidii allowed in
                                 each direction
        PyramidSurface        -- Similar to spherical surface diferent sizes
                                 allowed in each direction
Random
=========
    Classes for generating random surfaces based on filtering of random signals
        NoiseBasedSurface     -- Generate and filter a noisy surface, several 
                                 methods for this and the make_like method that
                                 Makes a surface 'like' the input surface

FFTBased
=========
    Classes for generating pseudo-random periodic surfaces based on describing 
    the FFT either deterministically or probabilisticly
        DiscFreqSurface       -- Generate a surface conating only specific 
                                 frequency componenets
        ProbFreqSurface       -- Generate a surface containing normally
                                 distributed amptitudes with a
                                 specified function for the varience of the 
                                 distribution based on the frequency
        DtmnFreqSurface       -- Generate a surface containing frequency
                                 components with amptitude specifed by a 
                                 function of the frequency
"""
import numpy as np
import warnings

from . import Surface

from .Geometric import *
from .Random import *
from .FFTBased import *