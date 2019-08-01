"""
===========================================================
Contact mechanics models (:mod:`slippy.contact`)
===========================================================

.. currentmodule:: slippy.contact

This module contains functions and classes for contact mechanics.

General utilities
=================

.. autosummary::
   :toctree: generated
   
   convert_array
   convert_dict
   combined_modulus

Utilities for converting plotting and analysing results

Elastic B.E.M.
==============

Functions for elastic BEM soultions assuming the small angle assumption is 
valid

.. autosummary::
   :toctree: generated
   
   elastic_loading      -- Flat or sloping surface.
   elastic_displacement -- Round surfaces
   elastic_im           -- Square based pyramid surface

Elastic perfectly plastic B.E.M.
================================

   
"""

from .elastic_bem import *
from .hertz import *
from .dummy_classes import *
from .steps import *
from .static_step import *
from .models import *
from .steps import *

__all__ = [s for s in dir() if not s.startswith("_")]