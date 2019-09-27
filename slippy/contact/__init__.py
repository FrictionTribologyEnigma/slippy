"""
===========================================================
Contact mechanics models (:mod:`slippy.contact`)
===========================================================

.. currentmodule:: slippy.contact

This module contains functions and classes for contact mechanics.

Analytical solutions to common problems
=======================================

.. autosummary::
   :toctree: generated

   hertz_full
   solve_hertz_line
   solve_hertz_point

Functions and classes for generating multistep contact models
=============================================================

Functions for elastic BEM soultions assuming the small angle assumption is
valid

.. autosummary::
   :toctree: generated

   ContactModel
   Step
   Material

XXXXXXXX put some examples here!
================================

   
"""

from .hertz import *
from .materials import *
from .friciton_models import *
from .adhesion_models import *
from ._material_utils import Loads, Displacements
from .steps import *
from .static_step import *
from .models import *


__all__ = [s for s in dir() if not s.startswith("_")]