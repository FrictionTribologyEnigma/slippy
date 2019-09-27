"""
===========================================================
Contact mechanics models (:mod:`slippy.contact`)
===========================================================

.. currentmodule:: slippy.contact

This module contains functions and classes for contact mechanics.

Functions and classes for generating multistep contact models
=============================================================

.. autosummary::
   :toctree: generated

   ContactModel   --A multistep contact model
   Step           --
   Elastic

Analytical solutions to common problems
=======================================

.. autosummary::
   :toctree: generated

   hertz_full
   solve_hertz_line
   solve_hertz_point

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