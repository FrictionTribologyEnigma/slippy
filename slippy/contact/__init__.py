"""
===========================================================
Contact mechanics models (:mod:`slippy.contact`)
===========================================================

.. currentmodule:: slippy.contact

This module contains functions and classes for contact mechanics.

Functions and classes for generating multi-step contact models
=============================================================

.. autosummary::
   :toctree: generated

   ContactModel   -- A multi-step contact model

   Step                     -- An abstract base class for steps in a contact model
   SurfaceDisplacement      -- Specified displacements on each grid point of a surface
   SurfaceLoading           -- Specified pressures on each grid point of a surface
   StaticNormalInterference -- Specified interference between two surfaces
   StaticNormalLoad         -- Specified loading between two surfaces
   ClosurePlot              -- Generate the data for a closure plot/ load separation curve for two surfaces
   PullOff                  -- Generate the data for an adhesive pull off test between two surfaces

   IterSemiSystemLoad       -- Iterative semi system lubrication step

   Material       -- An abstract base class for materials, these are assigned to surfaces
   Elastic        -- An elastic material

Adhesion models which can be added to dry contacts

Sub models which can be added to a contact simulation
=====================================================

#TODO

Output requests for long simulations
====================================

#TODO

Analytical solutions to common problems
=======================================

.. autosummary::
   :toctree: generated

   hertz_full
   solve_hertz_line
   solve_hertz_point

Examples
========
Examples can be found in the examples folder on github:
https://github.com/FrictionTribologyEnigma/SlipPY/tree/master/examples

"""

from ._material_utils import Loads, Displacements
from .adhesion_models import *
from .hertz import *
from .lubricant import *
from .lubricant_models import *
from .lubrication_steps import *
from .materials import *
from .models import *
from .outputs import *
from .static_step import *
from .steps import *
from .unified_reynolds_solver import *

__all__ = [s for s in dir() if not s.startswith("_")]
