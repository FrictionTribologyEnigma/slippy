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
# from .adhesion_models import *
from .hertz import hertz_full, solve_hertz_line, solve_hertz_point
from .lubricant import Lubricant
from .lubrication_steps import IterSemiSystemLoad
from .materials import Elastic, Rigid, rigid
from .models import ContactModel
from .outputs import OutputRequest
from .static_step import StaticNormalLoad, StaticNormalInterference, SurfaceDisplacement, SurfaceLoading
# from .steps import InitialStep
from .unified_reynolds_solver import UnifiedReynoldsSolver

__all__ = ['Loads', 'Displacements', 'hertz_full', 'solve_hertz_line', 'solve_hertz_point', 'Lubricant',
           'lubricant_models', 'IterSemiSystemLoad', 'Elastic', 'Rigid', 'rigid', 'ContactModel', 'OutputRequest',
           'StaticNormalLoad', 'StaticNormalInterference', 'SurfaceDisplacement', 'SurfaceLoading',
           'UnifiedReynoldsSolver']
