r"""
================================================
Contact mechanics models (:mod:`slippy.contact`)
================================================

.. currentmodule:: slippy.contact

This module contains functions and classes for making contact mechanics models. These models allow the user to model
complex contacts including user defined behaviours for material deformations, wear, friction, fluid flow etc..
For users of abaqus the API will be somewhat familiar: a ContactModel object contains all the information about the
model. Within this object there are Surface objects which contain the geometry information, these have Materials
assigned to them which control how the surfaces deform under load. The actual solving is done by ModelSteps which are
again held within the ContactModel object. When the model is solved these steps are solved in order.

The Contact Model object
========================

.. autosummary::
   :toctree: generated

   ContactModel   -- A multi-step contact model

Model Steps
===========

There are ModelSteps for many different situations such as dry normal loading, mixed lubricated etc.. Some of these also
require a Lubricant to be defined.

.. autosummary::
   :toctree: generated

   SurfaceDisplacement      -- Specified displacements on each grid point of a surface
   SurfaceLoading           -- Specified pressures on each grid point of a surface
   StaticNormalInterference -- Specified interference between two surfaces
   StaticNormalLoad         -- Specified loading between two surfaces

   IterSemiSystemLoad       -- Iterative semi system lubrication step

Materials
=========

Slippy currently contains solvers for Elastic and Rigid materials, more materials will be added in future releases, it
is also possible to add your own materials.

.. autosummary::
   :toctree: generated

   Elastic        -- An elastic material
   Rigid          -- The rigid material class\*

\*note as the Rigid class has no options, an instance (rigid) is also provided for convenience

Lubricants
==========

Lubricants in slippy are defined through the Lubricant object and lubricant sub models. A lubricant object is just a
container for the sub models which define the behaviour of the lubricant. Sub models must be added for the behaviours
required by a particular solver. These can be constants as for a newtonian lubricant or they can depend on other
variables found during the solution eg pressure.

.. autosummary::
   :toctree: generated

   Lubricant
   lubricant_models


Sub models
==========

Slippy is quasi static, meaning that the normal contact problem is always solved for the static system. Transient
behaviour such as wear, temperature change, tribofilm growth, plastic deformation etc. is dealt with by sub models.
These are solved after the contact problem has been solved for each step.

.. autosummary::
   :toctree: generated

   sub_models

Output requests for long simulations
====================================

By default a contact model will return the final state when .solve() is called on it. However often this is not
sufficient. Output requests allow the user to save all or part of the model state at set time points or after set steps.

.. autosummary::
   :toctree: generated

   OutputRequest

Analytical solutions to common problems
=======================================

Lastly slippy.contact also contains analytical solutions for common contacts. These are often useful as initial guesses
to more complex contact problems.

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
from .lubrication_steps import IterSemiSystem
from .materials import Elastic, Rigid, rigid, elastic_influence_matrix
from .models import ContactModel
from .outputs import OutputRequest, OutputReader, OutputSaver, read_output
from .static_step import StaticStep
# from .steps import InitialStep
from .unified_reynolds_solver import UnifiedReynoldsSolver
from .quasi_static_step import QuasiStaticStep
from . import sub_models
from .influence_matrix_utils import guess_loads_from_displacement, bccg, plan_convolve

__all__ = ['Loads', 'Displacements', 'hertz_full', 'solve_hertz_line', 'solve_hertz_point', 'Lubricant',
           'lubricant_models', 'IterSemiSystem', 'Elastic', 'Rigid', 'rigid', 'elastic_influence_matrix',
           'ContactModel', 'OutputRequest', 'OutputReader', 'OutputSaver', 'read_output',
           'StaticStep', 'UnifiedReynoldsSolver', 'sub_models', 'QuasiStaticStep', 'sub_models',
           'guess_loads_from_displacement', 'bccg', 'plan_convolve'
           ]
