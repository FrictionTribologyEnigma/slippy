"""
===============================================================
Contact mechanics sub models (:mod:`slippy.contact.sub_models`)
===============================================================

.. currentmodule:: slippy.contact.sub_models

In slippy, things that are not solved in the main normal contact loop are dealt with using sub models. These could be
anything from flash temperature to tribofilm growth to friction etc.. These sub models cover behaviour that doesn't
the normal contact problem or acts on a slower time scale. For example: wear of the surfaces does affect the normal
contact problem but over a very slow time scale (compared to the reaction of the material) which mean that wear should
be solved as a sub model. Likewise, in many situations the friction force can be solved separately to the normal contact
problem, thus friction can also be a sub model. However, when solving a lubrication model for example the pressure-
viscosity response of the lubricant strongly affects the normal contact problem and acts in the same time scale to the
material response, thus this should be solved as part of the main contact loop and is not a valid sub model.

Currently implemented sub models are:

.. autosummary::
   :toctree: generated

   ContactStiffnessSubModel   -- Find contact stiffness in any direction
   EPPWear                    -- Wear the surfaces where there is excess interference after elastic deformation

"""

from .contact_stiffness import ContactStiffness
from .epp_wear import EPPWear

__all__ = ['ContactStiffness', 'EPPWear']
