.. _Extensions:

Extensions
==========

Slippy is built to be simple to extend. This means that, if done correctly, new functionality can be added that works with existing code.
However it can ber difficult to see how a new model can fit in. Typically adding functionality would involve making a new sub class for one of the base classes below.

Note that these base classes can change without breaking compatibility with the users code, however these changes may break compatibility with extensions.
If your extension has been added to the main code, it will not be broken by future updates.
In general we will try to add depreciation warnings and make updates simple for user built extensions.
However it is worth keeping a record of the version of slippy that your extension was developed for as this version will always be available on pypi.

Surface profiles
----------------

New profile types can be added by sub classing the _AnalyticalSurface abstract base class found in slippy.surface.
To implement a new analytically defined surface type you must implement the __init__ and _height methods.
The __init__ method should include a call to the super's init:

    super().__init__(generate=generate, rotation=rotation, shift=shift,
                     grid_spacing=grid_spacing, extent=extent, shape=shape)

The _height method should take as arguments an array of x coordinates and an array of y coordinates and return the height of the profile at the specified points.

Materials
---------

Materials in slippy are defined by their influence matrix: the surface displacements caused by a unit pressure on one
grid cell. Two routes are available for adding a new material.

**The recommended route: a frequency response function.** If the surface response of the material can be written in the
frequency domain, u(q) = C(q) p(q) — true for any linear, homogeneous-in-plane half space model (layered, graded,
surface tensed...) — sub class ``_FrequencyDomainMaterial`` from slippy.core and implement two methods:

- ``__init__``, which should store the material parameters and call the super's init:

    ``super().__init__(name, max_load=max_load)``

- ``_frf(components, q_y, q_x, q_norm)``, which evaluates C on the supplied wavenumber grids and returns a dict with
  one array per requested component. Components are named as load direction then displacement direction ('zz' is the
  normal displacement from a normal load); normal contact solvers only request 'zz'. The zero frequency (q = 0) element
  is handled for you: pass ``zero_frequency_value`` to the super init if your material has a finite compliance at
  q = 0 (the default, matching the elastic half space, is 0).

For reference, the elastic half space has C_zz(q) = 2 / (E* |q|), and ``SurfaceTensedMaterial``
(slippy/core/surface_tensed_material.py) is a complete worked example in ~70 lines.

**The general route.** Sub class ``_IMMaterial`` directly and implement ``_influence_matrix_spatial`` and/or
``_influence_matrix_frequency`` (at least one is required, enforced at class creation). The base class provides the
public ``influence_matrix`` method, including memoization, periodic strides and DC-term handling — do not override it.
The two sub surface stress methods (``sss_influence_matrices_normal`` and ``sss_influence_matrices_tangential_x``) must
be defined but may raise NotImplementedError if sub surface stresses are not available for your material.

In both cases:

- Material names must be unique per process; influence matrices are memoized by name and grid arguments, so treat all
  parameters as immutable after construction — make a new instance rather than mutating.
- Set ``max_load`` if the material should saturate at a maximum pressure (elastic-perfectly-plastic approximation,
  pairs with the WearElasticPerfectlyPlastic sub model).
- Add an entry for your material to the ``material_parameters`` dict in slippy/core/tests/test_materials.py: the test
  suite automatically round-trip tests every registered material, and fails if parameters are missing. Materials which
  only support normal loading should set ``'_directions': 'z'`` there, and materials with a zero DC compliance (the
  default for frequency-only materials) should also set ``'_zero_mean': True``.

Steps
-----

Implementing a new model step is a major task, it requires the use to implement the __init__ and solve methods.
Each step must also have a .provides property which details what will be in the state dict at the end of the execution
For a clear picture of what a step should do have a look at the existing StaticStep and QuasiStaticStep.
Both of these sub class the _ModelStep abstract base class from slippy.abcs.

The results from each step of the simulation are passed around in a dictionary. Each step has a .provides property
which is a set of strings, these string are exactly and only the keys to the current state dictionary at the end of the
step. Many common items have names which should be respected if your new step is to work with existing sub models.
If your step provides additional parameters these should be given descriptive names.
In general parameters should be a single value or an array of values, anything else requires special treatment by the output system.
For example to store a set of coordinates points_x and points_y should be used rather than a single points parameter being a list of arrays.

If your step promises to provide something which it doesn't actually provide or provides extra parameters, slippy will raise an error when the model is solved.
This is intended to push errors into the development process and ultimately leave fewer confused users.
As such the provides property can be set by the init method and doesn't need to be the same for every possible version of the step.

Steps should include calls to: self.solve_sub_models and self.save_outputs after the main problem has been solved.

Sub models
----------

Making a new sub model is a simple task. The _SubModel abstract base class from slippy.abcs should be used.
To add a sub model the __init__ and solve methods must be implemented.
As above, a call to the super's init method should be included in the __init__ method:

    super().__init__(name, requires, provides)

This sets the name which should be a string, and the requires and provides properties of the sub model.
The requires and provides properties should be sets of strings. The requires property should include every item which
must be in the current state dict for the sub model to be executed successfully.
The provides property should detail exactly and only what the sub model will add to the current state dict.
Unexpected or missing items will cause an error when the model is solved.

As well as the current state each sub model has access to the main model and thus the surface profiles of each surface.
Because of this sub models can be used to apply wear to the surfaces. Other parameters can be retained by the sub model.
