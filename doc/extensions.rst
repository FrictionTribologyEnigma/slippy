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

If the influence matrix for a material is known, then to add it into slippy the _IMMaterial base class can be sub classed.
This requires the user to implement the __init__ and influence_matrix methods.
Practically it is also useful to memoize the influence matrix to save computing it for every time step of a simulation.

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
