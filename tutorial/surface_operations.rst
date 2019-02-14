Surface Operations
==================

Making a surface
----------------

SlipPy provides the Surface class as a convienient way to store information about surfaces. Modelling functionallity is written to integrate well with these objects. There are several ways to make one of these objects::

    >>> import numpy as np
    >>> import slippy.surface as S
    >>> profile=np.random.norm(size=[10,10])
    >>> # Directly invoking the surface class:
    >>> my_surface=S.Surface(profile=profile)
    >>> # Using a helper function:
    >>> my_surface=S.assurface(profile)

SlipPY also provides functionality for reading surfaces from common analysis machines or file types, again there are two ways to create a surface from a file::

    >>> # Directly invoking the Surface class:
    >>> my_surface=S.Surface(file_name='path_to_file')
    >>> # Using a helper function:
    >>> my_surface=S.read_surface('path_to_file')

Currently the read funcion supports .txt, .csv, .al3d and .mat files.

SlipPy can also generate analytical or random surfaces, this is covered in a later section of the documentation. 

Analysing a surface
-------------------

Roughness parameters for surface objects or profiles can be found using the roughness function::

    >>> profile=np.random.norm(size=[10,10])
    >>> my_surface=S.assurface(profile)
    >>> # The following 3 lines produce identical results:
    >>> # Using the roughness function with a Surface object:
    >>> S.roughness(my_surface, 'sa')
    >>> # Using the roughness function with a raw profile:
    >>> S.roughness(profile, 'sa')
    >>> # Using the roughness method of a Surface object:
    >>> my_surface.roughness('sa')

The roughness function also supports masking of the input for many but not all of the parameters. If the mask keyword is set to a single value that value is excluded, if it is set to a boolean array with the same shape as the surface profile, the profile is ignored where the mask is false::

    >>> # Ignore values smaller than or equal to 1:
    >>> S.roughness(my_surface, 'sa', mask=profile>1)
    >>> # Ignore nan values:
    >>> my_surface.roughness('sa', mask=float('nan'))

