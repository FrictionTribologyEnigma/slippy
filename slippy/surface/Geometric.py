"""
Classes for generating geometric surfaces:
    ===========================================================================
    ===========================================================================
    Each class inherits functionality from the _AnalyticalSurface class but changes the
    __init__, _height and __repr__ functions
    ===========================================================================
    ===========================================================================

    FlatSurface:
        For generating flat surfaces, slopes are allowed in both directions
    PyramidSurface:
        For generating square pyramid surfaces like indenters
    RoundSurface:
        For generating ball type surfaces different radii allowed in each
        direction

    ===========================================================================
    ===========================================================================

"""

__all__ = ['FlatSurface', 'RoundSurface', 'PyramidSurface']

import collections.abc
import typing
import warnings
from numbers import Number

import numpy as np

from .Surface_class import _AnalyticalSurface


class FlatSurface(_AnalyticalSurface):
    """ Flat surface can be angled in any direction by changing slope

    Parameters
    ----------
    slope : {tuple, float}, optional (0,0)
        The gradient of the surface in the x and y directions
    rotation: float, optional (None)
        If set the surface will be rotated by this number of radians
    shift: tuple, optional (None)
        If set the surface will be shifted by this distance in the and y directions, tuple should be length 2, if not
        set the default is to shift by half the extent, meaning that the origin becomes the centre.
    generate: bool, optional (False)
        If True the surface profile is discretised on instantiation.
    grid_spacing: float, optional (None)
        The distance between grid points on the surface profile
    extent: tuple, optional (None)
        The overall size of the surface
    shape: tuple, optional (None)
        The number of grid points in each direction on the surface


    Attributes
    ----------

    Methods
    -------
    height

    See Also
    --------
    Surface

    Notes
    -----
    This is a subclass of Surface all functionality and attributes are available
    within this class.

    All keyword arguments allowed for Surface are also
    allowed on instantiation of this class apart from the profile key word.

    """
    surface_type = 'flat'
    analytic = True

    def __init__(self, slope: typing.Union[tuple, float] = (0, 0), rotation: float = None,
                 shift: typing.Optional[tuple] = None,
                 generate: bool = False, grid_spacing: float = None,
                 extent: tuple = None, shape: tuple = None):
        if isinstance(slope, collections.abc.Sequence):
            self._slope = slope
        elif isinstance(slope, Number):
            # noinspection PyTypeChecker
            slope = float(slope)
            self._slope = [slope, 0]
            if self.dimensions == 2:
                warnings.warn("Assumed 0 slope in Y direction for"
                              " analytical flat surface")
        super().__init__(generate=generate, rotation=rotation, shift=shift,
                         grid_spacing=grid_spacing, extent=extent, shape=shape)

    def _height(self, x_mesh, y_mesh):
        """Analytically determined height of the surface at specified points

        Parameters
        ----------
        x_mesh, y_mesh : array-like
            Arrays of X and Y points, must be the same shape

        Returns
        -------
        Array of surface heights, with the same shape as the input arrays

        Notes
        -----
        This is an alternative to discretise which may be more
        appropriate for some applications

        Examples
        --------
        >>> import numpy as np
        >>> my_surface=FlatSurface(slope=(1,1))
        >>> x, y = np.arange(10), np.arange(10)
        >>> X, Y = np.meshgrid(x,y)
        >>> Z=my_surface.height(X, Y)
        """
        return x_mesh * self._slope[0] + y_mesh * self._slope[1]

    def __repr__(self):
        string = self._repr_helper()
        return 'FlatSurface(slope=' + repr(self._slope) + string + ')'


class RoundSurface(_AnalyticalSurface):
    """ Round surfaces with any radii

    Parameters
    ----------
    radius : Sequence
        The radius of the surface in the X Y and Z directions, or in all
        directions if a float is given
    rotation: float, optional (None)
        If set the surface will be rotated by this number of radians
    shift: tuple, optional (None)
        If set the surface will be shifted by this distance in the and y directions, tuple should be length 2, if not
        set the default is to shift by half the extent, meaning that the origin becomes the centre.
    generate: bool, optional (False)
        If True the surface profile is discretised on instantiation.
    grid_spacing: float, optional (None)
        The distance between grid points on the surface profile
    extent: tuple, optional (None)
        The overall size of the surface
    shape: tuple, optional (None)
        The number of grid points in each direction on the surface


    Attributes
    ----------

    Methods
    -------
    height

    See Also
    --------
    Surface

    Notes
    -----
    This is a subclass of Surface all functionality and attributes are available
    within this class.

    All keyword arguments allowed for Surface are also
    allowed on instantiation of this class apart from the profile key word.

    """
    radius: tuple

    def __init__(self, radius: typing.Sequence, rotation: float = None,
                 shift: typing.Optional[tuple] = None,
                 generate: bool = False, grid_spacing: float = None,
                 extent: tuple = None, shape: tuple = None):

        if isinstance(radius, Number):
            radius = (radius,)*3
        if isinstance(radius, collections.abc.Sequence) and len(radius) == 3:
            self._radius = radius
        else:
            msg = ('Radius must be either scalar or list of radii equal in '
                   'length to number of dinmetions of the surface +1')
            raise ValueError(msg)
        super().__init__(generate=generate, rotation=rotation, shift=shift,
                         grid_spacing=grid_spacing, extent=extent, shape=shape)

    def _height(self, x_mesh, y_mesh):
        """Analytically determined height of the surface at specified points

        Parameters
        ----------
        x_mesh, y_mesh : array-like
            Arrays of x and y points, must be the same shape

        Returns
        -------
        Array of surface heights, with the same shape as the input arrays

        Notes
        -----
        This is an alternative to discretise the surface which may be more
        appropriate for some applications

        Examples
        --------
        >>> import numpy as np
        >>> my_surface=RoundSurface(radius=(1,1,1))
        >>> x, y = np.arange(10), np.arange(10)
        >>> x_mesh, y_mesh = np.meshgrid(x,y)
        >>> Z=my_surface.height(x_mesh, y_mesh)
        """
        # noinspection PyTypeChecker
        z = ((1 - (x_mesh / self._radius[0]) ** 2 - (y_mesh / self._radius[1]) ** 2) ** 0.5) * self._radius[-1] - \
            self._radius[-1]
        return np.nan_to_num(z, False)

    def __repr__(self):
        string = self._repr_helper()
        return 'RoundSurface(radius=' + repr(self._radius) + string + ')'


class PyramidSurface(_AnalyticalSurface):
    """ Pyramid surface with any slopes

    Parameters
    ----------
    lengths : {Sequence, float}
        The characteristic lengths of the pyramid in each direction, if a scalar is given the results is a square based
        pyramid with 45 degree sides
    rotation: float, optional (None)
        If set the surface will be rotated by this number of radians
    shift: tuple, optional (None)
        If set the surface will be shifted by this distance in the and y directions, tuple should be length 2, if not
        set the default is to shift by half the extent, meaning that the origin becomes the centre.
    generate: bool, optional (False)
        If True the surface profile is discretised on instantiation.
    grid_spacing: float, optional (None)
        The distance between grid points on the surface profile
    extent: tuple, optional (None)
        The overall size of the surface
    shape: tuple, optional (None)
        The number of grid points in each direction on the surface

    See Also
    --------
    Surface

    Notes
    -----
    This is a subclass of Surface all functionality and attributes are available
    within this class.

    All keyword arguments allowed for Surface are also
    allowed on instantiation of this class apart from the profile key word.

    """
    surface_type = 'pyramid'

    def __init__(self, lengths: typing.Union[typing.Sequence], rotation: float = None,
                 shift: typing.Optional[tuple] = None,
                 generate: bool = False, grid_spacing: float = None,
                 extent: tuple = None, shape: tuple = None):
        if isinstance(lengths, Number):
            lengths = (lengths, ) * 3

        if type(lengths) is tuple:
            if len(lengths) == (self.dimensions + 1):
                self._lengths = lengths
            else:
                msg = ('Lengths must be either scalar or list of Lengths equal'
                       ' in length to number of dinmetions of the surface +1')
                raise ValueError(msg)
        super().__init__(generate=generate, rotation=rotation, shift=shift,
                         grid_spacing=grid_spacing, extent=extent, shape=shape)

    def _height(self, x_mesh, y_mesh):
        """Analytically determined height of the surface at specified points

        Parameters
        ----------
        x_mesh, y_mesh : array-like
            Arrays of X and Y points, must be the same shape

        Returns
        -------
        Array of surface heights, with the same shape as the input arrays

        Notes
        -----
        This is an alternative to discretise the surface which may be more
        appropriate for some applications

        Examples
        --------
        >>> import numpy as np
        >>> my_surface=PyramidSurface(lengths=[20,20,20])
        >>> x, y = np.arange(10), np.arange(10)
        >>> X, Y = np.meshgrid(x,y)
        >>> Z=my_surface.height(X, Y)
        """
        return (0 - np.abs(x_mesh) / self._lengths[0] - np.abs(y_mesh) / self._lengths[1]) * self._lengths[-1]

    def __repr__(self):
        string = self._repr_helper()
        return 'PyramidSurface(lengths=' + repr(self._lengths) + string + ')'
