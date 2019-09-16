"""
Classes for generating geometric surfaces:
    ===========================================================================
    ===========================================================================
    Each class inherits functionallity from the _AnalyticalSurface calss but changes the
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

from .Surface_class import _AnalyticalSurface
import warnings
import numpy as np
from numbers import Number
import typing


class FlatSurface(_AnalyticalSurface):
    """ Flat surface can be angled in any direction by changing slope
    
    Parameters
    ----------
    slope : tuple, optional (0,0)
        The gradient of the surface in the x and y directions
    
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

    def __init__(self, slope: tuple = (0, 0), rotation: float = 0,
                 shift: typing.Union[tuple, str] = 'origin to centre',
                 generate: bool = False, grid_spacing: float = None,
                 extent: tuple = None, shape: tuple = None):
        if type(slope) is tuple:
            self._slope = slope
        elif type(slope) is int or type(slope) is float:
            self._slope = [slope, 0]
            if self.dimentions == 2:
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
        This is an alternative to descretising the surface which may be more 
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
    radius : tuple
        The radius of the surface in the X Y and Z directions, or in all 
        directions if a float is given
    
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

    def __init__(self, radius: tuple, rotation: float = 0,
                 shift: typing.Union[tuple, str] = 'origin to centre',
                 generate: bool = False, grid_spacing: float = None,
                 extent: tuple = None, shape: tuple = None):

        if isinstance(radius, Number):
            radius = (radius,)*3
        if type(radius) is tuple and len(radius) == 3:
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
        This is an alternative to descretising the surface which may be more 
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
    
    Keyword parameters
    ------------------
    lengths : list or float
        The characteristic lengths of the pyramid in each direction, if a 
        scalar is given the results is a square based pyramid with 45 degre 
        sides
    
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
    surface_type = 'pyramid'

    def __init__(self, lengths, rotation: float = 0,
                 shift: typing.Union[tuple, str] = 'origin to centre',
                 generate: bool = False, grid_spacing: float = None,
                 extent: tuple = None, shape: tuple = None):
        if isinstance(lengths, Number):
            lengths = (lengths, ) * 3

        if type(lengths) is tuple:
            if len(lengths) == (self.dimentions + 1):
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
        This is an alternative to descretising the surface which may be more 
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
