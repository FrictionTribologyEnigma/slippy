import numpy as np
import types
import scipy.signal
import scipy.interpolate
from slippy.abcs import _SurfaceABC

__all__ = ['ACF']


class ACF(object):
    """ A helper class for auto correlation functions

    Produces ACFs that are independent ot the original grid spacing, these can
    be used to generate random surfaces with any grid spacing.

    Parameters
    ----------
    source : Surface object, array, function or str
        The source for the ACF see notes
    grid_spacing : float optional (None)
        The distance between grid points in the surface profile, only needed
        if source is an array

    Other Parameters
    ----------------
    args : float optional (None)
        If the source is a string, indicating the acf follows an equation, arg
        are the constants in the equation

    Attributes
    ----------

    method
    original
    acf_type

    Methods
    -------

    call

    Notes
    -----

    Valid options for the source are:

        - A surface object
        - An array, the grid_spacing parameter must also be set
        - A function
        - 'exp' other args must be passed

    If the source is 'exp' three additional arguments must be passed. The
    centre value of the acf (sigma) and the decay length in the x and y
    directions (beta_x and beta_y). The acf is then calculated as:

    sigma**2*np.exp(-2.3*np.sqrt((X/beta_x)**2+(Y/beta_y)**2))

    If a function is given as the input it must take as arguments arrays of X
    and Y coordinates and return an array of Z coordinates for example:

    X,Y=np.meshgrid(range(10),range(10))

    Z=input_function(X,Y)

    Examples
    --------

    >>> # Generate an ACF with an exponential shape:
    >>> ACF('exp', 2, 0.1, 0.2)

    >>> # Generate an ACF from a surface object:
    >>> ACF(my_surface)

    >>> # Generate an ACF from an array:
    >>> my_acf=ACF(array, grid_spacing=1)
    >>> # Get the original acf points:
    >>> np.array(my_acf)

    """

    method = None
    """The method which is called to generate points"""
    original = None
    """If the source is an array or a Surface object, the original acf array"""
    acf_type = ''
    """A description of the acf source"""

    def __init__(self, source, *args, grid_spacing=1):
        if type(source) is str:
            self._input_check_string(source, args)
            self.acf_type = "string"
        elif isinstance(source, _SurfaceABC):
            self._input_check_array(source.profile, grid_spacing=source.grid_spacing)
            self.acf_type = "surface"
        elif isinstance(source, types.FunctionType):
            self._input_check_method(source)
            self.acf_type = "function"
        else:
            if grid_spacing is None:
                msg = "grid spacing positional argument must be supplied if ACF source is an array"
                raise ValueError(msg)
            self._input_check_array(source, grid_spacing)
            # args should contain the grid_spacing of the array
            self.acf_type = "array"

    def _input_check_string(self, source, args):
        supported_functions = ['exp', 'polynomial']
        if not (source in supported_functions):
            msg = ("Function type not supported, supported types are:\n" +
                   "\n".join(supported_functions) + "\nfor custom functions pass "
                                                    "the function object to this constructor")
            raise NotImplementedError(msg)
        if source == 'exp':
            sigma = args[0]
            beta_x = args[1]
            beta_y = args[2]
            self.method = _exp_acf(sigma, beta_x, beta_y)
        elif source == 'polynomial':
            pass
            raise NotImplementedError("polynomial functions are not implemented")

    def _input_check_array(self, source, grid_spacing, origin='centre'):
        try:
            profile = np.asarray(source)
        except ValueError:
            msg = ("invalid input, input should be either a surface, function "
                   "handle, function name as a string with relevant params or "
                   "array-like")
            raise ValueError(msg)
        x = profile.shape[0]
        y = profile.shape[1]
        self.o_grid_spacing = grid_spacing
        self.original = (scipy.signal.correlate(profile, profile, 'same') / (x * y))

        x = np.arange(x) * grid_spacing
        y = np.arange(y) * grid_spacing

        if type(origin) is str:
            origin = origin.lower()
            if origin == 'centre':
                origin = [np.mean(x), np.mean(y)]
            else:
                description = origin
                origin = [0, 0]
                if description[1] == 'r' or description[1] == 'e':
                    origin[0] = max(x)
                if description[0] == 't' or description[0] == 'n':
                    origin[1] = max(y)
        x = x - origin[0]
        y = y - origin[1]

        self.method = scipy.interpolate.RectBivariateSpline(x, y, self.original)

    def _input_check_method(self, source):
        self.method = source

    def __call__(self, x, y):
        """
        Evaluate the ACF at specified grid points

        Parameters
        ----------
        x,y : 1D np.array
            The x and y coordinates of the grid points to be evaluated, see notes for usage

        Returns
        -------
        np.array
            The 'height' of the acf at the grip points requested

        Notes
        -----
        The x and y parameters are turned into a grid before evaluating the ACF function

        Examples
        --------
        >>> my_acf = ACF('exp', None, 1, 1) # Make an ACF object with an exponential ACf
        >>> # evaluate the acf a the points x=[-10,-8....,10], y==[-10,-8....,10]
        >>> discrete_acf = my_acf(np.arange(-10, 11, 2), np.arange(-10, 11, 2))
        >>> # The results is evaluated over the grid, not just at the specified points
        >>> discrete_acf.shape
        (11,11)
        """
        # feed to self.method
        if self.acf_type in ['array', 'surface']:
            return self.method(x, y)
        else:
            mesh_x, mesh_y = np.meshgrid(x, y)
            return self.method(mesh_x, mesh_y)

    def __array__(self):
        if self.original is not None:
            return self.original
        else:
            raise ValueError("Could not return ACF as array, ACF must be made from"
                             " an array or surface for this to work in stead use "
                             "array(this(x_pts,y_pts)), see documentation for __call__ for more information")


def _exp_acf(sigma, beta_x, beta_y):
    def inner(x_mesh, y_mesh):
        raw = sigma ** 2 * np.exp(-2.3 * np.sqrt((x_mesh / beta_x) ** 2 + (y_mesh / beta_y) ** 2))
        return raw
    return inner
