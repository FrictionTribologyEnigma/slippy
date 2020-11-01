# TODO mesh function

# TODO documentation

# TODO make list of all functionality

import abc
import copy
import csv
import os
import typing
import warnings
from numbers import Number
from collections.abc import Sequence

import numpy as np
import scipy.interpolate
import scipy.signal

from slippy.abcs import _MaterialABC, _SurfaceABC
from .ACF_class import ACF
from .roughness_funcs import get_height_of_mat_vr, low_pass_filter
from .roughness_funcs import get_mat_vr, get_summit_curvatures
from .roughness_funcs import roughness, subtract_polynomial, find_summits

__all__ = ['Surface', 'assurface', 'read_surface', '_Surface', '_AnalyticalSurface']


def assurface(profile, grid_spacing=None):
    """ make a surface from a profile

    Parameters
    ----------
    profile : array-like
        The surface profile
    grid_spacing : float optional (None)
        The spacing between grid points on the surface

    Returns
    -------
    surface : Surface object
        A surface object with the specified profile and grid size

    See Also
    --------
    Surface
    read_surface

    Notes
    -----


    Examples
    --------

    >>> profile=np.random.normal(size=[10,10])
    >>> my_surface=assurface(profile, 0.1)
    >>> my_surface.extent
    [1,1]

    """
    return Surface(profile=profile, grid_spacing=grid_spacing)


def read_surface(file_name, **kwargs):
    """ Read a surface from a file

    Parameters
    ----------
    file_name : str
        The full path to the data file

    Other Parameters
    ----------------

    delim : str optional (',')
        The delimiter used in the data file, only needed for csv or txt files
    p_name : str optional ('profile')
        The name of the variable containing the profile data, needed if a .mat
        file is given
    gs_name : str optional ('grid_spacing')
        The name of the variable containing the grid_spacing, needed if a .mat
        file is given

    Returns
    -------
    A surface object generated from the file

    See Also
    --------
    Surface
    alicona_read
    scipy.io.loadmat

    Notes
    -----
    This function directly invokes the surface class, any other keywords that
    can be passed to that class can be passed to this function

    Examples
    --------

    >>> # Read a csv file with tab delimiters
    >>> my_surface=read_surface('data.csv', delim='\t')

    >>> # Read a .al3d file
    >>> my_surface=read_surface('data.al3d')

    >>> # Read a .mat file with variables called prof and gs
    >>> my_surface=read_surface('data.mat', p_name='prof', gs_name='gs')

    """
    return Surface(file_name=file_name, **kwargs)


class _Surface(_SurfaceABC):
    """
    An abstract base class for surface types, this class should be extended to given new types of surface. To create an
    analytical surface please subclass _AnalyticalSurface
    """

    # The surface class for discrete surfaces (typically experimental)
    is_discrete: bool = False
    """ A bool flag, True if there is a profile present """
    acf: typing.Optional[ACF] = None
    """ The auto correlation function of the surface profile """
    psd: typing.Optional[np.ndarray] = None
    """ The power spectral density of the surface """
    fft: typing.Optional[np.ndarray] = None
    """ The fast fourier transform of the surface """
    surface_type: str = "Generic"
    """ A description of the surface type """
    dimensions: typing.Optional[int] = 2
    """ The number of spatial dimensions that """
    is_analytic: bool = False
    """ A bool, true if the surface can be described by an equation and a  Z=height(X,Y) method is provided"""
    invert_surface: bool = False

    _material: typing.Optional[_MaterialABC] = None
    unworn_profile: typing.Optional[np.ndarray] = None
    _profile: typing.Optional[np.ndarray] = None
    _grid_spacing: typing.Optional[float] = None
    _shape: typing.Optional[tuple] = None
    _extent: typing.Optional[tuple] = None
    _inter_func = None
    _allowed_keys = {}
    _mask: typing.Optional[np.ndarray] = None
    _size: typing.Optional[int] = None
    _subclass_registry = []
    _original_extent = None

    def __init__(self, grid_spacing: typing.Optional[float] = None, extent: typing.Optional[tuple] = None,
                 shape: typing.Optional[tuple] = None, is_discrete: bool = False):
        if grid_spacing is not None and extent is not None and shape is not None:
            raise ValueError("Up to two of grid_spacing, extent and size should be set, all three were set")

        self.is_discrete = is_discrete

        if grid_spacing is not None:
            self.grid_spacing = grid_spacing
        if extent is not None:
            self.extent = extent
        if shape is not None:
            self.shape = shape

        self.wear_volumes = dict()

    @classmethod
    def __init_subclass__(cls, is_abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            _Surface._subclass_registry.append(cls)

    @property
    def size(self):
        """The total number of points in the surface"""
        return self._size

    @property
    def mask(self):
        """A mask used to exclude some values from analysis, a single float or an array of bool the same size as profile
        Either a boolean array of size self.size or a float of the value to be excluded
        """
        return self._mask

    @mask.setter
    def mask(self, value: typing.Union[float, np.ndarray]):

        if type(value) is float:
            if np.isnan(value):
                mask = np.isnan(self.profile)
            else:
                mask = self.profile == value
        elif isinstance(value, np.ndarray):
            mask = np.asarray(value, dtype=bool)
            if not mask.shape == self.shape:
                msg = ("profile and mask shapes do not match: profile is"
                       "{profile.shape}, mask is {mask.shape}".format(**locals()))
                raise TypeError(msg)
        elif isinstance(value, str):
            raise TypeError('Mask cannot be a string')
        elif isinstance(value, Sequence):
            mask = np.zeros_like(self.profile, dtype=bool)
            for item in value:
                self.mask = item
                mask = np.logical_and(self._mask, mask)
        else:
            raise TypeError("Mask type is not recognised")
        self._mask = mask

    @mask.deleter
    def mask(self):
        self._mask = None

    @property
    def extent(self):
        """ The overall dimensions of the surface in the same units as grid spacing
        """
        return self._extent

    @extent.setter
    def extent(self, value: typing.Sequence[float]):
        if not isinstance(value, Sequence):
            msg = "Extent must be a Sequence, got {}".format(type(value))
            raise TypeError(msg)
        if len(value) > 2:
            raise ValueError("Too many elements in extent, must be a maximum of two dimensions")

        if self.profile is not None:
            p_aspect = (self.shape[0]) / (self.shape[1])
            e_aspect = value[0] / value[1]
            if abs(e_aspect - p_aspect) > 0.0001:
                msg = "Extent aspect ratio doesn't match profile aspect ratio"
                raise ValueError(msg)
            else:
                self._extent = tuple(value)
                self._grid_spacing = value[0] / (self.shape[0])
        else:
            self._extent = tuple(value)
            self.dimensions = len(value)
            if self.grid_spacing is not None:
                self._shape = tuple([int(v / self.grid_spacing) for v in value])
                self._size = np.product(self._shape)
            if self._shape is not None:
                self._grid_spacing = self._extent[0] / self._shape[0]
                self._extent = tuple([sz * self._grid_spacing for sz in self._shape])
        return

    @extent.deleter
    def extent(self):
        self._extent = None
        self._grid_spacing = None
        if self.profile is None:
            self._shape = None
            self._size = None

    @property
    def shape(self):
        """The shape of the surface profile array, the number of points in each direction
        """
        return self._shape

    @shape.setter
    def shape(self, value: typing.Sequence[int]):
        if not isinstance(value, Sequence):
            raise ValueError(f"Shape should be a Sequence type, got: {type(value)}")

        if self._profile is not None:
            raise ValueError("Cannot set shape when profile is present")

        self._shape = tuple([int(x) for x in value])
        self._size = np.product(self._shape)
        if self.grid_spacing is not None:
            self._extent = tuple([v * self.grid_spacing for v in value])
        elif self.extent is not None:
            self._grid_spacing = self._extent[0] / self._shape[0]
            self._extent = tuple([sz * self.grid_spacing for sz in self.shape])

    @shape.deleter
    def shape(self):
        if self.profile is None:
            self._shape = None
            self._size = None
            self._extent = None
            self._grid_spacing = None
        else:
            msg = "Cannot delete shape with a surface profile set"
            raise ValueError(msg)

    @property
    def profile(self):
        """The height data for the surface profile
        """
        if self.invert_surface:
            return -1 * self._profile
        else:
            return self._profile

    @profile.setter
    def profile(self, value: np.ndarray):
        """Sets the profile property
        """
        if value is None:
            return

        try:
            self.unworn_profile = np.asarray(value, dtype=float).copy()
            self._profile = np.asarray(value, dtype=float).copy()
        except ValueError:
            msg = "Could not convert profile to array of floats, profile contains invalid values"
            raise ValueError(msg)

        self._shape = self._profile.shape
        self._size = self._profile.size
        self.dimensions = len(self._profile.shape)
        self.wear_volumes = dict()

        if self.grid_spacing is not None:
            self._extent = tuple([self.grid_spacing * p for p in self.shape])
        elif self.extent is not None:
            if self.dimensions == 1:
                self._grid_spacing = (self.extent[0] / self.shape[0])
            if self.dimensions == 2:
                e_aspect = self.extent[0] / self.extent[1]
                p_aspect = self.shape[0] / self.shape[1]

                if abs(e_aspect - p_aspect) < 0.0001:
                    self._grid_spacing = (self.extent[0] / self.shape[0])
                else:
                    warnings.warn("Global size does not match profile size,"
                                  " global size has been deleted")
                    self._extent = None

    @profile.deleter
    def profile(self):
        self.unworn_profile = None
        self._profile = None
        del self.shape
        del self.extent
        del self.mask
        self.wear_volumes = dict()
        self.is_discrete = False

    @property
    def grid_spacing(self):
        """The distance between grid points in the x and y directions
        """
        return self._grid_spacing

    @grid_spacing.setter
    def grid_spacing(self, grid_spacing: float):
        if grid_spacing is None:
            return

        if not isinstance(grid_spacing, float):
            try:
                # noinspection PyTypeChecker
                grid_spacing = float(grid_spacing)
            except ValueError:
                msg = ("Invalid type, grid spacing of type {} could not be "
                       "converted into float".format(type(grid_spacing)))
                raise ValueError(msg)

        if np.isinf(grid_spacing):
            msg = "Grid spacing must be finite"
            raise ValueError(msg)

        self._grid_spacing = grid_spacing

        if self.profile is None:
            if self.extent is not None:
                self._shape = tuple([int(sz / grid_spacing) for sz in self.extent])
                self._size = np.product(self._shape)
                self._extent = tuple([sz * grid_spacing for sz in self._shape])
            elif self.shape is not None:
                self._extent = tuple([grid_spacing * pt for pt in self.shape])
        else:
            self._extent = tuple([s * grid_spacing for s in self.shape])

    @grid_spacing.deleter
    def grid_spacing(self):
        self._extent = None
        self._grid_spacing = None
        if self.profile is None:
            del self.shape

    @property
    def material(self):
        """ A material object describing the properties of the surface """
        return self._material

    @material.setter
    def material(self, value):
        if isinstance(value, _MaterialABC):
            self._material = value
        else:
            raise ValueError("Unable to set material, expected material object"
                             " received %s" % str(type(value)))

    @material.deleter
    def material(self):
        self._material = None

    def wear(self, name: str, x_pts: np.ndarray, y_pts: np.ndarray, depth: np.ndarray):
        """
        Add wear / geometry changes to the surface profile

        Parameters
        ----------
        name: str
            Name of the source of wear
        x_pts: np.ndarray
            The x locations of the worn points in length units
        y_pts: np.ndarray
            The y locations of the worn points in length units
        depth: np.ndarray
            The depth to wear each point, negative values will add height

        """
        if not x_pts.size == y_pts.size == depth.size:
            raise ValueError(f"X, Y locations and wear depths are not the same size for wear '{name}':\n"
                             f"x:{x_pts.size}\n"
                             f"y:{y_pts.size}\n"
                             f"depth:{depth.size}")

        if np.any(np.isnan(depth)):
            raise ValueError(f"Some wear depth values are nan for wear {name}")

        if name not in self.wear_volumes:
            self.wear_volumes[name] = np.zeros_like(self._profile)

        # equivalent to rounding and applying wear to nearest node
        x_ind = np.array(x_pts / self.grid_spacing + self.grid_spacing/2, dtype=np.uint16)
        y_ind = np.array(y_pts / self.grid_spacing + self.grid_spacing/2, dtype=np.uint16)

        self.wear_volumes[name][y_ind, x_ind] += depth
        self._profile[y_ind, x_ind] -= depth
        self._inter_func = None  # force remaking the interpolator if the surface has been worn

    def get_fft(self, profile_in=None):
        """ Find the fourier transform of the surface

        Finds the fft of the surface and stores it in your_instance.fft

        Parameters
        ----------
        profile_in : array-like optional (None)
            If set the fft of profile_in will be found and returned otherwise
            instances profile attribute is used

        Returns
        -------
        transform : array
            The fft of the instance's profile or the profile_in if one is
            supplied

        See Also
        --------
        get_psd
        get_acf
        show

        Notes
        -----
        Uses numpy fft.fft or fft.fft2 depending on the shape of the profile


        Examples
        --------
        >>># Set the fft property of the surface
        >>> import slippy.surface as s
        >>> my_surface  = s.assurface([[1,2],[3,4]], grid_spacing=1)
        >>>my_surface.get_fft()

        >>># Return the fft of a provided profile
        >>>fft_of_profile_2=my_surface.get_fft(np.array([[1,2],[3,4]]))

        """
        if profile_in is None:
            profile = self.profile
        else:
            profile = profile_in
        try:
            if len(profile.shape) == 1:
                transform = np.fft.fft(profile)
                if type(profile_in) is bool:
                    self.fft = transform
            else:
                transform = np.fft.fft2(profile)
                if type(profile_in) is bool:
                    self.fft = transform
        except AttributeError:
            raise AttributeError('Surface must have a defined profile for fft'
                                 ' to be used')
        if profile_in is None:
            self.fft = transform
        else:
            return transform

    def get_acf(self, profile_in=None):
        """ Find the auto correlation function of the surface

        Finds the ACF of the surface and stores it in your_instance.acf

        Parameters
        ----------
        profile_in : array-like optional (None)

        Returns
        -------
        output : ACF object
            An acf object with the acf data stored, the values can be extracted
            by numpy.array(output)

        See Also
        --------
        get_psd
        get_fft
        show
        slippy.surface.ACF

        Notes
        -----
        ACF data is kept in ACF objects, these can then be interpolated or
        evaluated at specific points with a call:


        Examples
        --------
        >>> import slippy.surface as s
        >>> my_surface  = s.assurface([[1,2],[3,4]], grid_spacing=1)
        >>> # Sets the acf property of the surface with an ACF object
        >>> my_surface.get_acf()
        >>> # The acf values are then given by the following
        >>> np.array(my_surface.acf)
        >>> # The acf can be shown using the show function:
        >>> my_surface.show('acf', 'image')

        >>> # Finding the ACF of a provided profile:
        >>> ACF_object_for_profile_2=my_surface.get_acf(np.array([[4, 3], [2, 1]]))
        >>> # equivalent to ACF(profile_2)

        """

        if profile_in is None:
            # noinspection PyTypeChecker
            self.acf = ACF(self)
        else:
            profile = np.asarray(profile_in)
            # noinspection PyTypeChecker
            output = np.array(ACF(profile))
            return output

    def get_psd(self):
        """ Find the power spectral density of the surface

        Finds the PSD of the surface and stores it in your_instance.psd

        Parameters
        ----------

        (None)

        Returns
        -------

        (None), sets the psd attribute of the instance

        See Also
        --------
        get_fft
        get_acf
        show

        Notes
        -----
        Finds the psd by fourier transforming the ACF, in doing so looks for
        the instance's acf property. If this is not found the acf is calculated
        and set.

        Examples
        --------

        >>> # sets the psd attribute of my_surface
        >>> import slippy.surface as s
        >>> my_surface  = s.assurface([[1,2],[3,4]], grid_spacing=1)
        >>> my_surface.get_psd()

        """
        # PSD is the fft of the ACF (https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density)
        if self.acf is None:
            self.get_acf()
        # noinspection PyTypeChecker
        self.psd = self.get_fft(np.asarray(self.acf))

    def subtract_polynomial(self, order, mask=None):
        """ Flatten the surface by subtracting a polynomial

        Alias for :func:`~slippy.surface.subtract_polynomial` function


        """
        if mask is None:
            mask = self.mask

        new_profile, coefs = subtract_polynomial(self.profile, order, mask)
        self.profile = new_profile

        return coefs

    def roughness(self, parameter_name, mask=None, curved_surface=False,
                  no_flattening=False, filter_cut_off=None,
                  four_nearest=False):
        """ Find areal roughness parameters

        Alias for :func:`~slippy.surface.roughness` function

        """
        if mask is None:
            mask = self.mask

        out = roughness(self, parameter_name, mask=mask,
                        curved_surface=curved_surface,
                        no_flattening=no_flattening,
                        filter_cut_off=filter_cut_off,
                        four_nearest=four_nearest)
        return out

    def get_mat_vr(self, height, void=False, mask=None, ratio=True):
        """ Find the material or void volume ratio for a given height

        Alias for :func:`~slippy.surface.get_mat_vr` function

        """
        if mask is None:
            mask = self.mask

        return get_mat_vr(height, profile=self.profile, void=void, mask=mask,
                          ratio=ratio)

    def get_height_of_mat_vr(self, ratio, void=False, mask=None,
                             accuracy=0.001):
        """ Find the height of a given material or void volume ratio

        Alias for :func:`~slippy.surface.get_height_of_mat_vr` function

        """
        if mask is None:
            mask = self.mask

        return get_height_of_mat_vr(ratio, self.profile, void=void, mask=mask,
                                    accuracy=accuracy)

    def get_summit_curvature(self, summits=None, mask=None,
                             filter_cut_off=None, four_nearest=False):
        """ Get summit curvatures

        Alias for :func:`~slippy.surface.get_summit_curvature` function

        """
        if mask is None:
            mask = self.mask

        return get_summit_curvatures(self.profile, summits=summits, mask=mask,
                                     filter_cut_off=filter_cut_off,
                                     four_nearest=four_nearest, grid_spacing=self.grid_spacing)

    def find_summits(self, mask=None, four_nearest=False, filter_cut_off=None,
                     invert=False):
        """ Find summits after low pass filtering

        Alias for :func:`~slippy.surface.find_summits` function

        """
        if mask is None:
            mask = self.mask

        if invert:
            return find_summits(self.profile * -1,
                                grid_spacing=self.grid_spacing, mask=mask,
                                four_nearest=four_nearest,
                                filter_cut_off=filter_cut_off)
        else:
            return find_summits(self, mask=mask, four_nearest=four_nearest,
                                filter_cut_off=filter_cut_off)

    def low_pass_filter(self, cut_off_freq, return_copy=False):
        """ Low pass FIR filter the surface profile

        Alias for :func:`~slippy.surface.low_pass_filter` function

        """
        if return_copy:
            return low_pass_filter(self, cut_off_freq)
        else:
            self.profile = low_pass_filter(self, cut_off_freq)

    def resample(self, new_grid_spacing=None, return_profile=False, remake_interpolator=False):
        """ Resample or crop the profile by interpolation

        Parameters
        ----------

        new_grid_spacing : float, optional (None)
            The grid spacing on the new surface, if the grid_spacing is not set on the current surface it is assumed to
            be 1
        return_profile : bool, optional (False)
            If true the interpolated profile is returned otherwise it is set as the profile of the instance
        remake_interpolator : bool, optional (False)
            If true any memoized interpolator will be deleted and remade based on the current profile before
            interpolation, see notes.

        Returns
        -------

        new_profile : array
            If return_profile is True the interpolated profile is returned

        See Also
        --------
        rotate
        fill_holes
        surface_like

        Notes
        -----
        On the first call this function will make an interpolator object which
        is used to interpolate, on subsequent calls this object is found and
        used resulting in no loss of quality. If the remake_interpolator key
        word is set to true this interpolator is remade. This will result in a
        loss of quality for subsequent calls but is necessary if the profile
        property has changed.

        This method does not support masking.

        The profile should have nan or inf values removed by the fill_holes
        method before running this

        Examples
        --------

        >>> import numpy as np
        >>> import slippy.surface as s
        >>> profile=np.random.normal(size=(101,101))
        >>> my_surface=s.assurface(profile, grid_spacing=1)
        >>> # interpolate on a coarse grid:
        >>> my_surface.resample(10)
        >>> # check shape:
        >>> my_surface.shape
        (11,11)
        >>> # restore original profile:
        >>> my_surface.resample(1)
        >>> my_surface.shape
        (101,101)
        """
        gs_changed = False
        if self.grid_spacing is None:
            gs_changed = True
            self.grid_spacing = 1

        if remake_interpolator or self._inter_func is None:
            self._original_extent = self.extent
            x0 = np.arange(0, self.extent[0], self.grid_spacing)
            y0 = np.arange(0, self.extent[1], self.grid_spacing)
            self._inter_func = scipy.interpolate.RectBivariateSpline(x0, y0, self.profile)
        x1 = np.arange(0, self._original_extent[0], new_grid_spacing)
        y1 = np.arange(0, self._original_extent[1], new_grid_spacing)
        new_profile = self._inter_func(x1, y1)

        if gs_changed:
            del self.grid_spacing

        if return_profile:
            return new_profile
        else:
            self.profile = new_profile
            if not gs_changed:
                self.grid_spacing = new_grid_spacing

    def __add__(self, other):
        if not isinstance(other, _Surface):
            return Surface(profile=self.profile + other, grid_spacing=self.grid_spacing)

        if self.grid_spacing is not None and other.grid_spacing is not None and self.grid_spacing != other.grid_spacing:
            if self.grid_spacing < other.grid_spacing:
                prof_2 = other.resample(self.grid_spacing, return_profile=True)
                prof_1 = self.profile
                new_gs = self.grid_spacing
            else:
                prof_1 = self.resample(other.grid_spacing, return_profile=True)
                prof_2 = other.profile
                new_gs = other.grid_spacing
        else:
            prof_1 = self.profile
            prof_2 = other.profile
            if self.grid_spacing is not None:
                new_gs = self.grid_spacing
            else:
                new_gs = other.grid_spacing

        new_shape = [min(p1s, p2s) for p1s, p2s in zip(prof_1.shape, prof_2.shape)]
        new_profile = prof_1[0:new_shape[0], 0:new_shape[1]] + prof_2[0:new_shape[0], 0:new_shape[1]]
        return Surface(profile=new_profile, grid_spacing=new_gs)

    def __mul__(self, other):
        if isinstance(other, Number):
            return Surface(profile=self.profile*other, grid_spacing=self.grid_spacing)
        else:
            raise NotImplementedError("Multiplication not implement for Surfaces unless other parameter is number")

    def __div__(self, other):
        if isinstance(other, Number):
            return Surface(profile=self.profile/other, grid_spacing=self.grid_spacing)
        else:
            raise NotImplementedError("Division not implement for Surfaces unless other parameter is number")

    def __sub__(self, other):
        if not isinstance(other, _Surface):
            return Surface(profile=self.profile + other, grid_spacing=self.grid_spacing)

        if self.grid_spacing is not None and other.grid_spacing is not None and self.grid_spacing != other.grid_spacing:
            if self.grid_spacing < other.grid_spacing:
                prof_2 = other.resample(self.grid_spacing, return_profile=True)
                prof_1 = self.profile
                new_gs = self.grid_spacing
            else:
                prof_1 = self.resample(other.grid_spacing, return_profile=True)
                prof_2 = other.profile
                new_gs = other.grid_spacing
        else:
            prof_1 = self.profile
            prof_2 = other.profile
            if self.grid_spacing is not None:
                new_gs = self.grid_spacing
            else:
                new_gs = other.grid_spacing

        new_shape = [min(p1s, p2s) for p1s, p2s in zip(prof_1.shape, prof_2.shape)]
        new_profile = prof_1[0:new_shape[0], 0:new_shape[1]] - prof_2[0:new_shape[0], 0:new_shape[1]]
        return Surface(profile=new_profile, grid_spacing=new_gs)

    def __eq__(self, other):
        if not isinstance(other, _Surface) or self.is_discrete != other.is_discrete:
            return False
        if self.is_discrete:
            return np.array_equal(self.profile, other.profile) and self.grid_spacing == other.grid_spacing
        else:
            return repr(self) == repr(other)

    def show(self, property_to_plot: typing.Union[str, typing.Sequence[str]] = 'profile',
             plot_type: typing.Union[str, typing.Sequence[str]] = 'default', ax=False, *, dist=None, stride=None,
             **figure_kwargs):
        """ Plot surface properties

        Parameters
        ----------
        property_to_plot : str or list of str length N optional ('profile')
            The property to be plotted see notes for supported names
        plot_type : str or list of str length N optional ('default')
            The type of plot to be produced, see notes for supported types
        ax : matplotlib axes or False optional (False)
            If supplied the plot will be added to the axis
        dist : a scipy probability distribution, optional (None)
            Only used if probplot is requested, the probability distribution
            to plot against
        stride : float, optional (None)
            Only used if a wire frame plot is requested, the stride between
            wires
        figure_kwargs : optional (None)
            Keyword arguments sent to the figure function in matplotlib

        Returns
        -------
        ax : matplotlib axes or list of matplotlib axes length N
            The axis with the plot

        See Also
        --------
        get_fft
        get_psd
        get_acf
        ACF

        Notes
        -----
        If fft, psd or acf are requested the field of the surface is filled
        by the relevant get_ method before plotting.

        The grid spacing attribute should be set before plotting

        2D and 1D plots can be produced. 2D properties are:

            - profile         - surface profile
            - unworn_profile  - the surface profile with no wear applied
            - fft2d           - fft of the surface profile
            - psd             - power spectral density of the surface profile
            - acf             - auto correlation function of the surface
            - apsd            - angular power spectral density of the profile

        Plot types allowed for 2D plots are:

            - surface (default)
            - image
            - mesh

        If a mesh plot is requested the distance between lines in the mesh can
        be specified with the stride keyword

        1D properties are:

            - histogram - histogram of the profile heights
            - fft1d     - 1 dimentional fft of the surface
            - qq        - quartile quartile plot of the surface heights

        If qq or dist hist are requested the distribution to be plotted against
        the height values can be specified by the dist keyword

        Each of the 1D properties can only be plotted on it's default plot type

        Examples
        --------
        >>> # show the surface profile as an image:
        >>> import slippy.surface as s
        >>> import numpy as np
        >>> my_surface=s.assurface(np.random.rand(10,10))
        >>> my_surface.show('profile', 'image')

        >>> # show the 2D fft of the surface profile with a range of plot types
        >>> my_surface.show(['fft2D','fft2D','fft2D'], ['mesh', 'image', 'default'])

        """
        import matplotlib.pyplot as plt
        # noinspection PyUnresolvedReferences
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from scipy.stats import probplot

        if self.profile is None:
            raise AttributeError('The profile of the surface must be set before it can be shown')

        types2d = ['profile', 'fft2d', 'psd', 'acf', 'apsd', 'unworn_profile']
        types1d = ['histogram', 'fft1d', 'qq', 'hist']

        # using a recursive call to deal with multiple plots on the same fig
        if isinstance(property_to_plot, Sequence) and not isinstance(property_to_plot, str):
            number_of_subplots = len(property_to_plot)
            if not type(ax) is bool:
                msg = ("Can't plot multiple plots on single axis, "
                       'making new figure')
                warnings.warn(msg)
            if isinstance(plot_type, Sequence) and not isinstance(plot_type, str):
                plot_type = list(plot_type)
                if len(plot_type) < number_of_subplots:
                    plot_type.extend(['default'] * (number_of_subplots - len(plot_type)))
            else:
                plot_type = [plot_type, ] * number_of_subplots
            # 11, 12, 13, 22, then filling up rows of 3 (unlikely to be used)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            if len(property_to_plot) < 5:
                n_cols = [1, 2, 3, 2][number_of_subplots - 1]
            else:
                n_cols = 3
            n_rows = int(np.ceil(number_of_subplots / 3))
            fig = plt.figure(**figure_kwargs)
            ax = []
            sub_plot_number = 100 * n_rows + 10 * n_cols + 1
            for i in range(number_of_subplots):
                if property_to_plot[i].lower() in types2d and not plot_type[i] in ('image', 'default'):
                    ax.append(fig.add_subplot(sub_plot_number + i, projection='3d'))
                else:
                    ax.append(fig.add_subplot(sub_plot_number + i))
                self.show(property_to_plot[i], plot_type[i], ax[i])
            return fig, ax
        #######################################################################
        # main method
        #######################################################################
        # 2D plots
        try:
            property_to_plot = property_to_plot.lower()
        except AttributeError:
            msg = "Property to plot must be a string or a list of strings"
            raise ValueError(msg)

        if not (property_to_plot in types2d or property_to_plot in types1d):
            msg = ('Unsupported property to plot see documentation for details'
                   ', type given: \n' + str(property_to_plot) + ' \nsupported ty'
                                                                'pes: \n' + ' '.join(types2d + types1d))
            raise ValueError(msg)

        if not ax:
            fig = plt.figure(**figure_kwargs)

        if property_to_plot in types2d:
            if not ax and (plot_type == 'image' or plot_type == 'default'):
                # noinspection PyUnboundLocalVariable
                ax = fig.add_subplot(111)
            elif not ax:
                # noinspection PyUnboundLocalVariable
                ax = fig.add_subplot(111, projection='3d')

            if property_to_plot == 'profile':
                labels = ['Surface profile', 'x', 'y', 'Height']
                x = self.grid_spacing * np.arange(self.shape[0])
                y = self.grid_spacing * np.arange(self.shape[1])
                z = self.profile

            elif property_to_plot == 'unworn_profile':
                labels = ['Surface profile (unworn)', 'x', 'y', 'Height']
                x = self.grid_spacing * np.arange(self.shape[0])
                y = self.grid_spacing * np.arange(self.shape[1])
                z = self.unworn_profile

            elif property_to_plot == 'fft2d':
                labels = ['Fourier transform of surface', 'u', 'v', '|F(x)|']
                if self.fft is None:
                    self.get_fft()
                z = np.abs(np.fft.fftshift(self.fft))
                x = np.fft.fftfreq(self.shape[0], self.grid_spacing)
                y = np.fft.fftfreq(self.shape[1], self.grid_spacing)

            elif property_to_plot == 'psd':
                labels = ['Power spectral density', 'u', 'v', 'Power/ frequency']
                if self.psd is None:
                    self.get_psd()
                # noinspection PyTypeChecker
                z = np.log(np.abs(np.fft.fftshift(self.psd)))
                x = np.fft.fftfreq(self.shape[0], self.grid_spacing)
                y = np.fft.fftfreq(self.shape[1], self.grid_spacing)

            elif property_to_plot == 'acf':
                labels = ['Auto correlation function', 'x', 'y',
                          'Surface auto correlation']
                if self.acf is None:
                    self.get_acf()
                # noinspection PyTypeChecker
                z = np.abs(np.asarray(self.acf))
                x = self.grid_spacing * np.arange(self.shape[0])
                y = self.grid_spacing * np.arange(self.shape[1])
                x = x - max(x) / 2
                y = y - max(y) / 2

            elif property_to_plot == 'apsd':
                labels = ['Angular power spectral density', 'x', 'y']
                if self.fft is None:
                    self.get_fft()
                p_area = (self.shape[0] - 1) * (self.shape[1] - 1) * self.grid_spacing ** 2
                z = self.fft * np.conj(self.fft) / p_area
                x = self.grid_spacing * np.arange(self.shape[0])
                y = self.grid_spacing * np.arange(self.shape[1])
                x = x - max(x) / 2
                y = y - max(y) / 2
            else:
                raise ValueError("Property not recognised")

            mesh_x, mesh_y = np.meshgrid(x, y)

            if plot_type == 'surface':
                ax.plot_surface(mesh_x, mesh_y, np.transpose(z))
                # plt.axis('equal')
                ax.set_zlabel(labels[3])
            elif plot_type == 'mesh':
                if property_to_plot == 'psd' or property_to_plot == 'fft2d':
                    mesh_x, mesh_y = np.fft.fftshift(mesh_x), np.fft.fftshift(mesh_y)
                if stride:
                    ax.plot_wireframe(mesh_x, mesh_y, np.transpose(z), rstride=stride,
                                      cstride=stride)
                else:
                    ax.plot_wireframe(mesh_x, mesh_y, np.transpose(z), rstride=25,
                                      cstride=25)
                ax.set_zlabel(labels[3])
            elif plot_type == 'default' or plot_type == 'image':
                ax.imshow(z, extent=[min(y), max(y), min(x), max(x)], aspect=1)
            else:
                ValueError('Unrecognised plot type')

            ax.set_title(labels[0])
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[2])
            return ax

        #######################################################################
        # 1D plots
        #######################################################################

        elif property_to_plot in types1d:
            if not ax:
                # noinspection PyUnboundLocalVariable
                ax = fig.add_subplot(111)

            if property_to_plot == 'histogram' or property_to_plot == 'hist':
                # do all plotting in this loop for 1D plots
                labels = ['Histogram of surface heights', 'height', 'counts']
                ax.hist(self.profile.flatten(), 100)

            elif property_to_plot == 'fft1d':
                if self.dimensions == 1:
                    labels = ['FFt of surface', 'frequency', '|F(x)|']

                    if type(self.fft) is bool:
                        self.get_fft()
                    x = np.fft.fftfreq(self.shape[0], self.grid_spacing)
                    y = np.abs(self.fft / self.shape[0])
                    # line plot for 1d surfaces
                    ax.plot(x, y)
                    ax.xlim(0, max(x))
                else:
                    labels = ['Scatter of frequency magnitudes',
                              'frequency', '|F(x)|']
                    u = np.fft.fftfreq(self.shape[0], self.grid_spacing)
                    v = np.fft.fftfreq(self.shape[1], self.grid_spacing)
                    u_mesh, v_mesh = np.meshgrid(u, v)
                    frequencies = u_mesh + v_mesh
                    if type(self.fft) is bool:
                        self.get_fft()
                    mags = np.abs(self.fft)
                    # scatter plot for 2d frequencies
                    ax.scatter(frequencies.flatten(), mags.flatten(), 0.5, None, 'x')
                    ax.set_xlim(0, max(frequencies.flatten()))
                    ax.set_ylim(0, max(mags.flatten()))
            elif property_to_plot == 'qq':

                labels = ['Probability plot', 'Theoretical quantities',
                          'Ordered values']
                if dist:
                    probplot(self.profile.flatten(), dist=dist, fit=True,
                             plot=ax)
                else:
                    probplot(self.profile.flatten(), fit=True, plot=ax)
            else:
                raise ValueError(f"Property to plot {property_to_plot}, not recognised.")

            ax.set_title(labels[0])
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[2])
            return ax
        #######################################################################
        #######################################################################

    def __array__(self):
        return np.asarray(self.profile)

    @abc.abstractmethod
    def __repr__(self):
        return "Surface(profile=" + self.profile.__repr__() + ", grid_spacing=" + self.grid_spacing.__repr__() + ")"

    def get_points_from_extent(self, extent=None, grid_spacing=None, shape=None):
        """
        Gets the grid points from the extent and the grid spacing

        Returns
        -------
        mesh_x, mesh_y : np.ndarray
            arrays of the grid points (result from mesh grid)
        """
        if extent is None and grid_spacing is None and shape is None:
            if self.grid_spacing is None or self.extent is None:
                raise AttributeError('Grid points cannot be found until the surface is fully defined, the grid spacing '
                                     'and extent must be findable.')

            # I know this looks stupid, using arrange will give the wrong number of elements because of rounding error
            x = np.linspace(0, self.grid_spacing*(self.shape[1]-1), self.shape[1])
            y = np.linspace(0, self.grid_spacing*(self.shape[0]-1), self.shape[0])

            mesh_x, mesh_y = np.meshgrid(x, y)

        else:
            dum = Surface(grid_spacing=grid_spacing, shape=shape, extent=extent)
            try:
                mesh_y, mesh_x = dum.get_points_from_extent()
            except AttributeError:
                raise ValueError('Exactly two parameters must be supplied')

        return mesh_y, mesh_x

    def mesh(self, depth, method='grid', parameters=None):
        """
        Returns a Mesh object for the surface

        Equivalent to Mesh(surface)

        Parameters
        ----------

        # TODO
        """
        pass
        # raise NotImplementedError("No mesh yet, Sorry!")
        # if not self.is_discrete:
        #     raise ValueError("Surface must be discrete before meshing")

    def interpolate(self, y_points: np.ndarray, x_points: np.ndarray, mode: str = 'nearest',
                    remake_interpolator: bool = False):
        """
        Easy memoized interpolation on surface objects

        Parameters
        ----------
        x_points: np.ndarray
            N by M array of x points, in the same units as the grid spacing
        y_points: np.ndarray
            N by M array of y points, in the same units as the grid spacing
        mode: str {'nearest', 'linear', 'cubic'}, optional ('nearest')
            The mode of the interpolation
        remake_interpolator: bool, optional (False)
            If True the interpolator function will be remade, otherwise the existing one will be used, if no
            interpolator function is found it will be made automatically

        Returns
        -------
        sub_profile: np.ndarray
            The surface heights at the grid points requested, same shape as x_points and y_points
        """
        assert (x_points.shape == y_points.shape)

        if mode == 'nearest':
            x_index = np.array(x_points / self.grid_spacing, dtype='int32')
            y_index = np.array(y_points / self.grid_spacing, dtype='int32')
            return np.reshape(self.profile[y_index, x_index], newshape=x_points.shape)
        elif mode == 'linear':
            if remake_interpolator or self._inter_func is None or self._inter_func.degrees != (1, 1):
                x0 = np.arange(0, self.extent[0], self.grid_spacing)
                y0 = np.arange(0, self.extent[1], self.grid_spacing)
                self._inter_func = scipy.interpolate.RectBivariateSpline(x0, y0, self.profile, kx=1, ky=1)
        elif mode == 'cubic':
            if remake_interpolator or self._inter_func is None or self._inter_func.degrees != (3, 3):
                x0 = np.arange(0, self.extent[0], self.grid_spacing)
                y0 = np.arange(0, self.extent[1], self.grid_spacing)
                self._inter_func = scipy.interpolate.RectBivariateSpline(x0, y0, self.profile, kx=3, ky=3)
        else:
            raise ValueError(f'{mode} is not a recognised mode for the interpolation function')

        return self._inter_func(x_points, y_points, grid=False)


class Surface(_Surface):
    """ Object for reading, manipulating and plotting surfaces

        The Surface class contains methods for setting properties,
        examining measures of roughness and descriptions of surfaces, plotting,
        fixing and editing surfaces.

        Parameters
        ----------
        profile: np.ndarray, optional (None)
            The height profile of the surface, the units should be the same as used for the grid spacing parameter
        grid_spacing: float, optional (None)
            The distance between the grid points in the surface profile
        shape: tuple, optional (None)
            The number of grid points in the surface in each direction, should not be set if a profile is given
        extent: tuple, optional (None)
            The total extent of the surface in the same units as the grid spacing, either this or the grid spacing can
            be set if a profile is given (either as the profile argument or from a file)
        file_name: str, optional (None)
            The full path including the file extension to a supported file type, supported types are .txt, .csv, .al3d,
            .mat
        csv_delimiter: str, optional (None)
            The delimiter used in the .csv or .txt file, only used if the file name is given and the file is a .txt or
            .csv file
        csv_dialect: {csv.Dialect, str), optional ('sniff')
            The dialect used to read the csv file, only used if a file is supplied and the file is a csv file, defaults
            to 'sniff' meaning that the csv. sniffer will be used.
        csv_sniffer_n_bytes: int, optional (2048)
            The number of bytes used by the csv sniffer, only used if 'sniff' is given as the dialect and a csv file is
            given as the file name
        mat_profile_name: str, optional ('profile')
            The name of the profile variable in the .mat file, only used if the file_name is given and the file is a
            .mat file
        mat_grid_spacing_name: str, optional (None)
            The name of the grid_spacing variable in the .mat file, only used if the file_name is given and the file is
            a .mat file. If unset the grid_spacing property is not read from the file.

        See Also
        --------
        ACF
        roughness

        Notes
        -----
        Roughness functions are aliased from the functions provided in the surface
        module

        Examples
        --------


        """

    def rotate(self, radians):
        raise NotImplementedError("Cannot rotate this surface")

    surface_type = 'Experimental'

    def __init__(self, profile: typing.Optional[np.ndarray] = None, grid_spacing: typing.Optional[float] = None,
                 shape: typing.Optional[tuple] = None, extent: typing.Optional[tuple] = None,
                 file_name: typing.Optional[str] = None,
                 mat_profile_name: typing.Optional[str] = None, mat_grid_spacing_name: typing.Optional[str] = None,
                 csv_delimiter: str = None, csv_dialect: typing.Union[csv.Dialect, str] = 'sniff',
                 csv_sniffer_n_bytes: int = 2048):

        if profile is not None or file_name is not None:
            if shape is not None:
                raise ValueError("The shape cannot be set if the profile is also set, please set either the "
                                 "grid_spacing or the extent only")
            if grid_spacing is not None and extent is not None:
                raise ValueError("Either the grid_spacing or the extent should be set with a profile, not both")
            self.profile = profile

        if file_name is not None:
            if profile is not None:
                raise ValueError("The profile and a file name cannot be set")
            file_ext = os.path.splitext(file_name)[1]
            if file_ext == '.mat':
                self.read_mat(file_name, mat_profile_name, mat_grid_spacing_name)
            elif file_ext == '.al3d':
                self.read_al3d(file_name)
            elif file_ext == '.txt' or file_ext == '.csv':
                self.read_csv(file_name, delimiter=csv_delimiter, dialect=csv_dialect, sniff_bytes=csv_sniffer_n_bytes)
            # read file replace profile

        super().__init__(grid_spacing=grid_spacing, extent=extent, shape=shape, is_discrete=True)

    def read_al3d(self, file_name: str, return_data: bool = False):
        """
        Reads an alicona al3d file and sets the profile and grid_spacing property of the surface

        Parameters
        ----------
        file_name: str
            The full path including the extension of the .al3d file
        return_data: bool, optional (False)
            If True the data from the al3d file is returned as a dict

        Returns
        -------
        data: dict
            data read from the al3d file, only returned if return_data is set to True
        """
        from .alicona import alicona_read
        data = alicona_read(file_name)
        self.profile = data['DepthData']
        self.grid_spacing = data['Header']['PixelSizeXMeter']
        if return_data:
            return data

    def read_csv(self, file_name: str, delimiter: str = None, return_profile: bool = False,
                 dialect: typing.Union[csv.Dialect, str] = 'sniff', sniff_bytes: int = 2048):
        """
        Read a profile from a csv or txt file, header lines are automatically skipped

        Parameters
        ----------
        file_name: str
            The full path to the .txt or .csv file including the file extension
        delimiter: str, optional (None)
            The delimiter used in by csv reader
        return_profile: bool, optional (False)
            If true the profile will be returned
        dialect: {csv.Dialect, str}, optional ('sniff')
            A csv dialect object or 'sniff' if the dialect is to be found by the csv sniffer
        sniff_bytes: int, optional (2048)
            The number of bytes read from the file for the csv.Sniffer, only used if the delimiter is 'sniff'
        """

        with open(file_name) as file:
            if delimiter is not None:
                reader = csv.reader(file, delimiter=delimiter)
            else:
                if dialect == 'sniff':
                    dialect = csv.Sniffer().sniff(file.read(sniff_bytes))
                    file.seek(0)
                reader = csv.reader(file, dialect=dialect)
            profile = []
            for row in reader:
                if row:
                    if type(row[0]) is float:
                        profile.append(row)
                    else:
                        if len(row) == 1:
                            try:
                                row = [float(x) for x in row[0].split()
                                       if not x == '']
                                profile.append(row)
                            except ValueError:
                                pass
                        else:
                            try:
                                row = [float(x) for x in row if not x == '']
                                profile.append(row)
                            except ValueError:
                                pass
        if return_profile:
            return np.array(profile, dtype=float)
        self.profile = profile

    def read_mat(self, path: str, profile_name: str = 'profile', grid_spacing_name: str = None):
        """ Reads .mat files as surfaces

        Parameters
        ----------
        path : str
            full path including file name to a .mat file
        profile_name : srt, optional ('profile')
            The name of the profile variable in the .mat file
        grid_spacing_name : str, optional (None)
            The name of the grid_spacing variable in the .mat file, if set to none the grid spacing variable is not set

        Notes
        -----
        This method will search the .mat file for the given keys. If no keys
        are given, and the .mat file contains variables called grid_spacing or
        profile these are set as the relevant attributes. Otherwise, if the
        .mat file only contains one variable this is set as the profile.
        If none of the above are true, or if the given keys are not found
        an error is raised

        """
        from scipy.io import loadmat
        # load file
        mat = loadmat(path)
        keys = [key for key in mat if not key.startswith('_')]

        if grid_spacing_name is not None:
            try:
                self.grid_spacing = mat[grid_spacing_name]
            except KeyError:
                msg = ("Name {} not found in .mat file,".format(grid_spacing_name) +
                       " names found were: ".join(keys))
                raise ValueError(msg)

        try:
            self.profile = mat[profile_name]
        except KeyError:
            msg = ("Name {} not found in .mat file,".format(profile_name) +
                   " names found were: ".join(keys))
            raise ValueError(msg)

    def fill_holes(self, hole_value='auto', mk_copy=False, remove_boarder=True,
                   b_thresh=0.99):
        """ Replaces specified values with filler

        Removes boarder then uses biharmonic equations algorithm to fill holes

        Parameters
        ----------
        hole_value: {'auto' or float}
            The value to be replaced, 'auto' replaces all -inf, inf and nan
            values
        mk_copy : bool
            if set to true a new surface object will be returned with the holes
            filled otherwise the profile property of the current surface is
            updated
        remove_boarder : bool
            Defaults to true, removes the boarder from the image until the
            first row and column that have
        b_thresh : float
            (0>, <=1) If the boarder is removed, the removal will continue until the row
            or column to be removed contains at least this proportion of real values

        Returns
        -------
        If mk_copy is true a new surface object with holes filled else resets
        profile property of the instance and returns nothing

        See Also
        --------
        skimage.restoration.inpaint.inpaint_biharmonic

        Notes
        -----
        When alicona images are imported the invalid pixel value is
        automatically set to nan so this will work in auto mode

        Holes are filled with bi harmonic equations

        Examples
        --------

        >>> import slippy.surface as s
        >>> # make a dummy profile
        >>> x=np.arange(12, dtype=float)
        >>> X,_=np.meshgrid(x,x)
        >>> # pad with nan values
        >>> X2=np.pad(X,2,'constant', constant_values=float('nan'))
        >>> # add hole to centre
        >>> X2[6,6]=float('nan')
        >>> # make surface
        >>> my_surface=s.Surface(profile=X2)
        >>> my_surface.fill_holes()
        >>> my_surface.profile[6,6]
        6.0
        """
        from skimage.restoration import inpaint

        profile = self.profile

        if hole_value == 'auto':
            holes = np.logical_or(np.isnan(profile), np.isinf(profile))
        else:
            holes = profile == hole_value
        if sum(sum(holes)) == 0:
            warnings.warn('No holes detected')

        profile[holes] = 0

        if remove_boarder:
            # find rows
            good = [False] * 4

            start_r = 0
            end_r = None  # len(profile)
            start_c = 0
            end_c = None  # len(profile[0])

            # iterate ove removing cols and rows if they have too many holes
            while not all(good):
                if np.mean(holes[start_r, start_c:end_c]) > b_thresh:
                    start_r += 1
                else:
                    good[0] = True

                if np.mean(holes[-1 if end_r is None else end_r - 1, start_c:end_c]) > b_thresh:
                    end_r = -1 if end_r is None else end_r - 1
                else:
                    good[1] = True

                if np.mean(holes[start_r:end_r, start_c]) > b_thresh:
                    start_c += 1
                else:
                    good[2] = True

                if np.mean(holes[start_r:end_r, -1 if end_c is None else end_c - 1]) > b_thresh:
                    end_c = -1 if end_c is None else end_c - 1
                else:
                    good[3] = True

            # add back in if they are ok
            while any(good):
                if start_r > 0 and not np.mean(holes[start_r - 1, start_c:end_c]) > b_thresh:
                    start_r -= 1
                else:
                    good[0] = False

                if end_r is not None and not np.mean(holes[end_r, start_c:end_c]) > b_thresh:
                    end_r = end_r + 1 if end_r + 1 < 0 else None
                else:
                    good[1] = False

                if start_c > 0 and not np.mean(holes[start_r:end_r, start_c - 1]) > b_thresh:
                    start_c -= 1
                else:
                    good[2] = False

                if end_c is not None and not np.mean(holes[start_r:end_r, end_c]) > b_thresh:
                    end_c = end_c + 1 if end_c + 1 < 0 else None
                else:
                    good[3] = False

            profile = profile[start_r:end_r, start_c:end_c]
            holes = holes[start_r:end_r, start_c:end_c]

        profile_out = inpaint.inpaint_biharmonic(profile, holes,
                                                 multichannel=False)

        if mk_copy:
            new_surf = Surface(profile=profile_out, grid_spacing=self.grid_spacing)
            return new_surf
        else:
            self.profile = profile_out

    def __repr__(self):
        string = ''
        if self.profile is not None:
            string += 'profile = ' + repr(self.profile) + ', '
        elif self.shape is not None:
            string += 'shape = ' + repr(self.shape) + ', '
        if self.grid_spacing is not None:
            string += 'grid_spacing = ' + repr(self.grid_spacing) + ', '
        if self.material is not None:
            string += 'material = ' + repr(self.material) + ', '
        if self.mask is not None:
            string += 'mask = ' + repr(self.mask) + ', '
        string = string[:-2]

        return 'Surface(' + string + ')'


class _AnalyticalSurface(_Surface):
    """
    A abstract base class for analytical surfaces, to extend the height and __repr__ methods must be overwritten
    """
    _total_shift: tuple = (0, 0)
    _total_rotation: float = 0
    is_analytic = True
    _analytic_subclass_registry = []
    is_discrete = False

    def __init__(self, generate: bool = False, rotation: Number = None,
                 shift: typing.Union[str, tuple] = None,
                 grid_spacing: float = None, extent: tuple = None, shape: tuple = None):
        super().__init__(grid_spacing=grid_spacing, extent=extent, shape=shape)
        if rotation is not None:
            self.rotate(rotation)

        self.shift(shift)

        if generate:
            self.discretise()

    def discretise(self):
        if self.is_discrete:
            msg = ('Surface is already discrete this will overwrite surface'
                   ' profile')
            warnings.warn(msg)
        if self.grid_spacing is None:
            msg = 'A grid spacing must be provided before discretisation'
            raise AttributeError(msg)

        if self.extent is None:
            msg = 'The extent or the shape of the surface must be set before discretisation'
            raise AttributeError(msg)
        if self.size > 10E7:
            warnings.warn('surface contains over 10^7 points calculations will'
                          ' be slow, consider splitting surface for analysis')

        x_mesh, y_mesh = self.get_points_from_extent()
        self.is_discrete = True
        self.profile = self.height(x_mesh, y_mesh)

    @abc.abstractmethod
    def _height(self, x_mesh, y_mesh):
        pass

    def height(self, x_mesh: typing.Union[np.ndarray, Number], y_mesh: typing.Union[np.ndarray, Number]) -> np.ndarray:
        """ Find the height of the surface at specified points

        Parameters
        ----------
        x_mesh: np.ndarray
            An n by m array of x co-ordinates
        y_mesh: np.ndarray
            An n by m array of y co-ordinates

        Returns
        -------
        height: np.ndarray
            An n by m array of surface heights

        Notes
        -----
        If a shift and rotation are specified, the rotation is applied first about the origin, the shift is then applied

        Examples
        --------
        >>>import slippy.surface as s
        >>>my_surf = s.PyramidSurface((1,1,1))
        >>>my_surf.height(0,0)
        0
        """

        x = x_mesh * np.cos(self._total_rotation) - y_mesh * np.sin(self._total_rotation)
        y = y_mesh * np.cos(self._total_rotation) + x_mesh * np.sin(self._total_rotation)
        x_shift, y_shift = self._total_shift
        x += x_shift * np.cos(self._total_rotation) - y_shift * np.sin(self._total_rotation)
        y += y_shift * np.cos(self._total_rotation) + x_shift * np.sin(self._total_rotation)

        return self._height(x, y)

    def _repr_helper(self):
        string = ''
        if self._total_shift[0] or self._total_shift[1]:
            string += ', shift = ' + repr(self._total_shift)
        if self._total_rotation:
            string += ', rotation = ' + repr(self._total_rotation)
        if self.is_discrete:
            string += ', generate = True'
        if self.grid_spacing:
            string += f', grid_spacing = {self.grid_spacing}'
        if self.extent:
            string += f', extent = {self.extent}'
        return string

    @classmethod
    def __init_subclass__(cls, is_abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            _AnalyticalSurface._analytic_subclass_registry.append(cls)

    @abc.abstractmethod
    def __repr__(self):
        pass

    def rotate(self, radians: Number):
        self._total_rotation += radians

    def shift(self, shift: tuple = None):
        """ Translate the profile of the surface

        Parameters
        ----------
        shift: tuple, optional (None)
            The distance to move the surface profile in the x and y directions, defaults to moving the origin of the
            profile to the centre
        """

        if shift is None:
            if self.extent is None:
                return
            else:
                shift = tuple(ex / -2 for ex in self.extent)

        if len(shift) != 2:
            raise ValueError("Shift tuple should be length 2")
        self._total_shift = tuple([cs + s for cs, s in zip(self._total_shift, shift)])

    def __add__(self, other):
        if isinstance(other, Number):
            self_copy = copy.copy(self)
            if self.profile is not None:
                self_copy.profile = self.profile + other
            self_copy.height = lambda x_mesh, y_mesh: self.height(x_mesh, y_mesh) + other
            self_copy.surface_type = 'Combination'
            return self_copy

        if isinstance(other, _AnalyticalSurface):
            return SurfaceCombination(self, other)

        if isinstance(other, Surface):
            if not self.is_discrete:
                self_copy = copy.copy(self)
                self_copy.extent = other.extent
                self_copy.grid_spacing = other.grid_spacing
                self_copy.shape = other.shape
                self_copy.discretise()
                return other + self_copy

        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, Number):
            self_copy = copy.copy(self)
            if self.profile is not None:
                self_copy.profile = self.profile - other
            self_copy.height = lambda x_mesh, y_mesh: self.height(x_mesh, y_mesh) - other
            self_copy.surface_type = 'Combination'
            return self_copy

        if isinstance(other, _AnalyticalSurface):
            return SurfaceCombination(self, other, '-')

        return super().__sub__(other)

    def __mul__(self, other):
        if isinstance(other, Number):
            self_copy = copy.copy(self)
            if self.profile is not None:
                self_copy.profile = self.profile * other
            self_copy.height = lambda x_mesh, y_mesh: self.height(x_mesh, y_mesh)*other
            return self_copy
        else:
            raise NotImplementedError(f"Multiplication between analytical surfaces and {type(other)} not implemented")

    def __div__(self, other):
        if isinstance(other, Number):
            return self * (1.0/other)
        else:
            raise NotImplementedError(f"Division between analytical surfaces and {type(other)} not implemented")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.is_discrete and other.is_discrete:
            return super().__eq__(other)

        return self.__dict__ == other.__dict__

    def show(self, property_to_plot='profile', plot_type='default', ax=False, *, dist=None, stride=None, n_pts=100,
             **figure_kwargs):
        if self.is_discrete:
            return super().show(property_to_plot=property_to_plot, plot_type=plot_type, ax=ax, dist=dist, stride=stride,
                                **figure_kwargs)

        old_props = self.fft, self.psd, self.acf

        if self.grid_spacing is not None and self.shape is not None:
            set_gs = False
            profile = self.height(*self.get_points_from_extent())
        elif self.extent is not None:
            set_gs = True
            gs = min(self.extent) / n_pts
            profile = self.height(*self.get_points_from_extent(extent=self.extent, grid_spacing=gs))
            self._shape = tuple([int(sz / gs) for sz in self.extent])
            self._grid_spacing = gs
        else:
            raise AttributeError('The extent and grid spacing of the surface should be set before the surface can be '
                                 'shown')
        self._profile = profile
        try:
            return super().show(property_to_plot=property_to_plot, plot_type=plot_type, ax=ax, dist=dist,
                                stride=stride, **figure_kwargs)
        finally:
            self._profile = None
            self.fft, self.psd, self.acf = old_props
            if set_gs:
                self._grid_spacing = None
                self._shape = None


class SurfaceCombination(_AnalyticalSurface):
    surface_type = 'Analytical Combination'

    def __init__(self, surface_1: _AnalyticalSurface, surface_2: _AnalyticalSurface, mode: str = '+'):
        """A class for containing additions or subtractions of analytical surfaces

        Parameters
        ----------
        surface_1: _AnalyticalSurface
            The first surface
        surface_2: _AnalyticalSurface
            The second surface
        mode: str {'+', '-'}
            The combination mode

        """
        if surface_1.extent is not None and surface_2.extent is not None and surface_1.extent != surface_2.extent:
            raise ValueError('Surfaces have different extents, cannot add')
        if surface_1.grid_spacing is not None and surface_2.grid_spacing is not None \
           and surface_1.grid_spacing != surface_2.grid_spacing:
            raise ValueError('Surfaces have different extents, cannot add')
        new_extent = surface_1.extent if surface_1.extent is not None else surface_2.extent
        new_gs = surface_1.grid_spacing if surface_1.grid_spacing is not None else surface_2.grid_spacing

        super().__init__(grid_spacing=new_gs, extent=new_extent, shift=(0, 0))

        self.mode = mode
        self.surfaces = (surface_1, surface_2)
        if self.mode == '+':
            self._height = lambda x_mesh, y_mesh: surface_1.height(x_mesh, y_mesh) + surface_2.height(x_mesh, y_mesh)
        elif self.mode == '-':
            self._height = lambda x_mesh, y_mesh: surface_1.height(x_mesh, y_mesh) - surface_2.height(x_mesh, y_mesh)

    def __repr__(self):
        return ('SurfaceCombination(surface_1=' + repr(self.surfaces[0]) + ', surface_2=' + repr(self.surfaces[1]) +
                f', mode=\'{self.mode}\'')

    def _height(self, x_mesh, y_mesh):
        """This will be overwritten on init"""
        pass
