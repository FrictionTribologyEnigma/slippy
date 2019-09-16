# TODO mesh function

# TODO documentation

# TODO make list of all functionallity

import os
import warnings
import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.signal
import typing
import abc
from scipy.stats import probplot
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from skimage.restoration import inpaint
from slippy.abcs import _MaterialABC, _SurfaceABC
from numbers import Number

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
    S : Surface object
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

    # The surface class for descrete surfaces (typically experiemntal)
    is_descrete: bool = False
    """ A bool flag, true if there is a profile present """
    acf: typing.Optional[ACF] = None
    """ The autocorelation function of the surface profile """
    aacf = None
    psd: typing.Optional[np.ndarray] = None
    """ The power spectral density of the surface """
    fft: typing.Optional[np.ndarray] = None
    """ The fast fourier transform of the surface """
    surface_type: str = "Generic"
    """ A description of the surface type """
    dimentions: typing.Optional[int] = 2
    """ The number of spartial dimentions that """
    is_analytic: bool = False
    """ A bool, true if the surface can be described by an equaiton and a 
    Z=height(X,Y) method is provided"""
    invert_surface: bool = False

    _material: typing.Optional[_MaterialABC] = None
    _profile: typing.Optional[np.ndarray] = None
    _grid_spacing: typing.Optional[float] = None
    _shape: typing.Optional[tuple] = None
    _extent: typing.Optional[tuple] = None
    _inter_func = None
    _allowed_keys = {}
    _mask: typing.Optional[np.ndarray] = None
    _wear: typing.Optional[np.ndarray] = None
    _size: typing.Optional[int] = None
    _subclass_registry = []

    def __init__(self, grid_spacing: typing.Optional[float] = None, extent: typing.Optional[tuple] = None,
                 shape: typing.Optional[tuple] = None, is_descrete: bool = False):
        if grid_spacing is not None and extent is not None and shape is not None:
            raise ValueError("Up to two of grid_spacing, extent and size should be set, all three were set")

        self.is_descrete = is_descrete

        if grid_spacing is not None:
            self.grid_spacing = grid_spacing
        if extent is not None:
            self.extent = extent
        if shape is not None:
            self.shape = shape

    @classmethod
    def __init_subclass__(cls, is_abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            _Surface._subclass_registry.append(cls)

    @property
    def size(self):
        """The total numebr of points in the surface"""
        return self._size

    @property
    def mask(self):
        """A mask used to exclude some values from analysis

        Either a boolean array of size self.size or a float of the value to
        be excluded
        """
        return self._mask

    @mask.setter
    def mask(self, value):
        if value is None:
            self._mask = None

        elif type(value) is float:
            if np.isnan(value):
                mask = ~np.isnan(self.profile)
            else:
                mask = ~self.profile == value
        else:
            mask = np.asarray(value, dtype=bool)
            if not mask.shape == self.shape:
                msg = ("profile and mask shapes do not match: profile is"
                       "{profile.shape}, mask is {mask.shape}".format(**locals()))
                raise TypeError(msg)

    @mask.deleter
    def mask(self):
        self._mask = None

    @property
    def extent(self):
        """ The overall dimentions of the surface in the same units as grid
        spacing
        """
        return self._extent

    @extent.setter
    def extent(self, value):
        """
        The overall dimentions of the surface
        """
        if type(value) is not tuple:
            msg = "Extent must be a tuple, got {}".format(type(value))
            raise TypeError(msg)
        if len(value) > 2:
            raise ValueError("Too many elements in extent, must be a maximum of two dimentions")

        if self.profile is not None:
            p_aspect = self.shape[0] / self.shape[1]
            e_aspect = value[0] / value[1]
            if abs(e_aspect - p_aspect) > 0.0001:
                msg = "Extent aspect ratio doesn't match profile aspect ratio"
                raise ValueError(msg)
            else:
                self._extent = value
                self._grid_spacing = value[0] / self.shape[0]
        else:
            self._extent = value
            self.dimentions = len(value)
            if self.grid_spacing is not None:
                self._shape = tuple([v / self.grid_spacing for v in value])
                self._size = np.product(self._shape)
        return

    @extent.deleter
    def extent(self):
        self._extent = None
        self._grid_spacing = None
        if self.profile is None:
            del self.shape
            self._size = None

    @property
    def shape(self):
        """
        The shape of the surface profile array, the number of points in each
        direction
        """
        return self._shape

    @shape.setter
    def shape(self, value):
        if type(value) is not tuple:
            raise ValueError(f"Shape shuld be a tuple, got: {type(value)}")

        if self.profile is None:
            self._shape = tuple([int(x) for x in value])
            self._size = np.product(self._shape)
            if self.grid_spacing is not None:
                self._extent = tuple([v * self.grid_spacing for v in value])
            elif self.extent is not None:
                e_aspect = self.extent[0] / self.extent[1]
                p_aspect = value[0] / value[1]
                if abs(e_aspect - p_aspect) < 0.0001:
                    self._grid_spacing = (self.extent[0] /
                                          self.shape[0])
                else:
                    warnings.warn("Global size does not match profile size,"
                                  "global size has been deleted")
                    self._extent = None
        else:
            raise ValueError("Cannot set shape when profile is present")

    @shape.deleter
    def shape(self):
        if self.profile is None:
            self._shape = None
            self._size = None
        else:
            msg = "Cannot delete shape with a surface profile set"
            raise ValueError(msg)

    @property
    def profile(self):
        """
        The height data for the surface profile
        """
        if self.invert_surface:
            return -1 * self._profile
        else:
            return self._profile

    @profile.setter
    def profile(self, value):
        """Sets the profile property
        """
        try:
            self._profile = np.asarray(value, dtype=float)
        except ValueError:
            msg = ("Could not convert profile to array of floats, profile contai"
                   "ns invalid values")
            raise ValueError(msg)

        self._shape = self._profile.shape
        self._size = self._profile.size
        self.dimentions = len(self._profile.shape)
        self._wear = np.zeros_like(self._profile)

        if self.grid_spacing is not None:
            self._extent = tuple([self.grid_spacing * p for p in
                                  self.shape])
        elif self.extent is not None:
            if self.dimentions == 1:
                self._grid_spacing = (self.extent[0] / self.shape[0])
            if self.dimentions == 2:
                e_aspect = self.extent[0] / self.extent[1]
                p_aspect = self.shape[0] / self.shape[1]

                if abs(e_aspect - p_aspect) < 0.0001:
                    self._grid_spacing = (self.extent[0] /
                                          self.shape[0])
                else:
                    warnings.warn("Global size does not match profile size,"
                                  " global size has been deleted")
                    self._extent = None

    @profile.deleter
    def profile(self):
        self._profile = None
        del self.shape
        del self.extent
        del self.mask
        self._wear = None
        self.is_descrete = False

    @property
    def worn_profile(self):
        if self.invert_surface:
            return self._profile - self._wear
        else:
            return -1 * (self._profile - self._wear)

    @property
    def grid_spacing(self):
        return self._grid_spacing

    @grid_spacing.setter
    def grid_spacing(self, grid_spacing):
        """ Change the grid spacing of the surface

        Changes the grid spacing attribute while keeping all other dimentions
        up to date. Does not re interpolate on a different size grid,
        stretches the surface to the new grid size keeping all points the same.

        """
        if grid_spacing is not float:
            try:
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
                self._shape = tuple([sz / grid_spacing for sz in
                                     self.extent])
                self._size = np.product(self._shape)
            elif self.shape is not None:
                self._extent = tuple([grid_spacing * pt for pt in
                                      self.shape])
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
                             " recived %s" % str(type(value)))

    @material.deleter
    def material(self):
        self._material = None

    def get_fft(self, profile_in=None):
        """ Find the fourier transform of the surface

        Findes the fft of the surface and stores it in your_instance.fft

        Parameters
        ----------
        profile_in : array-like optional (None)
            If set the fft of profile_in will be found and returned otherwise
            instances profile attribue is used

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

        return transform

    def get_acf(self, profile_in=None):
        """ Find the auto corelation function of the surface

        Findes the ACF of the surface and stores it in your_instance.acf

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
        >>> # equvalent to ACF(profile_2)

        """

        if profile_in is None:
            # noinspection PyTypeChecker
            self.acf = ACF(self)
        else:
            profile = np.asarray(profile_in)
            x = profile.shape[0]
            y = profile.shape[1]
            output = (scipy.signal.correlate(profile, profile, 'same') / (x * y))
            return output

    def get_psd(self):
        """ Find the power spectral density of the surface

        Findes the PSD of the surface and stores it in your_instance.psd

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
                  no_flattening=False, filter_cut_off=False,
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

    def resample(self, new_grid_spacing, return_profile=False,
                 remake_interpolator=False):
        """ Resample the profile by interpolation

        Changes the grid spacing of a surface by intrpolation of the original
        surface profile

        Parameters
        ----------

        new_grid_spacing : float
            The new grid spacing to be interpolated to
        return_profile : bool optional (False)
            If true the interpolated profile is returned otherwise it is set as
            the profile attribute of the instance
        remake_interpolator : boo optional (False)
            If true the interpolator will be remade before interpolation
            see notes

        Returns
        -------

        inter_profile : array
            If return_profile is true the interpolated profile is returned

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
        >>> # interpolate on a corse grid:
        >>> my_surface.resample(10)
        >>> # check shape:
        >>> my_surface.shape
        (11,11)
        >>> # restore original profile:
        >>> my_surface.resample(1)
        >>> my_surface.shape
        (101,101)
        """
        if remake_interpolator or self._inter_func is None:
            x0 = np.arange(0, self.extent[0], self.grid_spacing)
            y0 = np.arange(0, self.extent[1], self.grid_spacing)
            self._inter_func = scipy.interpolate.RectBivariateSpline(x0, y0,
                                                                     self.profile)
        x1 = np.arange(0, self.extent[0], new_grid_spacing)
        y1 = np.arange(0, self.extent[1], new_grid_spacing)
        inter_profile = self._inter_func(x1, y1)

        if return_profile:
            return inter_profile
        else:
            self.profile = inter_profile
            self.grid_spacing = new_grid_spacing

    def __add__(self, other):
        if self.extent == other.extent and self.extent is not None:
            if self.grid_spacing == other.grid_spacing:
                out = Surface(profile=self.profile + other.profile,
                              grid_spacing=self.grid_spacing)
                return out
            else:
                msg = "Surface sizes do not match: resampling"
                warnings.warn(msg)
                # resample surface with coarser grid then add again
                if self.grid_spacing > other.grid_spacing:
                    self.resample(other.grid_spacing, False)
                else:
                    other.resample(self.grid_spacing)

                return self + other

        elif self.shape == other.shape:
            if self.grid_spacing is not None:
                if other.grid_spacing is not None:
                    if self.grid_spacing == other.grid_spacing:
                        gs = self.grid_spacing
                    else:
                        msg = ('Surfaces have diferent sizes and grid spacing is'
                               ' set for both for element wise adding delete the'
                               ' grid spacing for one of the surfaces using: '
                               'del surface.grid_spacing before adding')
                        raise AttributeError(msg)
                else:  # only self not other
                    gs = self.grid_spacing
            else:  # not self
                gs = other.grid_spacing

            out = Surface(profile=self.profile + other.profile,
                          grid_spacing=gs)
            return out
        else:
            raise ValueError('surfaces are not compatible sizes cannot add')

    def __sub__(self, other):
        if self.extent == other._extent and self.extent:
            if self.grid_spacing == other.grid_spacing:
                out = Surface(profile=self.profile - other.profile,
                              grid_spacing=self.grid_spacing)
                return out
            else:
                msg = "Surface sizes do not match: resampling"
                warnings.warn(msg)
                # resample surface with coarser grid then add again
                if self.grid_spacing > other.grid_spacing:
                    self.resample(other.grid_spacing, False)
                else:
                    other.resample(self.grid_spacing)

                return self - other

        elif self.shape == other.shape:
            if self.grid_spacing:
                if other.grid_spacing:
                    if self.grid_spacing == other.grid_spacing:
                        gs = self.grid_spacing
                    else:
                        msg = ('Surfaces have diferent sizes and grid spacing is'
                               ' set for both for element wise adding unset the '
                               'grid spacing for one of the surfaces using: '
                               'set_grid_spacing(None) before subtracting')
                        raise AttributeError(msg)
                else:  # only self not other
                    gs = self.grid_spacing
            else:  # not self
                gs = other.grid_spacing

            out = Surface(profile=self.profile - other.profile,
                          grid_spacing=gs)
            return out
        else:
            raise ValueError('surfaces are not compatible sizes cannot'
                             ' subtract')

    def __eq__(self, other):
        if not isinstance(other, _Surface) or self.is_descrete!=other.is_descrete:
            return False
        if self.is_descrete:
            return self.profile == other.profile and self.grid_spacing == other.grid_spacing
        else:
            return repr(self) == repr(other)

    def show(self, property_to_plot='profile', plot_type='default', ax=False, *, dist=None, stride=None):
        """ Polt surface properties

        Parameters
        ----------
        property_to_plot : str or list of str length N optional ('profile')
            The property to be plotted see notes for supported names
        plot_type : str or list of str length N optional ('defalut')
            The type of plot to be produced, see notes for supported types
        ax : matplotlib axes or False optional (False)
            If supploed the plot will be added to the axis

        Keyword Parameters
        ------------------
        dist : a scipy probability distribution optional (None)
            Only used if probplot is requested, the probability distribution
            to plot against
        stride : float optional (None)
            Only used if a wire frame plot is requested, the stride between
            wires

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
        by the relavent get_ method before plotting.

        The grid spacing attribute should be set before plotting

        2D and 1D plots can be produced. 2D properties are:

            - profile - surface profile
            - fft2d   - fft of the surface profile
            - psd     - power spectral density of the surface profile
            - acf     - auto correlation function of the surface
            - apsd    - angular powerspectral density of the profile

        Plot types allowed for 2D plots are:

            - surface (default)
            - image
            - mesh

        If a mesh plot is requested the distance between lines in the mesh can
        be specified with the stride keyword

        1D properties are:

            - histogram - histogram of the profile heights
            - fft1d     - 1 dimentional fft of the surface
            - qq        - quantile quantile plot of the surface heights

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

        if self.profile is None:
            raise AttributeError('The profile of the surface must be set befor'
                                 'e it can be shown')

        types2d = ['profile', 'fft2d', 'psd', 'acf', 'apsd']
        types1d = ['histogram', 'fft1d', 'qq']

        # using a recursive call to deal with multiple plots on the same fig
        if type(property_to_plot) is list:
            number_of_subplots = len(property_to_plot)
            if not type(ax) is bool:
                msg = ("Can't plot multiple plots on single axis, "
                       'making new figure')
                warnings.warn(msg)
            if type(plot_type) is list:
                if len(plot_type) < number_of_subplots:
                    plot_type.extend(['default'] * (number_of_subplots -
                                                    len(plot_type)))
            else:
                plot_type = [plot_type] * number_of_subplots
            # 11, 12, 13, 22, then filling up rows of 3 (unlikely to be used)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            if len(property_to_plot) < 5:
                n_cols = [1, 2, 3, 2][number_of_subplots - 1]
            else:
                n_cols = 3
            n_rows = int(np.ceil(number_of_subplots / 3))
            fig = plt.figure()
            ax = []
            sub_plot_number = 100 * n_rows + 10 * n_cols + 1
            print(sub_plot_number)
            for i in range(number_of_subplots):
                if property_to_plot[i].lower() in types2d and not plot_type[i] == 'image':
                    ax.append(fig.add_subplot(sub_plot_number + i, projection='3d'))
                else:
                    ax.append(fig.add_subplot(sub_plot_number + i))
                self.show(property_to_plot[i], plot_type[i], ax[i])
            fig.show()
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
            fig = plt.figure()

        if property_to_plot in types2d:
            if not ax and not plot_type == 'image':
                # noinspection PyUnboundLocalVariable
                ax = fig.add_subplot(111, projection='3d')
            elif not ax and plot_type == 'image':
                # noinspection PyUnboundLocalVariable
                ax = fig.add_subplot(111)

            if property_to_plot == 'profile':
                labels = ['Surface profile', 'x', 'y', 'Height']
                x = self.grid_spacing * np.arange(self.shape[0])
                y = self.grid_spacing * np.arange(self.shape[1])
                z = self.profile

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
                z = np.abs(np.fft.fftshift(self.psd))
                x = np.fft.fftfreq(self.shape[0], self.grid_spacing)
                y = np.fft.fftfreq(self.shape[1], self.grid_spacing)

            elif property_to_plot == 'acf':
                labels = ['Auto corelation function', 'x', 'y',
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

            if plot_type == 'default' or plot_type == 'surface':
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
            elif plot_type == 'image':
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

            if property_to_plot == 'histogram':
                # do all plotting in this loop for 1D plots
                labels = ['Histogram of sufrface heights', 'height', 'counts']
                ax.hist(self.profile.flatten(), 100)

            elif property_to_plot == 'fft1d':
                if self.dimentions == 1:
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
                    freqs = u_mesh + v_mesh
                    if type(self.fft) is bool:
                        self.get_fft()
                    mags = np.abs(self.fft)
                    # scatter plot for 2d frequencies
                    ax.scatter(freqs.flatten(), mags.flatten(), 0.5, None, 'x')
                    ax.set_xlim(0, max(freqs.flatten()))
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
                raise ValueError(f"Porperty to plot {property_to_plot}, not recognised.")

            ax.set_title(labels[0])
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[2])
            return ax
        #######################################################################
        #######################################################################

    def __array__(self):
        return np.asarray(self._profile)

    @abc.abstractmethod
    def __repr__(self):
        return "Surface(profile=" + self.profile.__repr__() + ", grid_spacing=" + self.grid_spacing.__repr__() + ")"

    def rotate(self, radians):
        """Rotate the surface relative to the grid and reinterpolate
        """
        raise NotImplementedError('Not implemented yet')

    def get_points_from_extent(self):
        """
        Gets the grid points from the extent and the grid spacing

        Returns
        -------
        mesh_x, mesh_y : np.ndarray
            arrays of the grid points (result from mesh grid)
        """

        x = np.arange(0, self.extent[0], self.grid_spacing)
        y = np.arange(0, self.extent[1], self.grid_spacing)

        mesh_x, mesh_y = np.meshgrid(x, y)

        return mesh_x, mesh_y

    def mesh(self, depth, method='grid', parameters=None):
        """
        Returns a Mesh object for the surface

        Equvalent to Mesh(surface)

        Parameters
        ----------

        # TODO
        """
        pass
        # raise NotImplementedError("No mesh yet, Sorry!")
        # if not self.is_descrete:
        #     raise ValueError("Surface must be descrete before meshing")

    def interpolate(self, x_points: np.ndarray, y_points: np.ndarray, mode: str = 'nearest',
                    remake_interpolator: bool = False):
        """
        Easy memoised interpolation on surface objects

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
        assert(x_points.shape == y_points.shape)

        if mode == 'nearest':
            x_index = np.array(x_points / self.grid_spacing + 0.5, dtype='int32')
            y_index = np.array(y_points / self.grid_spacing + 0.5, dtype='int32')
            return np.reshape(self.profile[x_index, y_index], newshape=x_points.shape)
        elif mode == 'linear':
            if remake_interpolator or self._inter_func is None or self._inter_func.degrees != (1, 1):
                x0 = np.arange(0, self.extent[0], self.grid_spacing)
                y0 = np.arange(0, self.extent[1], self.grid_spacing)
                self._inter_func = scipy.interpolate.RectBivariateSpline(x0, y0, self.profile, kx=1, ky=1)
        elif mode == 'cubic':
            if remake_interpolator or self._inter_func is None or self._inter_func.degrees != (3, 3):
                x0 = np.arange(0, self.extent[0], self.grid_spacing)
                y0 = np.arange(0, self.extent[1], self.grid_spacing)
                self._inter_func = scipy.interpolate.RectBivariateSpline(x0, y0, self.profile, kx=1, ky=1)
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
        grid_spacing: float, optioanl (None)
            The distance between the grid points in the surface profile
        shape: tuple, optional (None)
            The number of grid points in the surrface in each direction, should not be set if a profile is given
        extent: tuple, optional (None)
            The total extent of the surface in the same units as the grid spacing, either this or the grid spacing can
            be set if a profile is given (either as the profile argument or from a file)
        file_name: str, optional (None)
            The full path including the file extention to a supported file type, supported types are .txt, .csv, .al3d,
            .mat
        delimiter: str, optional (',')
            The delimeter used in the .csv or .txt file, only used if the file name is given and the file is a .txt or
            .csv file
        mat_profile_name: str, optional ('profile')
            The name of the profile variable in the .mat file, only used if the file_name is given and the file is a
            .mat file
        mat_grid_spacing_name: str, optional (None)
            The name of the grid_spacing variable in the .mat file, only used if the file_name is given and the file is
            a .mat file. If unset the grid_spacing property is not read from the file.

        Attributes
        ----------
        profile : array
            The height infromation of the surface

        shape : tuple
            The numbeer of points in each direction of the profile

        grid_spacing: float
            The distance between adjacent points in the profile

        extent : list
            The size of the profile in the same units as the grid spacing is set in

        Methods
        -------
        show
        subtract_polynomial
        roughness
        get_mat_vr
        get_height_of_mat_vr
        find_summits
        get_summit_curvatures
        low_pass_filter
        read_from_file
        fill_holes
        resample
        get_fft
        get_acf
        get_psd

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
                 file_name: typing.Optional[str] = None, delimiter: typing.Optional[str] = ',',
                 mat_profile_name: typing.Optional[str] = None, mat_grid_spacing_name: typing.Optional[str] = None):

        if profile is not None or file_name is not None:
            if shape is not None:
                raise ValueError("The shape cannot be set if the profiel is also set, please set either the grid_spacin"
                                 "g or the extent only")
            if grid_spacing is not None and extent is not None:
                raise ValueError("Either the grid_spacing or the extent should be set with a profile, not both")

        if file_name is not None:
            if profile is not None:
                raise ValueError("The profile and a file name cannot be set")
            file_ext = os.path.splitext(file_name)[1]
            if file_ext == '.mat':
                self.read_mat(file_name, mat_profile_name, mat_grid_spacing_name)
            elif file_ext == '.al3d':
                self.read_al3d(file_name)
            elif file_ext == '.txt' or file_ext == '.csv':
                self.read_csv(file_name, delimiter)
            # read file replace profiel

        self.profile = profile
        super().__init__(grid_spacing=grid_spacing, extent=extent, shape=shape, is_descrete=True)

    def read_al3d(self, file_name: str, return_data: bool = False):
        """
        Reads an alicona al3d file and sets the profile and grid_spacing property of the surface

        Parameters
        ----------
        file_name: str
            The full path including the extention of the .al3d file
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

    def read_csv(self, file_name: str, delimiter: str = ',', return_profile: bool = False):
        """
        Read a profile from a csv or txt file, header lines are automatically skipped

        Parameters
        ----------
        file_name: str
            The full path to the .txt or .csv file including the file extention
        delimiter: str, optional (',')
            The delimiter used in by csv reader
        return_profile: bool, optional (False)
            If true the profile will be returned
        """
        import csv

        with open(file_name) as file:
            reader = csv.reader(file, delimiter=delimiter)
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
            The name of the grid_spacing variable in the .mat file, if set to none the grid spacing variabel is not set

        Notes
        -----
        This method will search the .mat file for the given keys. If no keys
        are given, and the .mat file contains variables called grid_spacing or
        profile these are set as the relavent attributes. Otherwise, if the
        .mat file only contains one variable this is set as the profile.
        If none of the above are true, or if the given keys are not found
        an error is raised

        """
        # load file
        mat = loadmat(path)
        keys = [key for key in mat if not key.startswith('_')]

        if grid_spacing_name is not None:
            try:
                # noinspection PyAttributeOutsideInit
                self.grid_spcaing = mat[grid_spacing_name]
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
        """ Replaces specificed values with filler

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
            or coloumn to be removed contains at least this proportion of real values

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
            end_r = len(profile) - 1
            start_c = 0
            end_c = len(profile[0]) - 1

            while not (all(good)):
                # start row
                if 1 - sum(holes[start_r, start_c:end_c]) / (end_c - start_c) < b_thresh:
                    start_r += 1
                else:
                    good[0] = True

                # end row
                if 1 - sum(holes[end_r, start_c:end_c]) / (end_c - start_c) < b_thresh:
                    end_r -= 1
                else:
                    good[1] = True

                if 1 - sum(holes[start_r:end_r, start_c]) / (end_r - start_r) < b_thresh:
                    start_c += 1
                else:
                    good[2] = True

                if 1 - sum(holes[start_r:end_r, end_c]) / (end_r - start_r) < b_thresh:
                    end_c -= 1
                else:
                    good[3] = True

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
            string += 'profile = ' + repr(self.profile)
        elif self.shape is not None:
            string += 'shape = ' + repr(self.shape)
        if self.grid_spacing is not None:
            string += ', grid_spacing = ' + repr(self.grid_spacing)
        if self.material is not None:
            string += ', material = ' + repr(self.material)
        if self.mask is not None:
            string += ', mask = ' + repr(self.mask)

        return 'Surface(' + string + ')'


class _AnalyticalSurface(_Surface):
    """
    A abstract base class for analytical surfaces, to extend the height and __repr__ mehtods must be overwritten
    """
    _total_shift: tuple = (0, 0)
    _total_rotation: float = 0
    is_analytic = True

    def __init__(self, generate: bool = False, rotation: Number = 0,
                 shift: typing.Union[str, tuple] = 'origin to centre',
                 grid_spacing: float = None, extent: tuple = None, shape: tuple = None):
        super().__init__(grid_spacing=grid_spacing, extent=extent, shape=shape)
        if rotation:
            self.rotate(rotation)
        if any(shift):
            if shift == 'origin to centre':
                try:
                    shift = tuple(ex/-2 for ex in self.extent)
                except TypeError:
                    raise ValueError("The extent of the surface must be set to shift the origin to the centre")
                self.shift(shift)

        if generate:
            self.descretise()

    def descretise(self):
        if self.is_descrete:
            msg = ('Surface is already discrete this will overwrite surface'
                   ' profile')
            warnings.warn(msg)
        if self.grid_spacing is None:
            msg = 'A grid spacing must be provided before descretisation'
            raise AttributeError(msg)

        if self.extent is None:
            msg = 'The extent or the shape of the surface must be set before descretisation'
            raise AttributeError(msg)
        if self.size > 10E7:
            warnings.warn('surface contains over 10^7 points calculations will'
                          ' be slow, consider splitting surface for analysis')

        x_mesh, y_mesh = self.get_points_from_extent()
        self.is_descrete = True
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
        If a shift and rotation are specified, the shift is applied first, the rotation is then applied about the
        original origin

        Examples
        --------
        >>>import slippy.surface as s
        >>>my_surf = s.PyramidSurface((1,1,1))
        >>>my_surf.height(0,0)
        0
        """

        x_mesh += self._total_shift[0]
        y_mesh += self._total_shift[1]
        x = x_mesh*np.sin(self._total_rotation) - y_mesh*np.cos(self._total_rotation)
        y = y_mesh*np.sin(self._total_rotation) + x_mesh*np.cos(self._total_rotation)

        return self._height(x, y)

    def _repr_helper(self):
        string = ''
        if self._total_shift[0] or self._total_shift[1]:
            string += ', centre = ' + repr(self._total_shift)
        if self._total_rotation:
            string += ', rotation = ' + repr(self._total_rotation)
        if self.is_descrete:
            string += ', generate = True'
        if self.grid_spacing:
            string += f', grid_spacing = {self.grid_spacing}'
        if self.extent:
            string += f', extent = {self.extent}'
        return string

    @abc.abstractmethod
    def __repr__(self):
        pass

    def rotate(self, radians: Number):
        self._total_rotation += radians

    def shift(self, shift: tuple):

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

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if self.is_descrete and other.is_descrete:
            return super().__eq__(other)

        return self.__dict__ == other.__dict__

    def show(self, property_to_plot='profile', plot_type='default', ax=False, *, dist=None, stride=None):
        if self.is_descrete:
            return super().show(property_to_plot=property_to_plot, plot_type=plot_type, ax=ax, dist=dist, stride=stride)
        elif self.grid_spacing is not None and self.shape is not None:
            profile = self.height(*self.get_points_from_extent())
            self._profile = profile
            try:
                return super().show(property_to_plot=property_to_plot, plot_type=plot_type, ax=ax, dist=dist,
                                    stride=stride)
            finally:
                self._profile = None
        else:
            raise AttributeError('The extent and grid spacing of the surface should be set before the surface can be '
                                 'shown')


class SurfaceCombination(_AnalyticalSurface):
    surface_type = 'Analytical Combination'

    def __init__(self, surface_1: _AnalyticalSurface, surface_2: _AnalyticalSurface, mode: str = '+'):
        super().__init__(grid_spacing=surface_1.grid_spacing, shape=surface_1.shape, generate=surface_1.is_descrete)
        self.mode = mode
        self.surfaces = (surface_1, surface_2)
        if self.mode == '+':
            self._height = lambda x_mesh, y_mesh: surface_1.height(x_mesh, y_mesh) + surface_2.height(x_mesh, y_mesh)
        elif self.mode == '-':
            self._height = lambda x_mesh, y_mesh: surface_1.height(x_mesh, y_mesh) - surface_2.height(x_mesh, y_mesh)

    def __repr__(self):
        return ('SurfaceCombination(surface_1=' + repr(self.surfaces[0]) + ', surface_2=' + repr(self.surfaces[1]) +
                f', mode={self.mode}')

    def _height(self, x_mesh, y_mesh):
        """This will be overwritten on init"""
        pass


if __name__ == '__main__':
    A = Surface(file_name='C:\\Users\\44779\\code\\SlipPY\\data\\image1_no head'
                          'er_units in nm.txt', delimiter=' ', grid_spacing=0.001)
    # testing show()
    # types2d=['profile', 'fft2d', 'psd', 'acf', 'apsd']
    # types1d=['histogram','fft1d', 'qq', 'disthist']
    # A.show(['profile','fft2d','acf'], ['surface', 'mesh', 'image'])
    # A.show(['histogram', 'fft1d', 'qq', 'disthist'])
    # out=A.birmingham(['sa', 'sq', 'ssk', 'sku'])
    # out=A.birmingham(['sds','sz', 'ssc'])
    # out=A.birmingham(['std','sbi', 'svi', 'str'])
    # out=A.birmingham(['sdr', 'sci', 'str'])
