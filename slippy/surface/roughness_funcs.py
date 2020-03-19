import numpy as np
import itertools
from slippy.surface.ACF_class import ACF
import scipy.signal
import scipy.optimize
import scipy.special
import typing
from collections.abc import Sequence
from numbers import Number
from slippy.abcs import _SurfaceABC

__all__ = ['roughness', 'subtract_polynomial', 'get_mat_vr',
           'get_height_of_mat_vr', 'get_summit_curvatures',
           'find_summits', 'low_pass_filter']


# noinspection PyTypeChecker
def _check_surface(surface, grid_spacing):
    if isinstance(surface, _SurfaceABC):
        p = np.asarray(surface)
        if grid_spacing is None or grid_spacing == float('inf'):
            gs = surface.grid_spacing
            return p, gs
        else:
            return np.asarray(surface), grid_spacing
    else:
        if grid_spacing is None:
            return np.asarray(surface), None
        return np.asarray(surface), float(grid_spacing)


def roughness(profile_in: {np.ndarray, _SurfaceABC}, parameter_name: {str, typing.Sequence[str]},
              grid_spacing: typing.Optional[float] = None,
              mask: typing.Optional[typing.Union[np.ndarray, float]] = None,
              curved_surface: bool = False, no_flattening: bool = False,
              filter_cut_off: typing.Optional[float] = None,
              four_nearest: bool = False) -> {float, list}:
    r"""Find 3d surface roughness parameters

    Calculates and returns common surface roughness parameters also known
    as birmingham parameters

    Parameters
    ----------
    profile_in : array like or Surface
        The surface profile or surface object to be used
    parameter_name : str or Sequence[str]
        The name of the surface roughness parameter to be returned see notes
        for descriptions of each
    grid_spacing : float optional (None)
        The distance between adjacent grid points in the surface
        only required for some parameters, see notes
    mask : array-like same shape as profile or float (None)
        If an array, the array is used as a mask for the profile, it must be
        the same shape as the profile, if a float or list of floats is given,
        those values are excluded from the calculation. If None, no mask is
        used. Limited applicability, see notes
    curved_surface : bool optional (False)
        True if the measurement surface was curved, in this case a 2nd order
        polynomial is subtracted other wise a 1st order polynomial is
        subtracted before the measurement
    no_flattening : bool optional (False)
        If true, flattening will be skipped, no polynomial will be subtracted
        before calculation of parameters, used for periodic surfaces or to
        save time
    filter_cut_off: float, optional (None)
        The cut off frequency of the low pass filter applied to the surface before finding summits, only used for
        parameters which need summits, if not set no low pass filter is applied
    four_nearest: bool, optional (False)
        If true any point that is higher than it's 4 nearest neighbours will be
        counted as a summit, otherwise a point must be higher than it's 8
        nearest neighbours to be a summit. Only used if summit descriptions
        are required, passed to find_summits.

    Returns
    -------
    out : float or list[float]
        The requested parameters

    See Also
    --------
    Surface : a helper class with useful surface analysis functionality
    subtract_polynomial
    find_summits
    get_mat_vr
    get_summit_curvatures

    Notes
    -----

    Before calculation the least squares plane is subtracted if a periodic surface is used this can be prevented by
    setting the no_flattening key word to true. If a curved surface is used a bi quadratic polynomial is fitted and
    removed before analysis as described in the above text.

    If a list of valid parameter names is given this method will return a list of parameter values.

    If a parameter based on summit descriptions is needed the following key words can be set to refine what counts as a
    summit, see find_summits for more information. This is only used to find summits, calculations of curvature are run
    on the unfiltered profile:

    - filter_cut_off (default None)
    - and
    - four_nearest (default False)

    Descriptions of each of the surface roughness parameters are given below:

    Amplitude parameters:

    - Sq   - RMS deviation of surface height \*
    - Sz   - Ten point height (based on definition of summits) \*\-
    - Ssk  - Skew of the surface (3rd moment) \*
    - Sku  - Kurtosis of the surface (4th moment) \*
    - Sv   - Lowest valley in the sample \*

    Spatial parameters:

    - Sds  - Summit density*-, see note above on definition of summit
    - Str  - Texture aspect ratio defined using the aacf
    - Std  - Texture direction
    - Sal  - Fastest decay auto correlation length \+

    hybrid parameters:

    - Sdelq- RMS slope \+
    - Ssc  - Mean summit curvature, see note above on definition of summit \*\+
    - Sdr  - Developed interfacial area ratio \+

    functional parameters:

    - Sbi  - Bearing index \*
    - Sci  - Core fluid retention index \*
    - Svi  - Valley fluid retention index \*

    non 'core' parameters (implemented):

    - Sa   - Mean amplitude of surface \*
    - Stp  - Surface bearing ratio \*
    - Smr  - Material volume ratio of the surface \*
    - Svr  - Void volume ratio of the surface, as for previous \*

    non 'core' parameters (not implemented):

    - Sk   - Core roughness depth
    - Spk  - Reduced summit height
    - Svk  - Reduced valley depth
    - Sr1  - Upper bearing area
    - Sr2  - Lower bearing area

    \* masking supported

    \+ requires grid_spacing

    \- requires grid spacing only if filtering is used for summit definition

    Summit parameters only support masking if low pass filtering is not
    required

    Parameter names are not case sensitive

    Examples
    --------


    References
    ----------

    Stout, K., Sullivan, P., Dong, W., Mainsah, E., Luo, N., Mathia,
    T., & Zahouani, H. (1993).
    The development of methods for the characterisation of roughness in
    three dimensions. EUR(Luxembourg), 358.
    Retrieved from http://cat.inist.fr/?aModele=afficheN&cpsidt=49475
    chapter 12
    """
    profile, grid_spacing = _check_surface(profile_in, grid_spacing)

    needs_gs = ['scc', 'sdr', 'sal']
    no_mask = ['sdr', 'str', 'sal']

    if mask is not None:
        if type(mask) is float:
            if np.isnan(mask):
                mask = ~np.isnan(profile)
            else:
                mask = ~profile == mask
        else:
            mask = np.asarray(mask, dtype=bool)
            if not mask.shape == profile.shape:
                msg = ("profile and mask shapes do not match: profile is"
                       "{profile.shape}, mask is {mask.shape}".format(**locals()))
                raise TypeError(msg)

    # subtract polynomial
    if curved_surface:
        order = 2
    else:
        order = 1

    if no_flattening:
        eta = profile
    else:
        eta, _ = subtract_polynomial(profile, order, mask=mask)

    if mask is None:
        eta_masked = eta
    else:
        eta_masked = eta[mask]

    # recursive call to allow lists of parameters to be found at once
    if not isinstance(parameter_name, str):
        if not isinstance(parameter_name, Sequence):
            raise ValueError("Parameter name must be a string or a sequence of strings")
        out = []
        for par_name in parameter_name:
            out.append(roughness(eta, par_name, grid_spacing=grid_spacing,
                                 mask=mask, no_flattening=True,
                                 filter_cut_off=filter_cut_off,
                                 four_nearest=four_nearest))
        return out
    else:
        try:
            # noinspection PyUnresolvedReferences
            parameter_name = parameter_name.lower()
        except AttributeError:
            msg = "Parameters must be strings or list of strings"
            raise ValueError(msg)

    if parameter_name in needs_gs and grid_spacing is None:
        raise ValueError("Grid spacing required for {}".format(parameter_name))

    if parameter_name in no_mask and mask is not None:
        raise ValueError("Masking not supported for {}".format(parameter_name))

    # return parameter of interest
    num_pts_m = eta_masked.size

    if grid_spacing is not None:
        global_size = [grid_spacing * dim for dim in profile.shape]
        gs2 = grid_spacing ** 2
        p_area_m = num_pts_m * gs2
        p_area_t = eta.size * gs2
    else:
        gs2 = None
        p_area_m = None
        p_area_t = None

    if parameter_name == 'sq':  # root mean square checked
        out = np.sqrt(np.mean(eta_masked ** 2))

    elif parameter_name == 'sa':  # mean amplitude checked
        out = np.mean(np.abs(eta_masked))

    elif parameter_name == 'ssk':  # skewness checked
        sq = np.sqrt(np.mean(eta_masked ** 2))
        out = np.mean(eta_masked ** 3) / sq ** 3

    elif parameter_name == 'sku':  # kurtosis checked
        sq = np.sqrt(np.mean(eta_masked ** 2))
        out = np.mean(eta_masked ** 4) / sq ** 4

    elif parameter_name == 'sv':
        out = np.min(eta_masked)

    elif parameter_name in ['sds', 'sz', 'ssc']:  # all that require summits
        # summits is logical array of summit locations
        summits = find_summits(eta, grid_spacing, mask, four_nearest,
                               filter_cut_off)
        if parameter_name == 'sds':  # summit density
            out = np.sum(summits) / num_pts_m
        elif parameter_name == 'sz':
            valleys = find_summits(-1 * eta, grid_spacing, mask, four_nearest,
                                   filter_cut_off)
            summit_heights = eta[summits]
            valley_heights = eta[valleys]
            summit_heights = np.sort(summit_heights, axis=None)
            valley_heights = np.sort(valley_heights, axis=None)
            out = np.abs(valley_heights[:5]) + np.abs(summit_heights[-5:]) / 5
        else:  # ssc mean summit curvature
            out = np.mean(get_summit_curvatures(eta, summits, grid_spacing))

    elif parameter_name == 'sdr':  # developed interfacial area ratio
        # ratio between actual surface area and projected or apparent
        # surface area
        i_areas = [0.25 * (((gs2 + (eta[x, y] - eta[x, y + 1]) ** 2) ** 0.5 +
                            (gs2 + (eta[x + 1, y + 1] - eta[x + 1, y]) ** 2) ** 0.5) *
                           ((gs2 + (eta[x, y] - eta[x + 1, y]) ** 2) ** 0.5 +
                            (gs2 + (eta[x, y + 1] - eta[x + 1, y + 1]) ** 2) ** 0.5))
                   for x in range(eta.shape[0] - 1)
                   for y in range(eta.shape[1] - 1)]
        i_area = sum(i_areas)
        out = (i_area - p_area_t) / i_area

    elif parameter_name == 'stp':
        # bearing area curve
        eta_rel = eta_masked / np.sqrt(np.mean(eta_masked ** 2))
        heights = np.linspace(min(eta_rel), max(eta_rel), 100)
        ratios = [np.sum(eta_masked < height) / p_area_m for height in heights]
        out = [heights, ratios]

    elif parameter_name == 'sbi':  # bearing index
        index = int(eta_masked.size / 20)
        sq = np.sqrt(np.mean(eta_masked ** 2))
        out = sq / np.sort(eta_masked)[index]

    elif parameter_name == 'sci':  # core fluid retention index
        sq = np.sqrt(np.mean(eta_masked ** 2))
        eta_m_sorted = np.sort(eta_masked)
        index = int(eta_masked.size * 0.05)
        h005 = eta_m_sorted[index]
        index = int(eta_masked * 0.8)
        h08 = eta_m_sorted[index]

        v005 = get_mat_vr(h005, eta, void=True, mask=mask)
        v08 = get_mat_vr(h08, eta, void=True, mask=mask)

        out = (v005 - v08) / p_area_m / sq

    elif parameter_name == 'svi':  # valley fluid retention index
        sq = np.sqrt(np.mean(eta_masked ** 2))
        index = int(eta_masked.size * 0.8)
        h08 = np.sort(eta_masked)[index]
        v08 = get_mat_vr(h08, eta, void=True, mask=mask)
        out = v08 / p_area_m / sq

    elif parameter_name == 'str':  # surface texture ratio

        # noinspection PyTypeChecker
        acf = np.asarray(ACF(eta))

        x = np.arange(eta.shape[0] / -2, eta.shape[0] / 2)
        y = np.arange(eta.shape[1] / -2, eta.shape[1] / 2)
        x_mesh, y_mesh = np.meshgrid(x, y)
        distance_to_centre = np.sqrt(x_mesh ** 2 + y_mesh ** 2)
        min_dist = min(distance_to_centre[acf < 0.2]) - 0.5
        max_dist = max(distance_to_centre[acf > 0.2]) + 0.5

        out = min_dist / max_dist

    elif parameter_name == 'std':  # surface texture direction
        fft = np.fft.fft2(eta)

        apsd = fft * np.conj(fft) / p_area_t
        x = np.arange(eta.shape[0] / -2, eta.shape[0] / 2)
        y = np.arange(eta.shape[1] / -2, eta.shape[1] / 2)
        i, j = np.unravel_index(apsd.argmax(), apsd.shape)
        beta = np.arctan(i / j)

        if beta < (np.pi / 2):
            out = -1 * beta
        else:
            out = np.pi - beta

    elif parameter_name == 'sal':  # fastest decaying auto correlation length
        # shortest distance from center of ACF to point where R<0.2
        # noinspection PyTypeChecker
        acf = np.asarray(ACF(eta))

        x = grid_spacing * np.arange(eta.shape[0] / -2,
                                     eta.shape[0] / 2)
        y = grid_spacing * np.arange(eta.shape[1] / -2,
                                     eta.shape[1] / 2)
        x_mesh, y_mesh = np.meshgrid(x, y)

        distance_to_centre = np.sqrt(x_mesh ** 2 + y_mesh ** 2)

        out = min(distance_to_centre[acf < 0.2])

    else:

        msg = 'Parameter name not recognised'
        raise ValueError(msg)

    return out


def get_height_of_mat_vr(ratio: float, profile: np.ndarray, void=False, mask=None,
                         accuracy=0.001):
    """Finds the cut off height of a specified material or void volume ratio

    Parameters
    ----------
    ratio : float {from 0 to 1}
        the target material or void volume ratio
    profile : array-like
        The surface profile to be used in the calculation
    void : bool optional (False)
        If set to true the height for the void volume ratio will be calculated
        otherwise the height for the material volume ratio will be calculated
    mask : array-like (bool) same shape as profile or float (defaults to None)
        If an array, the array is used as a mask for the profile, must be the
        same shape as the profile, if a float is given, values which match are
        excluded from the calculation
    accuracy : float optional (0.0001)
        The threshold value to stop iterations


    Returns
    -------
    height : float
        the height at which the input surface has the specified material or
        void ratio

    See also
    --------
    get_mat_vr
    roughness
    subtract_polynomial

    Notes
    -----
    This function should not be used without first flattening the surface using
    subtract_polynomial

    This function uses a simplified algorithm assuming that each point in the
    surface can be modeled as a column of material.

    Examples
    --------

    """

    p = np.asarray(profile)

    if mask is not None:
        if type(mask) is float:
            if np.isnan(mask):
                mask = ~np.isnan(p)
            else:
                mask = ~p == mask
        else:
            mask = np.asarray(mask, dtype=bool)
            if not mask.shape == p.shape:
                msg = ("profile and mask shapes do not match: profile is"
                       "{p.shape}, mask is {mask.shape}".format(**locals()))
                raise TypeError(msg)

        p = p[~mask]
    else:
        p = p.flatten()

    min_h = min(p)
    max_h = max(p)

    if void:
        first_guess = min_h + ratio * (max_h - min_h)
    else:
        first_guess = max_h - ratio * (max_h - min_h)

    output = scipy.optimize.minimize(lambda h: (get_mat_vr(h, p, void) - ratio) ** 2, first_guess,
                                     bounds=(min_h, max_h), tol=accuracy)

    height = output.x[0]

    return height


def get_mat_vr(height: float, profile: np.ndarray, void: bool = False, mask: {float, np.ndarray}=None,
               ratio=True, grid_spacing=None):
    """ Finds the material or void volume ratio

    Finds the material or void volume for a given plane height, uses an
    approximation (that each point is a column of material)

    Parameters
    ----------
    profile : 2D array-like or Surface object
        The surface profile to be used in the calculation
    height : float
        The height of the cut off plane
    void : bool optional (False)
        If set to true the void volume will be calculated otherwise the
        material volume is calculated
    mask : array-like (bool) same shape as profile or float (defaults to None)
        If an array, the array is used as a mask for the profile, must be the
        same shape as the profile, if a float is given, values which match are
        excluded from the calculation
    ratio : bool optional (True)
        If true the material or void ratio will be returned, if false the
        absolute value will be returned, this requires the grid_spacing
        keyword to be set
    grid_spacing : float
        The distance between adjacent grid points in the surface


    Returns
    -------
    out : float
        The requested output parameter

    See also
    --------
    get_height_of_mat_vr
    roughness
    subtract_polynomial

    Notes
    -----
    This function should not be used without first flattening the surface using
    subtract_polynomial

    This function uses a simplified algorithm assuming that each point in the
    surface can be modeled as a column of material.


    Examples
    --------


    """

    p, grid_spacing = _check_surface(profile, grid_spacing)

    if not grid_spacing and not ratio:
        msg = ("Grid spacing keyword or property of input surface must be set "
               "for absolute results, see Surface.set_grid_spacing if you are"
               " using surface objects")
        raise ValueError(msg)

    if mask is not None:
        if type(mask) is float:
            if np.isnan(mask):
                mask = ~np.isnan(p)
            else:
                mask = ~p == mask
        else:
            mask = np.asarray(mask, dtype=bool)
            if not mask.shape == p.shape:
                msg = ("profile and mask shapes do not match: profile is"
                       "{p.shape}, mask is {mask.shape}".format(**locals()))
                raise TypeError(msg)

        p = p[~mask]
    else:
        p = p.flatten()

    max_height = max(p)
    min_height = min(p)

    n_pts = p.size
    total_vol = n_pts * (max_height - min_height)
    max_m = sum(p - min_height)

    material = sum(p - height) * (p > height)
    if void:
        all_above = (max_height - height) * n_pts
        void_out = all_above - material  # void not below height
        void = total_vol - max_m - void_out
        if ratio:
            out = void / (total_vol - max_m)
        else:
            out = void * grid_spacing ** 3
    else:
        if ratio:
            out = material / max_m
        else:
            out = material * grid_spacing ** 3
    return out


def get_summit_curvatures(profile: np.ndarray, summits: typing.Optional[np.ndarray] = None, grid_spacing: float = None,
                          mask: typing.Optional[typing.Union[np.ndarray, float]] = None,
                          filter_cut_off: typing.Optional[float] = None, four_nearest: bool = False):
    """ find the curvatures of the summits

    Parameters
    ----------
    profile: N by M array-like or Surface object
        The surface profile for analysis
    summits: N by M array (optional)
        A bool array True at the location of the summits, if not supplied the
        summits are found using find_summits first, see notes
    grid_spacing: float optional (False)
        The distance between points on the grid of the surface profile. Required
        only if the filter_cut_off is set and profile is not a surface object
    mask: array-like (bool)N by M or float optional (None)
        If an array, the array is used as a mask for the profile, must be the
        same shape as the profile, if a float is given, values which match are
        excluded from the calculation
    filter_cut_off: float, optional (None)
        The cutoff frequency of the low pass filter that is applied before finding summits
    four_nearest: bool, optional (False)
        If true a summit is found if it is higher than it's four nearest neighbours, else it must be higher than it's
        eight nearest neighbours
    Returns
    -------
    curves : array
        Array of summit curvatures of size sum(summits.flatten())

    Other parameters
    ----------------
    four_nearest : bool optional (False)
        If true any point that is higher than it's 4 nearest neighbours will be
        counted as a summit, otherwise a point must be higher than it's 8
        nearest neighbours to be a summit. Only used is summits are not given.
    filter_cut_off : float optional (None)
        If given the surface will be low pass filtered before finding summits.
        Only used if summits are not given

    See also
    --------
    find_summits
    roughness

    Notes
    -----
    If the summits parameter is not set, any key word arguments that can be
    passed to find_summits can be passed through this function.

    Examples
    --------

    """
    profile, grid_spacing = _check_surface(profile, grid_spacing)

    gs2 = grid_spacing ** 2

    if summits is None:
        summits = find_summits(profile, filter_cut_off=filter_cut_off,
                               grid_spacing=grid_spacing,
                               four_nearest=four_nearest, mask=mask)
    verts = np.transpose(np.nonzero(summits))
    curves = [-0.5 * (profile[vert[0] - 1, vert[1]] + profile[vert[0] + 1, vert[1]] +
                      profile[vert[0], vert[1] - 1] + profile[vert[0], vert[1] + 1]
                      - 4 * profile[vert[0], vert[1]]) / gs2 for vert in verts]
    return curves


def find_summits(profile, grid_spacing: float = None, mask: typing.Union[np.ndarray, float] = None,
                 four_nearest=False, filter_cut_off=None):
    """ Finds high points after low pass filtering

    Parameters
    ----------
    profile : N by M array-like
        The surface profile for analysis
    grid_spacing : float, optional (None)
        The distance between points on the grid of the surface profile. required
        only if the filter_cut_off is set
    mask : array-like (bool) N by M or float optional (None)
        If an array, the array is used as a mask for the profile, must be the
        same shape as the profile, if a float is given, values which match are
        excluded from the calculation
    four_nearest : bool optional (False)
        If true any point that is higher than it's 4 nearest neighbours will be
        counted as a summit, otherwise a point must be higher than it's 8
        nearest neighbours to be a summit
    filter_cut_off : float optional (None)
        If given the surface will be low pass filtered before finding summits

    Returns
    -------
    summits : N by M bool array
        True at location of summits

    See Also
    --------


    Notes
    -----


    Examples
    --------

    """
    profile, grid_spacing = _check_surface(profile, grid_spacing)

    if mask is not None:
        if type(mask) is float:
            if np.isnan(mask):
                mask = ~np.isnan(profile)
            else:
                mask = ~profile == mask
        else:
            mask = np.asarray(mask, dtype=bool)
            if not mask.shape == profile.shape:
                msg = ("profile and mask shapes do not match: profile is"
                       "{profile.shape}, mask is {mask.shape}".format(**locals()))
                raise TypeError(msg)

        profile[mask] = float('nan')

    if filter_cut_off is not None:
        filtered_profile = low_pass_filter(profile, filter_cut_off, grid_spacing)
    else:
        filtered_profile = profile
    summits = np.ones(profile[1:-1, 1:-1].shape, dtype=bool)
    if four_nearest:
        x = [-1, +1, 0, 0]
        y = [0, 0, -1, +1]
    else:
        x = [-1, +1, 0, 0, -1, -1, +1, +1]
        y = [0, 0, -1, +1, -1, +1, -1, +1]

    for i in range(len(x)):
        summits = np.logical_and(summits, (filtered_profile[1:-1, 1:-1] > filtered_profile[1 + x[i]:-1 + x[i] or None,
                                           1 + y[i]:-1 + y[i] or None]))

    # pad summits with False to make same size as original
    summits = np.pad(summits, 1, 'constant', constant_values=False)
    return summits


def low_pass_filter(profile: typing.Union[_SurfaceABC, np.ndarray], cut_off_freq: float, grid_spacing: float = None):
    """2d low pass FIR filter with specified cut off frequency

    Parameters
    ----------
    profile : N by M array-like or Surface
        The Surface object or profile to be filtered
    cut_off_freq : Float
        The cut off frequency of the filter in the same units as the
        grid_spacing of the profile
    grid_spacing : float optional (None)
        The distance between adjacent points of the grid of the surface profile
        not required if the grid spacing of the Surface object is set, always
        required when an array-like profile is used

    Returns
    -------
    filtered_profile : N by M array
        The filtered surface profile

    See Also
    --------
    Surface

    Notes
    -----


    Examples
    --------


    References
    ----------

    """
    profile, grid_spacing = _check_surface(profile, grid_spacing)

    if grid_spacing is None:
        msg = "Grid spacing must be set"
        raise ValueError(msg)

    sz = profile.shape
    x = np.arange(1, sz[0] + 1)
    y = np.arange(1, sz[1] + 1)
    x_mesh, y_mesh = np.meshgrid(x, y)
    distance_to_centre = np.sqrt(x_mesh ** 2 + y_mesh ** 2)
    ws = 2 * np.pi / grid_spacing
    wc = cut_off_freq * 2 * np.pi
    h = (wc / ws) * scipy.special.j1(2 * np.pi * (wc / ws) * distance_to_centre) / distance_to_centre
    filtered_profile = scipy.signal.convolve2d(profile, h, 'same')

    return filtered_profile


def subtract_polynomial(profile: np.ndarray, order: int = 1,
                        mask: typing.Optional[typing.Union[np.ndarray, float]] = None):
    """ Flattens the surface by fitting and subtracting a polynomial

    Fits a polynomial to the surface the subtracts it from the surface, to
    remove slope or curve from imaging machines

    Parameters
    ----------

    profile : array-like or Surface
        The surface or profile to be used
    order : int
        The order of the polynomial to be fitted
    mask : np.ndarray (dtype=bool) or float, optional (None)
        If an array, the array is used as a mask for the profile, must be the same shape as the profile, if a float or
        list of floats is given, those values are excluded from the calculation, if None all the values are included in
        the calculation

    Returns
    -------
    adjusted : array
        The flattened profile
    coefs : array
        The coefficients of the polynomial

    Examples
    --------
    >>> import slippy.surface as s
    >>> import numpy as np
    >>>my_surface = s.assurface(np.random.rand(10,10))
    >>>flat_profile, coefs = subtract_polynomial(my_surface, 2)
    Subtract a quadratic polynomial from the profile of my_surface the result
    is returned but the profile property of the surface is not updated

    >>>flat_profile, coefs = subtract_polynomial(my_surface.profile, 2)
    Identical to the above operation

    >>>profile_2 = np.random.rand(100,100)
    >>>flat_profile_2, coefs = subtract_polynomial(profile_2, 1)
    Subtract a plane of best fit from profile_2 and return the result

    >>>flat_profile, coefs = subtract_polynomial(profile_2, 1, mask=float('nan'))
    Subtract the profile from the surface ignoring nan height values

    >>>mask=np.zeros_like(profile, dtype=bool)
    >>>mask[5:-5,5:-5]=True
    >>>flat_profile, coefs = subtract_polynomial(profile_2, 1, mask=mask)
    Subtract a polynomial from the surface ignoring a 5 deep boarder

    See Also
    --------
    roughness
    numpy.linalg.lstsq

    Notes
    -----
    In principal polynomials of any integer order are supported however higher
    order polynomials will take more time to fit

    """

    profile = np.asarray(profile)
    x = np.arange(profile.shape[1], dtype=float)
    y = np.arange(profile.shape[0], dtype=float)
    x_mesh_full, y_mesh_full = np.meshgrid(x, y)
    z_full = profile

    if mask is not None:
        if isinstance(mask, Number):
            if np.isnan(mask):
                mask = ~np.isnan(profile)
            else:
                mask = ~(profile == mask)
        else:
            mask = np.asarray(mask, dtype=bool)
            if not mask.shape == profile.shape:
                msg = ("profile and mask shapes do not match: profile is"
                       "{profile.shape}, mask is {mask.shape}".format(**locals()))
                raise TypeError(msg)
        z_masked = z_full[mask]
        x_masked = x_mesh_full[mask]
        y_masked = y_mesh_full[mask]

    else:
        z_masked = z_full.flatten()
        x_masked = x_mesh_full.flatten()
        y_masked = y_mesh_full.flatten()

    # fit polynomial
    n_cols = (order + 1) ** 2
    g = np.zeros((z_masked.size, n_cols))
    ij = itertools.product(range(order + 1), range(order + 1))

    for k, (i, j) in enumerate(ij):
        g[:, k] = x_masked ** i * y_masked ** j

    try:
        coefs, _, _, _ = np.linalg.lstsq(g, z_masked, rcond=None)

    except np.linalg.LinAlgError:
        msg = ("np.linalg.lstsq failed to converge, it is likely that there are Nan or inf values in the profile these"
               " should be masked, see the documentation for this function for more details")
        raise ValueError(msg)

    if any(np.isnan(coefs)) or any(np.isinf(coefs)):
        msg = ("Could not fit polynomial to surface. The surface likely contains nan or inf values, these should be "
               "masked before fitting, for more information see the documentation of this function")
        raise ValueError(msg)

    poly = np.zeros_like(profile)
    # must reset to iterate again
    ij = itertools.product(range(order + 1), range(order + 1))

    for a, (i, j) in zip(coefs, ij):
        poly += a * x_mesh_full ** i * y_mesh_full ** j
    poly = poly.reshape(profile.shape)
    adjusted = profile - poly

    if mask is not None:
        adjusted[~mask] = profile[~mask]

    return adjusted, coefs
