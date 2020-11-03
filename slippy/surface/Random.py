# change random iteration methods to work with scipy optimize way more methods avalible
"""
#TODO:
        Sort out documentation for each method

"""

import typing
import warnings
from collections import defaultdict

import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from numba import njit

from .ACF_class import ACF
from .Surface_class import Surface, _Surface
from ._johnson_utils import _fit_johnson_by_moments, _fit_johnson_by_quantiles

__all__ = ['RandomFilterSurface', 'RandomPerezSurface']


class RandomPerezSurface(_Surface):
    """ Surfaces with set height distribution and PSD found by the Perez method

    Parameters
    ----------
    target_psd: np.ndarray
        The PSD which the surface will approximate, the shape of the surface will be the same as the psd array
        (the same number of points in each direction)
    height_distribution: {scipy.stats.rv_continuous, sequence}
        Either a scipy.stats distribution or a sequence of the same size as the required output
    accuracy: float, optional (1e-3)
        The accuracy required for the solution to be considered converged, see the notes of the discretise method for
        more information
    max_it: int, optional (100)
        The maximum number of iterations used to discretise a realisation
    min_speed: float, optional (1e-10)
        The minimum speed of the iterations, if the iterations are converging slower than this they are deemed not to
        converge
    generate: bool, optional (False)
        If True the surface profile is found on instantiation
    grid_spacing: float, optional (None)
        The distance between grid points on the surface
    exact: {'psd', 'heights', 'best'}, optional ('best')


    Notes
    -----
    This method iterates between a surface with the exact right height distribution and one with the exact right PSD
    this method is not guaranteed to converge for all surfaces, for more details see the reference.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.stats as stats
    >>> import slippy.surface as s
    >>> # making a surface with an exponential ACF as described in the original paper:
    >>>beta = 10 # the drop off length of the acf
    >>>sigma = 1 # the roughness of the surface
    >>>qx = np.arange(-128,128)
    >>>qy = np.arange(-128,128)
    >>>Qx, Qy = np.meshgrid(qx,qy)
    >>>Cq = sigma**2*beta/(2*np.pi*(beta**2+Qx**2+Qy**2)**0.5) # the PSD of the surface
    >>>Cq = np.fft.fftshift(Cq)
    >>>height_distribution = stats.norm()
    >>>my_surface = s.RandomPerezSurface(target_psd = Cq, height_distribution=height_distribution,
    >>>                                  grid_spacing=1,
    >>>                                  generate=True)
    >>>my_surface.show()

    References
    ----------
    Based on the method and code given in:

    Francesc Pérez-Ràfols, Andreas Almqvist,
    Generating randomly rough surfaces with given height probability distribution and power spectrum,
    Tribology International,
    Volume 131,
    2019,
    Pages 591-604,
    ISSN 0301-679X,
    https://doi.org/10.1016/j.triboint.2018.11.020.
    (http://www.sciencedirect.com/science/article/pii/S0301679X18305607)

    """

    dist: scipy.stats.rv_continuous = None
    _rvs: np.ndarray = None

    def __init__(self, target_psd: np.ndarray,
                 height_distribution: typing.Union[scipy.stats.rv_continuous, typing.Sequence] = None,
                 accuracy: float = 1e-3, max_it: int = 100, min_speed: float = 1e-10,
                 generate: bool = False, grid_spacing: float = None, exact: str = 'best'):

        super().__init__(grid_spacing, None, None)
        self._target_psd = target_psd
        if height_distribution is None:
            height_distribution = scipy.stats.norm()
        try:
            if hasattr(height_distribution, 'rvs'):
                self.dist = height_distribution
            elif len(height_distribution) == target_psd.size or height_distribution.size == target_psd.size:
                self._rvs = np.array(height_distribution).flatten()
        except AttributeError:
            raise ValueError('Unrecognised type for height distribution')
        self._accuracy = accuracy
        self._max_it = max_it
        self._min_speed = min_speed
        if exact not in {'psd', 'heights', 'best'}:
            raise ValueError("exact not recognised should be one of {'psd', 'heights', 'best'}")
        self._exact = exact
        if generate:
            self.discretise()

    def __repr__(self):
        pass

    def discretise(self, return_new: bool = False, accuracy: float = None, max_it: int = None, min_speed: float = None,
                   suppress_errors: bool = False, return_original: bool = False):
        """Discretise the surface with a realisation

        Parameters
        ----------
        return_new: bool, optional (False)
            If True a new surface is returned else Nothing is returned and the profile property of the surface is set
        accuracy: float, optional (None)
            The tolerance used to detect convergence of the psd and height distribution, if not set defaults to the
            value set on initialisation
        max_it: int, optional (None)
            The maximum number of iterations used to fit the PSD and height spectra, if not set defaults to the value
            set on initialisation
        min_speed: float, optional (None)
            The minimum speed which the iterations can be converging before they are stopped (assumed to be not
            converging), if not set defaults to the value set on initialisation
        suppress_errors: bool, optional (False)
            If True convergence errors are suppressed and the profile realisation is made even if the solution has not
            converged, warnings are produced
        return_original: bool, optional (False)
            If True the variables returned by the original code given in the paper are returned, these are: the
            estimated surface with the correct height distribution, the estimated surface with the correct PSD and a
            dict of errors with each element being a list of error values with one value for each iteration.

        Returns
        -------
        Will return a new surface object if the return_new argument is True, otherwise sets the profile property of the
        parent surface
        Will return two surface estimates and a dict of errors  if return_original is True

        Examples
        --------
        >>> import numpy as np
        >>> import scipy.stats as stats
        >>> import slippy.surface as s
        >>> # making a surface with an exponential ACF as described in the original paper:
        >>>beta = 10 # the drop off length of the acf
        >>>sigma = 1 # the roughness of the surface
        >>>qx = np.arange(-128,128)
        >>>qy = np.arange(-128,128)
        >>>Qx, Qy = np.meshgrid(qx,qy)
        >>>Cq = sigma**2*beta/(2*np.pi*(beta**2+Qx**2+Qy**2)**0.5) # the PSD of the surface
        >>>height_distribution = stats.norm()
        >>>my_surface = s.RandomPerezSurface(target_psd = Cq, height_distribution=height_distribution, grid_spacing=1)
        >>>my_surface.discretise()
        >>>my_surface.show()
        Zh, Zs, error = fractal_surf_generator(np.fft.ifftshift(Cq),
                                               np.random.randn(256,256),
                                               min_speed = 1e-10,max_error=0.01)

        Notes
        -----
        During iteration of this solution two surfaces are maintained, one which has the correct PSD and one which has
        the correct height distribution, the one returned is the one which was first deemed to have converged. To over
        ride this behaviour set return original to True

        References
        ----------

        Francesc Pérez-Ràfols, Andreas Almqvist,
        Generating randomly rough surfaces with given height probability distribution and power spectrum,
        Tribology International,
        Volume 131,
        2019,
        Pages 591-604,
        ISSN 0301-679X,
        https://doi.org/10.1016/j.triboint.2018.11.020.
        (http://www.sciencedirect.com/science/article/pii/S0301679X18305607)
        """
        if return_original and return_new:
            raise ValueError("Only one of return_new and return_original can be set to True")

        accuracy = self._accuracy if accuracy is None else accuracy
        max_it = self._max_it if max_it is None else max_it
        min_speed = self._min_speed if min_speed is None else min_speed

        # scale and centre the PSD
        power_spectrum = self._target_psd
        m, n = power_spectrum.shape
        power_spectrum[0, 0] = 0
        power_spectrum = power_spectrum / np.sqrt(np.sum(power_spectrum.flatten() ** 2)) * m * n

        # generate random values for the surface
        if self.dist is not None:
            height_distribution = self.dist.rvs(power_spectrum.shape).flatten()
        elif self._rvs is not None:
            height_distribution = self._rvs
        else:
            raise ValueError("Height distribution not set, cannot descretise")

        # scale and centre the height probability distribution
        mean_height = np.mean(height_distribution.flatten())
        height_guess = height_distribution - mean_height
        sq_roughness = np.sqrt(np.mean(height_guess.flatten() ** 2))
        height_guess = height_guess / sq_roughness

        # sort height dist
        index_0 = np.argsort(height_guess.flatten())
        sorted_target_heights = height_guess.flatten()[index_0]

        # find bins for height distribution error
        bin_width = 3.5 * sorted_target_heights.size ** (-1 / 3)  # Scott bin method *note that the
        # values are normalised to have unit standard deviation
        n_bins = int(np.ceil((sorted_target_heights[-1] - sorted_target_heights[0]) / bin_width))
        bin_edges = np.linspace(sorted_target_heights[0], sorted_target_heights[-1], n_bins + 1)
        n0, bin_edges = np.histogram(sorted_target_heights, bins=bin_edges, density=True)
        error = defaultdict(list)

        height_guess = height_guess.reshape(power_spectrum.shape)
        fft_height_guess = np.fft.fft2(height_guess)

        best = 'psd'

        while True:
            # Step 1: fix power spectrum by FFT filter
            # Zs = np.fft.ifft2(zh*power_spectrum/Ch)
            phase = np.angle(fft_height_guess)
            # phase = _conj_sym(phase, neg=True)
            psd_guess = np.fft.ifft2(power_spectrum * np.exp(1j * phase)).real

            # find error in height distribution
            n_hist, _ = np.histogram(psd_guess.flatten(), bin_edges, density=True)
            error['H'].append(np.sum(np.abs(n_hist - n0) * (bin_edges[1:] - bin_edges[:-1])))

            # Step 2: Fix the height distribution rank ordering
            height_guess = psd_guess.flatten()
            index = np.argsort(height_guess)
            height_guess[index] = sorted_target_heights
            height_guess = height_guess.reshape((m, n))
            fft_height_guess = np.fft.fft2(height_guess)
            # find error in the power spectrum
            fft_hg_abs = np.abs(fft_height_guess)
            # Ch = np.abs(zh**2)#*grid_spacing**2/(n*m*(2*np.pi)**2)
            error['PS'].append(np.sqrt(np.mean((1 - fft_hg_abs[power_spectrum > 0] /
                                                power_spectrum[power_spectrum > 0]) ** 2)))
            error['PS0'].append(np.sqrt(np.mean(fft_hg_abs[power_spectrum == 0] ** 2)) /
                                np.mean(power_spectrum[power_spectrum > 0]))

            if len(error['H']) >= max_it:
                msg = 'Iterations for fractal surface failed to converge in the set number of iterations'
                break
            if len(error['H']) > 2 and abs(error['H'][-1] - error['H'][-2]) / error['H'][-1] < min_speed:
                msg = ('Solution for fractal surface convering is converging '
                       'slower than the minimum speed, solution failed to converge')
                break
            if len(error['H']) > 2 and (error['H'][-2] - error['H'][-1]) < 0:
                msg = 'Solution is diverging, solution failed to converge'
                break
            if error['H'][-1] < accuracy:
                msg = ''
                best = 'heights'
                break
            if error['PS'][-1] < accuracy and error['PS0'][-1] < accuracy:
                msg = ''
                best = 'psd'
                break  # solution converged

        if msg:
            if suppress_errors:
                warnings.warn(msg)
            else:
                raise StopIteration(msg)

        exact = self._exact if self._exact != 'best' else best

        if exact == 'psd':
            profile = psd_guess * sq_roughness + mean_height
        else:
            profile = height_guess * sq_roughness + mean_height

        if return_new:
            return Surface(profile=profile, grid_spacing=self.grid_spacing)

        if return_original:
            return height_guess, psd_guess, error

        self.profile = profile


class RandomFilterSurface(_Surface):
    """ Surfaces based on transformations of random sequences by a filter

    Attributes
    ----------
    dist : scipy.stats.rv_continuous
        The statistical distribution which the random sequence is drawn from

    Methods
    -------
    linear_transforms
    set_moments
    set_quantiles
    fir_filter
    discretise

    See Also
    --------
    surface_like

    Notes
    -----
    This is a subclass of Surface and inherits all methods. All key words that
    can be passed to Surface on instantiation can also be passed to this class
    apart from 'profile'

    Examples
    --------
    In the following example we will generate a randomly rough surface with an exponential ACF and a non gaussian height
    distribution.

    >>> import slippy.surface as s  # surface generation and manipulation
    >>> import numpy as np          # numerical functions
    >>> np.random.seed(0)
    >>> target_acf = s.ACF('exp', 2, 0.1, 0.2)  # make an example ACF
    >>> # Finding the filter coefficients
    >>> lin_trans_surface = s.RandomFilterSurface(target_acf=target_acf, grid_spacing=0.01)
    >>> lin_trans_surface.linear_transform(filter_shape=(40,20), gtol=1e-5, symmetric=True)
    >>> # Setting the skew and kurtosis of the output surface
    >>> lin_trans_surface.set_moments(skew = -0.5, kurtosis=5)
    >>> # generating and showing a realisation of the surface
    >>> my_realisation = lin_trans_surface.discretise([512,512], periodic=False, create_new=True)
    >>> fig, axes = my_realisation.show(['profile', 'acf', 'histogram'], ['image', 'image'], figsize=(15,5))
    """

    surface_type = 'Random'
    dist = scipy.stats.norm(loc=0, scale=1)
    _filter_coefficients: np.ndarray = None
    target_acf: ACF = None
    is_discrete: bool = False
    _moments = None
    _method_keywords = None
    target_acf_array = None
    "An array of acf values used as the target for the fitting procedure"

    def __init__(self,
                 target_acf: ACF = None,
                 grid_spacing: typing.Optional[float] = None,
                 extent: typing.Optional[typing.Sequence] = None,
                 shape: typing.Optional[typing.Sequence] = None,
                 moments: typing.Sequence = None,
                 quantiles: typing.Sequence = None):

        super().__init__(grid_spacing=grid_spacing, extent=extent, shape=shape)

        if target_acf is not None:
            self.target_acf = target_acf

        if moments is not None:
            if quantiles is not None:
                raise ValueError("Cannot set moments and quantiles")
            self.set_moments(*moments)
        if quantiles is not None:
            self.set_quantiles(quantiles)

    def __repr__(self):
        string = 'RandomSurface('
        if self.target_acf is not None:
            string += f'target_acf={repr(self.target_acf)}, '
            string += f'method={self.surface_type}, '
            string += f'grid_spacing={self.grid_spacing}, '
            string += f'**{repr(self._method_keywords)}, '
        if self._moments is not None:
            string += f'moments = {self._moments}, '
        if self.shape is not None:
            string += f'shape = {self.shape}, '
        if self.is_discrete:
            string += 'generate = True, '
        string = string[:-2]
        return string + ')'

    def linear_transform(self, filter_shape: typing.Sequence = (14, 14), symmetric: bool = True, max_it: int = None,
                         gtol: float = 1e-5, method='BFGS', **minimize_kwargs):
        r"""
        Generates a linear transform matrix

        Solves the non linear optimisation problem to generate a
        moving average filter that when convolved with a set of normally
        distributed random numbers will generate a surface profile with the
        specified ACF

        Parameters
        ----------

        filter_shape: Sequence, optional (14, 14)
            The dimensions of the filter coefficient matrix to be generated the default is (35, 35), must be exactly 2
            elements both elements must be ints
        symmetric: bool, optional (True)
            If true a symmetric filter will be fitted to the target ACF, this typically produces more realistic surfaces
            for the same filter shape but takes longer to fit the filter
        max_it: int, optional (100)
            The maximum number of iterations used
        gtol: float, optional (1e-11)
            The accuracy of the iterated solution
        method: str, optional ('BFGS')
            Type of solver. In most situations this should be one of the following:
            - Nelder-Mead
            - Powell
            - CG
            - BFGS
            - Newton-CG
            However other options exist, see the notes for more details
        minimize_kwargs
            Extra key word arguments which are passed to scipy.optimise.minimize function, valid arguments will depend
            on the choice of method

        Returns
        -------

        None
            Sets the filter_coefficients property of the instance


        See Also
        --------

        RandomFilterSurface.set_moments
        RandomFilterSurface.FIRfilter

        Notes
        -----

        This problem has a unique solution for each grid spacing. This should
        be set before running this method, else it is assumed to be 1.

        For more information on each of the methods available the documentation of scipy.optimize.minimize should be
        consulted. Practically, for this problem only unconstrained, unbound solvers are appropriate, these are:

        - Nelder-Mead
        - Powell
        - CG
        - BFGS
        - Newton-CG
        - dogleg
        - trust-ncg
        - trust-krylov
        - trust-exact

        However, the dogleg, trust-ncg, trust-krylov, trust-exact additionally require the user to specify the hessian
        matrix for the problem which is currently unsupported.

        References
        ----------

        ..[1] N. Patir, "A numerical procedure for random generation of
        rough surfaces (1978)"
        Wear, 47(2), 263–277.
        '<https://doi.org/10.1016/0043-1648(78)90157-6>'_

        Examples
        --------

        In the following example we will generate a randomly rough surface with an exponential ACF and a non gaussian
        height distribution.

        >>> import slippy.surface as s  # surface generation and manipulation
        >>> import numpy as np          # numerical functions
        >>> np.random.seed(0)
        >>> target_acf = s.ACF('exp', 2, 0.1, 0.2)  # make an example ACF
        >>> # Finding the filter coefficients
        >>> lin_trans_surface = s.RandomFilterSurface(target_acf=target_acf, grid_spacing=0.01)
        >>> lin_trans_surface.linear_transform(filter_shape=(40,20), gtol=1e-5, symmetric=True)
        >>> # Setting the skew and kurtosis of the output surface
        >>> lin_trans_surface.set_moments(skew = -0.5, kurtosis=5)
        >>> # generating and showing a realisation of the surface
        >>> my_realisation = lin_trans_surface.discretise([512,512], periodic=False, create_new=True)
        >>> fig, axes = my_realisation.show(['profile', 'acf', 'histogram'], ['image', 'image'], figsize=(15,5))
        """
        self._method_keywords = {**locals()}
        del (self._method_keywords['self'])

        self.surface_type = 'linear_transform'

        if self.target_acf is None:
            raise ValueError("No target ACF given, a target ACF must be given before the filter coefficients can be "
                             "found")

        # n by m ACF
        n = filter_shape[0]
        m = filter_shape[1]

        if max_it is None:
            max_it = n * m * 100

        if self.grid_spacing is None:
            msg = ("Grid spacing is not set assuming grid grid_spacing is 1, the solution is unique for each grid "
                   "spacing")
            warnings.warn(msg)
            self.grid_spacing = 1

        # generate the acf array form the ACF object
        el = self.grid_spacing * np.arange(n)
        k = self.grid_spacing * np.arange(m)
        acf_array = self.target_acf(k, el)
        self.target_acf_array = acf_array
        # initial guess (n by m guess of filter coefficients)
        x0 = _initial_guess(acf_array)

        if symmetric:
            result = minimize(_min_fun_symmetric, x0/2, args=(acf_array,), method=method,
                              jac=_get_grad_min_fun_symmetric, tol=gtol,
                              **minimize_kwargs)
        else:
            result = minimize(_min_fun, x0, args=(acf_array,), method=method, jac=_get_grad_min_fun, tol=gtol,
                              **minimize_kwargs)

        if not result.success:
            warnings.warn(result.message)

        alpha = np.reshape(result.x, filter_shape)

        if symmetric:
            filter_coefficients_half = alpha
            n1, m1 = filter_coefficients_half.shape
            filter_coefficients = np.zeros((n1 * 2 - 1, m1))
            filter_coefficients[:n1, :m1] = np.flip(filter_coefficients_half, 0)
            filter_coefficients[n1 - 1:, :m1] = filter_coefficients_half
            alpha = filter_coefficients

        # un comment the next two lines for the root finding method
        # from scipy.optimise import fsolve
        # alpha, *optional_out = fsolve(_root_func, x0, args=(acf_array,),
        #                              xtol=gtol, maxfev=max_it, full_output=True)

        self._filter_coefficients = alpha

    def set_moments(self, skew=0, kurtosis=0):
        r"""
        Sets the skew and kurtosis of the output surface

        If a filter coefficients matrix is present, this method changes the dist
        property of this instance to a distribution that produces a series of
        johnson or normally distributed random numbers that will have the
        set skew and kurtosis when convolved with the filter coefficients matrix.

        Parameters
        ----------

        skew, kurtosis : float
            The desired moments of the surface profile

        Returns
        -------
        None
            Sets the dist parameter of the instance

        See Also
        --------
        RandomFilterSurface.linear_transform

        Notes
        -----

        The skew of the input sequence:
        :math:`Sk_\eta`
        can be related to the skew of the final surface:
        :math:`Sk_z`
        by the following:

        :math:`Sk_z=Sk_\eta \frac{\sum_{i=0}^{q} \alpha_{i}^{3}}{(\sum_{i=0}^{q}\alpha_i^2)^\frac{3}{2}}`

        The kurtosis of the input sequence can be related to the final surface
        by [1]:

        :math:`K_z= \frac{K_\eta \sum_{i=0}^q \alpha_i^2 + 6 \sum_{i=0}^{q-1}\sum_{j=i+1}^q\alpha_i^2 \alpha_j^2}{(\sum_{i=0}^q \alpha_i^2)^2}`

        References
        ----------

        [1] Manesh, K. K., Ramamoorthy, B., & Singaperumal, M. (2010). Numerical generation of anisotropic 3D
        non-Gaussian engineering surfaces with specified 3D surface roughness parameters. Wear, 268(11–12),
        1371–1379. https://doi.org/10.1016/j.wear.2010.02.005

        Examples
        --------
        In the following example we will generate a randomly rough surface with an exponential ACF and a non gaussian
        height distribution.

        >>> import slippy.surface as s  # surface generation and manipulation
        >>> import numpy as np          # numerical functions
        >>> np.random.seed(0)
        >>> target_acf = s.ACF('exp', 2, 0.1, 0.2)  # make an example ACF
        >>> # Finding the filter coefficients
        >>> lin_trans_surface = s.RandomFilterSurface(target_acf=target_acf, grid_spacing=0.01)
        >>> lin_trans_surface.linear_transform(filter_shape=(40,20), gtol=1e-5, symmetric=True)
        >>> # Setting the skew and kurtosis of the output surface
        >>> lin_trans_surface.set_moments(skew = -0.5, kurtosis=5)
        >>> # generating and showing a realisation of the surface
        >>> my_realisation = lin_trans_surface.discretise([512,512], periodic=False, create_new=True)
        >>> fig, axes = my_realisation.show(['profile', 'acf', 'histogram'], ['image', 'image'], figsize=(15,5))
        """  # noqa
        self._moments = (skew, kurtosis)

        if self._filter_coefficients is None:
            msg = ("Filter coefficients matrix not found, this must be found by"
                   " the linear_transforms or FIR_filter methods before this "
                   "method can be used")
            raise AttributeError(msg)

        alpha = self._filter_coefficients

        alpha = alpha.flatten()
        alpha2 = alpha ** 2  # alpha squared
        sal2 = np.sum(alpha2)  # sum alpha squared

        seq_skew = skew * (sal2 ** (3 / 2)) / np.sum(alpha2 * alpha)

        # making the mixed term
        quad_term = 0.0
        q = len(alpha)
        for i in range(0, q - 1):
            for j in range(i, q):
                quad_term += alpha2[i] * alpha2[j]

        seq_kurt = (kurtosis * sal2 ** 2 - 6 * quad_term) / np.sum(alpha2**2) + 3

        self.dist = _fit_johnson_by_moments(0, 1, seq_skew, seq_kurt, True)

    def set_quantiles(self, quantiles: typing.Sequence):
        """ Fit a johnson distribution to give a resulting surface with the supplied quantiles

        Parameters
        ----------
        quantiles: Sequence
            Quantile values, quantiles relate to normal quantiles of -1.5, -0.5, 0.5, 1.5
            roughly: 0.067, 0.309, 0.691, 0.933

        Notes
        -----
        The quantiles supplied should relate to the quantiles in the final surface not the distribution that will be
        filtered.

        See Also
        --------
        set_moments

        """
        dist = _fit_johnson_by_quantiles(quantiles)

        moments = np.array(dist.stats('sk'))

        self.set_moments(moments[0], moments[1])

        return

    def fir_filter(self, target_acf: ACF = None, filter_span: typing.Sequence = None):
        """
        Create a 2D FIR filter to produce a surface with the given ACF

        Parameters
        ----------

        target_acf: ACF
            The target ACF of the final surface.
        filter_span: Sequence, optional (None)
            The span of the filter which will be found, larger filters give better long range representation of the ACF
            but take longer to find and longer to apply


        See Also
        --------

        slippy.surface.ACF
        RandomFilterSurface.linear_transform
        RandomFilterSurface.discretise

        Notes
        -----

        1 For this function to work the grid_spacing of the final surface
            must be set.
        2 After running this method surface realisations can be generated by the
            discretise method
        3 Uses the method defined here:
            Hu, Y. Z., & Tonder, K. (1992). Simulation of 3-D random rough
            surface by 2-D digital filter and Fourier analysis.
            International Journal of Machine Tools and …, 32(1–2), 83–90.
            https://doi.org/10.1016/0890-6955(92)90064-N

        Examples
        --------

        #TODO
        """
        if target_acf is None:
            target_acf = self.target_acf

        if target_acf is None:
            raise ValueError('No ACF set')

        self._method_keywords = {**locals()}
        del (self._method_keywords['target_acf'])
        del (self._method_keywords['self'])

        self.surface_type = 'fir_filter'

        # initialise sizes
        if self.grid_spacing is None:
            warnings.warn("Grid grid_spacing is not set assuming grid"
                          " grid_spacing is 1")
            self.grid_spacing = 1

        if filter_span is None:
            if self.shape is None:
                raise ValueError('Either the shape and grid_spacing of the surface or the filter span and the '
                                 'grid_spacing must be set before the filter coefficients can be found by this method')
            filter_span = self.shape

        # generate ACF object if input is not ACF object
        if type(target_acf) is ACF:
            self.target_acf = target_acf
        if self.target_acf is None:
            raise ValueError("Target ACF must be set before the filter coefficients can be found")

        # Generate array of ACF
        el = self.grid_spacing * np.arange(filter_span[0])
        k = self.grid_spacing * np.arange(filter_span[1])
        acf_array = self.target_acf(k, el)

        # Find FIR filter coefficients
        self._filter_coefficients = np.sqrt(np.fft.fft2(acf_array))

    def discretise(self, output_shape: typing.Sequence = None, periodic: bool = False,
                   create_new: bool = False):
        """
        Create a random surface realisation based on preset parameters

        Parameters
        ----------

        output_shape : 2 element list of ints
            The size of the output in points, the grid_spacing of these points
            is set when the filter coefficients matrix is generated, see
            linear_transform for more information
        periodic : bool, (False)
            If true the resulting surface will be periodic in geometry, for
            this to work the filter coefficients matrix must have odd order in
            both directions
        create_new : bool, optional (False)
            If set to true the method will return a new surface object with the
            generated profile and the correct sizes/ grid_spacing otherwise the
            parent surface will given the generated profile

        Returns
        -------

        A new surface object if the create_new parameter is set to true else
        nothing, but sets profile property of surface

        See Also
        --------

        RandomFilterSurface.linear_transform
        RandomFilterSurface.fir_filter

        Notes
        -----

        Uses the method outlined in the below with fft based convolution:
        Liao, D., Shao, W., Tang, J., & Li, J. (2018). An improved rough
        surface modeling method based on linear transformation technique.
        Tribology International, 119(August 2017), 786–794.
        https://doi.org/10.1016/j.triboint.2017.12.008

        """

        if self._filter_coefficients is None:
            raise AttributeError('The filter coefficients matrix must be found by either the linear_transform or '
                                 'fir_filter method before surface realisations can be generated')

        filter_coefficients = self._filter_coefficients

        n, m = filter_coefficients.shape

        if output_shape is None:
            output_shape = self.shape

        if output_shape is None:
            raise ValueError("Output shape is not set")

        output_n, output_m = output_shape

        if periodic:
            eta = np.pad(self.dist.rvs(size=[output_n, output_m]), ((0, n-1), (0, m-1)), 'wrap')
        else:
            eta = self.dist.rvs(size=[output_n + n - 1, output_m + m - 1])

        profile = fftconvolve(eta, filter_coefficients, 'valid')

        if create_new:
            return Surface(grid_spacing=self.grid_spacing, profile=profile)
        else:
            self.profile = profile
            self.is_discrete = True
        return


def _initial_guess(target_acf):
    """Find the initial guess for the filter coefficient matrix in the linear transforms method

    Parameters
    ----------
    target_acf: np.ndarray

    Returns
    -------
    np.ndarray, initial guess of filter coefficients
    """
    n, m = target_acf.shape
    c = np.zeros_like(target_acf)
    for i in range(n):
        for j in range(m):
            c[i, j] = target_acf[i, j] / ((n - i) * (m - j))
    s_sq = target_acf[0, 0] / np.sum((c ** 2).flatten())
    return (c * s_sq ** 0.5).flatten()


def _root_func(alpha: np.ndarray, target_acf: np.ndarray):
    """Optimisation function for linear transforms method if using with a vector root finding algorithm

    Parameters
    ----------
    alpha: np.ndarray
        The current filter coefficient matrix
    target_acf: np.ndarray
        The target acf array (same shape as alpha)

    Returns
    -------
    np.ndarray of residuals
    """
    alpha = alpha.reshape(target_acf.shape)
    acf_estimate = _get_acf_estimate_fft(alpha)
    return (acf_estimate - target_acf).flatten()


def _min_fun(alpha: np.ndarray, target_acf: np.ndarray):
    """Optimisation function for linear transforms method if using with a minimisation function

    Parameters
    ----------
    alpha: np.ndarray
        The current filter coefficient matrix
    target_acf: np.ndarray
        The target acf array (same shape as alpha)

    Returns
    -------
    np.ndarray of residuals
    """
    alpha = alpha.reshape(target_acf.shape)
    acf_estimate = _get_acf_estimate_fft(alpha)
    return np.sum(((target_acf - acf_estimate).flatten()) ** 2)


def _min_fun_symmetric(alpha: np.ndarray, target_acf: np.ndarray):
    """Optimisation function for linear transforms method if using with a minimisation function

    Parameters
    ----------
    alpha: np.ndarray
        The current filter coefficient matrix
    target_acf: np.ndarray
        The target acf array (same shape as alpha)

    Returns
    -------
    np.ndarray of residuals
    """
    alpha = alpha.reshape(target_acf.shape)
    n1, m1 = alpha.shape
    alpha_2 = np.zeros((n1 * 2 - 1, m1))
    alpha_2[:n1, :m1] = np.flip(alpha, 0)
    alpha_2[n1 - 1:, :m1] = alpha
    acf_estimate = _get_acf_estimate_fft(alpha_2)
    return np.sum(((target_acf - acf_estimate[:n1, :m1]).flatten()) ** 2)


@njit
def _get_acf_estimate(alpha):
    n, m = alpha.shape
    acf_estimate = np.zeros_like(alpha)
    for p in range(n):
        for q in range(m):
            a_est = 0.0
            for k in range(n - p):
                for el in range(m - q):
                    a_est += alpha[k, el] * alpha[k + p, el + q]
            acf_estimate[p, q] = a_est
    return acf_estimate


def _get_acf_estimate_fft(alpha):
    """Gives the same results as the manual convolution method above but much faster for large filters"""
    n, m = alpha.shape
    alpha_pad = np.pad(np.flip(alpha, (0, 1)), ((0, n - 1), (0, m - 1)), mode='constant')
    return fftconvolve(alpha, alpha_pad, 'same')


def _get_grad_min_fun(alpha, target_acf):
    alpha = alpha.reshape(target_acf.shape)
    acf_estimate = _get_acf_estimate_fft(alpha)
    return _grad_min_fun(alpha, target_acf, acf_estimate).flatten()


def _get_grad_min_fun_symmetric(alpha, target_acf):
    alpha = alpha.reshape(target_acf.shape)
    n1, m1 = alpha.shape
    alpha_2 = np.zeros((n1 * 2 - 1, m1))
    alpha_2[:n1, :] = np.flip(alpha, 0)
    alpha_2[n1 - 1:, :m1] = alpha
    acf_estimate = _get_acf_estimate_fft(alpha_2)[:n1, :m1]
    grads_2 = _grad_min_fun(alpha_2, target_acf, acf_estimate)
    grads = grads_2[n1 - 1:, :m1]
    grads[1:, :] += np.flip(grads_2[:n1 - 1, :], 0)
    return grads.flatten()


@njit
def _grad_min_fun(alpha: np.ndarray, target_acf: np.ndarray, acf_estimate):
    """Gradient of the above optimisation function for minimisation methods

    Parameters
    ----------
    alpha: np.ndarray
        The filter coefficients
    target_acf: np.ndarray
        The the target acf

    Returns
    -------
    np.ndarray
        The gradient of the objective value wrt each of the filter coefficients
    """
    n, m = alpha.shape
    grads = np.zeros_like(alpha)
    n1, m1 = target_acf.shape
    difference = target_acf - acf_estimate
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            grad = 0.0
            for p in range(min(n - i + 1, n1)):
                for q in range(m - j + 1):
                    grad += (difference[p, q]) * (-alpha[i + p - 1, j + q - 1])
            for p in range(min(i, n1)):
                for q in range(j):
                    grad += (difference[p, q]) * (-alpha[i - p - 1, j - q - 1])
            grads[i - 1, j - 1] = 2.0 * grad

    return grads
