# change random itteration methods to work with scipy optimize way more methods
# avalible
"""
#TODO:
        Sort out documantation for each method

"""

import typing
import warnings
from collections import defaultdict

import numpy as np
import scipy.stats
from numpy.matlib import repmat
from scipy.optimize import fsolve
from scipy.signal import fftconvolve

from .ACF_class import ACF
from .Surface_class import Surface, _Surface
from ._johnson_utils import _fit_johnson_by_moments, _fit_johnson_by_quantiles

__all__ = ['RandomFilterSurface', 'RandomPerezSurface', 'surface_like']


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
        The accuracy required for the solution to be considered converged, see the notes of the descretise mathod for
        more information
    max_it: int, optional (100)
        The maximum number of iterations used to descretise a realisation
    min_speed: float, optional (1e-10)
        The minimum speed of the iterations, if the iterations are converging slower than this they are deemed not to
        converge
    generate: bool, optional (False)
        If True the surface profile is found on instantiation
    grid_spacing: float, optional (None)
        The distance between grid points on the surface

    Notes
    -----
    This method iterates between a surface with the exact right height distribution and one with the exact right PSD
    this method is not garanteed to converge for all surfaces, for more details see the reference.

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
    >>>my_surface = s.RandomPerezSurface(target_psd = Cq, height_distribution=height_distribution, grid_spacing=1
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
                 generate: bool = False, grid_spacing: float = None):

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
        if generate:
            self.descretise()

    def __repr__(self):
        pass

    def descretise(self, return_new: bool = False, accuracy: float = None, max_it: int = None, min_speed: float = None,
                   supress_errors: bool = False, return_original: bool = False):
        """Descretise the surface with a realisiation

        Parameters
        ----------
        return_new: bool, optional (False)
            If True a new surface is returned else Nothing is returned and the profile property of the surface is set
        accuracy: float, optional (None)
            The tollerance used to detect convergence of the psd and height distribution, if not set defaults to the
            value set on initialisation
        max_it: int, optional (None)
            The maximum number of iterations used to fit the PSD and height spectra, if not set defaults to the value
            set on initialisation
        min_speed: float, optional (None)
            The minimum speed which the iterations can be converging before they are stopped (assumed to be not
            converging), if not set defaults to the value set on initialisation
        supress_errors: bool, optional (False)
            If True convergence errors are supressed and the profile realisation is made even if the solution has not
            converged, warnings are produced
        return_original: bool, optional (False)
            If True the variables returned by the original code given in the paper are returned, these are: the
            estimated surface with the correct height distribution, the esitmatied surface with the correct PSD and a
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
        Durring iteration of this solution two surfaces are maintained, one which has the correct PSD and one which has
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
            if error['H'][-1] < accuracy or \
                    (error['PS'][-1] < accuracy and error['PS0'][-1] < accuracy):
                msg = ''
                break  # soltuon converged

        if msg:
            if supress_errors:
                warnings.warn(msg)
            else:
                raise StopIteration(msg)

        if error['PS'][-1] < accuracy and error['PS0'][-1] < accuracy:
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
    dist : scipy distribution
        The statistical distribution which the random sequenc is drawn from
    
    Methods
    -------
    linear_transforms
    set_moments
    set_quantiles
    fir_filter
    descretise
    
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
    
    
    """

    surface_type = 'Random'
    dist = scipy.stats.norm(loc=0, scale=1)
    _filter_coeficents: np.ndarray = None
    target_acf: ACF = None
    is_discrete: bool = False
    _moments = None
    _method_keywords = None

    def __init__(self, method: str = 'linear_transform',
                 target_acf: ACF = None,
                 target_psd: np.ndarray = None,
                 generate: bool = False,
                 grid_spacing: typing.Optional[float] = None,
                 extent: typing.Optional[typing.Sequence] = None,
                 shape: typing.Optional[typing.Sequence] = None,
                 moments: typing.Sequence = None,
                 quantiles: typing.Sequence = None, **method_keywords):
        """
        aims: should be able to generate in single line, needs to include the mode that the matrix will be
        Parameters
        ----------
        method
        """
        super().__init__(grid_spacing=grid_spacing, extent=extent, shape=shape)

        valid_methods = ['linear_transform', 'fir_filter']

        if target_acf is not None or target_psd is not None:
            self.target_psd = target_psd
            self.target_acf = target_acf
            if method in valid_methods:
                method = self.__getattribute__(method)
            else:
                raise ValueError(f"Method '{method}' is not valid, valid methods are {' '.join(valid_methods)}")

            method(**method_keywords)

        elif method != 'linear_transform':
            warnings.warn("Setting the method to generate the filter coeficents matrix has no effect if a target acf is"
                          " not given.")

        if moments is not None:
            if quantiles is not None:
                raise ValueError("Cannot set moments and quantiles")
            self.set_moments(*moments)
        if quantiles is not None:
            self.set_quantiles(quantiles)

        if generate:
            self.descretise()

    def __repr__(self):
        string = 'RandomSurface('
        if self.target_acf is not None:
            string += f'target_surface={repr(self.target_acf)}, '
            string += f'method={self.surface_type}, '
            string += f'grid_spacing={self.grid_spacing}, '
            string += f'**{repr(self._method_keywords)}, '
        if self._moments is not None:
            string += f'moments = {self._moments}, '
        if self.shape is not None:
            string += f'shape = {self.shape}, '
        if self.is_descrete:
            string += f'generate = True, '
        string = string[:-2]
        return string + ')'

    def linear_transform(self, target_acf: ACF = None, filter_size_n_m: typing.Sequence = (14, 14), max_it: int = 100,
                         accuracy: float = 1e-11, no_large_filter_error=False):
        r"""
        Generates a linear transform matrix
        
        Solves the non linear optimisation problem to generate a 
        moving average filter that when convoloved with a set of normally
        distributed random numbers will generate a surface profile with the 
        sepcified ACF
        
        Parameters
        ----------
        
        target_acf : ACF object or description
            The target ACF, the linear transfrom matrix will produce surfaces
            with this ACF.
        filter_size_n_m : 2 element sequence of int
            The dimensions of the filter coeficent matrix to be genrated the defaultis (35, 35)
        max_it : int, optional (100)
            The maximum number of iterations used
        accuracy : float, optional (1e-11)
            The accuracy of the itterated solution
        no_large_filter_error: bool, optional (False
            If Ture the program allows large filters to be used, large filters do not converge to physical solutions
            with this method
        
        Returns
        -------
        
        None
        Sets the filter_coeficents property of the instance
        
        Other parameters
        ----------------
        

        
        See Also
        --------
        
        RandomSurface.set_moments
        RandomSurface.FIRfilter
        
        Notes
        -----
        
        This problem has a unique solution for each grid spacing. This should 
        be set before running this method, else it is assumed to be 1.
        
        The itteration procedure used if newtonian is selected is not strictly 
        newtonian. As it is much more time consuming to invert the jacobian 
        martix than multiply the result a modified newtonian is used. If the 
        next itteration is not an imporovement on the previous itteration the 
        'distance moved' is halved. This halving is repeted until the 
        itteration results in an improvement. The minimum disctance that will 
        be tried can be set by setting the min_relax key word. This defaults to 
        10e-6. This is a deviation from the method described by [1]
        
        References
        ----------
        
        ..[1] N. Patir, "A numerical procedure for random generation of 
        rough surfaces (1978)"
        Wear, 47(2), 263–277. 
        '<https://doi.org/10.1016/0043-1648(78)90157-6>'_
        
        Examples
        --------
        
        """
        self._method_keywords = {**locals()}
        del (self._method_keywords['target_acf'])

        self.surface_type = 'linear_transform'

        if type(target_acf) is ACF:
            self.target_acf = target_acf

        if self.target_acf is None:
            raise ValueError("No target ACF given, a target ACF must be given before the filter coeficents can be "
                             "found")

        # n by m ACF
        n = filter_size_n_m[0]
        m = filter_size_n_m[1]

        if n * m > 15 ** 2:
            warnings.warn("Warning large filter sizes often do not converge")
        if n * m > 400 and not no_large_filter_error:
            raise ValueError("Large filter size used, this will not converge, for large surfaces it is best to "
                             "resample a lower resolution surface. To supress this error set the "
                             "no_large_filter_error to True, please check the result if this is done")

        if self.grid_spacing is None:
            msg = ("Grid spacing is not set assuming grid grid_spacing is 1, the soultion is unique for each grid "
                   "spacing")
            warnings.warn(msg)
            self.grid_spacing = 1

        # generate the acf array form the ACF object
        l = self.grid_spacing * np.arange(n)
        k = self.grid_spacing * np.arange(m)
        [k_mesh, l_mesh] = np.meshgrid(k, l)
        acf_array = self.target_acf(k, l)

        # initial guess (n by m guess of filter coefficents)
        x0 = _initial_guess(acf_array)

        alpha, *optional_out = fsolve(_opt_func, x0, args=(acf_array,), fprime=_jac, xtol=accuracy, maxfev=max_it)

        self._filter_coeficents = alpha

        if optional_out[-2] != 1:
            warnings.warn("Iterations for the filter coefficents failed to converge, processexited with error: " +
                          optional_out[-1])

    def set_moments(self, skew=0, kurtosis=3):
        r"""
        Sets the skew and kurtosis of the output surface
        
        If a filter coeficents matrix is present, this method changes the dist
        property of this instance to a distribution that produces a series of 
        johnson or normally distributed random numbers that will have the 
        set skew and kurtosis when convolved with the filter coeficents matrix.
        
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
        RandomSurface.linear_transform
        
        Notes
        -----
        
        The skew of the input sequence:
        ..math:: Sk_\eta
        can be related to the skew of the final surface:
        ..math:: Sk_z
        by the following:
        
        ..math::
            Sk_z=Sk_\eta \frac{\sum_{i=0}^{q} \alpha_{i}^{3}}{(\sum_{i=0}^{q}\alpha_i^2)^\frac{3}{2}}\\
        
        The kurtosis of the input sequence can be related to the final surface 
        by [1]:
        
        ..math:
            K_z= \frac{K_\eta \sum_{i=0}^q \alpha_i^2 + 6 \sum_{i=0}^{q-1}\sum_{j=i+1}^q \alpha_i^2 \alpha_j^2}{(\sum_{i=0}^q \alpha_i^2)^2}\\
        
        References
        ----------
        
        [1] Liao, D., Shao, W., Tang, J., & Li, J.
        An improved rough surface modeling method based on linear 
        transformation technique. Tribology International, 119(August 2017), 
        786–794. '<https://doi.org/10.1016/j.triboint.2017.12.008>'_
        """
        self._moments = (skew, kurtosis)

        if not hasattr(self, '_filter_coeficents'):
            msg = ("filter coeficents matrix not found, this must be found by"
                   " the linear_transforms or FIR_filter methods before this "
                   "method can be used")
            raise AttributeError(msg)

        alpha = self._filter_coeficents

        alpha = alpha.flatten()
        alpha2 = alpha ** 2  # alpha squared
        sal2 = np.sum(alpha2)  # sum alpha squared

        seq_skew = skew * sal2 ** (3 / 2) / np.sum(alpha2 * alpha)

        # The quadratic alpha term needs some speical treatment
        # pad with 0s
        alphapad2 = np.pad(alpha2, [0, len(alpha2)], 'constant')

        ai = repmat(alpha2[:-1], len(alpha2) - 1, 1)

        index_x_mesh, index_y_mesh = np.meshgrid(np.arange(len(alpha2) - 1), np.arange(len(alpha2) - 1))
        index = index_x_mesh + index_y_mesh + 1  # diagonally increaing matrix
        aj = alphapad2[index]

        quad_term = sum(ai.flatten() * aj.flatten())

        seq_kurt = (kurtosis * sal2 ** 2 - 6 * quad_term) / sal2

        print(seq_skew, seq_kurt)

        self.dist = _fit_johnson_by_moments(0, 1, seq_skew, seq_kurt, True)

    def set_quantiles(self, quantiles: typing.Sequence):
        """ Fit a johnson districution to give a resulting surface with the supplied quantiles

        Parameters
        ----------
        quantiles: Sequence
            Quantile values, quantiles relate to normal quantiles of -1.5, -0.5, 0.5, 1.5
            roughly: 0.067, 0.309, 0.691, 0.933

        Notes
        -----
        The quantiles suplied should relate to the quantiles in the final surface not the distribution that will be
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
        RandomSurface.linear_transform
        RandomSurface.descretise
        
        Notes
        -----
        
        1 For this function to work the grid_spacing ofthe final surface
            must be set.
        2 After runing this method surface realisations can be generated by the 
            descretise method
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
                                 'grid_spacing must be set before the filter coeficents can be found by this method')
            filter_span = self.shape

        # genreate ACF object if input is not ACF object
        if type(target_acf) is ACF:
            self.target_acf = target_acf
        if self.target_acf is None:
            raise ValueError("Target ACF must be set before the filter coeficents can be found")

        # Generate array of ACF
        l = self.grid_spacing * np.arange(filter_span[0])
        k = self.grid_spacing * np.arange(filter_span[1])
        acf_array = self.target_acf(k, l)

        # Find FIR filter coeficents
        self._filter_coeficents = np.sqrt(np.fft.fft2(acf_array))

    def descretise(self, output_shape: typing.Sequence = None, periodic: bool = False,
                   create_new: bool = False):
        """
        Create a random surface realisation based on preset paramters
        
        Parameters
        ----------
        
        output_shape : 2 elemeent list of ints, defaults to [512, 512]
            The size of the output in points, the grid_spacing of these points 
            is set when the filter coefficents matrix is genreated, see 
            linear_transform for more information
        periodic : bool, (False)
            If true the resulting surface will be periodic in geometry, for 
            this to work the filter coefficents matrix must have odd order in 
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
        
        RandomSurface.linear_transform
        RandomSurface.fir_filter
        
        Notes
        -----
        
        Uses the method outlined in the below with fft based convoluiton:
        Liao, D., Shao, W., Tang, J., & Li, J. (2018). An improved rough
        surface modeling method based on linear transformation technique.
        Tribology International, 119(August 2017), 786–794.
        https://doi.org/10.1016/j.triboint.2017.12.008
        
        """

        if self._filter_coeficents is None:
            raise AttributeError('The filter coeficents matrix must be found by either the linear_transform or '
                                 'fir_filter method before surface realisations can be generated')
        (n, m) = self._filter_coeficents.shape
        if output_shape is None:
            output_shape = self.shape

        if output_shape is None:
            raise ValueError("Output shape is not set")

        output_n, output_m = output_shape

        if periodic:
            if n % 2 == 0 or m % 2 == 0:
                msg = ('For a periodic surface the filter coeficents matrix must'
                       'have an odd number of elements in every dimention, '
                       'output profile will not be the expected size')
                warnings.warn(msg)
            pad_rows = int(np.floor(n / 2))
            pad_cols = int(np.floor(m / 2))
            eta = np.pad(self.dist.rvs(size=[output_n, output_m]), ((pad_rows, pad_rows), (pad_cols, pad_cols)), 'wrap')
        else:
            eta = self.dist.rvs(size=[output_n + n - 1, output_m + m - 1])

        profile = fftconvolve(eta, self._filter_coeficents, 'valid')

        if create_new:
            return Surface(grid_spacing=self.grid_spacing, profile=profile)
        else:
            self.profile = profile
            self.is_descrete = True
        return


# TODO get rid of kwargs here
def surface_like(target_surface: Surface, extent: typing.Union[str, tuple] = 'original',
                 grid_spacing: typing.Union[str, float] = 'original',
                 filter_shape: typing.Sequence = (35, 35), **kwargs):
    """
    Generates a surface similar to the input surface
    
    Generates a surface with the same ACF, skew and kurtosis as the input
    surface assuming the surface is johnson distributed
    
    Parameters
    ----------
    
    target_surface : Surface
        A surface object to be 'copied'
        
    extent : {'origninal' or 2 element list of ints}
        The size in each direction of the output surface, 
        if 'original' the dimensions of the input surface are used
        
    grid_spacing : {'original' or float}
        The spacing between grid points, if 'original' the grid spacing of the 
        input surface is used, if this property is not set for the input
        surface it is assumed that both are 1 and warns
        
    filter_shape : 2 element list of ints
        The size of the filter to be used defaults to [35, 35]
        
    
    Returns
    -------
    surf_out : Surface
        A surface object with the same properties as the original surface
        of the scale and size requested with keyword arguments
    
    Warns
    -----
    If the grid spacing property is not set on the input surface and 'original'
    is given as the grid spacing arg will assume both are 1 and produce a 
    warning
    
    Other Parameters
    ----------------
    
    periodic : bool default False
        If true the returned surface will have a periodic profile
        
    filter_kwargs : dict
        Keyword arguments that are passed to RandomSurface.linear_transforms 
        see that for more infromation
        
    dist_type: {'johnson', 'kernel'}
        Defaults to johnson, the distribution that will be used to draw random
        samples for the pre-filter sequence, if johnson a johnson distribution
        will be fitted to the input surface quartiles, if kernel a kernel 
        distribution will be made from the input histogram, this is 
        experimental
        
    See Also
    --------
    RandomSurface.linear_transforms
    RandomSurface
    Surface
    
    Notes
    -----
    
    If multiple realisations are needed of the same 'copied' surface it will
    be much faster to call the descretise function of the returned surface. 
    multiple surfaces of the same grid spacing but different grid sizes can be 
    generated this way.

    """
    # TODO sort out this function
    if 'filter_method' in kwargs:
        filter_method = kwargs['filter_method']
    else:
        filter_method = 'newtonian'

    if 'filter_kwargs' in kwargs:
        filter_kwargs = kwargs['filter_kwargs']
    else:
        filter_kwargs = {}

    if 'periodic' in kwargs:
        periodic = kwargs.pop('periodic')
    else:
        periodic = False

    if kwargs:
        msg = f'Unrecognised key word {kwargs}'
        ValueError(msg)

    if not isinstance(target_surface, _Surface):
        msg = "input must be of surface type"
        raise ValueError(msg)

    if grid_spacing is 'original':
        if target_surface.grid_spacing is None:
            warnings.warn("Grid spacing of the original surface is not set "
                          "assuming it is 1")
            target_surface.grid_spacing = 1
            grid_spacing = 1
        else:
            grid_spacing = target_surface.grid_spacing

    if extent is 'original':
        if target_surface.extent is not None:
            extent = target_surface.extent
        else:
            extent = [grid_spacing * dim for dim in
                      target_surface.shape]

    target_surface.subtract_polynomial(2)

    if not target_surface.acf:
        target_surface.get_acf()

    surf_out = RandomFilterSurface(extent=extent, grid_spacing=grid_spacing)

    surf_out.linear_transform(target_surface.acf, filter_shape, filter_method,
                              **filter_kwargs)

    quantiles = np.quantile(target_surface.profile,
                            [0.066807, 0.30854, 0.69146, 0.93319])

    surf_out.set_quantiles(quantiles)

    pts_each_dir = [int(sz / grid_spacing) for sz in extent]

    surf_out.descretise(pts_each_dir, periodic, False)

    return surf_out


def _initial_guess(target_acf):
    """Find the initial guess for the filter coeficent matrix in the linear transforms method

    Parameters
    ----------
    target_acf: np.ndarray

    Returns
    -------
    np.ndarray, initial guess of filter coefficents
    """
    n, m = target_acf.shape
    c = np.zeros_like(target_acf)
    for i in range(n):
        for j in range(m):
            c[i, j] = target_acf[i, j] / ((n - i) * (m - j))
    s_sq = target_acf[0, 0] / np.sum((c ** 2).flatten())
    return (c*s_sq**0.5).flatten()


def _jac(alpha: np.ndarray, target_acf: np.ndarray):
    """Jacobian of the filter coeficents matrix

    Parameters
    ----------
    alpha: np.array
        Filter coefficents matrix
    target_acf: np.array
        The target acf

    Returns
    -------

    np.array shape (n,n) where n is alpha.size

    """
    alpha = alpha.reshape(target_acf.shape)
    n, m = target_acf.shape
    alpha_0 = np.pad(alpha, ((n, n), (m, m)), 'constant')
    jacobian_plus = []
    jacobian_minus = []
    for p in range(n):
        jacobian_plus.extend([alpha_0[n + p:2 * n + p, m + q:2 * m + q].flatten() for q in range(m)])
        jacobian_minus.extend([alpha_0[n - p:2 * n - p, m - q:2 * m - q].flatten() for q in range(m)])
    return np.array(jacobian_plus) + np.array(jacobian_minus)


def _opt_func(alpha: np.ndarray, target_acf: np.ndarray):
    """Optimisation function for linear transforms method

    Parameters
    ----------
    alpha: np.ndarray
        The current filter coefficent matrix
    target_acf: np.ndarray
        The target acf array (same shape as alpha)

    Returns
    -------
    np.ndarray of residuals
    """
    alpha = alpha.reshape(target_acf.shape)
    n, m = target_acf.shape
    acf_estimate = np.zeros_like(target_acf)
    for p in range(n):
        for q in range(m):
            acf_estimate[p, q] = sum([sum([alpha[k, l] * alpha[k + p, l + q]
                                           for l in range(m - q)])
                                      for k in range(n - p)])
    return (acf_estimate - target_acf).flatten()
