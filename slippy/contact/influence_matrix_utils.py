import numpy as np
import typing
import warnings
import slippy
from ._material_utils import Loads, Displacements, memoize_components
from scipy.signal import fftconvolve

__all__ = ['guess_loads_from_displacement', 'elastic_influence_matrix', '_solve_im_loading', '_solve_im_displacement',
           'bccg', 'plan_convolve']


def guess_loads_from_displacement(displacements: Displacements, components: dict) -> Loads:
    """
    Defines the starting point for the default loads from displacement method

    This method should be overwritten for non IM based surfaces

    Parameters
    ----------
    displacements: Displacements
        The point wise displacement
    components: dict
        Dict of influence matrix components

    Returns
    -------
    guess_of_loads: Loads
        A named tuple of the loads
    """

    directions = 'xyz'

    loads = dict()
    for direction in directions:
        if displacements.__getattribute__(direction) is None:
            continue
        max_im = max(components[direction * 2].flatten())
        load = displacements.__getattribute__(direction) * max_im
        load = np.nan_to_num(load)
        loads[direction] = load

    return Loads(**loads)


def _solve_im_loading(loads, components) -> Displacements:
    """The meat of the elastic loading algorithm

    Parameters
    ----------
    loads : Loads
        Loads named tuple of N by M arrays of surface loads with fields 'x', 'y', 'z' or any combination
    components : dict
        Components of the influence matrix keys are 'xx', 'xy' ... 'zz' for
        each name the first character represents the displacement and the
        second represents the load, eg. 'xy' is the deflection in the x
        direction caused by a load in the y direction

    Returns
    -------
    displacements : Displacements
        namedtuple of N by M arrays of surface displacements with labels 'x', 'y',
        'z'

    See Also
    --------
    elastic_im
    elastic_loading

    Notes
    -----
    This function has very little error checking and may give unexpected
    results with bad inputs please use elastic_loading to check inputs
    before using this directly

    Components of the influence matrix can be found by elastic_im

    References
    ----------

    Complete boundary element formulation for normal and tangential contact
    problems

    """
    shape = [ld.shape for ld in loads if ld is not None][0]

    displacements = Displacements(np.zeros(shape), np.zeros(shape), np.zeros(shape))

    for c_name, component in components.items():
        load_name = c_name[1]
        dis_name = c_name[0]
        dis_comp = displacements.__getattribute__(dis_name)
        dis_comp += fftconvolve(loads.__getattribute__(load_name), component, mode='same')
    return displacements


def _solve_im_displacement(displacements: Displacements, components: dict, max_it: int,
                           tol: float, initial_guess: Loads) -> typing.Tuple[Loads, Displacements]:
    """ Given displacements find the loads needed to cause then for an influence matrix based material

    Split away from the main function to allow for faster computation avoiding
    calculating the influence matrix for each iteration

    Parameters
    ----------
    displacements : Displacements
        The deflections at the surface with fields 'x', 'y', 'z', use float('nan') to signify free points (eg points
        out of the contact)
    components : dict
        Components of the influence matrix keys are 'xx', 'xy' ... 'zz' for
        each name the first character represents the displacement and the
        second represents the load, eg. 'xy' is the deflection in the x
        direction caused by a load in the y direction
    max_it : int
        The maximum number of iterations before breaking the loop
    tol : float
        The tolerance on the iterations, the loop is ended when the norm of
        the residual is below tol
    initial_guess : Loads
        The initial guess of the surface loads

    Returns
    -------
    loads : Loads
        The loads on the surface to cause the specified displacement with fields 'x' 'y' and 'z'
    full_displacement : Displacements
        The displacement results for the full surface/ problem

    See Also
    --------

    Notes
    -----

    References
    ----------

    Complete boundary element formulation for normal and tangential contact
    problems

    """
    valid_directions = 'xyz'
    def_directions = [vd for vd, el in zip(valid_directions, displacements) if el is not None]

    sub_domains = {dd: np.logical_not(np.isnan(displacements.__getattribute__(dd))) for dd in def_directions}
    sub_domain_sizes = {key: np.sum(value) for (key, value) in sub_domains.items()}

    # find first residual
    loads = initial_guess
    calc_displacements = _solve_im_loading(loads, components)

    residual = np.array([])

    for dd in def_directions:
        residual = np.append(residual, displacements.__getattribute__(dd)[sub_domains[dd]] -
                             calc_displacements.__getattribute__(dd)[sub_domains[dd]])

    # start loop
    search_direction = residual
    itnum = 0
    resid_norm = np.linalg.norm(residual)

    while resid_norm >= tol:
        # put calculated values back into right place

        start = 0
        search_direction_full = dict()
        for dd in def_directions:
            end = start + sub_domain_sizes[dd]
            search_direction_full[dd] = np.zeros_like(displacements.__getattribute__(dd))
            search_direction_full[dd][sub_domains[dd]] = search_direction[start:end]
            start = end

        # find z (equation 25 in ref):
        z_full = _solve_im_loading(Loads(**search_direction_full), components)

        z = np.array([])
        for dd in def_directions:
            z = np.append(z, z_full.__getattribute__(dd)[sub_domains[dd]])

        # find alpha (equation 26)
        alpha = np.matmul(residual, residual) / np.matmul(search_direction, z)

        # update stresses (equation 27)
        update_vals = alpha * search_direction
        start = 0
        for dd in def_directions:
            end = start + sub_domain_sizes[dd]
            loads.__getattribute__(dd)[sub_domains[dd]] += update_vals[start:end]
            start = end

        r_new = residual - alpha * z  # equation 28

        # find new search direction (equation 29)
        beta = np.matmul(r_new, r_new) / np.matmul(residual, residual)
        residual = r_new
        resid_norm = np.linalg.norm(residual)

        search_direction = residual + beta * search_direction

        itnum += 1

        if itnum > max_it:
            msg = (f"Max iteration: ({max_it}) reached without convergence,"
                   f" residual was: {resid_norm}, convergence declared at: {tol}")
            warnings.warn(msg)
            break

    calc_displacements = _solve_im_loading(loads, components)

    return loads, calc_displacements


# noinspection PyTypeChecker
@memoize_components(True)
def elastic_influence_matrix(comp: str, span: typing.Sequence[int], grid_spacing: typing.Sequence[float],
                             shear_mod: float, v: float,
                             shear_mod_2: typing.Optional[float] = None,
                             v_2: typing.Optional[float] = None) -> np.array:
    """Find influence matrix components for an elastic contact problem

    Parameters
    ----------
    comp : str {'xx','xy','xz','yx','yy','yz','zx','zy','zz'}
        The component to be returned
    span: Sequence[int]
        The span of the influence matrix in the x and y directions
    grid_spacing: Sequence[float]
        The grid spacings in the x and y directions
    shear_mod : float
        The shear modulus of the surface material
    v : float
        The Poission's ratio of the surface material
    shear_mod_2: float (optional) None
        The shear modulus of the second surface for a combined stiffness matrix
    v_2: float (optional) None
        The Poisson's ratio of the second surface for a combined stiffness matrix

    Returns
    -------
    C : array
        The influence matrix component requested

    See Also
    --------
    elastic_im

    Notes
    -----

    Don't use this function, used by: elastic_im

    References
    ----------
    Complete boundary element method formulation for normal and tangential
    contact problems

    """
    span = tuple(span)
    try:
        # lets just see how this changes
        # i'-i and j'-j
        idmi = (np.arange(span[1]) - span[1] // 2 + (1 - span[1] % 2))
        jdmj = (np.arange(span[0]) - span[0] // 2 + (1 - span[0] % 2))
        mesh_idmi = np.zeros(span)
        for i in range(span[0]):
            mesh_idmi[i, :] = idmi
        mesh_jdmj = np.zeros(span)
        for i in range(span[1]):
            mesh_jdmj[:, i] = jdmj

    except TypeError:
        raise TypeError("Span should be a tuple of integers")

    k = mesh_idmi + 0.5
    el = mesh_idmi - 0.5
    m = mesh_jdmj + 0.5
    n = mesh_jdmj - 0.5

    hx = grid_spacing[0]
    hy = grid_spacing[1]

    second_surface = (shear_mod_2 is not None) and (v_2 is not None)
    if not second_surface:
        v_2 = 1
        shear_mod_2 = 1

    if (shear_mod_2 is not None) != (v_2 is not None):
        raise ValueError('Either both or neither of the second surface parameters must be set')

    if comp == 'zz':
        c_zz = (hx * (k * np.log((m + np.sqrt(k ** 2 + m ** 2)) / (n + np.sqrt(k ** 2 + n ** 2))) +
                      el * np.log((n + np.sqrt(el ** 2 + n ** 2)) / (m + np.sqrt(el ** 2 + m ** 2)))) +
                hy * (m * np.log((k + np.sqrt(k ** 2 + m ** 2)) / (el + np.sqrt(el ** 2 + m ** 2))) +
                      n * np.log((el + np.sqrt(el ** 2 + n ** 2)) / (k + np.sqrt(k ** 2 + n ** 2)))))

        const = (1 - v) / (2 * np.pi * shear_mod) + second_surface * ((1 - v_2) / (2 * np.pi * shear_mod_2))
        return const * c_zz
    elif comp == 'xx':
        c_xx = (hx * (1 - v) * (k * np.log((m + np.sqrt(k ** 2 + m ** 2)) / (n + np.sqrt(k ** 2 + n ** 2))) +
                                el * np.log(
                (n + np.sqrt(el ** 2 + n ** 2)) / (m + np.sqrt(el ** 2 + m ** 2)))) +
                hy * (m * np.log((k + np.sqrt(k ** 2 + m ** 2)) / (el + np.sqrt(el ** 2 + m ** 2))) +
                      n * np.log((el + np.sqrt(el ** 2 + n ** 2)) / (k + np.sqrt(k ** 2 + n ** 2)))))
        const = 1 / (2 * np.pi * shear_mod) + second_surface * (1 / (2 * np.pi * shear_mod_2))
        return const * c_xx
    elif comp == 'yy':
        c_yy = (hx * (k * np.log((m + np.sqrt(k ** 2 + m ** 2)) / (n + np.sqrt(k ** 2 + n ** 2))) +
                      el * np.log((n + np.sqrt(el ** 2 + n ** 2)) / (m + np.sqrt(el ** 2 + m ** 2)))) +
                hy * (1 - v) * (m * np.log((k + np.sqrt(k ** 2 + m ** 2)) / (el + np.sqrt(el ** 2 + m ** 2))) +
                                n * np.log((el + np.sqrt(el ** 2 + n ** 2)) / (k + np.sqrt(k ** 2 + n ** 2)))))
        const = 1 / (2 * np.pi * shear_mod) + second_surface * (1 / (2 * np.pi * shear_mod_2))
        return const * c_yy
    elif comp in ['xz', 'zx']:
        c_xz = (hy / 2 * (m * np.log((k ** 2 + m ** 2) / (el ** 2 + m ** 2)) +
                          n * np.log((el ** 2 + n ** 2) / (k ** 2 + n ** 2))) +
                hx * (k * (np.arctan(m / k) - np.arctan(n / k)) +
                      el * (np.arctan(n / el) - np.arctan(m / el))))
        const = (2 * v - 1) / (4 * np.pi * shear_mod) + second_surface * (
            (2 * v_2 - 1) / (4 * np.pi * shear_mod_2))
        return const * c_xz
    elif comp in ['yx', 'xy']:
        c_yx = (np.sqrt(hy ** 2 * n ** 2 + hx ** 2 * k ** 2) -
                np.sqrt(hy ** 2 * m ** 2 + hx ** 2 * k ** 2) +
                np.sqrt(hy ** 2 * m ** 2 + hx ** 2 * el ** 2) -
                np.sqrt(hy ** 2 * n ** 2 + hx ** 2 * el ** 2))
        const = v / (2 * np.pi * shear_mod) + second_surface * (v_2 / (2 * np.pi * shear_mod_2))
        return const * c_yx
    elif comp in ['zy', 'yz']:
        c_zy = (hx / 2 * (k * np.log((k ** 2 + m ** 2) / (n ** 2 + k ** 2)) +
                          el * np.log((el ** 2 + n ** 2) / (m ** 2 + el ** 2))) +
                hy * (m * (np.arctan(k / m) - np.arctan(el / m)) +
                      n * (np.arctan(el / n) - np.arctan(k / n))))
        const = (1 - 2 * v) / (4 * np.pi * shear_mod) + second_surface * (1 - 2 * v_2) / (
            4 * np.pi * shear_mod_2)
        return const * c_zy
    else:
        ValueError('component name not recognised: ' + comp + ', components must be lower case')


try:
    import cupy as cp

    def n_pow_2(a):
        return 2 ** int(np.ceil(np.log2(a)))

    def _plan_cuda_convolve(loads: np.ndarray, im: np.ndarray, domain: np.ndarray,
                            circular: typing.Sequence[bool]):
        """Plans an FFT convolution, returns a function to carry out the convolution
        CUDA implementation

        Parameters
        ----------
        loads: np.ndarray
            An example of a loads array, this is not altered or stored
        im: np.ndarray
            The influence matrix component for the transformation, this is not altered but it's fft is stored to
            save time during convolution, this must be larger in every dimension than the loads array
        domain: np.ndarray, optional
            Array with same shape as loads filled with boolean values. If supplied this function will return a
            function which first fills the supplied loads into the domain then computes the convolution.
            This is typically used for finding loads from set displacements as the displacements are often not set
            over the whole surface.
        circular: Sequence[bool], optional (False)
            If True the circular convolution will be calculated, to be used for periodic simulations

        Returns
        -------
        function
            A function which takes a single input of loads and returns the result of the convolution with the original
            influence matrix. If a domain was not supplied the input to the returned function must be exactly the same
            shape as the loads array used in this function. If a domain was specified the length of the loads input to
            the returned function must be the same as the number of non zero elements in domain.

        Notes
        -----
        This function uses CUDA to run on a GPU if your computer dons't have cupy installed this should not have loaded
        if it is for some reason, this can be manually overridden by first importing slippy then setting the CUDA
        variable to False:

        >>> import slippy
        >>> slippy.CUDA = False
        >>> import slippy.contact
        >>> ...

        Examples
        --------
        >>> import numpy as np
        >>> import slippy.contact as c
        >>> result = c.hertz_full([1,1], [np.inf, np.inf], [200e9, 200e9], [0.3, 0.3], 1e4)
        >>> X,Y = np.meshgrid(*[np.linspace(-0.005,0.005,256)]*2)
        >>> grid_spacing = X[1][1]-X[0][0]
        >>> loads = result['pressure_f'](X,Y)
        >>> disp_analytical = result['surface_displacement_b_f'][0](X,Y)['uz']
        >>> im = c.elastic_influence_matrix('zz', (512,512), (grid_spacing,grid_spacing), 200e9/(2*(1+0.3)), 0.3)
        >>> convolve_func = plan_convolve(loads, im, None, [False, False])
        >>> disp_numerical = convolve_func(loads)

        """
        loads = cp.asarray(loads)
        im = cp.asarray(im)
        im_shape_orig = im.shape
        if domain is not None:
            domain = cp.asarray(domain)
        input_shape = []
        for i in range(2):
            if circular[i]:
                assert loads.shape[i] == im.shape[i], "For circular convolution loads and im must be same shape"
                input_shape.append(loads.shape[i])
            else:
                input_shape.append(2 * n_pow_2(max(loads.shape[i], im.shape[i])))
        input_shape = tuple(input_shape)

        forward_trans = cp.fft.fft2
        backward_trans = cp.fft.ifft2
        shape_diff = [[0, (b - a)] for a, b in zip(im.shape, input_shape)]

        norm_inv = (input_shape[0] * input_shape[1]) ** 0.5
        norm = 1 / norm_inv
        im = cp.pad(im, shape_diff, mode='constant')
        im = cp.roll(im, tuple(-((sz - 1) // 2) for sz in im_shape_orig), (-2, -1))
        fft_im = forward_trans(im, s=input_shape) * norm
        shape = loads.shape
        dtype = loads.dtype

        def inner_with_domain(sub_loads, ignore_domain=False):
            full_loads = cp.zeros(shape, dtype=dtype)
            full_loads[domain] = sub_loads
            fft_loads = forward_trans(full_loads, s=input_shape)
            full = norm_inv * cp.real(backward_trans(fft_loads * fft_im))
            full = full[:full_loads.shape[0], :full_loads.shape[1]]
            if ignore_domain:
                return full
            return full[domain]

        def inner_no_domain(full_loads):
            full_loads = cp.asarray(full_loads)
            fft_loads = forward_trans(full_loads, s=input_shape)
            full = norm_inv * cp.real(backward_trans(fft_loads * fft_im))
            full = full[:full_loads.shape[0], :full_loads.shape[1]]
            return full

        if domain is None:
            return inner_no_domain
        else:
            return inner_with_domain

    def _cuda_bccg(f: typing.Callable, b: typing.Sequence, tol: float, max_it: int, x0: typing.Sequence,
                   min_pressure: float = 0.0, max_pressure: float = cp.inf, k_inn=1) -> typing.Tuple[cp.ndarray, bool]:
        """
        The Bound-Constrained Conjugate Gradient Method for Non-negative Matrices
        CUDA implementation

        Parameters
        ----------
        f: Callable
            A function equivalent to multiplication by a non negative n by n matrix must work with cupy arrays.
            Typically this function will be generated by slippy.contact.plan_convolve, this will guarantee
            compatibility with different versions of this function (FFTW and CUDA).
        b: array
            1 by n array of displacements
        tol: float
            The tolerance on the result
        max_it: int
            The maximum number of iterations used
        x0: array
            An initial guess of the solution
        min_pressure: float, optional (0)
            The minimum allowable pressure at each node, defaults to 0
        max_pressure: float, optional (inf)
            The maximum allowable pressure at each node, defaults to inf, for purely elastic contacts
        k_inn: int

        Returns
        -------
        x: cp.array
            The solution to the system f(x)-b = 0 with the constraints applied.

        Notes
        -----
        This function uses the method described in the reference below, with some modification.
        Firstly, this method allows both a minimum and maximum force to be set simulating quasi plastic regimes. The
        code has also been optimised in several places and importantly this version has also been modified to run
        on a GPU through cupy.

        If you do not have a CUDA compatible GPU, slippy can be imported while falling back to the fftw version
        by first importing slippy then patching the CUDA variable to False:

        >>> import slippy
        >>> slippy.CUDA = False
        >>> import slippy.contact
        >>> ...

        Though this should happen automatically if you don't have cupy installed.

        References
        ----------
        Vollebregt, E.A.H. The Bound-Constrained Conjugate Gradient Method for Non-negative Matrices. J Optim
        Theory Appl 162, 931–953 (2014). https://doi.org/10.1007/s10957-013-0499-x

        Examples
        --------

        """
        # if you use np or most built ins in this function at all it will slow it down a lot!

        # initialize
        b = cp.asarray(b)
        x = cp.clip(cp.asarray(x0), min_pressure, max_pressure)
        g = f(x) - b
        msk_bnd_0 = cp.logical_and(x <= 0, g >= 0)
        msk_bnd_max = cp.logical_and(x >= max_pressure, g <= 0)
        n_bound = cp.sum(msk_bnd_0) + cp.sum(msk_bnd_max)
        n = b.size
        n_free = n - n_bound
        small = 1e-14
        it = 0
        it_inn = 0
        rho_prev = cp.nan
        rho = 0.0
        failed = False

        while True:
            it += 1
            it_inn += 1
            x_prev = x
            if it > 1:
                rho_prev = rho
            r = -g
            r[msk_bnd_0] = 0
            r[msk_bnd_max] = 0
            rho = cp.dot(r, r)
            if it > 1:
                beta_pr = (rho - cp.dot(r, r)) / rho_prev
                p = r + np.max([beta_pr, 0])  # np ok here they are both just scalars
            else:
                p = r
            p[msk_bnd_0] = 0
            p[msk_bnd_max] = 0
            # compute tildex optimisation ignoring the bounds
            q = f(p)
            if it_inn < k_inn:
                q[msk_bnd_0] = cp.nan
                p[msk_bnd_max] = cp.nan
            alpha = cp.dot(r, p) / cp.dot(p, q)
            x = x + alpha * p

            rms_xk = cp.linalg.norm(x) / cp.sqrt(n_free)
            rms_upd = cp.linalg.norm(x - x_prev) / cp.sqrt(n_free)
            upd = rms_upd / rms_xk

            # project onto feasible domain
            changed = False
            outer_it = it_inn >= k_inn or upd < tol

            if outer_it:
                msk_prj_0 = x < -small
                if cp.any(msk_prj_0):
                    x[msk_prj_0] = 0
                    msk_bnd_0[msk_prj_0] = True
                    changed = True
                msk_prj_max = x >= max_pressure * (1 + small)
                if cp.any(msk_prj_max):
                    x[msk_prj_max] = max_pressure
                    msk_bnd_max[msk_prj_max] = True
                    changed = True

            if changed or (outer_it and k_inn > 1):
                g = f(x) - b
            else:
                g = g + alpha * q

            check_grad = outer_it

            if check_grad:
                msk_rel = cp.logical_or(cp.logical_and(msk_bnd_0, g < -small), cp.logical_and(msk_bnd_max, g > small))
                if cp.any(msk_rel):
                    msk_bnd_0[msk_rel] = False
                    msk_bnd_max[msk_rel] = False
                    changed = True

            if changed:
                n_free = n - cp.sum(msk_bnd_0) - cp.sum(msk_bnd_max)

            if not n_free:
                print("No free nodes")
                warnings.warn("No free nodes for BCCG iterations")
                failed = True
                break

            if outer_it:
                it_inn = 0

            if it > max_it:
                print("Max iterations")
                warnings.warn("Bound constrained conjugate gradient iterations failed to converge")
                failed = True
                break

            if outer_it and (not changed) and upd < tol:
                break

        return x, bool(failed)
except ImportError:
    _plan_cuda_convolve = None
    _cuda_bccg = None

try:
    import pyfftw

    def _plan_fftw_convolve(loads: np.ndarray, im: np.ndarray, domain: np.ndarray, circular: typing.Sequence[bool]):
        """Plans an FFT convolution, returns a function to carry out the convolution
        FFTW implementation

        Parameters
        ----------
        loads: np.ndarray
            An example of a loads array, this is not altered or stored
        im: np.ndarray
            The influence matrix component for the transformation, this is not altered but it's fft is stored to
            save time during convolution, this must be larger in every dimension than the loads array
        domain: np.ndarray, optional (None)
            Array with same shape as loads filled with boolean values. If supplied this function will return a
            function which first fills the supplied loads into the domain then computes the convolution.
            This is typically used for finding loads from set displacements as the displacements are often not set
            over the whole surface.
        circular: Sequence[bool]
            If True the circular convolution will be calculated, to be used for periodic simulations

        Returns
        -------
        function
            A function which takes a single input of loads and returns the result of the convolution with the original
            influence matrix. If a domain was not supplied the input to the returned function must be exactly the same
            shape as the loads array used in this function. If a domain was specified the length of the loads input to
            the returned function must be the same as the number of non zero elements in domain.

        Notes
        -----
        This function uses FFTW, if you want to use the CUDA implementation make sure that cupy is installed and
        importable. If cupy can be imported slippy will use the CUDA implementations by default

        Examples
        --------
        >>> import numpy as np
        >>> import slippy.contact as c
        >>> result = c.hertz_full([1,1], [np.inf, np.inf], [200e9, 200e9], [0.3, 0.3], 1e4)
        >>> X,Y = np.meshgrid(*[np.linspace(-0.005,0.005,256)]*2)
        >>> grid_spacing = X[1][1]-X[0][0]
        >>> loads = result['pressure_f'](X,Y)
        >>> disp_analytical = result['surface_displacement_b_f'][0](X,Y)['uz']
        >>> im = c.elastic_influence_matrix('zz', (512,512), (grid_spacing,grid_spacing), 200e9/(2*(1+0.3)), 0.3)
        >>> convolve_func = plan_convolve(loads, im, None, [False, False])
        >>> disp_numerical = convolve_func(loads)

        """
        loads = np.asarray(loads)
        im = np.asarray(im)
        im_shape_orig = im.shape
        if domain is not None:
            domain = np.asarray(domain, dtype=np.bool)
        input_shape = []
        for i in range(2):
            if circular[i]:
                assert loads.shape[i] == im.shape[i], "For circular convolution loads and im must be same shape"
                input_shape.append(loads.shape[i])
            else:
                input_shape.append(2 * pyfftw.next_fast_len(max(loads.shape[i], im.shape[i])))
        input_shape = tuple(input_shape)

        fft_shape = [input_shape[0], input_shape[1] // 2 + 1]
        in_empty = pyfftw.empty_aligned(input_shape, dtype=loads.dtype)
        out_empty = pyfftw.empty_aligned(fft_shape, dtype='complex128')
        ret_empty = pyfftw.empty_aligned(input_shape, dtype=loads.dtype)
        forward_trans = pyfftw.FFTW(in_empty, out_empty, axes=(0, 1),
                                    direction='FFTW_FORWARD', threads=slippy.CORES)
        backward_trans = pyfftw.FFTW(out_empty, ret_empty, axes=(0, 1),
                                     direction='FFTW_BACKWARD', threads=slippy.CORES)
        norm_inv = forward_trans.N ** 0.5
        norm = 1 / norm_inv

        shape_diff = [[0, (b - a)] for a, b in zip(im.shape, input_shape)]
        im = np.pad(im, shape_diff, 'constant')
        im = np.roll(im, tuple(-((sz - 1) // 2) for sz in im_shape_orig), (-2, -1))
        fft_im = forward_trans(im) * norm

        shape_diff_loads = [[0, (b - a)] for a, b in zip(loads.shape, input_shape)]

        def inner_no_domain(full_loads):
            loads_pad = np.pad(full_loads, shape_diff_loads, 'constant')
            full = backward_trans(forward_trans(loads_pad) * fft_im)
            return norm_inv * full[:full_loads.shape[0], :full_loads.shape[1]]

        shape = loads.shape
        dtype = loads.dtype

        def inner_with_domain(sub_loads, ignore_domain=False):
            full_loads = np.zeros(shape, dtype=dtype)
            full_loads[domain] = sub_loads
            loads_pad = np.pad(full_loads, shape_diff_loads, 'constant')
            full = backward_trans(forward_trans(loads_pad) * fft_im)
            same = norm_inv * full[:full_loads.shape[0], :full_loads.shape[1]]
            if ignore_domain:
                return same
            return same[domain]

        if domain is None:
            return inner_no_domain
        else:
            return inner_with_domain

    def _fftw_bccg(f: typing.Callable, b: np.ndarray, tol: float, max_it: int, x0: np.ndarray,
                   min_pressure: float = 0, max_pressure: float = np.inf, k_inn=1) -> typing.Tuple[np.ndarray, bool]:
        """
        The Bound-Constrained Conjugate Gradient Method for Non-negative Matrices
        FFTW implementation

        Parameters
        ----------
        f: Callable
            A function equivalent to multiplication by a non negative n by n matrix must work with cupy arrays.
            Typically this function will be generated by slippy.contact.plan_convolve, this will guarantee
            compatibility with different versions of this function (FFTW and CUDA).
        b: array
            1 by n array of displacements
        tol: float
            The tolerance on the result
        max_it: int
            The maximum number of iterations used
        x0: array
            An initial guess of the solution must be 1 by n
        min_pressure: float, optional (0)
            The minimum allowable pressure at each node, defaults to 0
        max_pressure: float, optional (inf)
            The maximum allowable pressure at each node, defaults to inf, for purely elastic contacts
        k_inn: int, optional (1)

        Returns
        -------
        x: cp.array
            The solution to the system f(x)-b = 0 with the constraints applied.

        Notes
        -----
        This function uses the method described in the reference below, with some modification.
        Firstly, this method allows both a minimum and maximum force to be set simulating quasi plastic regimes. The
        code has also been optimised in several places and updated to allow fft convolution in place of the large matrix
        multiplication step.

        References
        ----------
        Vollebregt, E.A.H. The Bound-Constrained Conjugate Gradient Method for Non-negative Matrices. J Optim
        Theory Appl 162, 931–953 (2014). https://doi.org/10.1007/s10957-013-0499-x

        Examples
        --------

        """
        # initialize
        x = np.clip(x0, min_pressure, max_pressure)
        g = f(x) - b
        msk_bnd_0 = np.logical_and(x <= 0, g >= 0)
        msk_bnd_max = np.logical_and(x >= max_pressure, g <= 0)
        n_bound = np.sum(msk_bnd_0) + np.sum(msk_bnd_max)
        n = b.size
        n_free = n - n_bound
        small = 1e-14
        it = 0
        it_inn = 0
        rho_prev = np.nan
        rho = 0.0
        failed = False

        while True:
            it += 1
            it_inn += 1
            x_prev = x
            if it > 1:
                rho_prev = rho
            r = -g
            r[msk_bnd_0] = 0
            r[msk_bnd_max] = 0
            rho = np.dot(r, r)
            if it > 1:
                beta_pr = (rho - np.dot(r, r)) / rho_prev
                p = r + np.max([beta_pr, 0])
            else:
                p = r
            p[msk_bnd_0] = 0
            p[msk_bnd_max] = 0
            # compute tildex optimisation ignoring the bounds
            q = f(p)
            if it_inn < k_inn:
                q[msk_bnd_0] = np.nan
                p[msk_bnd_max] = np.nan
            alpha = np.dot(r, p) / np.dot(p, q)
            x = x + alpha * p

            rms_xk = np.linalg.norm(x) / np.sqrt(n_free)
            rms_upd = np.linalg.norm(x - x_prev) / np.sqrt(n_free)
            upd = rms_upd / rms_xk

            # project onto feasible domain
            changed = False
            outer_it = it_inn >= k_inn or upd < tol

            if outer_it:
                msk_prj_0 = x < -small
                if any(msk_prj_0):
                    x[msk_prj_0] = 0
                    msk_bnd_0[msk_prj_0] = True
                    changed = True
                msk_prj_max = x >= max_pressure * (1 + small)
                if any(msk_prj_max):
                    x[msk_prj_max] = max_pressure
                    msk_bnd_max[msk_prj_max] = True
                    changed = True

            if changed or (outer_it and k_inn > 1):
                g = f(x) - b
            else:
                g = g + alpha * q

            check_grad = outer_it

            if check_grad:
                msk_rel = np.logical_and(msk_bnd_0, g < -small) + np.logical_and(msk_bnd_max, g > small)
                if any(msk_rel):
                    msk_bnd_0[msk_rel] = False
                    msk_bnd_max[msk_rel] = False
                    changed = True

            if changed:
                n_free = n - np.sum(msk_bnd_0) - np.sum(msk_bnd_max)

            if not n_free:
                print("No free nodes")
                warnings.warn("No free nodes for BCCG iterations")
                failed = True
                break

            if outer_it:
                it_inn = 0

            if it > max_it:
                warnings.warn("Bound constrained conjugate gradient iterations failed to converge")
                print("Max iterations")
                failed = True
                break

            if outer_it and (not changed) and upd < tol:
                break
        return x, bool(failed)

except ImportError:
    _plan_fftw_convolve = None
    _fftw_bccg = None


def plan_convolve(loads, im, domain: np.ndarray = None, circular: typing.Union[bool, typing.Sequence[bool]] = False):
    """Plans an FFT convolution, returns a function to carry out the convolution
    CUDA / FFTW implementation

    Parameters
    ----------
    loads: np.ndarray
        An example of a loads array, this is not altered or stored
    im: np.ndarray
        The influence matrix component for the transformation, this is not altered but it's fft is stored to
        save time during convolution, this must be larger in every dimension than the loads array
    domain: np.ndarray, optional (None)
        Array with same shape as loads filled with boolean values. If supplied this function will return a
        function which first fills the supplied loads into the domain then computes the convolution.
        This is typically used for finding loads from set displacements as the displacements are often not set
        over the whole surface.
    circular: bool or sequence of bool, optional (False)
        If True the circular convolution will be computed, to be used for periodic simulations. Alternatively a 2
        element sequence of bool can be provided specifying which axes are to be treated as periodic.

    Returns
    -------
    function
        A function which takes a single input of loads and returns the result of the convolution with the original
        influence matrix. If a domain was not supplied the input to the returned function must be exactly the same
        shape as the loads array used in this function. If a domain was specified the length of the loads input to
        the returned function must be the same as the number of non zero elements in domain.

    Notes
    -----
    By default this function uses CUDA to run on a GPU if your computer dons't have cupy installed this should not
    have loaded if it is for some reason, this can be manually overridden by first importing slippy then patching the
    CUDA variable to False:

    >>> import slippy
    >>> slippy.CUDA = False
    >>> import slippy.contact
    >>> ...

    If the CUDA version is used cp.asnumpy() will need to be called on the output for compatibility with np arrays,
    Likewise the inputs to the convolution function should be cupy arrays.

    Examples
    --------
    >>> import numpy as np
    >>> import slippy.contact as c
    >>> result = c.hertz_full([1,1], [np.inf, np.inf], [200e9, 200e9], [0.3, 0.3], 1e4)
    >>> X,Y = np.meshgrid(*[np.linspace(-0.005,0.005,256)]*2)
    >>> grid_spacing = X[1][1]-X[0][0]
    >>> loads = result['pressure_f'](X,Y)
    >>> disp_analytical = result['surface_displacement_b_f'][0](X,Y)['uz']
    >>> im = c.elastic_influence_matrix('zz', (512,512), (grid_spacing,grid_spacing), 200e9/(2*(1+0.3)), 0.3)
    >>> convolve_func = plan_convolve(loads, im)
    >>> disp_numerical = convolve_func(loads)

    """
    if isinstance(circular, int):
        circular = [circular, ]*2
    try:
        length = len(circular)
    except TypeError:
        raise TypeError('Type of circular not recognised, should be a bool or a 2 element sequence of bool')

    if length != 2:
        raise ValueError(f"Circular must be a bool or a 2 element list of bool, length was {length}")

    if slippy.CUDA:
        return _plan_cuda_convolve(loads, im, domain, circular)
    else:
        return _plan_fftw_convolve(loads, im, domain, circular)


def bccg(f: typing.Callable, b: np.ndarray, tol: float, max_it: int, x0: np.ndarray,
         min_pressure: float = 0.0, max_pressure: float = np.inf, k_inn=1) -> typing.Tuple[np.ndarray, bool]:
    """
    The Bound-Constrained Conjugate Gradient Method for Non-negative Matrices
    CUDA implementation

    Parameters
    ----------
    f: Callable
        A function equivalent to multiplication by a non negative n by n matrix must work with cupy arrays.
        Typically this function will be generated by slippy.contact.plan_convolve, this will guarantee
        compatibility with different versions of this function (FFTW and CUDA).
    b: array
        1 by n array of displacements
    tol: float
        The tolerance on the result
    max_it: int
        The maximum number of iterations used
    x0: array
        An initial guess of the solution
    min_pressure: float, optional (0)
        The minimum allowable pressure at each node, defaults to 0
    max_pressure: float, optional (inf)
        The maximum allowable pressure at each node, defaults to inf, for purely elastic contacts
    k_inn: int

    Returns
    -------
    x: cp.array/ np.array
        The solution to the system f(x)-b = 0 with the constraints applied.
    failed: bool
        True if the solution failed to converge, this will also produce a warning

    Notes
    -----
    This function uses the method described in the reference below, with some modification.
    Firstly, this method allows both a minimum and maximum force to be set simulating quasi plastic regimes. The
    code has also been optimised in several places and importantly this version has also been modified to run
    on a GPU through cupy.

    If you do not have a CUDA compatible GPU, slippy can be imported while falling back to the fftw version
    by first importing slippy then patching the CUDA variable to False:

    >>> import slippy
    >>> slippy.CUDA = False
    >>> import slippy.contact
    >>> ...

    Though this should happen automatically if you don't have cupy installed.

    References
    ----------
    Vollebregt, E.A.H. The Bound-Constrained Conjugate Gradient Method for Non-negative Matrices. J Optim
    Theory Appl 162, 931–953 (2014). https://doi.org/10.1007/s10957-013-0499-x

    Examples
    --------

    """
    if max_it is None:
        max_it = x0.size
    if slippy.CUDA:
        return _cuda_bccg(f, b, tol, max_it, x0, min_pressure, max_pressure, k_inn)
    return _fftw_bccg(f, b, tol, max_it, x0, min_pressure, max_pressure, k_inn)
