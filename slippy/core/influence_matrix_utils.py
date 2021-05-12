import numpy as np
import typing
import warnings
import slippy
import functools

__all__ = ['guess_loads_from_displacement', 'bccg', 'plan_convolve', 'plan_multi_convolve', 'plan_coupled_convolve']


def guess_loads_from_displacement(displacements_z: np.array, zz_component: np.array) -> np.array:
    """
    Defines the starting point for the default loads from displacement method

    Parameters
    ----------
    displacements_z: np.array
        The point wise displacement
    zz_component: dict
        Dict of influence matrix components

    Returns
    -------
    guess_of_loads: np.array
        A named tuple of the loads
    """

    max_im = max(zz_component)
    return displacements_z / max_im


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
        forward_trans = functools.partial(cp.fft.fft2, s=input_shape)
        backward_trans = functools.partial(cp.fft.ifft2, s=input_shape)
        shape_diff = [[0, (b - a)] for a, b in zip(im.shape, input_shape)]

        norm_inv = (input_shape[0] * input_shape[1]) ** 0.5
        norm = 1 / norm_inv
        im = cp.pad(im, shape_diff, mode='constant')
        im = cp.roll(im, tuple(-((sz - 1) // 2) for sz in im_shape_orig), (-2, -1))
        fft_im = forward_trans(im) * norm
        shape = loads.shape
        dtype = loads.dtype

        def inner_with_domain(sub_loads, ignore_domain=False):
            full_loads = cp.zeros(shape, dtype=dtype)
            full_loads[domain] = sub_loads
            fft_loads = forward_trans(full_loads)
            full = norm_inv * cp.real(backward_trans(fft_loads * fft_im))
            full = full[:full_loads.shape[0], :full_loads.shape[1]]
            if ignore_domain:
                return full
            return full[domain]

        def inner_no_domain(full_loads):
            full_loads = cp.asarray(full_loads)
            if full_loads.shape == shape:
                flat = False
            else:
                full_loads = cp.reshape(full_loads, loads.shape)
                flat = True
            fft_loads = forward_trans(full_loads)
            full = norm_inv * cp.real(backward_trans(fft_loads * fft_im))
            full = full[:full_loads.shape[0], :full_loads.shape[1]]
            if flat:
                full = full.flatten()
            return full

        if domain is None:
            return inner_no_domain
        else:
            return inner_with_domain


    def _plan_cuda_multi_convolve(loads: np.ndarray, ims: np.ndarray, domain: np.ndarray = None,
                                  circular: typing.Sequence[bool] = (False, False)):
        """Plans an FFT convolution, returns a function to carry out the convolution
        CUDA implementation

        Parameters
        ----------
        loads: np.ndarray
            An example of a loads array, this is not altered or stored
        ims: np.ndarray
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
        im = cp.asarray(ims[0])
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
        forward_trans = functools.partial(cp.fft.fft2, s=input_shape)
        backward_trans = functools.partial(cp.fft.ifft2, s=input_shape)
        shape_diff = [[0, (b - a)] for a, b in zip(im.shape, input_shape)]

        norm_inv = (input_shape[0] * input_shape[1]) ** 0.5
        norm = 1 / norm_inv
        fft_ims = cp.zeros((len(ims), *input_shape), dtype=cp.complex128)
        for i in range(len(ims)):
            im = cp.asarray(ims[i])
            im = cp.pad(im, shape_diff, mode='constant')
            im = cp.roll(im, tuple(-((sz - 1) // 2) for sz in im_shape_orig), (-2, -1))
            fft_ims[i] = forward_trans(im) * norm
        shape = loads.shape
        dtype = loads.dtype

        def inner_no_domain(full_loads):
            full_loads = cp.asarray(full_loads)
            all_results = cp.zeros((len(fft_ims), *full_loads.shape))
            if full_loads.shape == shape:
                flat = False
            else:
                full_loads = cp.reshape(full_loads, loads.shape)
                flat = True
            fft_loads = forward_trans(full_loads)
            for i in range(len(ims)):
                full = norm_inv * cp.real(backward_trans(fft_loads * fft_ims[i]))
                full = full[:full_loads.shape[0], :full_loads.shape[1]]
                if flat:
                    full = full.flatten()
                all_results[i] = full
            return all_results

        def inner_with_domain(sub_loads, ignore_domain=False):
            full_loads = cp.zeros(shape, dtype=dtype)
            full_loads[domain] = sub_loads

            if ignore_domain:
                all_results = cp.zeros((len(fft_ims), *full_loads.shape))
            else:
                all_results = cp.zeros((len(fft_ims), *sub_loads.shape))

            fft_loads = forward_trans(full_loads)
            for i in range(len(ims)):
                full = norm_inv * cp.real(backward_trans(fft_loads * fft_ims[i]))
                full = full[:full_loads.shape[0], :full_loads.shape[1]]
                if ignore_domain:
                    all_results[i] = full
                else:
                    all_results[i] = full[domain]
            return all_results

        if domain is None:
            return inner_no_domain
        else:
            return inner_with_domain


    def _cuda_bccg(f: typing.Callable, b: typing.Sequence, tol: float, max_it: int, x0: typing.Sequence,
                   min_pressure: float = 0.0, max_pressure: typing.Union[float, typing.Sequence] = cp.inf,
                   k_inn=1) -> typing.Tuple[cp.ndarray, bool]:
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
        try:
            float(max_pressure)
            max_is_float = True
        except TypeError:
            max_is_float = False
            max_pressure = cp.array(max_pressure)

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
        r, p, r_prev = 0, 0, 0
        failed = False

        while True:
            it += 1
            it_inn += 1
            x_prev = x
            if it > 1:
                r_prev = r
                rho_prev = rho
            r = -g
            r[msk_bnd_0] = 0
            r[msk_bnd_max] = 0
            rho = cp.dot(r, r)
            if it > 1:
                beta_pr = (rho - cp.dot(r, r_prev)) / rho_prev
                p = r + max([beta_pr, 0]) * p
            else:
                p = r
            p[msk_bnd_0] = 0
            p[msk_bnd_max] = 0
            # compute tildex optimisation ignoring the bounds
            q = f(p)
            if it_inn < k_inn:
                q[msk_bnd_0] = cp.nan
                q[msk_bnd_max] = cp.nan
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
                    if max_is_float:
                        x[msk_prj_max] = max_pressure
                    else:
                        x[msk_prj_max] = max_pressure[msk_prj_max]
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
    _plan_cuda_multi_convolve = None
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

        shape = loads.shape
        dtype = loads.dtype

        def inner_no_domain(full_loads):
            if full_loads.shape == shape:
                flat = False
            else:
                full_loads = np.reshape(full_loads, loads.shape)
                flat = True
            loads_pad = np.pad(full_loads, shape_diff_loads, 'constant')
            full = backward_trans(forward_trans(loads_pad) * fft_im)
            full = norm_inv * full[:full_loads.shape[0], :full_loads.shape[1]]
            if flat:
                full = full.flatten()
            return full

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


    def _plan_fftw_multi_convolve(loads: np.ndarray, ims: np.ndarray, domain: np.ndarray = None,
                                  circular: typing.Sequence[bool] = (False, False)):
        """Plans an FFT convolution, returns a function to carry out the convolution
        FFTW implementation

        Parameters
        ----------
        loads: np.ndarray
            An example of a loads array, this is not altered or stored
        ims: np.ndarray
            The influence matrix components for the transformation, this is not altered but it's fft is stored to
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
        im = np.asarray(ims[0])
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
        input_shape = (len(ims),) + tuple(input_shape)

        fft_shape = [input_shape[0], input_shape[1], input_shape[2] // 2 + 1]
        ims_in_empty = pyfftw.empty_aligned(input_shape, dtype='float64')
        ims_out_empty = pyfftw.empty_aligned(fft_shape, dtype='complex128')
        loads_in_empty = pyfftw.empty_aligned(input_shape[-2:], dtype='float64')
        loads_out_empty = pyfftw.empty_aligned(fft_shape[-2:], dtype='complex128')
        ret_empty = pyfftw.empty_aligned(input_shape, dtype='float64')

        forward_trans_ims = pyfftw.FFTW(ims_in_empty, ims_out_empty, axes=(1, 2),
                                        direction='FFTW_FORWARD', threads=slippy.CORES)
        forward_trans_loads = pyfftw.FFTW(loads_in_empty, loads_out_empty, axes=(0, 1),
                                          direction='FFTW_FORWARD', threads=slippy.CORES)
        backward_trans_ims = pyfftw.FFTW(ims_out_empty, ret_empty, axes=(1, 2),
                                         direction='FFTW_BACKWARD', threads=slippy.CORES)
        norm_inv = forward_trans_loads.N ** 0.5
        norm = 1 / norm_inv

        shape_diff = [[0, (b - a)] for a, b in zip((len(ims),) + im.shape, input_shape)]
        ims = np.pad(ims, shape_diff, 'constant')
        ims = np.roll(ims, tuple(-((sz - 1) // 2) for sz in im_shape_orig), (-2, -1))
        fft_ims = forward_trans_ims(ims) * norm

        shape_diff_loads = [[0, (b - a)] for a, b in zip(loads.shape, input_shape[1:])]

        shape = loads.shape
        dtype = loads.dtype

        def inner_no_domain(full_loads):
            if full_loads.shape == shape:
                flat = False
            else:
                full_loads = np.reshape(full_loads, shape)
                flat = True
            loads_pad = np.pad(full_loads, shape_diff_loads, 'constant')
            fft_loads = np.expand_dims(forward_trans_loads(loads_pad), 0)
            full = backward_trans_ims(fft_loads * fft_ims)
            full = norm_inv * full[:, :full_loads.shape[0], :full_loads.shape[1]]
            if flat:
                return full.reshape((len(fft_ims), -1))
            return full

        def inner_with_domain(sub_loads, ignore_domain=False):
            full_loads = np.zeros(shape, dtype=dtype)
            full_loads[domain] = sub_loads

            loads_pad = np.pad(full_loads, shape_diff_loads, 'constant')
            fft_loads = np.expand_dims(forward_trans_loads(loads_pad), 0)
            full = backward_trans_ims(fft_loads * fft_ims)
            same = norm_inv * full[:, :full_loads.shape[0], :full_loads.shape[1]]
            if ignore_domain:
                return same
            else:
                return same[:, domain]

        if domain is None:
            return inner_no_domain
        else:
            return inner_with_domain


    def _fftw_bccg(f: typing.Callable, b: np.ndarray, tol: float, max_it: int, x0: np.ndarray,
                   min_pressure: float = 0, max_pressure: typing.Union[float, typing.Sequence] = np.inf,
                   k_inn=1) -> typing.Tuple[np.ndarray, bool]:
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
        try:
            float(max_pressure)
            max_is_float = True
        except TypeError:
            max_is_float = False

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
        r, p, r_prev = 0, 0, 0
        failed = False

        while True:
            it += 1
            it_inn += 1
            x_prev = x
            if it > 1:
                r_prev = r
                rho_prev = rho
            r = -g
            r[msk_bnd_0] = 0
            r[msk_bnd_max] = 0
            rho = np.dot(r, r)
            if it > 1:
                beta_pr = (rho - np.dot(r, r_prev)) / rho_prev
                p = r + np.max([beta_pr, 0]) * p
            else:
                p = r
            p[msk_bnd_0] = 0
            p[msk_bnd_max] = 0
            # compute tildex optimisation ignoring the bounds
            q = f(p)
            if it_inn < k_inn:
                q[msk_bnd_0] = np.nan
                q[msk_bnd_max] = np.nan  # changed from p[... to q[... 8/12/20
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
                if np.any(msk_prj_0):
                    x[msk_prj_0] = 0
                    msk_bnd_0[msk_prj_0] = True
                    changed = True
                msk_prj_max = x >= max_pressure * (1 + small)
                if np.any(msk_prj_max):
                    if max_is_float:
                        x[msk_prj_max] = max_pressure
                    else:
                        x[msk_prj_max] = max_pressure[msk_prj_max]
                    msk_bnd_max[msk_prj_max] = True
                    changed = True

            if changed or (outer_it and k_inn > 1):
                g = f(x) - b
            else:
                g = g + alpha * q

            check_grad = outer_it

            if check_grad:
                msk_rel = np.logical_and(msk_bnd_0, g < -small) + np.logical_and(msk_bnd_max, g > small)
                if np.any(msk_rel):
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
    _plan_fftw_multi_convolve = None
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
        circular = [circular, ] * 2
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


def plan_multi_convolve(loads, ims: np.array, domain: np.ndarray = None,
                        circular: typing.Union[bool, typing.Sequence[bool]] = False):
    """Plans a set of FFT convolutions, returns a function to carry out the convolution
    CUDA / FFTW implementation

    Parameters
    ----------
    loads: np.ndarray
        An example of a loads array, this is not altered or stored
    ims: np.ndarray
        The influence matrix components for the transformation, this is not altered but it's fft is stored to
        save time during convolution, this must be larger or equal in every dimension than the loads array.
        The first axis should index the compoents, eg a shape of (8, 64, 64), represents 8 kernels which are 64 by 64
        each.
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
        A function which takes a single input of loads and returns the result of the convolutions with the original
        influence matrcies. If a domain was not supplied the input to the returned function must be exactly the same
        shape as the loads array used in this function. If a domain was specified the length of the loads input to
        the returned function must be the same as the number of non zero elements in domain. In every case the output
        will be the same shape as the input, with an added first axis with the same length as the number of influence
        matricies supplied.

    Notes
    -----
    By default this function uses CUDA to run on a GPU if your computer dons't have cupy installed this should not
    have loaded if it is for some reason, this can be manually overridden by first importing slippy then patching the
    CUDA variable to False:

    >>> import slippy
    >>> slippy.CUDA = False
    >>> import slippy.contact
    >>> ...

    If the CUDA version is used cp.asnumpy() or slippy.asnumpy() will need to be called on the output for compatibility
    with numpy arrays.

    Examples
    --------
    The example below shows how to compute the x, y and z displacements under a normal load
    >>> import numpy as np
    >>> import itertools
    >>> import slippy.core as core
    >>> comps = [a + b for a, b in itertools.product('xyz', 'z')]
    >>> e_im = core.elastic_material.elastic_influence_matrix
    >>> ims = [e_im(comp, (64,64), [1e-6]*2, 200e9, 0.3) for comp in comps]
    >>> convolve_func = plan_convolve(loads, ims)
    >>> disp_numerical = convolve_func(loads)
    """
    if isinstance(circular, int):
        circular = [circular, ] * 2
    try:
        length = len(circular)
    except TypeError:
        raise TypeError('Type of circular not recognised, should be a bool or a 2 element sequence of bool')

    if length != 2:
        raise ValueError(f"Circular must be a bool or a 2 element list of bool, length was {length}")

    if slippy.CUDA:
        return _plan_cuda_multi_convolve(loads, ims, domain, circular)
    else:
        return _plan_fftw_multi_convolve(loads, ims, domain, circular)


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


def plan_coupled_convolve(loads: dict, components: dict, domain=None,
                          periodic_axes: typing.Sequence[bool] = (False, False)):
    """Plans a set of convolutions between a dict of loads and a dict of im components

    Parameters
    ----------
    loads: dict
         A dict of arrays with keys 'x', 'y' and/or 'z'
    components: dict
        A dict of arrays with keys like 'xz' etc. These must describe a square system if the function is to be used with
        bccg iterations. the 'xz' component gives the z displacement caused by pressures in the x direction.
    domain: np.array, optional (None)
        A boolean array of contacting nodes
    periodic_axes: Sequence of bool, optional (False, False)
        Strictly 2 elements, convolutions will be periodic along axes corresponding to True values
    Returns
    -------
    callable:

    """
    component_names = list(components.keys())
    load_dirs = list(set(n[0] for n in component_names))
    load_dirs.sort()
    load_shape = loads[load_dirs[0]].shape
    conv_funcs = dict()
    component_names_d = dict()
    if domain is not None:
        domain_sum = np.sum(domain)
    else:
        domain_sum = 0

    for direction in load_dirs:
        names = [name for name in component_names if name.startswith(direction)]
        comps = np.array([components[name] for name in names])
        conv_funcs[direction] = plan_multi_convolve(loads[direction], comps, None, periodic_axes)
        component_names_d[direction] = names

    def inner_no_domain(loads_dict: dict):
        all_deflections = dict()
        for direction in load_dirs:
            all_deflections.update({n: d for n, d in zip(component_names_d[direction],
                                                         conv_funcs[direction](loads_dict[direction]))})
        deflections = dict()
        for key, value in all_deflections.items():
            if key[-1] in deflections:
                deflections[key[-1]] += value
            else:
                deflections[key[-1]] = value
        return deflections

    def inner_with_domain(loads_in_domain: np.array, ignore_domain: bool = False):
        full_loads = dict()
        for i in range(len(load_dirs)):
            full_loads[load_dirs[i]] = slippy.xp.zeros(load_shape, loads_in_domain.dtype)
            full_loads[load_dirs[i]][domain] = loads_in_domain[i * domain_sum:(i + 1) * domain_sum]
        full_deflections = inner_no_domain(full_loads)
        if ignore_domain:
            return full_deflections
        deflections_in_domain = slippy.xp.zeros_like(loads_in_domain)
        for i in range(len(load_dirs)):
            deflections_in_domain[i * domain_sum:(i + 1) * domain_sum] = full_deflections[load_dirs[i]][domain]
        return deflections_in_domain

    if domain is None:
        return inner_no_domain
    return inner_with_domain
