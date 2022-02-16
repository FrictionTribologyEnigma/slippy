import numpy as np
import typing
import warnings
import slippy
import functools
import abc

__all__ = ['guess_loads_from_displacement', 'bccg', 'plan_convolve', 'plan_multi_convolve', 'plan_coupled_convolve',
           'polonsky_and_keer', 'rey', 'ConvolutionFunction']


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


class ConvolutionFunction(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def inner_with_domain(self, sub_loads, ignore_domain=False):
        pass

    @abc.abstractmethod
    def inner_no_domain(self, full_loads, _):
        pass

    @abc.abstractmethod
    def inverse_conv(self, deformations, ignore_domain):
        pass

    @abc.abstractmethod
    def change_domain(self, new_domain):
        pass

    @abc.abstractmethod
    def __call__(self, loads, ignore_domain):
        pass


try:
    import cupy as cp

    def n_pow_2(a):
        return 2 ** int(np.ceil(np.log2(a)))

    class CudaConvolutionFunction(ConvolutionFunction):

        def __init__(self, loads: np.ndarray, im: np.ndarray, domain: np.ndarray,
                     circular: typing.Sequence[bool], fft_im: bool = True):
            """Plans an FFT convolution, CUDA implementation

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

            Notes
            -----
            This function uses CUDA to run on a GPU if your computer dons't have cupy installed this should not have
            loaded if it is for some reason, this can be manually overridden by first importing slippy then setting the
            CUDA variable to False:

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
            >>> im = c.elastic_influence_matrix_spatial('zz', (512,512), (grid_spacing,grid_spacing),
            >>>                                         200e9/(2*(1+0.3)), 0.3)
            >>> convolve_func = plan_convolve(loads, im, None, [False, False])
            >>> disp_numerical = convolve_func(loads)

            """
            super().__init__()
            loads = cp.asarray(loads)
            im = cp.asarray(im)
            im_shape_orig = im.shape
            if domain is not None:
                domain = cp.asarray(domain)
            self._domain = domain
            input_shape = []
            for i in range(2):
                if circular[i]:
                    assert loads.shape[i] == im.shape[i], "For circular convolution loads and im must be same shape"
                    input_shape.append(loads.shape[i])
                else:
                    msg = "For non circular convolution influence matrix must be double loads"
                    assert loads.shape[i] == im.shape[i] // 2, msg
                    input_shape.append(loads.shape[i])
            input_shape = tuple(input_shape)
            fft_shape = [max(i, f) for i, f in zip(input_shape, im_shape_orig)]
            self.forward_trans = functools.partial(cp.fft.fft2, s=fft_shape)
            self.backward_trans = functools.partial(cp.fft.ifft2, s=fft_shape)
            self.norm_inv = (input_shape[0] * input_shape[1]) ** 0.5
            norm = 1 / self.norm_inv
            self.norm = norm
            if fft_im:
                self.fft_im = im
            else:
                im = cp.roll(im, tuple(-((sz - 1) // 2) for sz in im_shape_orig), (-2, -1))
                self.fft_im = self.forward_trans(im)
            self.inv_fft_im = 1 / self.fft_im

            if cp.isinf(self.inv_fft_im[0, 0]):
                self.inv_fft_im[0, 0] = 0.0

            self.shape = loads.shape
            self.dtype = loads.dtype

            if domain is None:
                self.callback = self.inner_no_domain
            else:
                self.callback = self.inner_with_domain

        def inner_with_domain(self, sub_loads, ignore_domain=False):
            full_loads = cp.zeros(self.shape, dtype=self.dtype)
            full_loads[self._domain] = sub_loads
            fft_loads = self.forward_trans(full_loads)
            full = cp.real(self.backward_trans(fft_loads * self.fft_im))
            full = full[:full_loads.shape[0], :full_loads.shape[1]]
            if ignore_domain:
                return full
            return full[self._domain]

        def inner_no_domain(self, full_loads, _):
            full_loads = cp.asarray(full_loads)
            if full_loads.shape == self.shape:
                flat = False
            else:
                full_loads = cp.reshape(full_loads, self.shape)
                flat = True
            fft_loads = self.forward_trans(full_loads)
            full = cp.real(self.backward_trans(fft_loads * self.fft_im))
            full = full[:full_loads.shape[0], :full_loads.shape[1]]
            if flat:
                full = full.flatten()
            return full

        def inverse_conv(self, deformations, ignore_domain):
            if self._domain is not None:
                full_defs = cp.zeros(self.shape, dtype=self.dtype)
                full_defs[self._domain] = deformations
                flat = False
            else:
                full_defs = deformations
                if full_defs.shape == self.shape:
                    flat = False
                else:
                    full_defs = cp.reshape(full_defs, self.shape)
                    flat = True
            fft_defs = self.forward_trans(full_defs)
            full = cp.real(self.backward_trans(fft_defs * self.inv_fft_im))
            full = full[:full_defs.shape[0], :full_defs.shape[1]]
            if ignore_domain:
                return full
            if flat:
                return full.flatten()
            return full[self._domain]

        def change_domain(self, new_domain):
            self._domain = new_domain

        def __call__(self, loads, ignore_domain=False):
            return self.callback(loads, ignore_domain)

    def _plan_cuda_multi_convolve(loads: np.ndarray, ims: np.ndarray, domain: np.ndarray = None,
                                  circular: typing.Sequence[bool] = (False, False), fft_im: bool = True):
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
        fft_im: bool, optional (True)
            True if the supplied influence matrix is already in the frequency domain, note the full fft is expected,
            including redundant elements

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
        >>> im = c.elastic_influence_matrix_spatial('zz', (512,512), (grid_spacing,grid_spacing), 200e9/(2*(1+0.3)), 0.3)
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
                msg = "For non circular convolution influence matrix must be double loads"
                assert loads.shape[i] == im.shape[i] // 2, msg
                input_shape.append(n_pow_2(max(loads.shape[i], im.shape[i])))

        input_shape = tuple(input_shape)
        fft_shape = tuple([input_shape[0], input_shape[1]])
        forward_trans = functools.partial(cp.fft.fft2, s=fft_shape)
        backward_trans = functools.partial(cp.fft.ifft2, s=input_shape)
        shape_diff = [[0, (b - a)] for a, b in zip(im.shape, input_shape)]

        norm_inv = (input_shape[0] * input_shape[1]) ** 0.5
        norm = 1 / norm_inv
        if fft_im:
            fft_ims = cp.asarray(ims * norm)
        else:
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

except ImportError:
    cp = None
    _plan_cuda_convolve = None
    _plan_cuda_multi_convolve = None

try:
    import pyfftw

    class FftwConvolutionFunction(ConvolutionFunction):
        def __init__(self, loads: np.ndarray, im: np.ndarray, domain: np.ndarray, circular: typing.Sequence[bool],
                     fft_im: bool = True):
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
            fft_im: bool, optional (True)


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
            >>> im = c.elastic_influence_matrix_spatial('zz', (512,512), (grid_spacing,grid_spacing),
            >>>                                         200e9/(2*(1+0.3)), 0.3)
            >>> convolve_func = plan_convolve(loads, im, None, [False, False])
            >>> disp_numerical = convolve_func(loads)

            """
            super().__init__()
            loads = np.asarray(loads)
            self._loads_shape = loads.shape
            im = np.asarray(im)
            im_shape_orig = im.shape
            if domain is not None:
                domain = np.asarray(domain, dtype=bool)
            self._domain = domain
            input_shape = []
            for i in range(2):
                if circular[i]:
                    assert loads.shape[i] == im.shape[i], "For circular convolution loads and im must be same shape"
                    input_shape.append(im.shape[i])
                else:
                    msg = "For non circular convolution influence matrix must be double loads"
                    assert loads.shape[i] == im.shape[i] // 2, msg
                    input_shape.append(im.shape[i])
            input_shape = tuple(input_shape)

            fft_shape = [input_shape[0], input_shape[1] // 2 + 1]
            in_empty = pyfftw.empty_aligned(input_shape, dtype=loads.dtype)
            out_empty = pyfftw.empty_aligned(fft_shape, dtype='complex128')
            ret_empty = pyfftw.empty_aligned(input_shape, dtype=loads.dtype)
            self.forward_trans = pyfftw.FFTW(in_empty, out_empty, axes=(0, 1),
                                             direction='FFTW_FORWARD', threads=slippy.CORES)
            self.backward_trans = pyfftw.FFTW(out_empty, ret_empty, axes=(0, 1),
                                              direction='FFTW_BACKWARD', threads=slippy.CORES)
            self.norm_inv = self.forward_trans.N ** 0.5
            norm = 1 / self.norm_inv
            self.norm = norm

            if fft_im:
                self.fft_im = im[:fft_shape[0], :fft_shape[1]]
            else:
                im = np.roll(im, tuple(-((sz - 1) // 2) for sz in im_shape_orig), (-2, -1))
                self.fft_im = self.forward_trans(im).copy()

            with np.errstate(divide='ignore'):
                self.inv_fft_im = 1/self.fft_im
            if np.isinf(self.inv_fft_im[0, 0]):
                self.inv_fft_im[0, 0] = 0.0

            self._shape_diff_loads = [[0, (b - a)] for a, b in zip(loads.shape, input_shape)]

            self.shape = loads.shape
            self.dtype = loads.dtype
            self.all_circ = circular[0] and circular[1]

            if domain is None:
                self.callback = self.inner_no_domain
            else:
                self.callback = self.inner_with_domain

        def inner_with_domain(self, sub_loads, ignore_domain=False):
            full_loads = np.zeros(self.shape, dtype=self.dtype)
            full_loads[self._domain] = sub_loads
            loads_pad = np.pad(full_loads, self._shape_diff_loads, 'constant')
            full = self.backward_trans(self.forward_trans(loads_pad) * self.fft_im)
            same = full[:full_loads.shape[0], :full_loads.shape[1]].copy()
            if ignore_domain:
                return same
            return same[self._domain]

        def inner_no_domain(self, full_loads, _):
            if full_loads.shape == self.shape:
                flat = False
            else:
                full_loads = np.reshape(full_loads, self.shape)
                flat = True
            loads_pad = np.pad(full_loads, self._shape_diff_loads, 'constant')
            full = self.backward_trans(self.forward_trans(loads_pad) * self.fft_im)
            full = full[:full_loads.shape[0], :full_loads.shape[1]].copy()
            if flat:
                full = full.flatten()
            return full

        def inverse_conv(self, deformations, ignore_domain):
            if not self.all_circ:
                raise ValueError("Inverse convolution only possible with fully periodic contacts")
            if self._domain is not None:
                full_defs = np.zeros(self.shape, dtype=self.dtype)
                full_defs[self._domain] = deformations
                flat = False
            else:
                full_defs = deformations
                if full_defs.shape == self.shape:
                    flat = False
                else:
                    full_defs = np.reshape(full_defs, self.shape)
                    flat = True
            defs_pad = np.pad(full_defs, self._shape_diff_loads, 'constant')
            full = self.backward_trans(self.forward_trans(defs_pad) * self.inv_fft_im)
            full = full[:self.shape[0], :self.shape[1]].copy()
            if ignore_domain:
                return full
            if flat:
                return full.flatten()
            return full[self._domain]

        def change_domain(self, new_domain):
            self._domain = new_domain

        def __call__(self, loads, ignore_domain=False):
            return self.callback(loads, ignore_domain)

    def _plan_fftw_multi_convolve(loads: np.ndarray, ims: np.ndarray, domain: np.ndarray = None,
                                  circular: typing.Sequence[bool] = (False, False), fft_im: bool = True):
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
        fft_im: bool, optional (True)
            True if the supplied influence matix is already in the frequency domain, note, the full fft inlcuding
            redundant elementes is required

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
        >>> im = c.elastic_influence_matrix_spatial('zz', (512,512), (grid_spacing,grid_spacing), 200e9/(2*(1+0.3)), 0.3)
        >>> convolve_func = plan_convolve(loads, im, None, [False, False])
        >>> disp_numerical = convolve_func(loads)

        """
        loads = slippy.asnumpy(loads)
        im = np.asarray(ims[0])
        im_shape_orig = im.shape
        if domain is not None:
            domain = slippy.asnumpy(domain)
        input_shape = []
        for i in range(2):
            if circular[i]:
                assert loads.shape[i] == im.shape[i], "For circular convolution loads and im must be same shape"
                input_shape.append(loads.shape[i])
            else:
                msg = "For non circular convolution influence matrix must be double loads"
                assert loads.shape[i] == im.shape[i] // 2, msg
                input_shape.append(im.shape[i])
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

        if fft_im:
            fft_ims = ims[:, :fft_shape[1], :fft_shape[2]] * norm
        else:
            shape_diff = [[0, (b - a)] for a, b in zip((len(ims),) + im.shape, input_shape)]
            ims = np.pad(ims, shape_diff, 'constant')
            ims = np.roll(ims, tuple(-((sz - 1) // 2) for sz in im_shape_orig), (-2, -1))
            fft_ims = forward_trans_ims(ims) * norm

        shape_diff_loads = [[0, (b - a)] for a, b in zip(loads.shape, input_shape[1:])]

        shape = loads.shape
        dtype = loads.dtype

        def inner_no_domain(full_loads):
            if not isinstance(full_loads, np.ndarray):
                full_loads = slippy.asnumpy(full_loads)
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
            if not isinstance(sub_loads, np.ndarray):
                sub_loads = slippy.asnumpy(sub_loads)
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

except ImportError:
    _plan_fftw_convolve = None
    _plan_fftw_multi_convolve = None


def plan_convolve(loads, im, domain: np.ndarray = None, circular: typing.Union[bool, typing.Sequence[bool]] = False,
                  fft_im: bool = True) -> ConvolutionFunction:
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
    fft_im: bool, optional (True)
        True if the supplied influence matrix is in the frequency domain, false if it is in the spatial domain

    Returns
    -------
    ConvolutionFunction
        A function which takes a single input of loads and returns the result of the convolution with the original
        influence matrix. If a domain was not supplied the input to the returned function must be exactly the same
        shape as the loads array used in this function. If a domain was specified the length of the loads input to
        the returned function must be the same as the number of non zero elements in domain.

    Notes
    -----
    By default this function uses CUDA to run on a GPU if your computer dosn't have cupy installed this should not
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
    >>> im = c.elastic_influence_matrix_spatial('zz', (512,512), (grid_spacing,grid_spacing), 200e9/(2*(1+0.3)), 0.3)
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
        return CudaConvolutionFunction(loads, im, domain, circular, fft_im)
    else:
        return FftwConvolutionFunction(loads, im, domain, circular, fft_im)


def plan_multi_convolve(loads, ims: np.array, domain: np.ndarray = None,
                        circular: typing.Union[bool, typing.Sequence[bool]] = False,
                        cuda: bool = None, fft_ims: bool = True):
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
    cuda: bool, optional (None)
        If False the computation will be completed on the CPU, if True or None computation will be completed on the
        gpu if slippy.CUDA is True
    fft_ims: bool, optional (True)
        True if the supplied influence matricies are in the frequency domain, note that the full fft of the influnce
        matix is required, including redundant elements

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
    """
    if isinstance(circular, int):
        circular = [circular, ] * 2
    try:
        length = len(circular)
    except TypeError:
        raise TypeError('Type of circular not recognised, should be a bool or a 2 element sequence of bool')

    if length != 2:
        raise ValueError(f"Circular must be a bool or a 2 element list of bool, length was {length}")

    if cuda is None:
        cuda = slippy.CUDA

    if cuda:
        return _plan_cuda_multi_convolve(loads, ims, domain, circular, fft_im=fft_ims)
    else:
        return _plan_fftw_multi_convolve(loads, ims, domain, circular, fft_im=fft_ims)


def polonsky_and_keer(f: typing.Callable, p0: typing.Sequence, just_touching_gap: typing.Sequence,
                      target_load: float, grid_spacing: typing.Union[typing.Sequence[float], float],
                      eps_0: float = 1e-6, max_it: int = None):
    """ The Polonsky and Keer CG method for solving elastic contact (cuda and fftw versions)

    Parameters
    ----------
    f: Callable
        A function equivalent to multiplication by a non negative n by n matrix must work with cupy arrays.
        Typically this function will be generated by slippy.contact.plan_convolve, this will guarantee
        compatibility with different versions of this function (FFTW and CUDA).
    p0: array
        An initial guess of the pressure distribution, must not be all zeros
    just_touching_gap: array
        The gap function at the point of first contact between the surfaces, cen be generated by get_gap_from_model
    target_load: float
        The total target load (not average pressure)
    grid_spacing: float or sequence of float
        Either a float indicating a square grid, or a two element sequence indicating the dimensions of the rectangles
        in the y and x directions respectively
    eps_0: float
        The error used as a convergence criterion
    max_it: int, optional (None)
        The maximum number of iterations used, defaults to the problem size

    Returns
    -------
    failed: bool
        True if the process failed to converge
    pij: array
        The pressure result on each point of the surface
    gij: array
        The deformed gap function

    Notes
    -----
    This method does not directly calculate the rigid body approach

    Examples
    --------
    >>> import slippy.core as core
    >>> import slippy.surface as s
    >>> import slippy.contact as c
    >>> n = 128
    >>> total_load = 100
    >>> p = np.zeros((n,n))
    >>> flat_surface = s.FlatSurface(shift=(0,0))
    >>> round_surface = s.RoundSurface((1,1,1), extent = (0.0035, 0.0035),
    >>>                                shape = (n, n), generate = True)
    >>> gs = round_surface.grid_spacing
    >>> e1 = 200e9; v1 = 0.3
    >>> e2 = 70e9; v2 = 0.33
    >>> im_1 = core.elastic_influence_matrix_spatial('zz', span = (n*2,n*2), grid_spacing=(gs, gs),
    >>>                                      shear_mod=e1/(2*(1+v1)), v=0.3)
    >>> im_2 = core.elastic_influence_matrix_spatial('zz', span = (n*2,n*2), grid_spacing=(gs, gs),
    >>>                                      shear_mod=e2/(2*(1+v2)), v=0.33)
    >>> im = im_1+im_2
    >>> f = core.plan_convolve(p, im, domain=None, circular=False)
    >>> model = c.ContactModel('model_1', round_surface, flat_surface)
    >>> just_touching_gap = c._model_utils.get_gap_from_model(model)[0]
    >>> gs = [round_surface.grid_spacing, ]*2
    >>> p0 = np.ones_like(just_touching_gap)
    >>> failed, numerical_pressure, deformed_gap = polonsky_and_keer(f, p0, just_touching_gap,
    >>>                                                              total_load, gs, eps_0=1e-7)

    References
    ----------
    I.A. Polonsky, L.M. Keer,
    A numerical method for solving rough contact problems based on the multi-level multi-summation and conjugate
    gradient techniques, Wear, Volume 231, Issue 2, 1999, Pages 206-219, ISSN 0043-1648,
    https://doi.org/10.1016/S0043-1648(99)00113-1. (https://www.sciencedirect.com/science/article/pii/S0043164899001131)

    """
    if np.sum(slippy.asnumpy(p0)) == 0:
        raise ValueError("Initial pressure guess cannot sum to zero for polonsky_and_keer")
    try:
        float(grid_spacing)
        grid_spacing = [grid_spacing, ] * 2
    except TypeError:
        pass

    try:
        assert len(grid_spacing) == 2, "Grid spacing must be a two element sequence or a number"
    except TypeError:
        raise ValueError("Grid spacing must be a two element sequence or a number")

    if slippy.CUDA and cp is not None:
        xp = cp
    else:
        xp = np

    just_touching_gap = xp.array(just_touching_gap)
    p0 = xp.array(p0)
    if max_it is None:
        max_it = just_touching_gap.size
    # init
    pij = p0 / xp.mean(p0) * target_load
    delta = 0
    g_big_old = 1
    tij = 0
    it_num = 0
    element_area = grid_spacing[0] * grid_spacing[1]
    while True:
        uij = f(pij)
        gij = uij + just_touching_gap
        current_touching = pij > 0
        g_bar = xp.mean(gij[current_touching])
        gij = gij - g_bar
        g_big = xp.sum(gij[current_touching] ** 2)
        if it_num == 0:
            tij = gij
        else:
            tij = gij + delta * (g_big / g_big_old) * tij
        tij[xp.logical_not(current_touching)] = 0
        g_big_old = g_big
        rij = f(tij)
        r_bar = xp.mean(rij[current_touching])
        rij = rij - r_bar
        if not xp.linalg.norm(rij):
            tau = 0
        else:
            tau = (xp.dot(gij[current_touching], tij[current_touching]) /
                   xp.dot(rij[current_touching], tij[current_touching]))
        pij_old = pij
        pij = pij - tau * tij
        pij = xp.clip(pij, 0, xp.inf)
        iol = xp.logical_and(pij == 0, gij < 0)
        if xp.any(iol):
            delta = 0
            pij[iol] = pij[iol] - tau * gij[iol]
        else:
            delta = 1
        p_big = element_area * xp.sum(pij)
        pij = pij / p_big * target_load
        eps = (element_area / target_load) * xp.sum(xp.abs(pij - pij_old))
        if eps < eps_0:
            failed = False
            break
        it_num += 1
        if it_num > max_it:
            failed = True
            break
        if xp.any(xp.isnan(pij)):
            failed = True
            break
    return failed, pij, gij


def bccg(f: typing.Callable, b: np.ndarray, tol: float, max_it: int, x0: np.ndarray,
         min_pressure: float = 0.0, max_pressure: float = np.inf, k_inn=1) -> typing.Tuple[np.ndarray, bool]:
    """
    The Bound-Constrained Conjugate Gradient Method for Non-negative Matrices

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
    if slippy.CUDA and cp is not None:
        xp = cp
    else:
        xp = np

    try:
        float(max_pressure)
        max_is_float = True
    except TypeError:
        max_is_float = False
        max_pressure = xp.array(max_pressure)

    try:
        float(min_pressure)
        min_is_float = True
    except TypeError:
        min_is_float = False
        min_pressure = xp.array(min_pressure)

    # initialize
    b = xp.asarray(b)
    x = xp.clip(xp.asarray(x0), min_pressure, max_pressure)
    g = f(x) - b
    msk_bnd_0 = xp.logical_and(x <= 0, g >= 0)
    msk_bnd_max = xp.logical_and(x >= max_pressure, g <= 0)
    n_bound = xp.sum(msk_bnd_0) + xp.sum(msk_bnd_max)
    n = b.size
    n_free = n - n_bound
    small = 1e-14
    it = 0
    it_inn = 0
    rho_prev = xp.nan
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
        rho = xp.dot(r, r)
        if it > 1:
            beta_pr = (rho - xp.dot(r, r_prev)) / rho_prev
            p = r + max([beta_pr, 0]) * p
        else:
            p = r
        p[msk_bnd_0] = 0
        p[msk_bnd_max] = 0
        # compute tildex optimisation ignoring the bounds
        q = f(p)
        if it_inn < k_inn:
            q[msk_bnd_0] = xp.nan
            q[msk_bnd_max] = xp.nan
        alpha = xp.dot(r, p) / xp.dot(p, q)
        x = x + alpha * p

        rms_xk = xp.linalg.norm(x) / xp.sqrt(n_free)
        rms_upd = xp.linalg.norm(x - x_prev) / xp.sqrt(n_free)
        upd = rms_upd / rms_xk

        # project onto feasible domain
        changed = False
        outer_it = it_inn >= k_inn or upd < tol

        if outer_it:
            msk_prj_0 = x < min_pressure - small
            if xp.any(msk_prj_0):
                if min_is_float:
                    x[msk_prj_0] = min_pressure
                else:
                    x[msk_prj_0] = min_pressure[msk_prj_0]
                msk_bnd_0[msk_prj_0] = True
                changed = True
            msk_prj_max = x >= max_pressure * (1 + small)
            if xp.any(msk_prj_max):
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
            msk_rel = xp.logical_or(xp.logical_and(msk_bnd_0, g < -small), xp.logical_and(msk_bnd_max, g > small))
            if xp.any(msk_rel):
                msk_bnd_0[msk_rel] = False
                msk_bnd_max[msk_rel] = False
                changed = True

        if changed:
            n_free = n - xp.sum(msk_bnd_0) - xp.sum(msk_bnd_max)

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


def rey(h: np.ndarray, f: ConvolutionFunction, adhesion_energy_derivative: typing.Callable, error: float,
        max_it: int = None, mean_gap: float = None, mean_pressure: float = None):
    """ Rey adhesive solver

    Parameters
    ----------
    h: np.ndarray
        The just touching gap
    f: ConvolutionFunction
        A convolution function made by plan_convolve
    adhesion_energy_derivative: Callable
        A function which returns the derivative of the adhesion potential at a specified gap value
    error: float
        The error used for a convergence criterion
    max_it: int
        The maximum number of iterations, defaults to the problem size
    mean_gap, mean_pressure: float, optional (None)
        Exactly one of these must be set, the converged solution will fit this value exactly if mean gap is used.

    Returns
    -------
    failed: bool
        True if the iterations failed to converge
    pressure: array
        The normal contact pressure
    gap: array
        The deformed gap
    total_displacement: array
        The total normal displacement of the surfaces
    contact_nodes: array
        Boolean array of surface nodes which are in contact

    """
    if not ((mean_gap is None) ^ (mean_pressure is None)):
        raise ValueError("Either the mean pressure or mean gap must be set (not both)")
    if max_it is None:
        max_it = h.size

    if adhesion_energy_derivative is None:
        def adhesion_energy_derivative(_):
            return 0

    if slippy.CUDA and cp is not None:
        xp = cp
    else:
        xp = np

    if mean_gap is None:
        gap_constrained = False
        target = mean_pressure
    else:
        gap_constrained = True
        target = mean_gap

    h = xp.asarray(h)
    h = -h
    n = h.size
    std = xp.std(h)
    # g = -h/np.mean(-h)*mean_gap
    it_num = 0
    big_r_old = 1
    delta = 0
    t = xp.zeros_like(h)
    if gap_constrained:
        g = xp.ones_like(h) * target
    else:
        g = xp.ones_like(h) * std
    failed = True
    print("Begin rey solver")
    while True:
        # compute functional gradf
        u = g + h
        q = f.inverse_conv(u, True)
        q += adhesion_energy_derivative(g)
        # mean on unsaturated (primal is gap)
        non_contact_nodes = g > 0
        q_bar = xp.mean(q[non_contact_nodes])
        if gap_constrained:
            q -= q_bar
        else:
            q += 2 * target + q_bar
        # compute squared norm
        big_r = xp.sum(q[non_contact_nodes] ** 2)
        # update search direction
        t[xp.logical_not(non_contact_nodes)] = 0
        t[non_contact_nodes] = (q[non_contact_nodes] +
                                delta * (big_r / big_r_old) * t[non_contact_nodes])
        big_r_old = big_r
        # compute critical step
        r = f.inverse_conv(t, True)
        r_bar = xp.mean(r[non_contact_nodes])
        if gap_constrained:
            r -= r_bar
        else:
            r += 2 * target + r_bar
        tau = (xp.sum(q[non_contact_nodes] * t[non_contact_nodes]) /
               xp.sum(r[non_contact_nodes] * t[non_contact_nodes]))
        # update primal
        g = g - tau * t
        xp.clip(g, 0, None, g)
        non_admissible_nodes = xp.logical_and(g == 0, q < 0)
        if xp.any(non_admissible_nodes):
            delta = 0
            g[non_admissible_nodes] = g[non_admissible_nodes] - tau * q[non_admissible_nodes]
        else:
            delta = 1
        # enforce mean value
        if gap_constrained:
            g = target / xp.mean(g) * g
        # compute error
        q -= xp.min(q)
        er = xp.sum(g * q)
        norm = xp.sum(q) * std
        eps = er / (norm * n)

        # if not it_num % 100:
        #     cost = 0.5 * xp.sum(q * (g + h)) / n ** 2
        #     print(f"it:{it_num}\teps:{eps}\tcost:{cost}")
        if eps < error and it_num:
            failed = False
            print("Rey Converged")
            break
        it_num += 1
        if it_num > max_it:
            print(f"Rey failed to converge: Max it (iterations: {max_it}, error: {eps})")
            break
        if xp.isnan(eps):
            print("Rey failed to converge: Nan's detected")
            break
    u = g + h
    q1 = f.inverse_conv(u, True)
    q = q1 + adhesion_energy_derivative(g)
    min_q = xp.min(q)
    p = q1 - min_q
    return failed, p, g, u, g <= 0.0
