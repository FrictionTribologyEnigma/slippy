import numpy as np
import numpy.testing as npt

import slippy
import slippy.core as c

from scipy.signal import fftconvolve

try:
    import cupy as cp
    slippy.CUDA = True
except ImportError:
    cp = None
    slippy.CUDA = False

e_im = c.elastic_influence_matrix_spatial

loads_shapes = [(128, 128),
                (127, 128), (129, 128), (128, 127), (128, 129),
                (128, 128), (129, 128), (128, 128), (128, 128),
                (129, 129), (127, 127), (127, 129), (129, 127)]

shapes_circ = [(127, 127), (128, 128), (129, 129),
               (129, 127), (127, 129),
               (128, 129), (128, 127)]


# Test if our non circular convolution lines up with scipy's fft convolve

def test_non_circ_convolve_vs_scipy():
    try:
        import cupy as cp
        slippy.CUDA = True
    except ImportError:
        return
    for l_s in loads_shapes:
        # generate an influence matrix, pick a component which is not symmetric!
        im_s = tuple(s*2 for s in l_s)
        im = e_im('zx', im_s, (0.01, 0.01), 200e9, 0.3)
        loads = 1000*np.random.rand(*l_s)
        scipy_result = fftconvolve(loads, im, mode='same')
        conv_func = c.plan_convolve(loads, im, fft_im=False)
        slippy_result = cp.asnumpy(conv_func(loads))
        err_msg = f'Non circular convolution did not match scipy output for loads shape: {l_s} and IM shape: {im_s}'
        npt.assert_allclose(slippy_result, scipy_result, err_msg=err_msg)


# Test if our circular convolve gives a maximum in the right place
def test_circ_convolve_location():
    try:
        import cupy as cp
        slippy.CUDA = True
    except ImportError:
        return
    for im_s, l_s in zip(shapes_circ, shapes_circ):
        # generate an influence matrix, pick a component which is not symmetric!
        im = e_im('zz', im_s, (0.01, 0.01), 200e9, 0.3)
        loads = np.zeros(l_s)
        loads[64, 64] = 1000
        conv_func = c.plan_convolve(loads, im, circular=True)
        slippy_result = cp.asnumpy(conv_func(loads))
        loc_load = np.argmax(loads)
        loc_result = np.argmax(slippy_result)
        err_msg = f'Circular convolution, location of load dosn\'t match displacement' \
                  f'for loads shape: {l_s} and IM shape: {im_s} \n ' \
                  f'expected: {np.unravel_index(loc_load,l_s)}, found: {np.unravel_index(loc_result,l_s)}'
        assert loc_load == loc_result, err_msg


# Test if our non circular convolution gives a maximum in the right place
def test_non_circ_convolve_location():
    try:
        import cupy as cp
        slippy.CUDA = True
    except ImportError:
        return
    for l_s in loads_shapes:
        # generate an influence matrix, pick a component which is not symmetric!
        im_s = tuple(s * 2 for s in l_s)
        im = e_im('zz', im_s, (0.01, 0.01), 200e9, 0.3)
        loads = np.zeros(l_s)
        loads[64, 64] = 1000
        conv_func = c.plan_convolve(loads, im)
        slippy_result = cp.asnumpy(conv_func(loads))
        loc_load = np.argmax(loads)
        loc_result = np.argmax(slippy_result)
        err_msg = f'Non circular convolution, location of load dosn\'t match displacement' \
                  f'for loads shape: {l_s} and IM shape: {im_s} \n ' \
                  f'expected: {np.unravel_index(loc_load,l_s)}, found: {np.unravel_index(loc_result,l_s)}'
        assert loc_load == loc_result, err_msg


def test_mixed_convolve():
    try:
        import cupy as cp
        slippy.CUDA = True
    except ImportError:
        return
    for circ in [[True, False], [False, True]]:
        loads = np.zeros([128, 128])
        im_s = tuple((2-p)*s for p, s in zip(circ, loads.shape))
        im = e_im('zz', im_s, (0.01, 0.01), 200e9, 0.3)
        loads[64, 64] = 1000
        conv_func = c.plan_convolve(loads, im, circular=circ)
        slippy_result = cp.asnumpy(conv_func(loads))
        loc_load = np.argmax(loads)
        loc_result = np.argmax(slippy_result)
        err_msg = f'Mixed circular convolution, location of load dosn\'t match displacement' \
                  f'for circular: {circ} \n ' \
                  f'expected: {np.unravel_index(loc_load, loads.shape)}, ' \
                  f'found: {np.unravel_index(loc_result, loads.shape)}'
        assert loc_load == loc_result, err_msg


# Test that the fft cannot be planned to be circular and different shape inputs
def test_raises_uequal_shapes_circ():
    try:
        import cupy  # noqa: F401
        slippy.CUDA = True
    except ImportError:
        return
    im = e_im('zz', (128, 128), (0.01, 0.01), 200e9, 0.3)
    load_shapes = [(128, 129), (128, 129), (129, 128), (129, 128)]
    circulars = [True, (False, True), True, (True, False)]
    for l_s, circ in zip(load_shapes, circulars):
        with npt.assert_raises(AssertionError):
            loads = np.zeros(l_s)
            _ = c.plan_convolve(loads, im, circular=circ)


# Test that the convolution functions don't raise errors when the shapes are equal
def test_dont_raise_equal_shapes_circ():
    try:
        import cupy  # noqa: F401
        slippy.CUDA = True
    except ImportError:
        return
    im = e_im('zz', (128, 128), (0.01, 0.01), 200e9, 0.3)
    load_shapes = [(128, 128), (128, 64), (64, 64), (64, 128)]
    circulars = [True, (True, False), False, (False, True)]
    for l_s, circ in zip(load_shapes, circulars):
        loads = np.zeros(l_s)
        try:
            _ = c.plan_convolve(loads, im, circular=circ)
        except:  # noqa: E722
            raise AssertionError(f"Plan convolve raised wrong error for mixed "
                                 f"convolution load shape: {l_s}, circ: {circ}")


def test_inverse_conv():
    np.random.seed(0)
    try:
        import cupy  # noqa: F401
        slippy.CUDA = True
    except ImportError:
        return
    loads = np.random.rand(128, 128)
    im = e_im('zz', loads.shape, (1e-6, 1e-6), 200e9, 0.3)
    conv_func = c.plan_convolve(loads, im, circular=True, fft_im=False)
    recovered = conv_func.inverse_conv(conv_func(loads), True)
    npt.assert_allclose(loads, cp.asnumpy(recovered))
