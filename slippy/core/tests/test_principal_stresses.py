from slippy.core import solve_cubic, get_derived_stresses

from scipy.spatial.transform import Rotation as R
import numpy as np
import numpy.testing as npt

# note these must be in order for comparison at end to work
test_matrix = [[300,   200,  100],
               [200,   0,    0],
               [-100, -200, -300],
               [300,   0,   -300],
               [200,  -100, -100],
               [0,     0,    0],
               [100,   100,  0],
               [0,    -100, -100]]


def test_derived_stresses():
    np.random.seed(0)
    for stresses in test_matrix:
        s1, s2, s3 = stresses
        # make cauchy stress tensor
        tensor = np.array([[s1, 0, 0],
                           [0, s2, 0],
                           [0, 0, s3]])
        # rotate tensor
        rotation = R.from_rotvec(2 * np.pi * np.random.rand(3))
        rm = rotation.as_matrix()
        rm_dash = rm.transpose()
        rt = np.dot(rm, np.dot(tensor, rm_dash))
        # make tensor (have to expand dims to ensure each element is an array
        named_tensor = {'xx': np.expand_dims(rt[0, 0], 0), 'yy': np.expand_dims(rt[1, 1], 0),
                        'zz': np.expand_dims(rt[2, 2], 0), 'xy': np.expand_dims(rt[0, 1], 0),
                        'xz': np.expand_dims(rt[0, 2], 0), 'yz': np.expand_dims(rt[1, 2], 0)}
        # find principal stresses
        found = get_derived_stresses(named_tensor, ['1', '2', '3'])
        npt.assert_allclose(np.stack([found['1'], found['2'], found['3']]).flatten(),
                            np.array(stresses), atol=1e-6)


def test_solve_cubic_numba():
    dtypes = ['float64', 'float32']
    for dtype in dtypes:
        # designed to test all branches of the algo
        b = np.array([3, 2, 3, 3, 4, 4, 4, 0], dtype=dtype)
        c = np.array([1, 4 / 3, 1, 4, 7 / 3, 7, 0, 0], dtype=dtype)
        d = np.array([0, 1, -1, 2, 2 - 1.6296296296296293, 2, -6, 0], dtype=dtype)
        found_roots = np.transpose(np.stack(solve_cubic(b, c, d)))
        all_coeffs = np.transpose(np.stack([np.ones_like(b), b, c, d]))
        for coeffs, roots in zip(all_coeffs, found_roots):
            np_roots = np.roots(coeffs)
            np_roots = np.sort(np.real(np_roots[(np.abs(np_roots) - np.abs(np.real(np_roots))) < 1e-7]))
            if len(roots) == len(np_roots):
                npt.assert_allclose(roots, np_roots, atol=1e-7)
            elif len(np_roots) == 1:
                npt.assert_allclose(roots, [np_roots[0]] * 3, atol=1e-7)
            else:
                raise ValueError("This should never happen")


def test_solve_cubic_cuda():
    try:
        import cupy as cp
    except ImportError:
        return
    dtypes = ['float64', 'float32']
    for dtype in dtypes:
        # designed to test all branches of the algo
        b = cp.array([3, 2, 3, 3, 4, 4, 4, 0], dtype=dtype)
        c = cp.array([1, 4 / 3, 1, 4, 7 / 3, 7, 0, 0], dtype=dtype)
        d = cp.array([0, 1, -1, 2, 2 - 1.6296296296296293, 2, -6, 0], dtype=dtype)
        found_roots = cp.transpose(cp.stack(solve_cubic(b, c, d)))
        all_coeffs = cp.transpose(cp.stack([cp.ones_like(b), b, c, d]))
        for coeffs, roots in zip(all_coeffs, found_roots):
            np_roots = np.roots(cp.asnumpy(coeffs))
            np_roots = np.sort(np.real(np_roots[(np.abs(np_roots) - np.abs(np.real(np_roots))) < 1e-7]))
            if len(roots) == len(np_roots):
                npt.assert_allclose(cp.asnumpy(roots), np_roots, atol=1e-7)
            elif len(np_roots) == 1:
                npt.assert_allclose(cp.asnumpy(roots), [np_roots[0]] * 3, atol=1e-7)
            else:
                raise ValueError("This should never happen")
