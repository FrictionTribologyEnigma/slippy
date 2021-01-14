import numpy as np
import numpy.testing as npt
from slippy.contact._lubrication_utils import tdma, cyclic_tdma


def test_cyclic_matches_base():
    # test that for non cyclic it gives the same result as the base tdma
    size = 4
    a = -1 * np.ones(size)
    b = 4 * np.ones(size)
    c = -1 * np.ones(size)
    d = np.arange(1, 5)
    x_normal = tdma(a[:-1], b, c[:-1], d)
    a[0] = 0
    c[-1] = 0
    x_cyclic = cyclic_tdma(a, b, c, d)
    npt.assert_allclose(x_normal, x_cyclic)


def test_cyclic_inversion():
    # test that a random matrix inverts properly
    np.random.seed(0)  # side stepping the most annoying possible error
    matrix = np.random.rand(4, 4)
    matrix[0, 2] = 0
    matrix[2, 0] = 0
    matrix[1, 3] = 0
    matrix[3, 1] = 0
    result = matrix @ np.arange(1, 5)

    b = np.diag(matrix, 0).copy()
    a = np.array([matrix[0, 3], *np.diag(matrix, -1)])
    c = np.array([*np.diag(matrix, 1), matrix[3, 0]])
    x_cyclic = cyclic_tdma(a, b, c, result)
    npt.assert_allclose(x_cyclic, np.arange(1, 5))
