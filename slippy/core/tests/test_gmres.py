import numpy.testing as npt
import numpy as np
import pytest
import slippy
from slippy.core import gmres


def test_gmres_cpu():
    n = 6
    a = np.tril(np.ones((n, n)))
    b = np.ones(n)
    x0 = b * 0
    x, failed = gmres(lambda y: np.dot(a, y), x0, b, 4, n, 1e-6, override_cuda=True)
    assert not failed, "CPU gmres iterations failed to converge"
    x_true = np.zeros(n)
    x_true[0] = 1
    npt.assert_allclose(x, x_true, atol=3e-6, err_msg="CPU gmres iterations converged to incorrect result")


def test_gmres_cuda():
    cp = pytest.importorskip('cupy')
    slippy.CUDA = True
    n = 6
    a = cp.tril(cp.ones((n, n)))
    b = cp.ones(n)
    x0 = b * 0
    x, failed = gmres(lambda y: cp.dot(a, y), x0, b, 4, n, 1e-6, override_cuda=False)
    assert not failed, "GPU gmres iterations failed to converge"
    x_true = np.zeros(n)
    x_true[0] = 1
    npt.assert_allclose(cp.asnumpy(x), x_true, atol=3e-6,
                        err_msg="GPU gmres iterations converged to incorrect result")
