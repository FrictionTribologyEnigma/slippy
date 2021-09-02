import numpy.testing as npt
import numpy as np
import slippy
from slippy.core import gmres


def test_gmres():
    all_x = []
    msgs = []
    try:
        import cupy as cp
        slippy.CUDA = True
        n = 6
        a = cp.tril(cp.ones((n, n)))
        b = cp.ones(n)
        x0 = b * 0
        x, failed = gmres(lambda y: cp.dot(a, y), x0, b, 4, n, 1e-6, override_cuda=False)
        assert not failed, "GPU gmres iterations failed to converge"
        all_x.append(cp.asnumpy(x))
        msgs.append("GPU gmres iterations converged to incorrect result")
    except ImportError:
        pass

    n = 6
    a = np.tril(np.ones((n, n)))
    b = np.ones(n)
    x0 = b * 0
    x, failed = gmres(lambda y: np.dot(a, y), x0, b, 4, n, 1e-6, override_cuda=True)
    assert not failed, "CPU gmres iterations failed to converge"
    all_x.append(x)
    msgs.append("CPU gmres iterations converged to incorrect result")

    x_true = x0*0
    x_true[0] = 1
    for x, msg in zip(all_x, msgs):
        npt.assert_allclose(x, x_true, atol=3e-6, err_msg=msg)
