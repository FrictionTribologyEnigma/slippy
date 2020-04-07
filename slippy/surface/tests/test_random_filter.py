from scipy.optimize import check_grad
import numpy as np
import numpy.testing as npt

import slippy.surface as s
from slippy.surface.Random import _min_fun, _grad_min_fun


def test_grad():
    np.random.seed(0)
    acf = np.random.rand()
