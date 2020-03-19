"""
testing for roughness functionality
"""

import numpy as np
import numpy.testing as npt
from pytest import raises as assert_raises
import slippy.surface as S


def test_fit_polynomial_masking():
    a = np.arange(10)
    b = np.arange(11)
    mesh_a, mesh_b = np.meshgrid(a, b)

    c = 1
    ac = 0.2
    bc = 0.3
    abc = 1.4

    profile = c + ac * mesh_a + bc * mesh_b + abc * mesh_a * mesh_b

    flattened_profile, coefficients = S.subtract_polynomial(profile, 1)

    npt.assert_allclose(coefficients, [c, bc, ac, abc])

    profile[0, 0] = np.inf

    assert_raises(ValueError, S.subtract_polynomial, profile, 1)

    flattened_profile, coefficients = S.subtract_polynomial(profile, 1, mask=np.inf)

    npt.assert_allclose(coefficients, [c, bc, ac, abc])

    profile[0, 0] = 1

    mask = np.logical_or(profile > 100, profile < 5)

    flattened_profile, coefficients = S.subtract_polynomial(profile, 1, mask=mask)

    npt.assert_allclose(coefficients, [c, bc, ac, abc])
