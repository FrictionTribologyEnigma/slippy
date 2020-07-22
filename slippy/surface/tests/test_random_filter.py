import numpy as np
import numpy.testing as npt

import slippy.surface as S


def test_random_filter():
    np.random.seed(0)
    sigma = 2
    target_acf = S.ACF('exp', sigma, 0.1, 0.2)
    lin_trans_surface = S.RandomFilterSurface(target_acf=target_acf, grid_spacing=0.01)
    lin_trans_surface.linear_transform(filter_shape=(20, 10), gtol=1e-5, symmetric=True)
    my_realisation = lin_trans_surface.discretise([512, 512], periodic=True, create_new=True)
    npt.assert_almost_equal(my_realisation.roughness('Sq'), sigma, 1)

    target = lin_trans_surface.target_acf_array
    my_realisation.get_acf()
    actual = np.array(my_realisation.acf)
    n, m = actual.shape
    tn, tm = target.shape
    actual_comparable = actual[n // 2:n // 2 + tn, m // 2:m // 2 + tm]
    npt.assert_allclose(actual_comparable[0, :], target[0, :], 0.05, 0.2)
    npt.assert_allclose(actual_comparable[:, 0], target[:, 0], 0.05, 0.2)
