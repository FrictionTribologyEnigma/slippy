"""
Johnson utils tests
"""
import numpy as np
import numpy.testing as npt
from scipy.stats._continuous_distns import _norm_cdf

import slippy.surface._johnson_utils as j_util

# precision of tests
decimal = 1e-5

# noinspection SpellCheckingInspection
fit_params = [['norm', (0, 1, 0, 3)],
              ['johnsonsb', (5, 2, np.sqrt(2), 4)],
              ['johnsonsu', (0, 2, 1, 6)],
              [False, (5, 2, np.sqrt(2), 3)],
              ['lognorm', (5, 2, np.sqrt(2), -4)]]


def test_johnson_fit():
    for params in fit_params:
        if type(params[0]) is str:
            my_dist = j_util._fit_johnson_by_moments(*params[1])
            # noinspection SpellCheckingInspection
            moments = list(my_dist.stats('mvsk'))
            moments[1] = np.sqrt(moments[1])
            moments[3] = moments[3] + 3
            # noinspection SpellCheckingInspection
            if params[0] == 'lognorm':
                npt.assert_allclose(moments[:-1], params[1:][0][:-1], decimal, atol=0.01,
                                    err_msg=f"Johnson fitting by moments failed for {params[0]}")
                npt.assert_equal(my_dist.dist.name, params[0],
                                 err_msg=f"Johnson fitting by moments failed for {params[0]}")
            else:
                npt.assert_allclose(moments, params[1:][0], decimal, atol=0.01,
                                    err_msg=f"Johnson fitting by moments failed for {params[0]}")
                npt.assert_equal(my_dist.dist.name, params[0],
                                 err_msg=f"Johnson fitting by moments failed for {params[0]}")

        else:
            npt.assert_raises(NotImplementedError,
                              j_util._fit_johnson_by_moments, *params[1])


def test_johnson_quantile_fit():
    quantile_pts = np.array(_norm_cdf([-1.5, -0.5, 0.5, 1.5]))
    test_quantiles = [[1, 1.2, 1.4, 10],
                      [1, 1.2, 1.4, 3],
                      [-2, -1, 1, 2]]

    for quantiles_in in test_quantiles:
        dist = j_util._fit_johnson_by_quantiles(quantiles_in)
        quantiles_out = dist.ppf(quantile_pts)
        npt.assert_allclose(quantiles_in, quantiles_out, decimal,
                            err_msg=f'Fitting johnson distribution by quartiles failed for test '
                                    f'quartiles {quantiles_in}')


if __name__ == '__main__':
    test_johnson_fit()
    test_johnson_quantile_fit()
