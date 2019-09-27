"""
Johnson utils tests
"""
import numpy as np
import numpy.testing as npt
from pytest import raises as assert_raises
import slippy.surface as S
from scipy.stats._continuous_distns import _norm_cdf



# precision of tests
decimal = 1e-5 

fit_params=[['norm',(0,1,0,3)],
            ['johnsonsb',(5,2,np.sqrt(2),4)],
            ['johnsonsu',(5,2,1,6)],
            [False,(5,2,np.sqrt(2),3)],
            ['johnsonsl',(5,2,np.sqrt(2),-3)]]

def test_johnson_fit():
    for params in fit_params:
        if type(params[0]) is str:
            myDist=S._fit_johnson_by_moments(*params[1])
            moments=myDist.stats('mvsk')
            moments[1]=np.sqrt(moments[1])
            moments[3]=moments[3]+3
            npt.assert_allclose(moments, params, decimal)
            npt.assert_equal(params[0], myDist.dist.name)
        else:
            assert_raises(NotImplementedError, 
                          S.fit_johnson_by_moments(*params[1]))

fit_quantiles=[[]]

def test_johnson_quant_fit():
    quantile_pts=np.array(_norm_cdf([-1.5,0.5,0.5,1.5]))
    test_quantiles=[[1,1.2,1.4,10],
                    [1,1.2,1.4,3],
                    [-2,-1,1,2]]
    
    for quantiles_in in test_quantiles:
        dist=S._fit_johnson_by_quantiles(quantiles_in)
        quantiles_out=dist.ppf(quantile_pts)
        npt.assert_allclose(quantiles_in, quantiles_out, decimal)
        
if __name__ is '__main__':
    test_johnson_fit()
    test_johnson_quant_fit()