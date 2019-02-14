"""
testing for roughness functionality
"""


import numpy as np
import numpy.testing as npt
from pytest import raises as assert_raises
import slippy.surface as S

def test_fit_polynomial_masking():
    a=np.arange(10)
    b=np.arange(11)
    A,B=np.meshgrid(a,b)
    
    c=1
    ac=0.2
    bc=0.3
    abc=1.4
    
    profile=c+ac*A+bc*B+abc*A*B
    
    P,C=S.subtract_polynomial(profile,1)
    
    npt.assert_allclose(C,[c,bc,ac,abc])
    
    profile[0,0]=float('nan')
    
    assert_raises(ValueError,S.subtract_polynomial,profile,1)
    
    P,C=S.subtract_polynomial(profile,1,mask=float('nan'))
    
    npt.assert_allclose(C,[c,bc,ac,abc])
    
    profile[0,0]=1
    
    mask=np.logical_or(profile>100, profile<5)
    
    P,C=S.subtract_polynomial(profile,1,mask=mask)
    
    npt.assert_allclose(C,[c,bc,ac,abc])
    

    