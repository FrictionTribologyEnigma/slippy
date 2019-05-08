import numpy as np
import numpy.testing as npt

from slippy.contat import material

def test_elastic():
    E=material('elastic', {'E':200, 'v':0.3})
    assert(E.E==200.0)
    assert(E.G>75.9 and E.G<77.0)

def test_plastic():
    P=material('elastic_plastic', {'E':200, 'v':0.3, 'yield_stress':250}, model='perfect')
    assert(P.stress(0)==0)
    assert(P.stress(500)==250)
    
    strain=np.arange(5, dtype=float)
    stress=strain**0.5+200
    
    P2=material('elastic_plastiC', {'E':200, 'v':0.5, 'stress':stress, 
                                     'plastic_strain':strain}, 
                model='table')
    assert(P2.Lam==np.inf)
    assert(P2.stress(0)==0)
    npt.assert_approx_equal(P2.stress(strain[-1]+1)==stress[-1])