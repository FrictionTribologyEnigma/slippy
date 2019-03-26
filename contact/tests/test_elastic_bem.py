import numpy as np
import numpy.testing as npt
#from pytest import raises as assert_raises
import slippy.contact as C

#['convert_array', 'convert_dict', 'elastic_displacement', '_solve_ed',
#         'elastic_loading', '_solve_el', 'elastic_im']
def test_convert():
    loads={'x':np.random.normal(size=(11,11)),
           'y':np.random.normal(size=(11,11)),
           'z':np.random.normal(size=(11,11))}
    
    loads_array=C.convert_dict(loads)
    
    ld=C.convert_array(loads_array)
    npt.assert_array_equal(ld['x'],loads['x'])
    npt.assert_array_equal(ld['y'],loads['y'])
    npt.assert_array_equal(ld['z'],loads['z'])
    
def test_elastic_loading():
    loads={'x':np.zeros((11,11)),
           'y':np.zeros((11,11)),
           'z':np.zeros((11,11))}
    loads['z'][5,5]=100
    
    # does simple work
    
    displacements1 = C.elastic_loading(loads, (1,1), v=0.3, G=200, 
                                      deflections='xyz', span=None, 
                                      simple=True)
    
    assert np.sum(displacements1['x'])<0.0001
    assert np.sum(displacements1['y'])<0.0001
    
    # is v=0.5 give 0 values in influence matrix
    
    displacements = C.elastic_loading(loads, (1,1), v=0.5, G=200, 
                                      deflections='xyz', span=None, 
                                      simple=False)
    
    assert np.sum(displacements['x'])<0.0001
    assert np.sum(displacements['y'])<0.0001
    # check central value
    npt.assert_approx_equal(displacements['z'][5,5], 0.14, significant=2)
    
    # check that not simple works properly
    
    displacements = C.elastic_loading(loads, (1,1), v=0.3, G=200, 
                                      deflections='xyz', span=None, 
                                      simple=False)
    
    
    assert np.sum(displacements['x'][:,5])<0.0001
    assert np.sum(displacements['y'][5,:])<0.0001
    
    assert displacements1['z'][5,5]-displacements['z'][5,5]<0.001
    
def test_elastic_displacement():
    loads={'x':np.zeros((11,11)),
           'y':np.zeros((11,11)),
           'z':np.zeros((11,11))}
    loads['z'][5,5]=100
    
    displacements = C.elastic_loading(loads, (1,1), v=0.3, G=200, 
                                      deflections='xyz', span=None, 
                                      simple=True)
    
    l2 = C.elastic_displacement(displacements, span=None, 
                                grid_spacing=(1,1), G=200, v=0.3, tol=1e-6, 
                                simple=True, max_it=100, components=None)
    
    npt.assert_allclose(l2['x'], loads['x'], atol=1)
    npt.assert_allclose(l2['y'], loads['y'], atol=1)
    npt.assert_allclose(l2['z'], loads['z'], atol=1)
    
    displacements = C.elastic_loading(loads, (1,1), v=0.5, G=200, 
                                      deflections='xyz', span=None, 
                                      simple=False)
    
    l2 = C.elastic_displacement(displacements, span=None, 
                                grid_spacing=(1,1), G=200, v=0.5, tol=1e-6, 
                                simple=False, max_it=100, components=None)
    
    npt.assert_allclose(l2['x'], loads['x'], atol=1)
    npt.assert_allclose(l2['y'], loads['y'], atol=1)
    npt.assert_allclose(l2['z'], loads['z'], atol=1)
    
    displacements = C.elastic_loading(loads, (1,1), v=0.3, G=200, 
                                      deflections='xyz', span=None, 
                                      simple=False)
    
    l2 = C.elastic_displacement(displacements, span=None, 
                                grid_spacing=(1,1), G=200, v=0.3, tol=1e-3, 
                                simple=False, max_it=100, components=None)
    
    npt.assert_allclose(l2['x'], loads['x'], atol=5)
    npt.assert_allclose(l2['y'], loads['y'], atol=5)
    npt.assert_allclose(l2['z'], loads['z'], atol=5)

if __name__ =='__main__':
    test_convert()
    test_elastic_loading()
    test_elastic_displacement()