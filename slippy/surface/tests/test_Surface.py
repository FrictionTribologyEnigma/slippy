"""
Tests for the surface class

tests for fft, psd and acf are done in test for frequency and random surface 
classes

roughness functions are tested in more detail in their own tests

"""
import numpy as np
import numpy.testing as npt
from pytest import raises as assert_raises
import slippy.surface as S

def test_assurface():
    profile=np.random.normal(size=[10,10])
    ms=S.assurface(profile, grid_spacing=0.1)
    npt.assert_equal(ms.profile, profile)
    npt.assert_equal(ms.extent, (1,1))
    

def test_read_surface():
    ms=S.read_surface('C:\\Users\\44779\\code\\SlipPY\\data\\image1_no '
                   'header_units in nm.txt', delimiter=' ')
    
    npt.assert_equal(ms.shape, (512,512))
    
    ms=S.read_surface('C:\\Users\\44779\\code\\SlipPY\\data\\dem.al3d')
    
    npt.assert_equal(ms.shape, (7556, 2048))

def test_roughness():
    profile=np.random.normal(size=[500,500])
    A=S.assurface(profile, 1)
    actual=A.roughness(['sq', 'ssk', 'sku'])
    expected=[1,0,3]
    npt.assert_allclose(actual, expected, rtol=1e-2)
    
def test_fill_holes():
    x=np.arange(12, dtype=float)
    y=np.arange(12, dtype=float)
    X,Y=np.meshgrid(x,y)
    X2=np.pad(X,2,'constant', constant_values=float('nan'))
    X2[6,6]=float('nan')
    A=S.Surface(profile=X2)
    A.fill_holes()
    npt.assert_array_almost_equal(X, A.profile)
    
def test_mask():
    profile=np.zeros((10,10))
    positions=[1,4,7,44,75,99]
    values=[float('nan'), float('inf'), float('-inf'), 1.1]
    
    for i in range(len(values)):
        profile[positions[i]]=values[i]
        A=S.Surface(profile=profile)
        A.mask=values[i]
        assert A.mask[positions[i]]==True
        assert np.sum(A.mask.flatten())==1
    

def test_combinations():
    x=np.arange(6, dtype=float)
    y=np.arange(12, dtype=float)
    X,Y=np.meshgrid(x,y)
    Z=np.zeros_like(X)
    
    A=S.Surface(profile=X)
    
    B=np.array(A+A)
    npt.assert_array_equal(B,X+X)
    
    B=np.array(A-A)
    npt.assert_array_equal(B,Z)
    
    x2=np.arange(0,6,2, dtype=float)
    y2=np.arange(0,12,2, dtype=float)
    X2,Y2=np.meshgrid(x2,y2)
    
    B=S.Surface(profile=Y2)
    
    A.grid_spacing=1
    B.grid_spacing=2
    
    C=np.array(A+B)
    npt.assert_array_almost_equal(C,X+Y)
    
    C=np.array(A-B)
    npt.assert_array_almost_equal(C,X-Y)
    
    B=S.Surface(profile=Y)
    
    C=A+B
    npt.assert_array_almost_equal(np.array(C),X+Y)
    assert C.grid_spacing==float(1)
    
    C=A-B
    npt.assert_array_almost_equal(np.array(C),X-Y)
    assert C.grid_spacing==float(1)
    
    B.grid_spacing=2
    
    with assert_raises(ValueError):
        A+B
        
    with assert_raises(ValueError):
        A-B
    
    
def test_dimentions():
    # setting grid spacing with a profile
    profile=np.random.normal(size=[10,10])
    ms=S.Surface(profile=profile)
    npt.assert_equal(ms.shape, (10, 10))
    ms.grid_spacing=0.1
    npt.assert_allclose(ms.extent, [1,1])
    assert ms.is_descrete==True
    
    # deleting 
    
    del ms.profile
    
    assert ms.is_descrete==False
    assert ms.profile is None
    assert ms.extent is None
    assert ms.shape is None
    assert ms.size is None
    
    assert ms.grid_spacing is 0.1
    
    del ms.grid_spacing
    
    assert ms.grid_spacing is None
    
    ms.extent=[10,11]
    
    assert ms.profile is None
    assert ms.extent is [10,11]
    assert ms.shape is None
    assert ms.size is None
    assert ms.grid_spacing is None
    
    ms.grid_spacing=1
    
    assert ms.profile is None
    assert ms.extent is [10,11]
    assert ms.shape==(10,11)
    assert ms.size==110
    assert ms.grid_spacing==float(1)
    
    del ms.shape
    assert ms.shape is None
    assert ms.size is None
    
    del ms.extent
    assert ms.extent is None
    assert ms.grid_spacing is None
    
    ms.extent=[10,11]
    assert ms.profile is None
    assert ms.extent is [10,11]
    assert ms.shape==(10,11)
    assert ms.size==110
    
    del ms.grid_spacing
    assert ms.extent is None
    assert ms.grid_spacing is None
    assert ms.shape is None
    assert ms.size is None
    
    ms.shape=[5,10]
    assert ms.size==50
    
    ms.extent=[50,100]
    assert ms.grid_spacing==10
    
    del ms.grid_spacing
    ms.profile=profile
    with assert_raises(ValueError):
        ms.extent=[10,9]

def test_array():
    profile=np.random.normal(size=[10,10])
    ms=S.Surface(profile=profile)
    npt.assert_array_equal(profile, np.asarray(ms))
    
    ms.profile=[[1,2],[2,1]]
    assert type(ms.profile) is np.ndarray
