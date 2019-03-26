import slippy.surface as S
import numpy as np
from itertools import product
from scipy.signal import fftconvolve

def elastic_displacement(surface, deflections: dict, span:tuple, 
                         grid_spacing:float=None, G:float=None, 
                         v:float=None, tol=1e-4, simple:bool=False):
    """Find surface loading from displacements of an elastic half space
    
    Parameters
    ----------
    surface : Surface object or array
        The surface to be analysed (N by M)
    deflections : dict 
        dict of arrays of deflections with keys 'x' 'y' or 'z' allowed
    span : tuple
        The span of the influence matrix in grid points defaults to same as 
        surface
    grid_spacing : float optional (None)
        The grid spacing only needed if surface is an array
    v : float optional (None)
        The Poission's ratio of the surface material, 
        only needed if surface is an array
    G : float optional (None)
        The shear modulus of the surface material,
        only needed if surface is an array
    simple : bool optional (False)
        If true only deflections in the directions of the loads are calculated,
        only the Cxx, Cyy and Czz components of the influcence matrix are used
    
    Returns
    -------
    Loads : array
    
    See Also
    --------
    elastic_loading
    
    Notes
    -----
    
    Examples
    --------
    
    Reference
    ---------
    
    Complete boundry element formulation for normal and tangential contact
    
    """
    set_defs=list(deflections.keys())
    #There has to be some better initial guess than this but what ever
    sigma={'x':np.zeros_like(deflections[set_defs[0]]),
           'y':np.zeros_like(deflections[set_defs[0]]),
           'z':np.zeros_like(deflections[set_defs[0]])}
    r=
    

def elastic_loading(surface, loads: dict, deflections='xyz', span: tuple=None, 
                    grid_spacing=None, v=None, G=None, simple=False):
    """Find surface displacments from a set of loads on an elastic half-space
    
    Parameters
    ----------
    surface : Surface object or array
        The surface to be analysed (N by M)
    loads : dict
        dict of loads with keys 'x' 'y' or 'z' allowed
    deflections : str {'xyz', 'x', 'y', 'z' or any combination}
        The components of the surface deflections to be calculated
    span : tuple
        The span of the influence matrix in grid points defaults to same as 
        surface
    grid_spacing : float optional (None)
        The grid spacing only needed if surface is an array
    v : float optional (None)
        The Poission's ratio of the surface material, 
        only needed if surface is an array
    G : float optional (None)
        The shear modulus of the surface material,
        only needed if surface is an array
    simple : bool optional (False)
        If true only deflections in the directions of the loads are calculated,
        only the Cxx, Cyy and Czz components of the influcence matrix are used
    
    Returns
    -------
    displacements : array
        The surface deflections
        
    See Also
    --------
    elastic_im
    elastic_displacement
    
    Notes
    -----
    
    
    Examples
    --------
    
    """
    load_directions=list(loads.keys())
    
    if not issubclass(surface, S.Surface):
        surface=S.assurface(surface, grid_spacing)
    if not hasattr(surface, 'material'):
        if G is None or v is None:
            raise ValueError("No material found")
    else:
        if not surface.material.type in ['elastic', 'elastic-plastic', 'epp']:
            raise ValueError("Material is not elastic")
        G=surface.material.G
        v=surface.material.v
    
    if G is None or v is None:
        raise ValueError("No material found G={}, v={}".format(G,v))
    
    if len(deflections)>3:
        raise ValueError("Too many deflection directions")
    if len(load_directions)>3:
        raise ValueError("Too many load directions")
    
    if span is None:
        span=surface.shape
    #u_a=ifft(fft(K_ab) * fft(sigma_b))
    
    displacements={'x':np.zeros(list(loads.shape[0:2])),
                   'y':np.zeros(list(loads.shape[0:2])),
                   'z':np.zeros(list(loads.shape[0:2]))}
    
    if simple:
        comp_names=[2*ld for ld in load_directions]
    else:
        comp_names=product(deflections,load_directions)
    
    components=elastic_im(grid_spacing, span, G, v, comp_names)
    
    for component,c_name in zip(components,comp_names):
        load_name=c_name[1]
        dis_name=c_name[0]
        displacements[dis_name]+=fftconvolve(loads[load_name], 
                                             component, mode='same')
    return displacements
    

def elastic_im(grid_spacing: tuple, span: tuple, 
                    shear_mod: float, v: float, 
                    component):
    """Influence matrix for a elastic contact problems including 
    
    Parameters
    ----------
    grid_spacing : tuple
        The spacing between grid points in the x and y directions
    span : tuple
        The span required in the x and y directions in number of grid points
    shear_mod : float
        The shear modulus of the surface material
    v : float
        The Poisson's ratio of the surface material
    component : str or list {'xx','xy','xz','yx','yy','yz','zx','zy','zz'}
        The required components
    
    Returns
    -------
    numpy.array
        Array of the requested influence matrix or matricies if 'all' requested
    
    See Also
    --------
    
    
    Notes
    -----
    span is automatically rounded up to the next odd number to ensure 
    symmetrical results
    
    K^{ij i'j'}_zz=(1-v)/(2*pi*G)*Czz
    
    Czz=(hx*(k*log((m+sqrt(k**2+m**2))/(n+sqrt(k**2+n**2)))+
             l*log((n+sqrt(l**2+n**2))/(m+sqrt(l**2+m**2))))+
         hy*(m*log((k+sqrt(k**2+m**2))/(l+sqrt(l**2+m**2)))+
             n*log((l+sqrt(l**2+n**2))/(k+sqrt(k**2+n**2)))))
    
    In which:
    
    k=i'-i+0.5
    l=i'-i-0.5
    m=j'-j+0.5
    n=j'-j-0.5
    hx=grid_spacing[0]
    hy=grid_spacing[1]
    
    Examples
    --------
    
    
    References
    ----------
    Complete boundry element method formulation for normal and tangential 
    contact problems
    
    """
    if len(span)==1:
        span=span*2
    if len(grid_spacing)==1:
        grid_spacing=grid_spacing*2
    
    try:
        # lets just see how this changes
        #i-i' and j-j'
        idmi=(np.arange(span[0]+~span[0]%2)-int(span[0]/2))
        jdmj=(np.arange(span[1]+~span[1]%2)-int(span[1]/2))
        Idmi,Jdmj=np.meshgrid(idmi,jdmj)
        
    except TypeError:
        msg="span should be a tuple of integers, {} found".format(
                type(span[0]))
        raise TypeError(msg)
    
    if component=='all':
        component=['xx','xy','xz','yx','yy','yz','zx','zy','zz']
    
    k=Idmi+0.5
    l=Idmi-0.5
    m=Jdmj+0.5
    n=Jdmj-0.5
    
    hx=grid_spacing[0]
    hy=grid_spacing[1]
    
    if type(component) is list:
        full={}
        for comp in component:
            full[comp]=(_elastic_im_getter(k,l,m,n,hx,hy,shear_mod,v,comp))
        return full
    else: 
        return _elastic_im_getter(k,l,m,n,hx,hy,shear_mod,v,component)
    
    

def _elastic_im_getter(k,l,m,n,hx,hy,shear_mod,v,comp) -> np.array:
    """Find influence matrix components for an elastic contact problem
    
    Parameters
    ----------
    k : array
        k=i'-i+0.5
    l : array
        l=i'-i-0.5
    m : array
        m=j'-j+0.5
    n : array
        n=j'-j-0.5
    hx : float
        The grid spacing in the x direction
    hy : float
        The grid spacing in the y direction
    shear_mod : float
        The shear modulus of the surface materuial
    v : float
        The Poission's ratio of the surface material
    comp : str {'xx','xy','xz','yx','yy','yz','zx','zy','zz'}
        The component to be returned
    
    Returns
    -------
    C : array
        The influence matrix component requested
    
    See Also
    --------
    elastic_im
    
    Notes
    -----
    span is automatically rounded up to the next odd number to ensure 
    symmetrical results
    Must be divided by the combined modulus before use
    
    K^{ij i'j'}_zz=(1-v)/(2*pi*G)*Czz
    
    
    
    Don't use this function, used by: elastic_im
    
    References
    ----------
    Complete boundry element method formulation for normal and tangential 
    contact problems
    
    """
    if comp=='zz':
        Czz=(hx*(k*np.log((m+np.sqrt(k**2+m**2))/(n+np.sqrt(k**2+n**2)))+
                 l*np.log((n+np.sqrt(l**2+n**2))/(m+np.sqrt(l**2+m**2))))+
            hy*(m*np.log((k+np.sqrt(k**2+m**2))/(l+np.sqrt(l**2+m**2)))+
                n*np.log((l+np.sqrt(l**2+n**2))/(k+np.sqrt(k**2+n**2)))))
        const=(1-v)/(2*np.pi*shear_mod)
        return const*Czz
    elif comp=='xx':
        Cxx=(hx*(1-v)*(k*np.log((m+np.sqrt(k**2+m**2))/(n+np.sqrt(k**2+n**2)))+
                      l*np.log((n+np.sqrt(l**2+n**2))/(m+np.sqrt(l**2+m**2))))+
                  hy*(m*np.log((k+np.sqrt(k**2+m**2))/(l+np.sqrt(l**2+m**2)))+
                      n*np.log((l+np.sqrt(l**2+n**2))/(k+np.sqrt(k**2+n**2)))))
        const=1/(2*np.pi*shear_mod)
        return const*Cxx
    elif comp=='yy':
        Cyy=     (hx*(k*np.log((m+np.sqrt(k**2+m**2))/(n+np.sqrt(k**2+n**2)))+
                      l*np.log((n+np.sqrt(l**2+n**2))/(m+np.sqrt(l**2+m**2))))+
            hy*(1-v)*(m*np.log((k+np.sqrt(k**2+m**2))/(l+np.sqrt(l**2+m**2)))+
                      n*np.log((l+np.sqrt(l**2+n**2))/(k+np.sqrt(k**2+n**2)))))
        const=1/(2*np.pi*shear_mod)
        return const*Cyy
    elif comp=='xz':
        Cxz=(hy/2*(m*np.log((k**2+m**2)/(l**2+m**2))+
                   n*np.log((l**2+n**2)/(k**2+n**2)))+
             hx*(k*(np.arctan(m/k)-np.arctan(n/k))+
                 l*(np.arctan(n/l)-np.arctan(m/l))))
        const=(2*v-1)/(4*np.pi*shear_mod)
        return const*Cxz
    elif comp=='zx':
        return -1*_elastic_im_getter(k,l,m,n,hx,hy,shear_mod,v,'xz')
    elif comp in ['yx','xy']:
        Cyx=(np.sqrt(hy**2*n**2+hx**2*k**2)-
             np.sqrt(hy**2*m**2+hx**2*k**2)+
             np.sqrt(hy**2*m**2+hx**2*l**2)-
             np.sqrt(hy**2*n**2+hx**2*l**2))
        const=v/(2*np.pi*shear_mod)
        return const*Cyx
    elif comp in ['zy','yz']:
        Czy=(hx/2*(k*np.log((k**2+m**2)/(n**2+k**2))+
                   l*np.log((l**2+n**2)/(m**2+l**2)))+
             hy*(m*(np.arctan(k/m)-np.arctan(l/m))+
                 n*(np.arctan(l/n)-np.arctan(k/n))))
        const=(1-2*v)/(4*np.pi*shear_mod)
        return const*Czy
    else:
        msg=('component name not recognised: '
             '{}, components are lower case'.format(comp))
        raise ValueError(msg)
    return 
    
def combined_modulus(stiff1: float, v1: float, stiff2: float=None, 
                     v2: float=None, modulus: str='elastic') -> float :
    """Find the combined elastic or shear modulus of a material pair
    
    Parameters
    ----------
    stiff1,2 : float
        The stiffness or youngs modulus of the materials
    v1,2 : float
        The Poisson's ratio of the materials
    mod : str optional {'elastic', 'shear'}
        The type of modulus to be calculated
        
    Return
    ------
    float
        The requested combined modulus
        
    See Also
    --------
    
    Notes
    -----
    If no material is given for material 2 it is assumed that the pair have 
    identical properties
    
    Examples
    --------
    
    >>> # find the combined elastic modulus for a steel on steel pair:
    >>> combined_modulus(200E9, 0.3)
    
    >>> # find the combined shear modulus for a aluminum-steel pair:
    >>> combined_modulus(200E9, 0.3, 63E9. 0.34, 'shear')
    
    
    Reference
    ---------
    
    
    """
    supported_types=['elastic', 'shear']
    
    try:
        mod=modulus.lower()
    except AttributeError:
        msg="Modulus must be a string, {} supplied".format(type(modulus))
    
    if stiff2 is None:
        stiff2=stiff1
    if v2 is None:
        v2=v1
    
    if mod=='elastic':
        return 1/((1-v1**2)/stiff1+(1-v2**2)/stiff2)
    elif mod=='shear':
        return 1/((1+v1)*(1-2*v2)/2/stiff1-(1+v2)*(1-2*v1)/2/stiff2)
    else:
        msg=("Unrecognised modulus '{}' supported types are: ".format(mod)
             + ''.join(supported_types))
        raise ValueError(msg)
        
if __name__=='__main__':
    ms=S.FlatSurface([0,0])
    ms.extent=[5.11,5.11]
    ms.grid_spacing=0.01
    ms.descretise()
    
    loads=S.RoundSurface([1])
    loads.extent=[5.11,5.11]
    loads.grid_spacing=0.01
    loads.descretise()
    
    loads=np.array(loads)*1000
    
    elastic_im((0.01,),(1,),200e9,0.3,'all')
    
    
    
    #a=combined_modulus(200, 0.3)
    
    