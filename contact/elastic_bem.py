import slippy.surface as S
import numpy as np
from itertools import product
from scipy.signal import fftconvolve
import warnings

__all__=['convert_array', 'convert_dict', 'elastic_displacement', '_solve_ed',
         'elastic_loading', '_solve_el', 'elastic_im', 'combined_modulus']

#todo 
def contact_rigid(surface1: S.Surface, rigid_surface: S.surface=0, 
                  displacement: list=None, load: list=None, material=None):
    """ Contact between a rigid surface and an elastic surface
    can be load or displacement controlled
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    
    See Also
    --------
    
    
    Notes
    -----
    
    
    Examples
    --------
    
    
    """

def convert_array(loads_or_displacements: np.array):
    """ Converts an array of loads or displacements to a dict of loads
    
    Parameters
    ----------
    loads_or_displacements: numpy.array
        Loads or displacemnts (3 by N by M array)
    
    Returns
    -------
    dict
        The same loads or displacements in a dict with keys 'x' 'y' and 'z'
    
    See Also
    --------
    convert_dict
    
    Notes
    -----
    
    Examples
    --------
    """
    valid_directions=['x','y','z']
    out=dict()
    for direc,i in zip(valid_directions,range(3)):
        out[direc]=loads_or_displacements[i]
    return out
    
def convert_dict(loads_or_displacements: dict):
    """ Converts a dict of loads or displacements to an array of loads
    
    Parameters
    ----------
    loads_or_displacements: dict
        Dict with keys 'x' 'y' and 'z' or any combination, each value must be
        N by M array
    
    Returns
    -------
    numpy.array
        3 by N by M array of loads or displacements 
    
    See Also
    --------
    convert_array
    
    Notes
    -----
    
    Examples
    --------
    """
    ld={key:np.squeeze(value) for (key,value) in 
        loads_or_displacements.items()}
    
    valid_directions=['x','y','z']
    directions=list(ld.keys())
        
    if not set(directions).issubset(set(valid_directions)):
        msg=("invalid keys in dict, keys found:" +
             ' '.join(directions) + " Valid keys are:" + 
             ' '.join(valid_directions))
        raise ValueError(msg)
    
    shapes=[value.shape for (key,value) in ld.items()]
    
    if len(set(shapes))!=1:
        raise ValueError("Vectors are not all the same shape")
    
    out=np.zeros([3]+list(shapes[0]))
    
    for direc,i in zip(valid_directions,range(3)):
        if direc in directions:
            out[i]=ld[direc]
    
    return out

def elastic_displacement(deflections: dict, grid_spacing: tuple, G: float, 
                         v: float, span:tuple=None, tol=1e-4, 
                         simple: bool=False, max_it: int=100, 
                         components: dict=None):
    """Find surface loading from displacements of an elastic half space
    
    Parameters
    ----------
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
    max_it : int optional (100)
        The maximum number of itterations before aborting the loop
    components : dict optional (None)
        Components of the influence matrix, if none they are found 
        automatically
    
    Returns
    -------
    Loads : dict 
        dict of arrays of surface loads with keys 'x' 'y' and 'z'
    
    See Also
    --------
    elastic_loading
    
    Notes
    -----
    This function is much slower than elastic_loading. 
    
    Examples
    --------
    
    Reference
    ---------
    
    Complete boundry element formulation for normal and tangential contact
    
    """
    valid_directions=['x','y','z']
    def_directions=list(deflections.keys())
        
    if not set(def_directions).issubset(set(valid_directions)):
        msg=("invalid keys in loads dict, supplied keys are: " +
             ''.join(def_directions) + " Valid keys are: " + 
             ''.join(valid_directions))
        raise ValueError(msg)
    
    deflections={key:np.array(value) for (key,value) in deflections.items()}
    shapes=[value.shape for (key,value) in deflections.items()]
    
    if len(set(shapes))!=1:
        raise ValueError("Deflection vectors are not all the same shape")
    
    if span is None:
        span=shapes[0]
    
    if len(grid_spacing)==1:
        grid_spacing=2*grid_spacing
    elif len(grid_spacing)>2:
        raise ValueError("Too many elements in grid_spacing, should be 1 or 2")
    
    if components is None:
        if simple or len(deflections)==1:
            comp_names=[dd*2 for dd in def_directions] # ['xx','yy' etc.]
        else:
            
            comp_names=list(product(deflections,valid_directions))
            comp_names=[a+b for a,b in comp_names]
            
        #get componets of the influence matrix
        components=elastic_im(grid_spacing, span, G, v, comp_names)
    else:
        comp_names=list(components.keys())
        components=list(components.values())
    
    loads=_solve_ed(deflections, components, max_it, tol)
    
    return loads
    
def _solve_ed(deflections: dict, components: dict, max_it:int, tol:float):
    """ The meat of the elastic deflection algorithm 
    
    Split away from the main function to allow for faster compuation avoiding 
    calculating the influence matrix for each itteration
    
    Parameters
    ----------
    deflections : dict
        The deflections at the surface with keys 'x' 'y' or 'z' or any 
        combination
    components : dict
        Components of the influence matrix keys are 'xx', 'xy' ... 'zz' for 
        each name the first character represents the displacement and the 
        second represents the load, eg. 'xy' is the deflection in the x 
        direction caused by a load in the y direction
    max_it : int
        The maximum number of itterations before breaking the loop
    tol : float
        The tolerance on the itterations, the loop is ended when the norm of 
        the residual is below tol
    
    Returns
    -------
    loads : dict
        The loads on the surface to cause the specified displacement with keys
        'x' 'y' and 'z'
    
    See Also
    --------
    elastic_displacement
    elastic_loading
    _solve_el
    elastic_im
    
    Notes
    -----
    Components of the influence matrix can be found using elastic_im
    
    This function is much slower than _solve_el, as it calls it in an 
    optimisation routine.
    
    This function has very little error checking and may give unexpected 
    results with bad inputs please use elastic_displacement to check inputs 
    before using this directly
    
    References
    ----------
    
    Complete boundry element formulation for normal and tangential contact 
    problems
    
    """
    def_directions=list(deflections.keys())
    
    sub_domains={key:np.logical_not(np.isnan(de)) for 
            (key,de) in deflections.items()}
    sub_domain_sizes={key:np.sum(value) for (key,value) in sub_domains.items()}
    
    #find first residual
    loads={'x':np.zeros_like(deflections[def_directions[0]]),
           'y':np.zeros_like(deflections[def_directions[0]]),
           'z':np.zeros_like(deflections[def_directions[0]])}
    calc_deflections=_solve_el(loads, components)
    
    r=np.array([])
    
    for dd in def_directions:
        r=np.append(r,deflections[dd][sub_domains[dd]]-
                         calc_deflections[dd][sub_domains[dd]])
    
    #start loop
    d=r
    D={'x':np.zeros_like(deflections[def_directions[0]]),
       'y':np.zeros_like(deflections[def_directions[0]]),
       'z':np.zeros_like(deflections[def_directions[0]])}
    
    itnum=0
    resid_norm=np.linalg.norm(r)
    
    while resid_norm>tol:
        # put calculated values back into right place
        start=0
        for dd in def_directions:
            end=start+sub_domain_sizes[dd]
            D[dd][sub_domains[dd]]=d[start:end]
            start=end
        
        # find z (equation 25 in ref)
        calc_deflections=_solve_el(D, components)
        z=np.array([])
        for dd in def_directions:
            z=np.append(z,calc_deflections[dd][sub_domains[dd]])
        
        # find alpha (equation 26)
        alpha=np.matmul(r,r)/np.matmul(d,z)
        
        # update stresses (equation 27)
        update_vals=alpha*d
        start=0
        for dd in def_directions:
            end=start+sub_domain_sizes[dd]
            loads[dd][sub_domains[dd]]+=update_vals[start:end]
            start=end
            
        r_new=r-alpha*z #equation 28
        
        # find new search direction (equation 29)
        beta=np.matmul(r_new,r_new)/np.matmul(r,r)
        r=r_new
        resid_norm=np.linalg.norm(r)
        
        d=r+beta*d
        
        itnum+=1
        
        if itnum>max_it:
            msg=(f"Max itteration ({max_it}) reached without convergence"
                 f" residual was: {resid_norm}, convergence declared at {tol}")
            warnings.warn(msg)
            break
    
    return loads

def elastic_loading(loads: dict, grid_spacing: tuple, v: float, G: float, 
                    deflections: str='xyz', span: tuple=None, simple=False):
    """Find surface displacments from a set of loads on an elastic half-space
    
    Parameters
    ----------
    loads : dict
        dict of loads with keys 'x' 'y' or 'z' allowed
    deflections : str {'xyz', 'x', 'y', 'z' or any combination}
        The components of the surface deflections to be calculated
    span : tuple
        The span of the influence matrix in grid points defaults to same as 
        surface
    grid_spacing : tuple or float
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
    displacements : dict
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
    valid_directions=['x','y','z']
    load_directions=list(loads.keys())
    
    if len(deflections)>3:
        raise ValueError("Too many deflection directions")
    if len(load_directions)>3:
        raise ValueError("Too many load directions")
        
    if not set(load_directions).issubset(set(valid_directions)):
        msg=("invalid keys in loads dict, supplied keys are: " +
             ''.join(load_directions) + " Valid keys are: " + 
             ''.join(valid_directions))
        raise ValueError(msg)
    
    loads={key:np.array(value) for (key,value) in loads.items()}
    shapes=[value.shape for (key,value) in loads.items()]
    
    if len(set(shapes))!=1:
        raise ValueError("Load vectors are not all the same shape")
    
    if span is None:
        span=shapes[0]
    
    if len(grid_spacing)==1:
        grid_spacing=2*grid_spacing
    elif len(grid_spacing)>2:
        raise ValueError("Too many elements in grid_spacing, should be 1 or 2")
        
    #u_a=ifft(fft(K_ab) * fft(sigma_b))
    
    if simple:
        comp_names=[2*ld for ld in load_directions]
    else:
        comp_names=list(product(deflections,load_directions))
        comp_names=[a+b for a,b in comp_names]
        
    components=elastic_im(grid_spacing, span, G, v, comp_names)
    
    displacements=_solve_el(loads, components)
    
    return displacements

def _solve_el(loads:dict, components:dict):
    """The meat of the elastic loading algorithm
    
    Parameters
    ----------
    loads : dict
        dict of N by M arrays of surface loads with labels 'x', 'y', 'z' or 
        any combination
    components : dict
        Components of the influence matrix keys are 'xx', 'xy' ... 'zz' for 
        each name the first character represents the displacement and the 
        second represents the load, eg. 'xy' is the deflection in the x 
        direction caused by a load in the y direction
        
    Returns
    -------
    displacements : dict
        dict of N by M arrays of surface displacements with labels 'x', 'y', 
        'z'
        
    See Also
    --------
    elastic_im
    elastic_loading
    
    Notes
    -----
    This function has very little error checking and may give unexpected 
    results with bad inputs please use elastic_loading to check inputs 
    before using this directly
    
    Components of the influence matrix can be found by elastic_im
    
    References
    ----------
    
    Complete boundry element formulation for normal and tangential contact 
    problems
    
    """
    shape=list(loads.values())[0].shape
    
    displacements={'x':np.zeros(shape),
                   'y':np.zeros(shape),
                   'z':np.zeros(shape)}
    
    for c_name,component in components.items():
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
    component: str or list {'xx','xy','xz','yx','yy','yz','zx','zy','zz','all'}
        The required components eg the 'xy' component represents the x 
        deflection caused by loads in the y direction
    
    Returns
    -------
    dict
        dict of the requested influence matrix or matricies
    
    See Also
    --------
    elastic_loading
    elastic_deflection
    
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
        out={component:_elastic_im_getter(k,l,m,n,hx,hy,shear_mod,v,component)}
        return out
    

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
    
    loads=S.RoundSurface(1)
    loads.extent=[5.11,5.11]
    loads.grid_spacing=0.01
    loads.descretise()
    
    loads=np.array(loads)*1000
    
    elastic_im((0.01,),(1,),200e9,0.3,'all')
    
    
    
    #a=combined_modulus(200, 0.3)
    
    