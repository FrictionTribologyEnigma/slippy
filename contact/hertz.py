from math import sqrt, pi, log
from warnings import warn

__all__=['hertz']

def hertz(r1:list, r2:list, E:list, v:list, load:float, angle:float=0,
          stress:str='basic', grid:list=[101,101]):
    """Find the hertzian stress solution for the given system
    
    Parameters
    ----------
    r1, r2 : list
        Two element list of the radii in the first body (r1) and the second 
        body (r2) of the radii of the first body in the x and y directions. 
        Each element should be a float, use float('inf') to indicate a flat 
        surface. If a single number is supplied both elements are set to that
        number: r1=1 is equvilent to r1=[1,1]
    E : list
        Two element list of the youngs modulii of the first and second bodies. 
        See note on units.
    v : list 
        Two element list of the poisson's raios of the first and second bodies.
    load : float
        The load applied for a point contact or the load per unit length for a 
        line contact, see note on units.
    angle : float, optional (0)
    stress : str {'basic', 'full'}, optional ('basic')
        string describing the stresses that are to be returned options are: 
        - 'basic' - maximum and mean stresses are calculated
        - 'full' - surface stress profiles are calculated sub surface stresses 
            can be found with the hertz_subsurface_stress function.
    grid : list, optional ([101,101])
        Two element list of the number of grid points to be used in the surface
        stress profile only used if full stresses are requested. Stresses are 
        calculated from the origin to (contact_radius_x, contact_radius_y) as 
        result is always symmetical about x=0 and y=0.
    
    Returns
    -------
    results : dict
        Dictionary of the results:
            line contact : True is it is a line contact
            r_eff : The effective radius of the contact
            
    
    See Also
    --------
    hertz_subsurface_stress
    
    Notes
    -----
    Units must be consistent: if the youngs modulii is given in N/mm**2, the 
    radii should be given in mm and the load should be given in N. etc.
    
    References
    ----------
    
    """
    results=dict()
    #check inputs
    r1=_sanitise_radii(r1)
    r2=_sanitise_radii(r2)
    if r1[0]==r2[0]==float('inf') and r1[1]==r2[1]==float('inf'):
        raise ValueError("flat on flat contacts are not supported")
    
    # figure out which regime we are in
    line_contact = r1[0]==r2[0]==float('inf') or r1[1]==r2[1]==float('inf')
    results['line_contact']=line_contact
    
    # reduce the problem 
    r_reduce=[1/(1/R1+1/R2) for R1, R2 in zip(r1,r2)]
    r_eff=1/(1/r_reduce[0]+1/r_reduce[1])
    results['r_eff']=r_eff
    E_reduce= 2/((1-v[0]^2)/E[0] + (1-v[1]^2)/E[1])
    
    # calculate contact radii and maximum pressure
    if line_contact:
        contact_radii=[r if r==float('inf') else 
                       sqrt(8*load*r_eff/(pi*E_reduce)) for r in r_reduce]
        pressure_max=2*load/(pi*min(contact_radii))
        contact_area=contact_radii
    else:
        lam=min(r_reduce)/max(r_reduce)
        k = 1/(1+sqrt(log(16/lam)/(2*lam))-sqrt(log(4))+0.16*log(lam))
        m = 1-k^2
        
        if m==1:
            Em = 0
            ae = 1
            be = 1
        else:
            Em = pi/2*(1-m)*(1+2*m/(pi*(1-m))-1/8*log(1-m))
            ae = k**(1/3)*(2/pi*Em)**(1/3)
            be = k**(-2/3)*(2/pi*Em)**(1/3)
            
        contact_radii=[r*(3*load*r_eff/(E_reduce))**(1/3) for r in [ae,be]]

        contact_area = pi*contact_radii[0]*contact_radii[1]
        pressure_max = 3*load/(contact_area*2)
    results['contact_radii']=contact_radii
    results['contact_area']=contact_area
    results['pressure_max']=pressure_max
    #check assumptions
    if r_eff>(10*min([*r1,*r2])):
        warn("Effective radius is {0}, this is {1}".format(r_eff,
             r_eff/min([*r1,*r2])) + " times the smallest radius of the "
             "contacting bodies, results may be invalid.")  

    if stress.lower()!='full':
        return results
    # calculate stresses if needed
    
    
    
    return results
    
def _sanitise_radii(radii):
    """
    checks on the radii input to hertz
    """
    if type(radii) is not list:
        try:
            radii=[float(radii)]*2
        except ValueError:
            raise ValueError("radii must be list or number, not a"
                             " {}".format(type(radii)))
    else:
        if len(radii)!=2:
            raise ValueError("Radii must be a two element list supplied radii"
                             " list has {} elements".format(len(radii)))
        try:
            radii=[float(r) for r in radii]
        except ValueError:
            raise ValueError("Elements of radii are not convertable to floats:"
                             "{}".format(radii))
    if any([r==0 for r in radii]):
        raise ValueError("Radii contains zero values, use float('inf') for fla"
                         "t surfaces")
    return radii
            
            
            
            
            
            
            
            