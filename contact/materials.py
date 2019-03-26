import numpy as np

#def material(mat_type, parameters)

class _Material(object):
    """ A class for describing material behaviour
    
    """
    #def change_units(current_unit:str, new_unit:str):
    """Change the units of the material properties
    
    Parameters
    ----------
    current_unit : str
        The units which the material properties are currently in
    new_unit : str
        The desired units
        
    Notes
    -----
    This will change all the material properties to be in the new unit 
    systen. Unit systems are defined in the format "mass-force-length"
    For example "kg-N-m" defines the SI system of units.
    
    Valid masses are:
    - "kg"   - kilogram
    - "g"    - gram
    - "mg"   - miligram
    - "um"   - microgram
    - "lb"   - Avoirdupois pound (0.45359237 kg)
    - "oz"   - Avoirdupois ounce (0.02834952312 kg)
    - "slug" - A slug (14.59390 kg)
    
    Valid forces are:
    - "kN"  - kilonewton
    - "N"   - newton
    - "mN"  - milinewton
    - "uN"  - micronewton
    - "lbf" - pound force (4.448222 N)
    - "kgf" - kilogram force (9.80665 N)
    - "dyn" - dyne (10e-5 N)
    - "pda" - poundal (0.138255 N)
    
    Valid lengths are:
    - "km"   - kilometer
    - "m"    - meter
    - "mm"   - milimeter
    - "um"   - micrometer
    - "thou" - thou or mil (0.0000254 m)
    - "in"   - inch (0.0254 m)
    - "ft"   - foot (0.3048 m)
    - "y"    - yard (0.9144 m)
    - "mile" - mile (1609.344 m)
    
    Examples
    --------
    
    """

            
class Elastic(_Material):
    """ A Class for defining elastic materials
    
    Parameters
    ----------
    
    properties : dict
        dict of properties, dicts must have exactly 2 items. 
        Allowed keys are : 'E', 'v', 'G', 'K', 'M', 'Lam'
        See notes for definitions
    layer_thickness : optional ([inf])
        The material is a thin layer on a rigid substrate, the thickness of the
        layer. If this is set the layered form of the BEM equations will be
        used by default
    density : float optional (None)
        The density of the material
    
    Attributes
    ----------
    E
    v
    K
    Lam
    G
    M
    layer_thickness
    density
    
    Methods
    -------
    speed_of_sound
    
    See Also
    --------
    
    Notes
    -----
    
    Keys refer to:
        - E   - Young's modulus
        - v   - Poission's ratio
        - K   - Bulk Modulus
        - Lam - Lame's first parameter
        - G   - Shear modulus
        - M   - P wave modulus 
    
    Examples
    --------
    >>> # Make a material model for elastic steel on a rigid substarte with a 
    >>> # thickness of 1mm
    >>> steel=Elastic({'E':200e9, 'v':0.3}, layer_thickness=0.001, 
    >>>               density=7850)
    >>> # Find it's pwave modulus:
    >>> pwm=steel.M
    >>> # Find the speeds of sound:
    >>> sos=steel.speed_of_sound()
    """
    _properties={'E':None,
                 'v':None,
                 'G':None,
                 'K':None,
                 'Lam':None,
                 'M':None,}
    
    _last_set=None
    
    density=None
    
    layer_thickness=float('inf')
    
    def __init__(self, properties:dict, layer_thickness:float=float('inf'), 
                 density=None):
        """
        
        """
        if len(properties)==2:
            self._properties=_get_properties(properties)
        elif len(properties)==1:
            kv=list(properties.items())[0]
            self._set_props(*kv)
        elif len(properties)!=0:
            raise ValueError("Too many properties suplied, must be 1 or 2")
        
        self.density=density
        self.layer_thickness=layer_thickness
    
    
    
    def _del_props(self, prop):
        #delete any of the material properties
        keys=list(self._properties.keys())
        if self._last_set==prop:
            self._properties={key:None for key in keys}
            self._last_set=None
        else:
            self._properties={key:None for key in keys 
                              if not key==self._last_set}
        
    def _set_props(self, prop, value):
        allowed_props=['E','v','G','K','Lam','M']
        if not prop in allowed_props:
            msg=(f'property {prop} not recognised allowed propertied are: ' +
                 ' '.join(allowed_props))
            raise ValueError(msg)
        
        self._properties[prop]=value
        
        if self._last_set is None or self._last_set==prop:
            self._last_set=prop
        else:
            self._last_set=prop
            #first find E and v
            set_props={prop:value, 
                        self._last_set:self._properties[self._last_set]}
            
            self._properties=_get_properties(set_props)
        return
    
    
    @property
    def E(self):
        """The Young's modulus of the material"""
        return self._properties('E')
    @E.deleter
    def E(self):
        self._del_props('E')
    @E.setter
    def E(self,value):
        self._set_props('E',value)
        
    @property
    def v(self):
        """The Poissions's ratio of the material"""
        return self._properties('v')
    @v.deleter
    def v(self):
        self._del_props('v')
    @v.setter
    def v(self,value):
        self._set_props('v',value)
    
    @property
    def G(self):
        """The shear modulus of the material"""
        return self._properties('G')
    @G.deleter
    def G(self):
        self._del_props('G')
    @G.setter
    def G(self,value):
        self._set_props('G',value)
    
    @property
    def K(self):
        """The bulk modulus of the material"""
        return self._properties('K')
    @K.deleter
    def K(self):
        self._del_props('K')
    @K.setter
    def K(self,value):
        self._set_props('K',value)
        
    @property
    def Lam(self):
        """Lame's first parameter for the material"""
        return self._properties('Lam')
    @Lam.deleter
    def Lam(self):
        self._del_props('Lam')
    @Lam.setter
    def Lam(self,value):
        self._set_props('Lam',value)
        
    @property
    def M(self):
        """The p wave modulus of the material"""
        return self._properties('M')
    @M.deleter
    def M(self):
        self._del_props('M')
    @M.setter
    def M(self,value):
        self._set_props('M',value)
    
    def speed_of_sound(self, density: float=None):
        """find the speed of sound in the material
        
        Parameters
        ----------
        density : float optional (None)
            The density of the material
        
        Returns
        -------
        
        speeds : dict
            With keys 's' and 'p' giving the s and p wave speeds
            
        Notes
        -----
        
        Finds speeds according to the following equations:
        
        Vs=sqrt(G/rho)
        Vp=sqrt(M/rho)
        
        Where rho is the density, G is the shear modulus and M is the p wave
        modulus
        
        Examples
        --------
        >>> Find the speed of sound in steel
        >>> my_material=Elastic({'E':200e9, 'v':0.3})
        >>> my_material.speed_of_sound(7850)
        
        """
        if density is not None:
            self.density=density
        elif self.density is None:
            raise ValueError("Density not given or set")
        
        speeds={'s':np.sqrt(self.G/self.density), 
                'p':np.sqrt(self.M/self.density)}
        
        return speeds
    
class ElasticPlastic(Elastic):
    """ A Class for defining elastic materials
    
    Parameters
    ----------
    
    properties : dict
        dict of properties, dicts must have exactly 2 items. 
        Allowed keys are : 'E', 'v', 'G', 'K', 'M', 'Lam'
        See notes for definitions
    layer_thickness : optional ([inf])
        The material is a thin layer on a rigid substrate, the thickness of the
        layer. If this is set the layered form of the BEM equations will be
        used by default
    density : float optional (None)
        The density of the material
    behaviour : str {'perfet', 'table'}
        The type of behaviour
    params : tuple
        The paramters needed for the desired behaviuor
    
    Attributes
    ----------
    E
    v
    K
    Lam
    G
    M
    layer_thickness
    density
    
    Methods
    -------
    speed_of_sound
    
    See Also
    --------
    
    Notes
    -----
    
    Keys refer to:
        - E   - Young's modulus
        - v   - Poission's ratio
        - K   - Bulk Modulus
        - Lam - Lame's first parameter
        - G   - Shear modulus
        - M   - P wave modulus 
    
    Examples
    --------
    >>> # Make a material model for elastic steel on a rigid substarte with a 
    >>> # thickness of 1mm
    >>> steel=Elastic({'E':200e9, 'v':0.3}, layer_thickness=0.001, 
    >>>               density=7850)
    >>> # Find it's pwave modulus:
    >>> pwm=steel.M
    >>> # Find the speeds of sound:
    >>> sos=steel.speed_of_sound()
    """
    _plastic_method=None
    
    def __init__(self, properties:dict, layer_thickness:float=float('inf'), 
                 density=None, behaviour: str='perfect', **params):
        if len(properties)==2:
            self._properties=_get_properties(properties)
        elif len(properties)==1:
            kv=list(properties.items())[0]
            self._set_props(*kv)
        elif len(properties)!=0:
            raise ValueError("Too many properties suplied, must be 1 or 2")
        
        self.density=density
        self.layer_thickness=layer_thickness
        if behaviour=='perfect':
            self._yield_stress=params.pop('yield stress')
            self._plastic_method=self._perfect
        elif behaviour=='table':
            self._plastic_method=self._perfect
        
        if params:
            msg=("Unrecognised key: " +' '.join(params.keys()) +" in params")
            raise ValueError(msg)
        
            
    def _perfect_mm(self, plastic_strain):
        return self._yield_stress
        
    def plastic_stress(self, plastic_strain):
        return self._plastic_mathod(plastic_strain)
        

class ViscoElastic(Elastic):
    """ A class for describing visco elastic materials or layered visco elastic
    materials
    
    """
class _Bunch(object):
  """A class to tidy dict indexing: (my_dict['e']) into attribute access:
  my_class.e, makes equations shorter and more understandable    
  """
    
  def __init__(self, adict):
    self.__dict__.update(adict)
    
def _get_properties(set_props: dict):
    """Get all elastic properties from any pair
    
    Parameters
    ----------
    set_props : dict
        dict of properties must have exactly 2 members valid keys are: 'K', 
        'E', 'v', 'Lam', 'M', 'G'
    
    Returns
    -------
    out : dict
        dict of all material properties keys are: 'K', 'E', 'v', 'Lam', 'M', 'G'
    
    Notes
    -----
    
    Keys refer to:
        - E - Young's modulus
        - v - Poission's ratio
        - K - Bulk Modulus
        - Lam - Lame's first parameter
        - G - Shear modulus
        - M - P wave modulus 
    
    """
    if len(set_props)!=2:
        raise ValueError("Exactly 2 properties must be set,"
                         " {} found".format(len(set_props))) 
    
    valid_keys=['K', 'E', 'v', 'G', 'Lam', 'M']
    
    set_params=[key for key in list(set_props.keys()) if key in valid_keys]
    
    if len(set_params)!=2:
        msg=("Invalid keys in set_props keys found are: " +
             "{}".format(set_props.keys()) +
             ". Valid keys are: "+" ".join(valid_keys))
        raise ValueError(msg)
    
    out=set_props.copy()
    
    set_params=list(set_props.keys())
    set_params.sort()
    #p is properties this saves a lot of space
    p=_Bunch(set_props)
    
    if set_params[0]=='E':
        if set_params[1]=='G':
            out['K']=p.E*p.G/(3*(3*p.G-p.E))
            out['Lam']=p.G*(p.E-2*p.G)/(3*p.G-p.E)
            out['M']=p.G*(4*p.G-p.E)/(3*p.G-p.E)
            out['v']=p.E/(2*p.G)-1
        elif set_params[1]=='K':
            out['G']=3*p.K*p.E/(9*p.K-p.E)
            out['Lam']=3*p.K*(3*p.K-p.E)/(9*p.K-p.E)
            out['M']=3*p.K*(3*p.K+p.E)/(9*p.K-p.E)
            out['v']=(3*p.K-p.E)/(6*p.K)
        elif set_params[1]=='Lam':
            R=np.sqrt(p.E**2+9*p.Lam**2+2*p.E*p.Lam)
            out['G']=(p.E-3*p.Lam+R)/4
            out['K']=(p.E+3*p.Lam+R)/6
            out['M']=(p.E-p.Lam+R)/2
            out['v']=2*p.Lam/(p.E+p.Lam+R)
        elif set_params[1]=='M':
            S=np.sqrt(p.E**2+9*p.M**2-10*p.E*p.M)
            out['G']=(3*p.M+p.E-S)/8
            out['K']=(3*p.M-p.E+S)/6
            out['Lam']=(p.M-p.E+S)/4
            out['v']=(p.E-p.M+S)/(4*p.M)
        else: # set_params[1]=='v'
            out['G']=p.E/(2*(1+p.v))
            out['K']=p.E/(3*(1-2*p.v))
            out['Lam']=p.E*p.v/((1+p.v)*(1-2*p.v))
            out['M']=p.E*(1-p.v)/((1+p.v)*(1-2*p.v))
    elif set_params[0]=='G':
        if set_params[1]=='K':
            out['E']=9*p.K*p.G/(3*p.K+p.G)
            out['Lam']=p.K-2*p.G/3
            out['M']=p.K+4*p.G/3
            out['v']=(3*p.K-2*p.G)/(2*(3*p.K+p.G))
        elif set_params[1]=='Lam':
            out['E']=p.G*(3*p.Lam+2*p.G)/(p.Lam+p.G)
            out['K']=p.Lam+2*p.G/3
            out['M']=p.Lam+2*p.G
            out['v']=p.Lam/(2*(p.Lam+p.G))
        elif set_params[1]=='M':
            out['E']=p.G*(3*p.M-4*p.G)/(p.M-p.G)
            out['K']=p.M-4*p.G/3
            out['Lam']=p.M-2*p.G
            out['v']=(p.M-2*p.G)/(2*p.M-2*p.G)
        else: # set_params[1]=='v'
            out['E']=2*p.G*(1+p.v)
            out['K']=2*p.G*(1+p.v)/(3*(1-2*p.v))
            out['Lam']=2*p.G*p.v/(1-2*p.v)
            out['M']=2*p.G*(1-p.v)/(1-2*p.v)
    elif set_params[0]=='K':
        if set_params[1]=='Lam':
            out['E']=9*p.K*(p.K-p.Lam)/(3*p.K-p.Lam)
            out['G']=3*(p.K-p.Lam)/2
            out['M']=3*p.K-2*p.Lam
            out['v']=p.Lam/(3*p.K-p.Lam)
        elif set_params[1]=='M':
            out['E']=9*p.K*(p.M-p.K)/(3*p.K+p.M)
            out['G']=3*(p.M-p.K)/4
            out['Lam']=(3*p.K-p.M)/2
            out['v']=(3*p.K-p.M)/(3*p.K+p.M)
        else: # set_params[1]=='v'
            out['E']=3*p.K*(1-2*p.v)
            out['G']=(3*p.K*(1-2*p.v))/(2*(1+p.v))
            out['Lam']=3*p.K*p.v/(1+p.v)
            out['M']=3*p.K*(1-p.v)/(1+p.v)
    elif set_params[0]=='Lam':
        if set_params[1]=='M':
            out['E']=(p.M-p.Lam)*(p.M+2*p.Lam)/(p.M+p.Lam)
            out['G']=(p.M-p.Lam)/2
            out['K']=(p.M+2*p.Lam)/3
            out['v']=p.Lam/(p.M+p.Lam)
        else:
            out['E']=p.Lam*(1+p.v)*(1-2*p.v)/p.v
            out['G']=p.Lam(1-2*p.v)/(2*p.v)
            out['K']=p.Lam*(1+p.v)/(3*p.v)
            out['M']=p.Lam*(1-p.v)/p.v
    else:
        out['E']=p.M*(1+p.v)*(1-2*p.v)/(1-p.v)
        out['G']=p.M*(1-2*p.v)/(2*(1-p.v))
        out['K']=p.M*(1+p.v)/(3*(1-p.v))
        out['Lam']=p.M*p.v/(1-p.v)
    
    return out