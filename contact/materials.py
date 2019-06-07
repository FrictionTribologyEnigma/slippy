import numpy as np
from scipy.interpolate import interp1d
from slippy.contact._material_utils import _get_properties

__all__=["Elastic", "ElasticPlastic", "ViscoElastic", "material"]

def material(material_type:str, properties:dict, model:str=None):
    """Create a new material object
    
    Parameters
    ----------
    material_type : str {elastic, visco_elastic, elastic_plastic}
        The name of the type of material you want to set, not case 
        sensitive
    properties : dict
        A dictionary containing the properties of the material
    model : str optional
        The material model to be used, only required for plastic and visco
        elastic materials
        
    Returns
    -------
    A material object with the required properties
    
    See Also
    --------
    Elastic
    ElasticPlastic
    ViscoElastic
    
    Notes
    -----
    For a detailed description of the models avalible and the properties 
    required for each model see the class descriptions given above
    
    Examples
    --------
    >>> #TODO
    """
    pass
    

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
    
    def __init__(self, properties : dict):
        """
        
        """
        if len(properties)==2:
            self._properties=_get_properties(properties)
        elif len(properties)==1:
            kv=list(properties.items())[0]
            self._set_props(*kv)
        elif len(properties)!=0:
            raise ValueError("Too many properties suplied, must be 1 or 2")
        
        self.density=properties['density']
    
    
    
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
    plastic_params : tuple optional (None)
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
    plastic_stress
    
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
                 density=None, behaviour: str='perfect', plastic_params=None):
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
            self._yield_stress=plastic_params.pop('yield_stress')
            self._plastic_method=self._perfect
        elif behaviour=='table':
            self._plastic_method=interp1d(plastic_params.pop('stress'), 
                                          plastic_params.pop('plastic strain'))
        elif behaviour=='exponent':
            self._plastic_method=self._exponent_mm
            self._exponent=plastic_params.pop('exponent')
            self._strength_coefficent=plastic_params.pop('strength coefficent')
        
        if plastic_params:
            keys=' '.join(list(plastic_params.keys()))
            msg=f"Unrecognised keys : {keys} in plastic_params"
            raise ValueError(msg)
        
            
    def _perfect_mm(self, plastic_strain):
        return self._yield_stress
        
    def _exponent_mm(self, plastic_strain):
        return self._strength_coefficent*plastic_strain**self._exponent
        
    def plastic_stress(self, plastic_strain):
        return self._plastic_mathod(plastic_strain)
        

class ViscoElastic(_Material):
    """ A class for describing visco elastic materials or layered visco elastic
    materials
    
    """
