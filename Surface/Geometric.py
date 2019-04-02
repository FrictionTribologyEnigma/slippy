"""
Classes for generating geometric surfaces:
    ===========================================================================
    ===========================================================================
    Each class inherits functionallity from the Surface but changes the 
    __init__ and descretise functions
    ===========================================================================
    ===========================================================================
    
    FlatSurface:
        For generating flat surfaces, slopes are allowed in both directions
    PyramidSurface:
        For generating square pyramid surfaces like indenters
    RoundSurface:
        For generating ball type surfaces different radii allowed in each 
        direction
        
    ===========================================================================
    ===========================================================================

#TODO:
        Add comment blocks to each class with examples of use
        make so it can work with the 'generate' keyword arg
"""

__all__=['FlatSurface', 'RoundSurface', 'PyramidSurface']

from .Surface_class import Surface
import warnings
import numpy as np

class FlatSurface(Surface): 
    """ Flat surface can be angled in any direction by changing slope
    
    Parameters
    ----------
    Slope : list
        The gradient of the surface in the x and y directions
    
    Attributes
    ----------
    
    Methods
    -------
    height
    
    See Also
    --------
    Surface
    
    Notes
    -----
    This is a subclass of Surface all functionality and attributes are available
    within this class. 
    
    All keyword arguments allowed for Surface are also 
    allowed on instantiation of this class apart from the profile key word.
    
    """
    surface_type='flat'
    analytic=True
    
    def __init__(self, slope=[0,0], **kwargs):
        
        self._init_checks(kwargs)
        if type(slope) is list:
            self._slope=slope
        elif type(slope) is int or type(slope) is float:
            self._slope=[slope,0]
            if self.dimentions==2:
                warnings.warn("Assumed 0 slope in Y direction for"
                              " analytical flat surface")
        
    def descretise(self, grid_spacing=None, centre=[0,0]):
        
        if grid_spacing:
            self.grid_spacing=grid_spacing
        self._descretise_checks()
        grid_spacing=self._grid_spacing
        x=np.linspace(-0.5*self.extent[0],
                    0.5*self.extent[0],self.shape[0])
        if self.dimentions==1:
            self.profile=x*self._slope[0]
        else:
            y=np.linspace(-0.5*self.extent[1],
                    0.5*self.extent[1],self.shape[1])
            (X,Y)=np.meshgrid(x,y)
            self.profile=X*self._slope[0]+Y*self._slope[1]
        self.is_descrete=True
    
    def height(self, X, Y):
        """Analytically determined height of the surface at specified points
        
        Parameters
        ----------
        X, Y : array-like
            Arrays of X and Y points, must be the same shape
        
        Returns
        -------
        Array of surface heights, with the same shape as the input arrays
        
        Notes
        -----
        This is an alternative to descretising the surface which may be more 
        appropriate for some applications
        
        Examples
        --------
        >>> import numpy as np
        >>> my_surface=FlatSurface(slope=[1,1])
        >>> x, y = np.arange(10), np.arange(10)
        >>> X, Y = np.meshgrid(x,y)
        >>> Z=my_surface.height(X, Y)
        """
        return X*self._slope[0]+Y*self._slope[1]
        
class RoundSurface(Surface):
    """ Round surfaces with any radii
    
    Parameters
    ----------
    radius : list or float
        The radius of the surface in the X Y and Z directions, or in all 
        directions if a float is given
    
    Attributes
    ----------
    
    Methods
    -------
    height
    
    See Also
    --------
    Surface
    
    Notes
    -----
    This is a subclass of Surface all functionality and attributes are available
    within this class. 
    
    All keyword arguments allowed for Surface are also 
    allowed on instantiation of this class apart from the profile key word.
    
    """
    surface_type='round'
    analytic=True
    
    def __init__(self, radius=[10,10,10], dimentions=2, **kwargs):
        
        self._init_checks(kwargs)
        
        self.dimentions=dimentions
        if type(radius) is list:
            if len(radius)==(self.dimentions+1):
                self.radius=radius
            else:
                msg=('Radius must be either scalar or list of radii equal in '
                'length to number of dinmetions of the surface +1')
                raise ValueError(msg)   
        elif type(radius) is int or type(radius) is float:
            self.radius=[radius]*(self.dimentions+1)
            
    def descretise(self, grid_spacing=False, centre=[0,0]):
        if grid_spacing:
            self.grid_spacing=grid_spacing
        self._descretise_checks()
        x=np.linspace(-0.5*self.extent[0],
                    0.5*self.extent[0],self.shape[0])
        if self.dimentions==1:
            self.profile=((1-(x/self.radius[0])**2)**0.5)*self.radius[-1]
        else:
            y=np.linspace(-0.5*self.extent[1],
                    0.5*self.extent[1],self.shape[1])
            (X,Y)=np.meshgrid(x,y)
            self.profile=((1-(X/self.radius[0])**2-
                          (Y/self.radius[1])**2)**0.5)*self.radius[-1]
        np.nan_to_num(self.profile, False)
        self.is_descrete=True
    
    def height(self,X,Y):
        """Analytically determined height of the surface at specified points
        
        Parameters
        ----------
        X, Y : array-like
            Arrays of X and Y points, must be the same shape
        
        Returns
        -------
        Array of surface heights, with the same shape as the input arrays
        
        Notes
        -----
        This is an alternative to descretising the surface which may be more 
        appropriate for some applications
        
        Examples
        --------
        >>> import numpy as np
        >>> my_surface=RoundSurface(radius=[1,1,1])
        >>> x, y = np.arange(10), np.arange(10)
        >>> X, Y = np.meshgrid(x,y)
        >>> Z=my_surface.height(X, Y)
        """
        Z=((1-(X/self.radius[0])**2-
            (Y/self.radius[1])**2)**0.5)*self.radius[-1]
        return np.nan_to_num(Z, False)
        
        
class PyramidSurface(Surface):
    """ Pyramid surface with any slopes
    
    Keyword parameters
    ------------------
    lengths : list or float
        The characteristic lengths of the pyramid in each direction, if a 
        scalar is given the results is a square based pyramid with 45 degre 
        sides
    
    Methods
    -------
    height
    
    See Also
    --------
    Surface
    
    Notes
    -----
    This is a subclass of Surface all functionality and attributes are available
    within this class. 
    
    All keyword arguments allowed for Surface are also 
    allowed on instantiation of this class apart from the profile key word.
    
    """
    surface_type='pyramid'
    
    def __init__(self, lengths, dimentions=2, **kwargs):
        
        self._init_checks(kwargs)
        
        self.dimentions=dimentions
        if type(lengths) is list:
            if len(lengths)==(self.dimentions+1):
                self.lengths=lengths
            else:
                msg=('Lengths must be either scalar or list of Lengths equal'
                ' in length to number of dinmetions of the surface +1')
                raise ValueError(msg)   
        elif type(lengths) is int or type(lengths) is float:
            self.lengths=[lengths]*(self.dimentions+1)
            
    def descretise(self, grid_spacing=None):
        #TODO check that ther is no gap around the edge, if so scale so there is not 
        #x/xl+y/yl+z/zl=1
        #(1-x/xl-y/yl)*zl=z
        if grid_spacing:
            self.grid_spacing=grid_spacing
        self._descretise_checks()
        x=np.abs(np.arange(-0.5*self.extent[0],
                    0.5*self.extent[0],self.grid_spacing))
        if self.dimentions==1:
            self.profile=(1-x/self.lengths[0])*self.lengths[-1]
        else:
            y=np.abs(np.arange(-0.5*self.extent[1],
                        0.5*self.extent[1],self.grid_spacing))
            (X,Y)=np.meshgrid(x,y)
            self.profile=(1-X/self.lengths[0]-
                          Y/self.lengths[1])*self.lengths[-1]
        self.is_descrete=True
    
    def height(self,X,Y):
        """Analytically determined height of the surface at specified points
        
        Parameters
        ----------
        X, Y : array-like
            Arrays of X and Y points, must be the same shape
        
        Returns
        -------
        Array of surface heights, with the same shape as the input arrays
        
        Notes
        -----
        This is an alternative to descretising the surface which may be more 
        appropriate for some applications
        
        Examples
        --------
        >>> import numpy as np
        >>> my_surface=PyramidSurface(lengths=[20,20,20])
        >>> x, y = np.arange(10), np.arange(10)
        >>> X, Y = np.meshgrid(x,y)
        >>> Z=my_surface.height(X, Y)
        """
        return (1-X/self.lengths[0]-Y/self.lengths[1])*self.lengths[-1]