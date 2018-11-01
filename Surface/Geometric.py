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

from . import Surface
import warnings
import numpy as np

class FlatSurface(Surface): 
""""
    simple flat surface can be angled in any direction by changing slope
"""
    is_descrete=False
    surface_type='flat'
    
    def __init__(self, slope=0, dimentions=2, **kwargs):
        
        self.init_checks(kwargs)
        self.dimentions=dimentions
        if type(slope) is list:
            self.slope=slope
        elif type(slope) is int or type(slope) is float:
            self.slope=[slope,0]
            if self.dimentions==2:
                warnings.warn("Assumed 0 slope in Y direction for"
                              " analytical flat surface")
    def descretise(self, spacing, centre=[0,0]):
        self.grid_size=spacing
        self.descretise_checks()
        x=np.linspace(-0.5*self.global_size[0],
                    0.5*self.global_size[0],self.pts_each_direction[0])
        if self.dimentions==1:
            self.profile=x*self.slope[0]
        else:
            y=np.linspace(-0.5*self.global_size[1],
                    0.5*self.global_size[1],self.pts_each_direction[1])
            (X,Y)=np.meshgrid(x,y)
            self.profile=X*self.slope[0]+Y*self.slope[1]
        self.is_descrete=True

class RoundSurface(Surface):
    is_descrete=False
    surface_type='round'
    def __init__(self, radius, dimentions=2, **kwargs):
        
        self.init_checks(kwargs)
        
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
            
    def descretise(self, spacing=False, centre=[0,0]):
        if spacing:
            self.grid_size=spacing
        self.descretise_checks()
        x=np.linspace(-0.5*self.global_size[0],
                    0.5*self.global_size[0],self.pts_each_direction[0])
        if self.dimentions==1:
            self.profile=((1-(x/self.radius[0])**2)**0.5)*self.radius[-1]
        else:
            y=np.linspace(-0.5*self.global_size[1],
                    0.5*self.global_size[1],self.pts_each_direction[1])
            (X,Y)=np.meshgrid(x,y)
            self.profile=((1-(X/self.radius[0])**2-
                          (Y/self.radius[1])**2)**0.5)*self.radius[-1]
        np.nan_to_num(self.profile, False)
        self.is_descrete=True
        
class PyramidSurface(Surface):
    is_descrete=False
    surface_type='pyramid'
    
    def __init__(self, lengths, dimentions=2, **kwargs):
        
        self.init_checks(kwargs)
        
        self.dimentions=dimentions
        if type(lengths) is list:
            if len(lengths)==(self.dimentions+1):
                self.lengths=lengths
            else:
                msg=('Radius must be either scalar or list of radii equal in '
                'length to number of dinmetions of the surface +1')
                raise ValueError(msg)   
        elif type(lengths) is int or type(lengths) is float:
            self.lengths=[lengths]*(self.dimentions+1)
            
    def descretise(self, spacing):
        #TODO check that ther is no gap around the edge, if so scale so there is not 
        #x/xl+y/yl+z/zl=1
        #(1-x/xl-y/yl)*zl=z
        self.grid_size=spacing
        self.descretise_checks()
        x=np.arange(-0.5*self.global_size[0],
                    0.5*self.global_size[0],self.grid_size)
        if self.dimentions==1:
            self.profile=(1-x/self.lengths[0])*self.lengths[-1]
        else:
            y=np.arange(-0.5*self.global_size[1],
                        0.5*self.global_size[1],self.grid_size)
            (X,Y)=np.meshgrid(x,y)
            self.profile=(1-X/self.lengths[0]-
                          Y/self.lengths[1])*self.lengths[-1]
        self.is_descrete=True
