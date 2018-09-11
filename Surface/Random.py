"""
Classes for generating random surfaces based on filtering of random signals:
    ===========================================================================
    ===========================================================================
    Each class inherits functionallity from the Surface but changes the 
    __init__ and descretise functions
    ===========================================================================
    ===========================================================================
    NoiseBasedSurface:
        Generate and filter a noisy surface, several methods for this and the 
        make_like method that makes a surface 'like' the input surface
        
    ===========================================================================
    ===========================================================================

#TODO:
        Add comment blocks to each class with examples of use
        Add other surface generation methods
        add citation to relevent method
        add make like method,
        
"""

from . import Surface
import warnings
import numpy as np
from math import ceil, floor

__all__=['GausianNoiseSurface', 'make_like']


class GausianNoiseSurface(Surface): #done
    is_descrete=False
    need_to_filter=False
    surface_type='gausianNoise'
    def __init__(self, mu=0, sigma=1, dimentions=2, **kwargs):
        
        self.init_checks(kwargs)
        
        self.dimentions=dimentions
        self.gn_mu=mu
        self.gn_sigma=sigma
        
    def descretise(self, spacing=False):
        if spacing:    
            self.grid_size=spacing
        self.descretise_checks()
        nPts=self.pts_each_direction
        if self.dimentions==1:
            profile=np.random.randn(nPts[0],1)
        elif self.dimentions==2:
            profile=np.random.randn(nPts[0],nPts[1])
        self.profile=profile*self.gn_sigma+self.gn_mu
        self.is_descrete=True
    
    def specify_ACF_IFFT_FIR(self, ACF_or_type, *args):
        size=self.global_size
        spacing=self.grid_size
        nPts=self.pts_each_direction
        if type(ACF_or_type) is str:
            k=np.arange(-nPts[0]/2,nPts[0]/2)
            l=np.arange(-nPts[1]/2,nPts[1]/2)
            [K,L]=np.meshgrid(k,l)
            
            if ACF_or_type=='exp':
                sigma=args[0]
                beta_x=args[1]/spacing
                beta_y=args[2]/spacing
                ACF=sigma**2*np.exp(-2.3*np.sqrt((K/beta_x)**2+(L/beta_y)**2))
                self.surf(ACF)
            else:
                ValueError("ACF_or_type must be array like or valid type")
        else:
            ACF=np.asarray(ACF_or_type)
            if not ACF.shape==size:
                #pad ACF with 0s equally on both sides
                size_difference=[]
                is_neg=[]
                for i in range(len(size)):
                    size_difference.append(size[i],ACF.shape[i])
                    is_neg.append(size_difference[0]<0)
                if any(is_neg):
                    ValueError("ACF size should be smaller than the profile"
                               " size")
                np.pad(ACF,[ceil(size_difference[0]),floor(size_difference[0]), 
                          ceil(size_difference[0]), floor(size_difference[0])],
                       'constant')
        
        filter_tf=np.sqrt(np.fft.fft2(ACF))
        self.profile=np.abs(np.fft.ifft2((np.fft.fft2(self.profile)*filter_tf)))
        
def make_like(surface, copy=True):
    #pass a surface to init then use __call__ to generate like surfaces by the given method
    pass
