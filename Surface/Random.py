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
        johnson trancslator
        make work with 'generate' keyword 
        add make like method,
        
"""

from . import Surface
import warnings
import numpy as np
from math import ceil, floor

__all__=['RandomSurface']


class RandomSurface(Surface):
    
## surfaces based on transformations of random sequences

    is_descrete=False
    surface_type='Random'

    def __init__(self, dimentions=2, **kwargs):
        
        self.init_checks(kwargs)
        
        self.dimentions=dimentions
        
    def fill_gaussian(self, spacing=None):
        if spacing:    
            self.grid_size=spacing
        self.descretise_checks()
        nPts=self.pts_each_direction
        if self.dimentions==1:
            profile=np.random.randn(nPts[0],1)
        elif self.dimentions==2:
            profile=np.random.randn(nPts[0],nPts[1])
        self.is_descrete=True
        self.profile=profile
    
    def johnson_translation(self, params, **kwargs):
        # kwargs can be set to surface if a surface is being used as input, should fit johnson system first then transform the gausian array, filling first if necessary
        # if just params are given surface should be filled and translated
        pass 

    def linear_transform(self, target_ACF, itteration_procedure='CGD'):
        valid_itt=['CGD', 'newton']
        if not itteration_procedure in valid_itt:
            str=("Invalid itteration procedure, valid options are:\n".join(valid_itt))
            ValueError(str)
        return ################## and here
        if type(target_ACF) is ACF:
            self.target_ACF=target_ACF
        

    def newton_itt(self,previous_itteration=False):
    #taken from patir 1977 appendix 1
        if not previous_itteration:
            c=self.target_ACF######### continue from here


    def CGD_itt(self,previous_itteration):
        pass

    def beta_functions(self, target_ACF):
        pass

    def specify_ACF_IFFT_FIR(self, ACF_or_type, *args):
        size=self.global_size
        spacing=self.grid_size
        self.descretise_checks()
        nPts=self.pts_each_direction
        self.fill_gaussian()
        if type(ACF_or_type) is str:
            k=np.arange(-nPts[0]/2,nPts[0]/2)
            l=np.arange(-nPts[1]/2,nPts[1]/2)
            [K,L]=np.meshgrid(k,l)
            
            if ACF_or_type=='exp':
                sigma=args[0]
                beta_x=args[1]/spacing
                beta_y=args[2]/spacing
                ACF=sigma**2*np.exp(-2.3*np.sqrt((K/beta_x)**2+(L/beta_y)**2))
            else:
                ValueError("ACF_or_type must be array like or valid type")
        else:
            ACF=np.asarray(ACF_or_type)
            if not ACF.shape==size:
                #pad ACF with 0s equally on both sides
                size_difference=[]
                is_neg=[]
                for i in range(len(size)):
                    size_difference.append(size[i]-ACF.shape[i])
                    is_neg.append(size_difference[0]<0)
                if any(is_neg):
                    ValueError("ACF size should be smaller than the profile"
                               " size")
                ACF=np.pad(ACF,((ceil(size_difference[0]/2),floor(size_difference[0]/2)), 
                          (ceil(size_difference[1]/2), floor(size_difference[1]/2))),
                       'constant')
        
        filter_tf=np.sqrt(np.fft.fft2(ACF))
        
        self.profile=np.abs(np.fft.ifft2((np.fft.fft2(self.profile)*filter_tf)))