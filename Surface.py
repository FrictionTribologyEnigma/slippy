import numpy as np
import warnings

class Surface():
    """ the basic surface class contains methods for setting properties, 
    examining measures of roughness and descriptions of surfaces, plotting,
    fixing and editing surfaces.
    
    Adding surfaces together produces a surface with the same properties as the
    original surfaces with a profile that is the sum of the original profiles.
    
    Multiplying 1D surfaces produces a 2D surface with a profile of the 
    summation of surface 1 stretched in the x direction and surface 2 
    stretched in the y direction.
    """
    # The surface class for descrete surfaces
    global_size=[1,1]
    dimentions=0
    is_descrete=True
    profile=np.array([])
    def __init__(self,*args,**kwargs):
        # initialisation surface
        self.init_checks(kwargs)
        # check for file
        
    def init_checks(self, kwargs):
        # add anything you want to run for all surface types here
        a=1
    
    #def set_size(self, grid_size=0, global_size=0):
        
        
    def plot(self, plot_type='surface'):
        # plots the surfae options to use any of the plotting methods including 
        # PSD and FFT methods 
        if plot_type=='surface' or plot_type=='scatter' or plot_type=='wf':
            x=np.arange(-0.5*self.global_size[0],
                    0.5*self.global_size[0],self.grid_size)
            y=np.arange(-0.5*self.global_size[1],
                        0.5*self.global_size[1],self.grid_size)
            (X,Y)=np.meshgrid(x,y)
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X,Y,self.profile)
            
    def fft_surface(self):
        try:
            if self.dimentions==1:
                transform=np.fft.fft(self.profile)
            else:
                transform=np.fft.fft2(self.profile)
        except AttributeError:
            raise AttributeError('Surface must have a defined profile for fft'
                                 ' to be used')
        return transform
    
    def plot_FFT_surface(self):#TODO fill in this
        self.global_size
        self.grid_size
        

#    def psd_surface(self):
#        
#    def plot_PSD_surface(self):
#        
#    def fill_holes(self):
#        
#    def read_from_file(self, filename):
#        
#    def check_surface(self):
#        
#    def roughness(self, type):
#        

#
#    def stretch(self, ratio):
#    def re_sample(self, new_size):
#    def __add__(self, other):
#    def __subtract__(self, other):
#    def __times__(self, other): 

""" the follwing class definitions are all sub classes of surfaces for useful 
analytically described surfaces, each class overides the __init__ method and 
provides a descretise_surface method the is_descrete flag should also be set to 
false until the descretise method has been run. init_checks should be
called first in the init method to check for dimentions and material property 
assignments"""
    
class FlatSurface(Surface): #done
    is_descrete=False
    surface_type='plane'
    
    def __init__(self, slope, dimentions=2, **kwargs):
        
        self.init_checks(kwargs)
        
        self.dimentions=dimentions
        if type(slope) is list:
            self.slope=slope
        elif type(slope) is int or type(slope) is float:
            self.slope=[slope,0]
            if self.dimentions==2:
                warnings.warn("Assumed 0 slope in Y direction for"
                              " analytical flat surface")
    def descretise(self, spacing, centre):
        self.grid_size=spacing
        if self.dimentions==0:
            raise ValueError(
                    'Cannot descritise 0 dimentional surface')
        x=np.arange(-0.5*self.global_size[0],
                    0.5*self.global_size[0],self.grid_size)
        if self.dimentions==1:
            self.profile=x*self.slope[0]
        else:
            y=np.arange(-0.5*self.global_size[1],
                        0.5*self.global_size[1],self.grid_size)
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
            
    def descretise(self, spacing, centre):
        #1d
        #(x/xr)^2+(z/zr)^2=1
        #(1-(x/rx)^2)^0.5*rz
        #2d
        #(x/rx)^2+(y/ry)^2+(z/rz)^2=1
        #1-(x/rx)^2-(y/ry)^2=(z/rz)^2
        #(1--(y/ry)^2)^0.5*rz
        self.grid_size=spacing
        x=np.arange(-0.5*self.global_size[0],
                    0.5*self.global_size[0],self.grid_size)
        if self.dimentions==1:
            self.profile=((1-(x/self.radius[0])**2)**0.5)*self.radius[-1]
        else:
            y=np.arange(-0.5*self.global_size[1],
                        0.5*self.global_size[1],self.grid_size)
            (X,Y)=np.meshgrid(x,y)
            self.profile=self.profile=((1-(X/self.radius[0])**2-
                                      (Y/self.radius[1])**2)**0.5)*self.radius[-1]
        #TODO make nan values 0 (results in flat surrounding ball)
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
        
class GausianNoiseSurface(Surface): #done
    is_descrete=False
    surface_type='gausianNoise'
    def __init__(self, mu=0, sigma=1, dimentions=2, **kwargs):
        
        self.init_checks(kwargs)
        
        self.dimentions=dimentions
        self.gn_mu=mu
        self.gn_sigma=sigma
        
    def descretise(self, spacing):
        self.grid_size=spacing
        nPts=[round(length/spacing) for length in self.global_size]
        if self.dimentions==1:
            profile=np.random.randn(nPts[0])
        elif self.dimentions==2:
            profile=np.random.randn(nPts[0],nPts[1])
        self.profile=profile*self.gn_sigma+self.gn_mu
        self.is_descrete=True
        
class FlatNoiseSurface(Surface): #done
    is_descrete=False
    surface_type='flatNoise'
    def __init__(self, noise_range=[0,1], dimentions=2, **kwargs):
        
        self.init_checks(kwargs)
        
        self.dimentions=dimentions
        if type(noise_range) is list and len(noise_range)==2:
            self.fn_range=noise_range
        elif type(noise_range) is int or type(noise_range) is float:
            self.fn_range=[0,noise_range]
        else:
            msg='Noise range must be a int float or 2 element list'
            raise ValueError(msg)
        
    
    def descretise(self, spacing):
        self.grid_size=spacing
        nPts=[round(length/spacing) for length in self.global_size]
        
        if self.dimentions==1:
            profile=np.random.rand(nPts[0])
        elif self.dimentions==2:
            profile=np.random.rand(nPts[0],nPts[1])
        self.profile=profile*(self.fn_range[1]-self.fn_range[0])+self.fn_range[0]
        self.is_descrete=True

class DiscreteFrequencySurface(Surface):#TODO make work better with 2d surface, read into it first
    is_descrete=False
    surface_type='discreteFreq'
    
    def __init__(self, frequencies, amptitudes=[1], phases_rads=[0], dimentions=2, **kwargs):
        
        self.init_checks(kwargs)
        
        self.dimentions=dimentions
        if type(frequencies) is list or type(frequencies) is np.ndarray:
            self.frequencies=frequencies
        else:
            raise ValueError('Frequencies, amptitudes and phases must be equal'
                             'length lists or np.arrays')
        is_complex=[type(amp) is complex for amp in amptitudes]
        if any(is_complex):
            if not len(frequencies)==len(amptitudes):
                raise ValueError('Frequencies, amptitudes and phases must be'
                                 ' equal length lists or np.arrays')
            else:
                self.amptitudes=amptitudes
        else:
            if not len(frequencies)==len(amptitudes)==len(phases_rads):
                raise ValueError('Frequencies, amptitudes and phases must be'
                                 ' equal length lists or np.arrays')
            else:
                cplx_amps=[]
                for idx in range(len(amptitudes)):
                    cplx_amps.append(amptitudes[idx]*
                                     np.exp(1j*phases_rads[idx]))
                self.amptitudes=cplx_amps
            
    def descretise(self, spacing):
        self.grid_size=spacing
        #TODO write this section
        x=np.arange(-0.5*self.global_size[0],
                    0.5*self.global_size[0],self.grid_size)
        if self.dimentions==1:
            profile=np.zeros_like(x)
            for idx in range(len(self.frequency)):
                profile+=np.real(self.amptitudes[idx]*
                                 np.exp(-1j*self.frequency[idx]*x))
            self.profile=profile
        elif self.dimentions==2:
            y=np.arange(-0.5*self.global_size[1],
                        0.5*self.global_size[1],self.grid_size)
            (X,Y)=np.meshgrid(x,y)
            for idx in range(len(self.frequency)):
                profile+=np.real(self.amptitudes[idx]*
                                 np.exp(-1j*self.frequency[idx]*X)+
                                 self.amptitudes[idx]*
                                 np.exp(-1j*self.frequency[idx]*Y))
            self.profile=profile
            
        else:
            raise ValueError('Cannont descretise %d dimentional surface' % self.dimentions)
        self.is_descrete=True
    
class ContinuousFrequencySurface(DiscreteFrequencySurface): #make work better with 2d surfaces 
    is_descrete=False
    surface_type='continuousFreq'
        #TODO write this function
        
    def descretise(self, spacing):
        self.grid_size=spacing
        x=np.arange(-0.5*self.global_size[0],
                    0.5*self.global_size[0],self.grid_size)
        #TODO write this function probably use an inverse FFT need to read up 
        # for 2d surfacesfor now just used np.eye * amps
        
if __name__ == "__main__":
    A=Surface(material_type="elastic", properties=(200E9, 0.3))