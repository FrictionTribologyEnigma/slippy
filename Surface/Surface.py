import numpy as np
import warnings
import scipy.special

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
    grid_size=0.01
    dimentions=0
    is_descrete=True
    profile=np.array([])
    pts_each_direction=[]
    
    def __init__(self,*args,**kwargs):
        # initialisation surface
        self.init_checks(kwargs)
        # check for file
        
    def init_checks(self, kwargs=False):
        # add anything you want to run for all surface types here
        a=1
    
    def descretise_checks(self):
        if self.is_descrete:
            msg='Surface is already discrete'
            raise ValueError(msg)
        if not self.dimentions or self.dimentions>2:
            msg='Number of dimensions should be 1 or 2'
            raise ValueError(msg)
        try: 
            spacing=self.grid_size
        except AttributeError:
            msg='A grid size must be provided before descretisation'
            raise AttributeError(msg)
        try:
            pts_each_direction=[np.floor(gs/spacing) for gs in 
                                    self.global_size]
            total_pts=1
            for pts in pts_each_direction:
                total_pts *= pts
            self.total_pts=total_pts
            self.pts_each_direction=pts_each_direction
        except AttributeError:
            msg='Global size must be set before descretisation'
            raise AttributeError(msg)
        if total_pts>10E7:
            warnings.warn('surface contains over 10^7 points')
        
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
            self.surf(self.profile,X,Y)
            
    def fft(self):
        try:
            if self.dimentions==1:
                transform=np.fft.fft(self.profile)
            else:
                transform=np.fft.fft2(self.profile)
        except AttributeError:
            raise AttributeError('Surface must have a defined profile for fft'
                                 ' to be used')
        return transform
    
    def plot_fft(self):#TODO fill in this
        self.global_size
        self.grid_size
    
#TODO make below pretty
    def acf(self, surf_in=False):
        import scipy.signal
        if not surf_in:
            surf_in=self.profile
        surf_in=np.asarray(surf_in)
        x=surf_in.shape[0]
        y=surf_in.shape[1]
        output=(scipy.signal.correlate(surf_in,surf_in,'same')/(x*y))
        return output
    
    def plot_acf(self):
        acf=self.acf()
        self.surf(acf)
#TODO all below
    def birmingham(self, parameter_name, curved_surface=False, *args): #TODO finish this!
# ================================================================= ============
#         Taken from: Metrology and Properties of Engineering Surfaces
#         Editors: Mainsah, E., Greenwood, james, Chetwynd, Derek (Eds.)
#         surface measurment and characterisation pg 23 onwards
# =============================================================================
        
        # recursive call to allow lists of parmeters to be retived at once
        
        if type(parameter_name) is list:
            out=[]
            for par_name in parameter_name:
                out.append(self.birmingham(parameter_name))
            return out
        
        # First remove the base line, linear fit for flat surfaces or 
        # biquadratic polynomial for curved surfaces 
        
        parameter_name=parameter_name.lower()
        N=self.global_size[0]/self.grid_size
        M=self.global_size[0]/self.grid_size
        x = np.arange(N)
        y = np.arange(M)
        X, Y = np.meshgrid(x, y, copy=False)
        X = X.flatten()
        Y = Y.flatten()
        if curved_surface:
            A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
            B = self.profile.flatten()
            coeff, r, rank, s = np.linalg.lstsq(A, B)
        else:
            A = np.array([X*0+1, X, Y]).T
            B = self.profile.flatten()
            coeff, r, rank, s = np.linalg.lstsq(A, B)
        fit=np.reshape(np.dot(A,coeff),[N,M])
        eta=self.profile-fit
        
        # return parameter of interst 
        
        if parameter_name=='sq': #root mean square
            out=np.sqrt(np.mean(eta**2))
        elif parameter_name=='sa': #mean amptitude
            out=np.sqrt(np.mean(eta))
        elif parameter_name=='ssk': #skewness
            sq=np.sqrt(np.mean(eta**2))
            out=np.mean(eta**3)/sq**3
        elif parameter_name=='sku': #kurtosis
            sq=np.sqrt(np.mean(eta**2))
            out=np.mean(eta**4)/sq**4
        elif parameter_name=='sz': #tenpoint height
            sorted_heights=np.sort(eta.flatten())
            out=(np.sum(np.abs(sorted_heights[0:5])+np.abs(sorted_heights[-6:-1])))/5
        elif parameter_name in ['','','','']:
            summits=self.find_summits()
            
    def find_summits(self, filter_cut_off):
        # summits are found by low pass filtering at the required scale then 
        # finding summits
        filtered_profile=self.low_pass_filter(True)
        
        
    def low_pass_filter(self, cut_off_freq, copy=False):
        import scipy.signal
        sz=self.pts_each_direction
        x=np.arange(1, sz[0]+1)
        y=np.arange(1, sz[1]+1)
        X,Y=np.meshgrid(x,y)
        D=np.sqrt(X**2+Y**2)
        ws=2*np.pi/self.grid_size
        wc=cut_off_freq*2*np.pi
        h=(wc/ws)*scipy.special.j1(2*np.pi*(wc/ws)*D)/D
        filtered_profile=scipy.signal.convolve2d(self.profile,h,'same','wrap')
        self.surf(filtered_profile)
        if copy:
            return filtered_profile
        else:
            self.profile=filtered_profile
            return
        
        ############# put low pass surface filter in here
        
        # step 1 remove a least squares linear fit or second order polynomal fit(if curved surface)
        #sq is root mean square
        #sa is average amptitude
        #sz is ten point height (sum|5 highest| +sum|5 lowest|)/5
        #ssk is skewness (3rd moment):
        #1/(MNsq^3)*sum(sum(eta^3))
        #Sku is kurtosis (4th moment)
        #1/(MNsq^4)*sum(sum(eta^4))
        
#    def plot_psd()
#    def psd(self)
#    def plotQQ(self, distribution)
#    def histogram(self):
#    def plot_histogram
#    def fill_holes(self):
#    def read_from_file(self, filename):
#    def match(self, filename, **kwargs)
#    def check_surface(self):
#    def stretch(self, ratio):
#    def re_sample(self, new_size):
#    def __add__(self, other):
#    def __subtract__(self, other):
#    def __times__(self, other): 
    def surf(self,Z,xmax=0,ymax=0):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x=range(Z.shape[0])
        y=range(Z.shape[1])
        if xmax:
            x=x/max(x)*xmax
        if ymax:
            y=y/max(y)*ymax
        (X,Y)=np.meshgrid(x,y)
        ax.plot_surface(X,Y,Z)
""" the follwing class definitions are all sub classes of surfaces for useful 
analytically described surfaces, each class overides the __init__ method and 
provides a descretise method the is_descrete flag should also be set to 
false until the descretise method has been run. init_checks should be
called first in the init method to check for dimentions and material property 
assignments each should also call the descretise_checks method before
affter assignemnt of inputs to descretise() to self.XXXXX but befor the rest of
the process

"""
    
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
        self.descretise_checks()
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
        self.descretise_checks()
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
        nPts=[round(length/spacing) for length in self.global_size]
        if self.dimentions==1:
            profile=np.random.randn(nPts[0],1)
        elif self.dimentions==2:
            profile=np.random.randn(nPts[0],nPts[1])
        self.profile=profile*self.gn_sigma+self.gn_mu
        self.is_descrete=True
    
    def specify_ACF(self, ACF_or_type, *args):
        size=self.global_size
        spacing=self.grid_size
        nPts=[int(sz/spacing) for sz in size]
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
        
class FlatNoiseSurface(GausianNoiseSurface): #done
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
        
    
    def descretise(self, spacing=False):
        if spacing:    
            self.grid_size=spacing
        nPts=[round(length/spacing) for length in self.global_size]
        self.descretise_checks()
        if self.dimentions==1:
            profile=np.random.rand(nPts[0],1)
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
        if spacing:    
            self.grid_size=spacing
        self.descretise_checks()
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
    
class ContinuousFrequencySurface(DiscreteFrequencySurface): #make work better with 2d surfaces 
    is_descrete=False
    surface_type='continuousFreq'
        #TODO write this function
        
    def descretise(self, spacing):
        self.grid_size=spacing#
        self.descretise_checks()
        x=np.arange(-0.5*self.global_size[0],
                    0.5*self.global_size[0],self.grid_size)
        #TODO write this function probably use an inverse FFT need to read up 
        # for 2d surfacesfor now just used np.eye * amps
        
if __name__ == "__main__":
    A=GausianNoiseSurface()
    #A.plot_acf()
    A.global_size=[1.01,1.01]
    A.descretise(0.003)
    A.low_pass_filter(15)