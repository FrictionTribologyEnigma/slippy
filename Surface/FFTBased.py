"""
Classes for generating pseudo-random surfaces based on description of FFT:
    ===========================================================================
    ===========================================================================
    Each class inherits functionallity from the Surface but changes the 
    __init__ and descretise functions
    ===========================================================================
    ===========================================================================
    DiscFreqSurface:
        Generate a surface conating only specific frequency componenets
    ProbFreqSurface:
        Generate a surface containing normally distributed amptitudes with a
        specified function for the varience of the distribution based on the 
        frequency
    DtmnFreqSurface:
        Generate a surface containing frequency components with amptitude 
        specifed by a function of the frequency
        
    ===========================================================================
    ===========================================================================

#TODO:
        Add comment blocks to each class with examples of use
        add DtmnFreqSurface
        cheange Prob and dtm to allow for more descriptios of FFTs inc passing
        actual functions and some built in functions
        overide resampling function should resample based on the FFT not simple intep
        make work with generate keyword
"""

from . import Surface
import warnings
import numpy as np

__all__=["DiscFreqSurface", "ProbFreqSurface", "HurstFractalSurface"]#, "DtmnFreqSurface"]

class DiscFreqSurface(Surface):
    """
    Generates a surface containg discrete frequncy components
    
    Usage:
    
    DiscFreqSurface(frequencies, amptitudes=[1], phases_rads=[0], dimentions=2)
    
    Generates a surface with the specified frequencies, amptitudes and phases 
    any kwargs that can be passed to surface can also be passed to this
    
    mySurf=DiscFreqSurface(10, 0.1) 
    mySurf.global_size=[0.5,0.5]
    mySurf.descretise(0.001)
    
    Generates and descretises a 2D surface with a frequency of 10 rads/unit
    of global size, descretised on a grid with a spacing of 0.001
    """
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
            
    def descretise(self, spacing=None):
        if spacing:    
            self.grid_size=spacing
        self.descretise_checks()
        #TODO write this section
        x=np.linspace(-0.5*self.global_size[0],
                    0.5*self.global_size[0],self.pts_each_direction[0])
        if self.dimentions==1:
            profile=np.zeros_like(x)
            for idx in range(len(self.frequencies)):
                profile+=np.real(self.amptitudes[idx]*
                                 np.exp(-1j*self.frequencies[idx]*x*2*np.pi))
            self.profile=profile
        elif self.dimentions==2:
            y=np.linspace(-0.5*self.global_size[1],
                        0.5*self.global_size[1],self.pts_each_direction[1])
            (X,Y)=np.meshgrid(x,y)
            profile=np.zeros_like(X)
            for idx in range(len(self.frequencies)):
                profile+=np.real(self.amptitudes[idx]*
                                 np.exp(-1j*self.frequencies[idx]*X*2*np.pi)+
                                 self.amptitudes[idx]*
                                 np.exp(-1j*self.frequencies[idx]*Y*2*np.pi))
            self.profile=profile
    
class ProbFreqSurface(Surface):
    """
    ProbFreqSurface(H, qr, qs)
    
    Generates a surface with all possible frequencies in the fft represented 
    with amptitudes described by the probability distrribution given as input.
    Defaults to the parameters used in the contact mechanics challenge
    
    This class only works for square 2D domains
    
    For more infromation on the definations of the input parameters refer to 
    XXXXXX contact mechanics challenge paper
    
    """
    is_descrete=False
    surface_type='continuousFreq'
        #TODO write this function
    def __init__(self, H=2, qr=0.05, qs=10, **kwargs):
        #q is frequency
        self.init_checks(kwargs)
        self.H=H
        self.qs=qs
        self.qr=qr
        
    def descretise(self, global_size, grid_spacing):
        self.global_size=global_size
        self.grid_spacing=grid_spacing
        self.descretise_checks()
        if self.global_size[0]!=self.global_size[1]:
            ValueError("This method is only defined for square domains")
        qny=np.pi/grid_spacing
        
        u=np.linspace(0,qny,self.pts_each_direction[0])
        U,V=np.meshgrid(u,u)
        Q=np.abs(U+V)
        varience=np.zeros(Q.shape)
        varience[np.logical_and((1/Q)>(1/self.qr),
                                (2*np.pi/Q)<=(global_size[0]))]=1
        varience[np.logical_and((1/Q)>=(1/self.qs),
                                (1/Q)<(1/self.qr))]=(Q[np.logical_and(
                1/Q>=1/self.qs,1/Q<1/self.qr)]/self.qr
                    )**(-2*(1+self.H))
        FT=np.array([np.random.normal()*var**0.5 for var in varience.flatten()])
        FT.shape=Q.shape
        self.profile=np.real(np.fft.ifft2(FT))
       
class HurstFractalSurface(Surface):
    """
    HurstFractalSurface(q0,q0 amptitude,cut off frequency,Hurst parameter)
    
    generates a hurst fratal surface with frequency components from q0 to 
    cut off frequency in even steps of q0.
    
    amptitudes are given by:
        q0 amptitude**2 *((h**2+k**2)/2)^(1-Hurst parameter)
    where h,k = -N...N 
    where N=cut off frequency/ q0
    phases are randomly generated on construction of the surface object,
    repeted calls to the descretise function will descretise on the same surface
    but repeted calls to this class will generate diferent realisations
    
    Example:
        #create the surface object with the specified fractal prameters
        my_surface=HurstFractalSurface(1,0.1,1000,2)
        #descrtise the surface over a grid 1 unit by 1 unit with a spacing of 0.01
        heights=my_surface.descretise('grid', [[0,1],[0,1]], [0.01,0.01])
        #interpolate over a previously made grid
        heights=my_surface.descretise('interp', X, Y, **kwargs) ** kwargs for remaking interpolator and interpolator options
        #generate new points (e.g. for a custom grid)
        my_surface.descretise('points', X, Y)
        
        A new efficient numerical method for contact mechanics of rough surfaces
        C.Putignano L.Afferrante G.Carbone G.Demelio
    """
    is_descrete=False
    surface_type="hurstFractal"
    
    def __init__(self,q0,q0_amp,q_cut_off,hurst,**kwargs):
        self.init_checks(kwargs)
        N=int(round(q_cut_off/q0))
        h, k=range(-1*N,N+1), range(-1*N,N+1)
        H,K=np.meshgrid(h,k)
        H[N,N]=1
        mm2=q0_amp**2*((H**2+K**2)/2)**(1-hurst)
        mm2[N,N]=0
        pha=2*np.pi*np.random.rand(mm2.shape[0], mm2.shape[1])
        
        mean_mags2=np.zeros((2*N+1,2*N+1))
        phases=np.zeros_like(mean_mags2)
        
        mean_mags2[:][N:]=mm2[:][N:]
        mean_mags2[:][0:N+1]=np.flipud(mm2[:][N:])
        self.mean_mags=np.sqrt(mean_mags2).flatten()
    
        phases[:][N:]=pha[:][N:]
        phases[:][0:N+1]=np.pi*2-np.fliplr(np.flipud(pha[:][N:]))
        phases[N,0:N]=np.pi*2-np.flip(phases[N,N+1:])
        #######next line added
        phases=2*np.pi*np.random.rand(phases.shape[0], phases.shape[1])
        self.phases=phases.flatten()
        self.mags=self.mean_mags*np.cos(self.phases)+1j*self.mean_mags*np.sin(self.phases)
        K=np.transpose(H)
        self.qkh=np.transpose(np.array([q0*H.flatten(), q0*K.flatten()]))
        
        
    def descretise(self, global_size=None, spacing=None):
        if not global_size:
            global_size=self.global_size
        if not spacing:
            spacing=self.grid_size
        
        self.descretise_checks()
        
        X,Y=self._get_points_from_extent(global_size, spacing)
        input_shape=X.shape
        coords=np.array([X.flatten(), Y.flatten()])
        
        Z=np.zeros(X.size,dtype=np.float32)

        for idx in range(len(self.qkh)):
            Z+=np.real(self.mags[idx]*np.exp(-1j*np.dot(self.qkh[idx],coords)*2*np.pi))

        self.profile=Z.reshape(input_shape)