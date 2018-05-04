
class Surface():
    # The surface class for descrete surfaces 
    def __init__(self,args,kwargs):
        # initialisation surface
        
        
    def set_properties(self, materialType='elastic', E=200E9, v=0.3):
        # allowed types are rigid, elastic, plastic, epp, ve, ? 
        
    def set_units(self, interval=0, globalX=0, globalY=0):
        
    def fft_surface(self):
        
    def plot_FFT_surface(self):
        
    def psd_surface(self):
        
    def plot_PSD_surface(self):
        
    def fill_holes(self):
        
    def read_from_file(self, filename):
        
    def check_surface(self):
        
    def roughness(self):
        
    def __add__(self, other):
        
        
        
    
        

class AnalyticalSurface(Surface):
    # analytical surface inherits Surface methods with additional methods for
    # descretising and functions for building surfaces
    def __init__(self,args,kwargs):
        #initialise as a flat rigid surface
        
        
    def set_type(self, surfaceType=flat, params=[]):
        
        
    def descretise_surface(self, spacing, centre):
         