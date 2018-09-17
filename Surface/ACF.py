import numpy as np
import warnings
import types
import scipy.signal


__all__=['ACF']

class ACF():
    last_output=False
    last_output_grid_size=False
    last_output_pts_each_dir=False
    method=False # the method that gets called 
    acf_type=''
     

    """ A class to specify auto corelation functions
    used with all surfaces
    functions should be usable independent of length
    can be directly specified or made from a function (again independent of base grid size)
    must ulitimately be expressed on a base grid

    should be able to supply:
    a surface, a name and params, a function handle(?)
    or a grid of values (to interpolate between)
    #TODO all
    """
    def __init__(self, source, *args):
        if type(source) is str:
            input_check_string(source, args)
            self.acf_type="string"
        elif type(source) is Surface:
            input_check_surface(source)
            self.acf_type="surface"
        elif type(source) is types.FunctionType:
            input_check_method(source)
            self.acf_type="function"
        else:
            input_check_array(source)
            self.acf_type="array"
           

    def input_check_string(self, source, args):
        supported_funcions=['exp', 'polynomial']
        if not(souce in supported_funcions):
            msg=("Function type not supported, supported types are:\n" +
                 "\n".join(supported_funcions) + "\nfor custom functions pass "
                 "the function handle to this constructor")
            ValueError(msg)
        if source=='exp':
            sigma = args[0]
            beta_x = args[1]/spacing
            beta_y = args[2]/spacing
            method = lambda X, Y : sigma**2*np.exp(-2.3*np.sqrt((X/beta_x)**2+(Y/beta_y)**2))
        elif source=='polynomial':
            pass
        self.method=method


    def input_check_surface(self, source, args):
		# first find the surface ACF then hand that to an interpolation function and set the interpolation function as the self.method
        profile=source.profile
        x=profile.shape[0]
        y=profile.shape[1]
        output=(scipy.signal.correlate(profile,profile,'same')/(x*y))
        self.source=source
        self.original=output
        self.method=surface_method

    def input_check_array(self,source, args):
        try:    
            profile=np.asarray(source)
        except ValueError:
            msg=("invalid input, input should be either a surface, function " 
                 "handle, funtion name as a string with relavent params or "
                 "array-like")
            ValueError(msg)
        x=profile.shape[0]
        y=profile.shape[1]
        self.source=source
        self.original=(scipy.signal.correlate(profile,profile,'same')/(x*y))
        self.method=array_method
    
    def input_check_method(self, source):
        self.method=source

    def surface_method(self, X, Y): # these 2 shold be pretty similar
        pass
    def array_method(self, X, Y):
        pass
        
    def __call__(self, spacing=False, pts_each_dir=False, origin='centre'):
        if not spacing or pts_each_dir:
             if self.last_output:
                 return self.last_output
             elif self.acf_type in ["array", "surface"]:
                 warnings.warn("ACF returned default values based on input "
                               "dimentions, use np.asarray(this) to surpress")
                 self.last_output=self.__array__()
                 return self.last_output
             else:
                 msg=("")
                 ValueError()
                 
        ######carry on from here
        
    def __array__(self):