import numpy as np
import types
import scipy.signal
import scipy.interpolate

__all__=['ACF']

class ACF():
    """ TODO make pretty and test
    A class to specify auto corelation functions
    used with all surfaces
    functions should be usable independent of length
    can be directly specified or made from a function (again independent of base grid size)
    must ulitimately be expressed on a base grid

    should be able to supply:
    a surface, a name and params, a function handle(?)
    or a grid of values (to interpolate between)
    leave a funcion in self.methos that takes x,y points and the grid_spacing then returnds the height of the ACF at those points
    
    __Call__(X,Y) returns the height of the acf at each of the x,y points
    """
    
    method=None # the method that gets called 
    original=None
    acf_type=''
    interpolator=None

    
    def __init__(self, source, *args):
        if type(source) is str:
            self.input_check_string(source, args)
            self.acf_type="string"
        elif hasattr(source, 'profile'):
            self.input_check_array(source.profile, source._grid_spacing)
            self.acf_type="surface"
        elif type(source) is types.FunctionType:
            self.input_check_method(source)
            self.acf_type="function"
        else:
            self.input_check_array(source, args[0])
            # args should contain the grid_spacing of the array
            self.acf_type="array"
           

    def input_check_string(self, source, args):
        supported_funcions=['exp', 'polynomial']
        if not(source in supported_funcions):
            msg=("Function type not supported, supported types are:\n" +
                 "\n".join(supported_funcions) + "\nfor custom functions pass "
                 "the function object to this constructor")
            raise NotImplementedError(msg)
        if source=='exp':
            sigma = args[0]
            beta_x = args[1]
            beta_y = args[2]
            method = lambda X, Y : sigma**2*np.exp(-2.3*np.sqrt((X/beta_x)**2+(Y/beta_y)**2))
        elif source=='polynomial':
            pass
        self.method=method

    def input_check_array(self, source, grid_spacing, origin='centre'):
        try:    
            profile=np.asarray(source)
        except ValueError:
            msg=("invalid input, input should be either a surface, function " 
                 "handle, funtion name as a string with relavent params or "
                 "array-like")
            raise ValueError(msg)
        x=profile.shape[0]
        y=profile.shape[1]
        self.o_grid_spacing=grid_spacing
        self.original=(scipy.signal.correlate(profile,profile,'same')/(x*y))
        
        x=np.arange(x)*grid_spacing
        y=np.arange(y)*grid_spacing
        
        if type(origin) is str:
            origin=origin.lower()
            if origin=='centre':
                origin=[np.mean(x), np.mean(y)]
            else:
                description=origin
                origin=[0,0]
                if description[1]=='r' or description[1]=='e':
                    origin[0]=max(x)
                if description[0]=='t' or description[0]=='n':
                    origin[1]=max(y)
        x=x-origin[0]
        y=y-origin[1]
        
        self.method=scipy.interpolate.RectBivariateSpline(x, y, self.original)
    
    def input_check_method(self, source):
        self.method=source
        
    def __call__(self, x,y):
        """
        x and y are vectors that get meshgridded befroe returning matrix of points, dunno why but rect b spline works this way
        """
        #####TODO redo docs
        
        """ 
        Retuns part of the ACF descretised on the requested grid
        
        Parameters
        ----------
        grid_spacing : 2 element list or scalar
            The grid_spacing of the grid for descretisation in each direction 
            if a scalar is given the grid_spacing will be the same in each direction
        extent : list of lists or list
            The extent to be returned in each dimention:
            [[xstart, xstop], [ystart, ystop]]
            or [start, stop] which is interpreted as the same for all dimentions
        
        Returns
        -------
        out : np.array
            A grid of values corresponding to the height of the ACF at each of
            the requested grid points
            
        See also
        --------
        this.__array__ : returns the acf of the original input with no 
                         interpolation
                         
        """
        # feed to self. method 
        if self.acf_type in ['array', 'surface']:
            return self.method(x,y)
        else:
            X,Y=np.meshgrid(x,y)
            return self.method(X,Y)
        
    def __array__(self):
        if self.original is not None:    
            return self.original
        else:
            raise ValueError("Could not return ACF as array, ACF must be made from"
                       " an array or surface for this to work in stead use "
                       "array(this(grid_grid_spacing, extent))")
