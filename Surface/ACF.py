import numpy as np
import warnings
import types
import scipy.signal


__all__=['ACF']

 class ACF():
 	profile=False
 	profile_grid_size=False
 	profile_pts_each_dir=False
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
 			input_check_surface(source, args)
 			self.acf_type="surface"
 		elif type(source) is types.FunctionType:
 			input_check_method(source, args)
 			self.acf_type="function"
 		pass

 	def input_check_string(self, source, args):
 		supported_funcions=['exp', 'somthing else']
 		if not souce in supported_funcions:
 			msg=("Function type not supported, supported types are:\n".join(supported_funcions)
 				"\nfor custom functions pass the function handle to this constructor")
 			ValueError(msg)
 		if source=='exp':
            sigma = args[0]
            beta_x = args[1]/spacing
            beta_y = args[2]/spacing
            method = lambda X, Y : sigma**2*np.exp(-2.3*np.sqrt((X/beta_x)**2+(Y/beta_y)**2))
        elif source=='somthing else':
        	pass
        self.method=method


 	def input_check_surface(self, source, args):
 		# first find the surface ACF then hand that to an interpolation function and set the interpolation function as the self.method
        if type(source) is Surface:## carry on from here
            profile=source.profile
        profile=np.asarray(surf_in)
        x=profile.shape[0]
        y=profile.shape[1]
        output=(scipy.signal.correlate(profile,profile,'same')/(x*y))
        if type(surf_in) is bool:
            self.acf=output
        return output

 	def surface_method(self, X, Y):
        

	def input_check_method(self, source, args):
		

 	def __call__(spacing=False, pts_each_dir=False, origin='centre'):
 		if 