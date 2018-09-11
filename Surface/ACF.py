import numpy as np
import warnings

__all__=['ACF']

 class ACF():
 	""" A class to specify auto corelation functions
 	used with random surface
 	functions should be usable independent of length
 	can be directly specified or made from a function (again independent of base grid size)
 	must ulitimately be expressed on a base grid

 	should be able to supply:
 	a surface, a name and params, a function handle(?)
 	or a grid of values (to interpolate between)
 	#TODO all
 	"""
 	def __init__(self, *params):
 		pass