# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:26:18 2018

@author: mike and lisa
"""

class Error(exception):
    """class for exceptions in the slipPy module"""
    pass

class ReadError(Error):
    """Exception raised when user tries to read unrecognised file
    Attributes:
        
    """
    
class SurfaceOperationError(Error):
    """Exception rasied when a user tried to perform an invalid surface 
    operation
    """    
    
class InputError(Error):
    
    