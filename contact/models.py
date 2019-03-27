"""
model object
"""
__all__=["Model", 'Step', 'Results']

class Model(object):
    """ A container for contact mechanics problems
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    Methods
    -------
    
    
    """
    steps=[]
    surface_1=None
    surface_2=None
    sets={}
    lubricants={}
    materials={}
    amptitudes={}
    interactions={}
    fluid_motions={}
    laods={}
    historyOutputs={}
    fieldOutputs={}
    
    def __init__():
        pass

class Results(object):
    """ A results class for contact mechanics models
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    Methods
    -------
    
    
    """
    model=None
    sets={}
    steps=[]
    
class ResultsStep(object):
    """ A step in a contact mechanics results set
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    Methods
    -------
    
    
    """
    frames=[]
    historyOutputs???