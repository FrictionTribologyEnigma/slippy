"""
Steps including solve functions
"""

class _ModelStep(object):
    """ A step in a contact mechanics problem
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    Methods
    -------
    
    
    """
    """
    each sub class should provide it's own:
        init
        read inputs into the step dont really do anything just for setting up
        _solve
        takes the present state of the 2 surfaces in (lacation and meshes) and adict of output
        adds to the output dict with the results
        
        _analysis_checks
        Checks that each of the outputc can be found and other checks (each surface has a material)
        
    """
    loads=None
    displacements=None
    solver=None
    rotation_centre=None
    rotation=None
    outputs=None
    name=None
    lubrication=False
    lubrication_model=False
    subsurface_stress=False
    wear=False
    wear_model=None
    
    def __init__(self, step_name : str, 
                 loads = None, 
                 displacements : dict = None, 
                 output_requests : list = None, 
                 solver = 'BEM',
                 wear : WearModel = None):
        
        self.name=step_name
        self.loads=loads
        self.displacements=displacements
        ouptputs=set([o_r['outputs'] for o_r in output_requests 
                      if step_name in o_r or 'all' in o_r])
        self.wear=not wear is None
        if self.wear:
            self.wear_model=wear
            self.subsurface_stress += wear.needs_subsurface_stress
        self.subsurface_stress+= 'sss' in outputs
        
            

class StaticStep(_ModelStep):
    """ A static contact step
    
    Keyword Parameters
    ------------------
    loads : dict optional 
        global loads with members 'x', 'y' and/or 'z' 
    displacements : dict optional
        global displacements with members 'x', 'y' and/or 'z'
    rotation_centre : array-like optional 
        The coordinates at the centre of rotation only required if rotation 
        keyword is also given
    rotation : dict optional
        global rotations with members 'x', 'y' and/or 'z'
    solver : str 
        The solver to be used
    """
    
    
        
    def _analysis_check(self):
        if not (loads is None ^ displacements is None):
            raise ValueError("Either Loads or displacement must be set for "
                             "static step")
        
    def _solve(self, surface_1, surface_2):
        
        
    
class QuasiStaticStep(_ModelStep):
    """ A quasi static step which alows changing loads/ displacements
    
    """
    def __init__(self, loads = None, displacements = None, heating = None, 
                 output_requests = None, solver = 'BEM'):
        pass
    
    def _analysis_check():
    
    def _solve():
        pass
    