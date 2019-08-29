from models.models import ContactModel

__all__ = ['step', '_ModelStep', '_InitialStep']

"""
Steps including solve functions, ecah actual step is a subclass of ModelStep should provide an __init__, _solve
 and _check method. thses do all the heavy lifting  
"""


def step(model, movement, load):
    """
    A helper function for creating step objects

    Parameters
    ----------

    Returns
    -------
    The step object for the relevent system
    """
    raise NotImplementedError("Steps are not implemented")


class _ModelStep:
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
    name = None
    """The name of the step"""
    _surfaces_required = None
    """The number of surfaces required to run the step (used in data checks)"""
    options = None
    """A named tuple options object should be different for each step type, specifies all of the analysis options"""

    def __init__(self, step_name: str, model: ContactModel):
        self.name = step_name
        self._model = model

    def _data_check(self):
        raise NotImplementedError("Data check have not been implemented for this step type!")

    def _solve(self, current_state, log_file, output_file):
        raise NotImplementedError("Solver not specified for this step!")


class _InitialStep(_ModelStep):
    """
    The initial step run at the start of each model
    """
    # Should calculate the just touching postion of two surfaces, set inital guesses etc.
    separation: float = 0.0

    def __init__(self, model: ContactModel, step_name: str = 'initial', separation: float = None):
        super().__init__(step_name=step_name, model=model)
        if separation is not None:
            self.separation = float(separation)

    def _data_check(self):
        pass

    def _solve(self, current_state, log_file, output_file):
        """
        Need to:
        find separtion, initialise current state(does this mean looking through the outputs? or should that be in the
        model solve function), ???? anthing else?
        """