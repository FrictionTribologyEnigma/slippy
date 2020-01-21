import abc
import pickle
import typing

from slippy.abcs import _ContactModelABC, _StepABC

__all__ = ['step', '_ModelStep', 'InitialStep']

"""
Steps including solve functions, ecah actual step is a subclass of ModelStep should provide an __init__, _solve
 and _check method. thses do all the heavy lifting  
"""


def step(model: _ContactModelABC):
    """ A text based inteface for generating steps

    Returns
    -------
    model_step: _ModelStep
        A model step object with the requested parameters
    """
    prompt = 'Please select one of the following step types but entering the number next to the item:'
    list_of_step_types = [f'\n{num}\t{item.__name__}\t{item.__doc__}' for num, item in
                          enumerate(_ModelStep._subclass_registry)]
    item = -1
    while True:
        try:
            item = int(input(prompt+''.join(list_of_step_types)))
        except ValueError:
            print("Please only enter the number of the item in the list")
            continue
        if len(list_of_step_types) > item >= 0:
            break
        else:
            print("Please only enter the number of the item in the list")
    return _ModelStep._subclass_registry[item].new_step(model)


class _ModelStep(_StepABC):
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
    _options = None
    """A named tuple options object should be different for each step type, specifies all of the analysis options"""
    _model: _ContactModelABC = None
    _subclass_registry = []

    def __init__(self, step_name: str):
        self.name = step_name
        if self._options is None or self.name is None:
            raise ValueError("Step has not been instantiated correctly")

    @classmethod
    def __init_subclass__(cls, is_abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            _ModelStep._subclass_registry.append(cls)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if self._model is not None:
            raise ValueError("The model cannot be changed after step instantiation")
        if isinstance(value, _ContactModelABC):
            self._model = value
        else:
            raise ValueError("Supplied model is not a contact model or no contact model supplied")

    @abc.abstractmethod
    def data_check(self, current_state: set):
        """
        Write potential errors and warnings to the log file, update the current state with what will be in it by the end
        of the step (warnings will be written by printing, the standard output will be changed before running).

        Parameters
        ----------
        current_state: set

        Returns
        -------
        errors: bool
            True if there were no errors false if there were errors
        current_state: set
            set of items in the current state afteer the step has run
        """
        raise NotImplementedError("Data check have not been implemented for this step type!")

    @abc.abstractmethod
    def solve(self, current_state, output_file):
        """ Take in the current state solve the step and update the current state and any field outputs, printing in the
        solve method will add to the log file, the standard output will be changed before running

        Parameters
        ----------
        current_state
        output_file

        Returns
        -------
        current_state
        """
        raise NotImplementedError("Solver not specified for this step!")

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def new_step(cls, model):
        """
        A helper function which provides a text based interface to making a new step of the given tyoe

        Returns
        -------
        An instance of the step
        """
        raise NotImplementedError()

    def save_outputs(self, current_state: typing.Union[dict, set], output_file=None, data_check=False):
        # TODO should check the ouput requests to see if there are any needed in this step (memoise this result) then
        #  save the requested ouputs if data_check is true just work like a data check (checking that the required
        #  things are present in the set,
        if data_check:
            return
        else:
            pickle.dump(current_state, output_file)

    def solve_sub_models(self, current_state: typing.Union[dict, set], data_check = False):
        # TODO this should solve the models like flash temperature, wear etc. should just update the current state dict
        return current_state


class InitialStep(_ModelStep):
    """
    The initial step run at the start of each model
    """

    _options = True

    @classmethod
    def new_step(cls, model):
        raise ValueError("Cannot make a new instace of the initial step")

    # Should calculate the just touching postion of two surfaces, set inital guesses etc.
    separation: float = 0.0

    def __init__(self, step_name: str = 'initial', separation: float = None):
        super().__init__(step_name=step_name)
        if separation is not None:
            self.separation = float(separation)

    def data_check(self, current_state: set):
        """
        Just check if this is the first step in the model
        """
        if current_state is not None:
            print("Error steps are out of order, initial step should be first")
        return {'off_set'}

    def solve(self, current_state, output_file):
        """
        Need to:
        find separation, initialise current state(does this mean looking through the outputs? or should that be in the
        model solve function), ???? anthing else?
        """
        if current_state is not None:
            raise ValueError("Steps have been run out of order, the initial step should always be run first")
        current_state = dict()
        current_state['off_set'] = (0, 0)
        current_state['interference'] = 0
        return current_state

    def __repr__(self):
        return f'InitialStep(model = {str(self.model)}, name = {self.name})'
