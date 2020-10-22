import abc
import inspect
import pickle
import typing

from slippy.abcs import _ContactModelABC, _StepABC, _SubModelABC
from .outputs import OutputRequest

__all__ = ['step', '_ModelStep', 'InitialStep']

"""
Steps including solve functions, each actual step is a subclass of ModelStep should provide an __init__, _solve
 and _check method. these do all the heavy lifting
"""


def step(model: _ContactModelABC):
    """ A text based interface for generating steps

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
            item = int(input(prompt + ''.join(list_of_step_types)))
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

    name = None
    """The name of the step"""
    _surfaces_required = None
    """The number of surfaces required to run the step (used in data checks)"""
    _options = None
    """A named tuple options object should be different for each step type, specifies all of the analysis options"""
    _model: _ContactModelABC = None
    _subclass_registry = []

    sub_models: typing.List[_SubModelABC]
    outputs: typing.List[OutputRequest]

    def __init__(self, step_name: str):
        assert isinstance(step_name, str), 'Step name must be string, this is used for all outputs related to this step'
        self.name = step_name
        self.sub_models = []
        self.outputs = []

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
    def data_check(self, previous_state: set):
        """
        Produce errors for predicted errors during simulation

        Parameters
        ----------
        previous_state: set
            The model state from the last model step

        Returns
        -------
        current_state: set
            set of items in the current state after this step has run

        Notes
        -----
        The methods check_sub_models and check_outputs should be called as part of this method
        """
        raise NotImplementedError("Data check have not been implemented for this step type!")

    def check_outputs(self, current_state: set) -> set:
        """Data check all outputs

        Parameters
        ----------
        current_state: set
            The model state at the point when the outputs will be called

        Returns
        -------
        current_state: set
            Unmodified from the input set

        Notes
        -----
        This should be called by each step in it's data check method
        """
        for output in self.outputs:
            if output.parameter not in current_state:
                raise ValueError(f"Output request {output.name}, requires {output.parameter} but this is not in the "
                                 "current state")
        return current_state

    def check_sub_models(self, current_state: set) -> set:
        """ Check all the sub models of the current step

        Parameters
        ----------
        current_state: set
            The state of the model when the sub models are evaluated

        Returns
        -------
        current_sate: set
            The input set updated with all values found by the sub models

        Notes
        -----
        This should be called by each step in it's data check method
        """
        for model in self.sub_models:
            full_arg_spec = inspect.getfullargspec(model.solve)
            args = full_arg_spec.args
            for requirement in args:
                if requirement not in current_state:
                    raise ValueError(f"Model step doesn't find required inputs for model: {model.name}")
                current_state.update(model.provides)

        return set(current_state)

    def add_sub_model(self, sub_model: _SubModelABC):
        if self.model is not None:
            sub_model.model = self.model
        self.sub_models.append(sub_model)

    @abc.abstractmethod
    def solve(self, current_state, output_file) -> dict:
        """ Take in the current state solve the step and update the current state and any field outputs, printing in the
        solve method will add to the log file, the standard output will be changed before running

        Parameters
        ----------
        current_state
        output_file

        Returns
        -------
        previous_state
        """
        raise NotImplementedError("Solver not specified for this step!")

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def new_step(cls, model):
        """
        A helper function which provides a text based interface to making a new step of the given type

        Returns
        -------
        An instance of the step
        """
        raise NotImplementedError()

    def save_outputs(self, current_state: dict, output_file=None):
        """Writes all outputs for the step into the output file

        Parameters
        ----------
        current_state
        output_file

        Returns
        -------

        """
        for output in self.outputs:
            if output.slices is None:
                pickle.dump({'step': self.name,
                             'output': output.name,
                             'parameter': output.parameter,
                             'data': current_state[output.parameter]}, output_file)
            else:
                # noinspection PyTypeChecker
                pickle.dump({'step': self.name,
                             'output': output.name,
                             'parameter': output.parameter,
                             'data': current_state[output.parameter[output.slices]]}, output_file)

    def solve_sub_models(self, current_state: dict):
        for model in self.sub_models:
            found_params = model.solve(current_state)
            current_state.update(found_params)
        return current_state


class InitialStep(_ModelStep):
    """
    The initial step run at the start of each model
    """

    _options = True

    @classmethod
    def new_step(cls, model):
        raise ValueError("Cannot make a new instance of the initial step")

    # Should calculate the just touching position of two surfaces, set initial guesses etc.
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
        model solve function), ???? anteing else?
        """
        if current_state is not None:
            raise ValueError("Steps have been run out of order, the initial step should always be run first")
        current_state = dict()
        current_state['off_set'] = (0, 0)
        current_state['interference'] = 0
        return current_state

    def __repr__(self):
        return f'InitialStep(model = {str(self.model)}, name = {self.name})'
