import abc
import inspect
import typing

from slippy.abcs import _ContactModelABC, _StepABC, _SubModelABC
from .outputs import OutputRequest

__all__ = ['step', '_ModelStep', 'InitialStep']

"""
Steps including solve functions, each actual step is a subclass of ModelStep should provide an __init__, solve
 and check method. these do all the heavy lifting
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
    max_time: float
    sub_models: typing.List[_SubModelABC]
    outputs: typing.List[OutputRequest]

    def __init__(self, step_name: str, max_time: float, provides: set):
        assert isinstance(step_name, str), 'Step name must be string, this is used for all outputs related to this step'
        self.name = step_name
        self.max_time = max_time
        self.provides = provides
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
            if output.parameters not in current_state:
                raise ValueError(f"Output request {output.name}, requires {output.parameters} but this is not in the "
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

    def save_outputs(self, current_state: dict, output_file):
        """Writes all outputs for the step into the output file

        Parameters
        ----------
        current_state
        output_file

        Returns
        -------

        """
        if not self.outputs:
            return
        params_to_save = {'time'}
        for output in self.outputs:
            if output.is_active(current_state['time'], self.max_time):
                params_to_save.update(output.parameters)
        # make an expanded current_state
        expanded_state = current_state.copy()
        for param in ['loads', 'total_displacement', 'surface_1_displacement', 'surface_2_displacement']:
            if param in expanded_state:
                expanded_state[param + '_x'] = expanded_state[param].x
                expanded_state[param + '_y'] = expanded_state[param].y
                expanded_state[param + '_z'] = expanded_state[param].z
                del expanded_state[param]
            if param in params_to_save:
                params_to_save.update({param + '_x', param + '_y', param + '_z'})
                params_to_save.remove(param)
        # if all in extended params to save
        if 'all' in params_to_save:
            params_to_save.update(expanded_state)
            params_to_save.remove('all')

        missing_outputs = params_to_save - set(expanded_state)

        for mo in missing_outputs:
            if mo.startswith('surface') and ';' not in mo:
                try:
                    exec('param = self.model.' + mo)
                except IndexError as e:
                    print(f"Output {mo}, {str(e)}")
                    param = None
                except AttributeError as e:
                    print(f"Output {mo}, {str(e)}")
                    param = None
                except SyntaxError as e:
                    print(f"Output {mo}, {str(e)}")
                    param = None
                except Exception as e:
                    print(f"Output {mo}, {str(e)}")
                    param = None
            if param is not None:
                expanded_state[mo] = param
                missing_outputs.remove(mo)

        if missing_outputs:
            print(f"WARNING: Step {self.name}, failed to find outputs: {', '.join(missing_outputs)}, "
                  f"available outputs are: {', '.join(set(expanded_state))}, as well as surface attributes "
                  f"eg: surface1.profile or surface2.material... etc.")
        for element in missing_outputs:
            params_to_save.remove(element)

        output_dict = {key: expanded_state[key] for key in params_to_save}
        output_file.write(output_dict)

    def solve_sub_models(self, current_state: dict):
        if self.provides != set(current_state):
            diff = self.provides-set(current_state)
            diff.update(set(current_state)-self.provides)
            raise ValueError(f"Step {self.name} dosn't provide what it should or provides things which are not "
                             f"declared, \nprovides = {self.provides} \ncurrent_state: {set(current_state)}"
                             f"\ndifference = {diff}")
        for model in self.sub_models:
            found_params = model.solve(current_state)
            current_state.update(found_params)
        return current_state

    def add_output(self, output):
        self.outputs.append(output)


class InitialStep(_ModelStep):
    """
    The initial step run at the start of each model
    """

    _options = True

    # Should calculate the just touching position of two surfaces, set initial guesses etc.
    separation: float = 0.0

    def __init__(self, step_name: str = 'initial', separation: float = None):
        super().__init__(step_name=step_name, max_time=0, provides={'off_set', 'interference', 'time'})
        if separation is not None:
            self.separation = float(separation)

    def data_check(self, current_state: set):
        """
        Just check if this is the first step in the model
        """
        if current_state is not None:
            print("Error steps are out of order, initial step should be first")
        return {'off_set', 'time', 'interference'}

    def solve(self, current_state, output_file):
        if current_state is not None:
            raise ValueError("Steps have been run out of order, the initial step should always be run first")
        current_state = dict()
        current_state['off_set'] = (0, 0)
        current_state['interference'] = 0
        current_state['time'] = 0
        return current_state

    def __repr__(self):
        return f'InitialStep(model = {str(self.model)}, name = {self.name})'
