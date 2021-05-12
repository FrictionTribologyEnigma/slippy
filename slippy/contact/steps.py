import abc
import typing
import slippy
import warnings

from slippy.core import _ContactModelABC, _StepABC, _SubModelABC
from slippy.core.outputs import OutputRequest

__all__ = ['_ModelStep', 'InitialStep']

"""
Steps including solve functions, each actual step is a subclass of ModelStep should provide an __init__, solve
 and check method. these do all the heavy lifting
"""


def _data_check_error_or_warn(msg: str):
    if slippy.ERROR_IN_DATA_CHECK:
        raise ValueError(msg)
    else:
        warnings.warn(msg)


class _ModelStep(_StepABC):

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

    def _data_check(self, previous_state: set):
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
        print(f"Checking step: {self.name}")
        self.data_check(previous_state)
        current_state = self.provides
        if 'time' not in current_state:
            _data_check_error_or_warn("All steps must provide a time")
        print(f"Checking sub models for step {self.name}")
        self.check_sub_models(current_state)
        print(f"Checking outputs for step {self.name}")
        self.check_outputs(current_state)

    def data_check(self, current_state):
        """To be overwritten by steps that need more complicated checking"""
        pass

    def check_outputs(self, current_state: set):
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
        params_to_save = {'time'}
        for output in self.outputs:
            params_to_save.update(output.parameters)

        expanded_state = current_state.copy()

        if 'all' in params_to_save:
            params_to_save.update(expanded_state)
            params_to_save.remove('all')

        for param in ['loads', 'total_displacement', 'surface_1_displacement', 'surface_2_displacement']:
            if param in expanded_state:
                expanded_state.add(param + '_x')
                expanded_state.add(param + '_y')
                expanded_state.add(param + '_z')
                expanded_state.remove(param)
            if param in params_to_save:
                params_to_save.update({param + '_x', param + '_y', param + '_z'})
                params_to_save.remove(param)

        missing_outputs = params_to_save - set(expanded_state)

        for mo in missing_outputs:
            if mo.startswith('surface') and ';' not in mo:
                warnings.warn(f"Could not check output: {mo}, not possible to check surface outputs at this time")
            else:
                _data_check_error_or_warn(f"Could not find output {mo} in state for step {self.name}.\n"
                                          f"State will be: {current_state}")

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
            if model.requires - current_state:
                _data_check_error_or_warn(f"Model step: {self.name} doesn't find required inputs for "
                                          f"model: {model.name}:\n"
                                          f"State will be: {current_state}\n"
                                          f"Model requires: {model.requires}")
            print(f"Passed: sub_model {model.name} in step {self.name}")
            current_state.update(model.provides)
        return set(current_state)

    def add_sub_model(self, sub_model: _SubModelABC):
        """
        Add a sub model to be exec
        :param sub_model:
        :return:
        """
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

        for param in ['surface_1_points', 'surface_2_points']:
            if param in expanded_state:
                expanded_state[param + '_y'] = expanded_state[param][0]
                expanded_state[param + '_x'] = expanded_state[param][1]
                del expanded_state[param]
            if param in params_to_save:
                params_to_save.update({param + '_x', param + '_y'})
                params_to_save.remove(param)

        # if all in extended params to save
        if 'all' in params_to_save:
            params_to_save.update(expanded_state)
            params_to_save.remove('all')

        missing_outputs = params_to_save - set(expanded_state)

        missing_outputs_copy = missing_outputs.copy()

        for mo in missing_outputs_copy:
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
        if self.provides != set(current_state) and not slippy.ERROR_IF_MISSING_MODEL:
            missing = self.provides-set(current_state)
            unexpected = set(current_state)-self.provides
            raise ValueError(f"Step {self.name} dosn't provide what it should or provides things which are not "
                             f"declared, \nprovides = {self.provides} \ncurrent_state: {set(current_state)}"
                             f"\nMissing from current state = {missing}"
                             f"\nUnexpected in current state = {unexpected}"
                             f"\nTo suppress this error set slippy.ERROR_IF_MISSING_MODEL to False")
        for model in self.sub_models:
            found_params = model.solve(current_state)
            if model.provides != set(found_params) and not slippy.ERROR_IF_MISSING_SUB_MODEL:
                unexpected = self.provides - set(current_state)
                missing = set(current_state) - self.provides
                raise ValueError(f"Sub model {model.name} dosn't provide what it should or provides things which are "
                                 f"not declared, \nprovides = {self.provides} \nresults: {set(current_state)}"
                                 f"\nMissing from results = {missing}"
                                 f"\nUnexpected in results = {unexpected}"
                                 f"\nTo suppress this error set slippy.ERROR_IF_MISSING_SUB_MODEL to False")
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

    def _data_check(self, current_state: set):
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
