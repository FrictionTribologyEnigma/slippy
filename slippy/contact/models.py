"""
model object just a container for step object that do the real work
"""
from slippy.abcs import _SurfaceABC, _LubricantModelABC, _ContactModelABC
from slippy.contact.steps import _ModelStep, InitialStep, step
from slippy.contact.outputs import FieldOutputRequest, HistoryOutputRequest, possible_field_outpts, \
    possible_history_outpts
from datetime import datetime
import typing
from collections import OrderedDict
from contextlib import redirect_stdout, ExitStack
import os

__all__ = ["ContactModel"]


class ContactModel(_ContactModelABC):
    """ A container for multi step contact mechanics and lubrication problems
    
    Parameters
    ----------
    name: str
        The name of the contact model, used for output and log files by default
    surface_1, surface_2: _SurfaceABC
        A surface object with the height profile and the material for the surface set. The first surface will be the
        master surface, when grid points are not aligned surface 2 will be interpolated on the grid points for
        surface 1.
    lubricant: _LubricantModelABC
        A lubricant model
    log_file_name: str
        The name of the log file to use, if not set the model name will be used instead
    output_file_name: str
        The name of the output file to use, if not set the model name will be used, can also be set when solve is called

    Attributes
    ----------
    steps: OrderedDict
        The model steps in the order they will be solved in, each value will be a ModelStep object.
    surface_1, surface_2: _SurfaceABC
        Surface objects of the two model surfaces
    history_outputs: dict
        Output request for history data from the model, non spatially resolved results such as total load etc.
    field_outputs: dict
        Output request for field data (spatial data from the results)

    Methods
    -------
    add_step
        Adds a step object to the model
    add_field_output
        Adds a field output to the model
    add_history_output
        Adds a history output to the model
    data_check
        Performs analysis checks for each of the steps and the model as a whole, prints the results to the log file
    solve
        Solves all of the model steps in sequence, writing history and field outputs to the output file. Writes progress
        to the log file
    
    """

    _domains = {'all': None}
    """Flag set to true if one of the surfaces is rigid"""
    _is_rigid: bool = False
    steps: OrderedDict
    log_file_name: str = None
    output_file_name: str = None

    def __init__(self, name: str, surface_1: _SurfaceABC, surface_2: _SurfaceABC = None,
                 lubricant: _LubricantModelABC = None, log_file_name: str = None,
                 output_file_name: str = None):
        self.surface_1 = surface_1
        self.surface_2 = surface_2
        self.name = name
        if lubricant is not None:
            self.lubricant_model = lubricant
        self.steps = OrderedDict({'Initial': InitialStep()})
        if log_file_name is None:
            log_file_name = name
        self.log_file_name = log_file_name + '.log'
        if output_file_name is None:
            output_file_name = name
        self.output_file_name = output_file_name + '.log'
        try:
            os.remove(self.log_file_name)
        except FileNotFoundError:
            pass
        self.history_outputs = dict()
        self.field_outputs = dict()

    def add_step(self, step_instance: _ModelStep = None, position: typing.Union[int, str] = None):
        """ Adds a solution step to the current model
        
        Parameters
        ----------
        step_instance: _ModelStep
            An instance of a model step
        position : {int, 'last'}, optional ('last')
            The position of the step in the existing order
        
        See Also
        --------
        step
        
        Notes
        -----
        Steps should only be added to the model using this method 
        #TODO detailed description of inputs
        
        Examples
        --------
        >>> #TODO
        """
        # note to self
        """
        New steps should be bare bones, all the relevant information should be
        added to the steps at the data check stage, essentially there should be 
        nothing you can do to make a nontrivial error here.
        """
        if step_instance is None:
            new_step = step(self)
        else:
            new_step = step_instance

        step_name = step_instance.name
        step_instance.model = self

        if position is None:
            self.steps[step_name] = new_step
        else:
            keys = list(self.steps.keys())
            values = list(self.steps.values())
            if type(position) is str:
                position = keys.index(position)
            keys.insert(position, step_name)
            values.insert(position, new_step)
            self.steps = OrderedDict()
            for k, v in zip(keys, values):
                self.steps[k] = v

    def add_field_output(self, name: str, domain: typing.Union[str, typing.Sequence], step_name: str,
                         time_points: typing.Sequence,
                         output: typing.Sequence[str]):
        f"""
        Adds a field output request to the model

        Parameters
        ----------
        name : str
            The name of the output request
        domain : {{str, Sequence}}
            The name of the node set or the node set to be used, node sets for 'all', 'surface_1' and 'surface_2' are
            created automatically
        step_name : str
            The name of the step that the field output is to be taken from, ues 'all' for all steps
        time_points : Sequence
            The time points for the field output. If the output is only required at the start of the step use (0,) if it
            is only required at the end of the step used (None,), otherwise pass a sequence of time points or a slice
            object.
        output : Sequence[str]
            Names of output parameters to be included in the request, for more information see the documentation of
            FieldOutputRequest

        See Also
        --------
        FieldOutputRequest

        Notes
        -----
        Valid output request are {', '.join(possible_field_outpts)}
        """
        # check that domain exists
        if domain not in self._domains:
            model_domains = ', '.join(self._domains.keys())
            raise ValueError(f"Unrecognised domain :{domain}, model domains are: {model_domains}")
        # check that name is str (doesn't start with _)
        if type(name) is not str:
            raise TypeError(f"Field output name should be string, not {type(name)}")
        elif name.startswith('_'):
            raise ValueError("Field output names cannot start with _")
        # check that step exists
        if step_name not in self.steps and step_name != 'all':
            raise ValueError(f"Step name {step_name} not found.")
        # check that all outputs are valid
        out_in = [o in possible_field_outpts for o in output]
        if not all(out_in):
            raise ValueError(f"Unrecognised output request: {output[out_in.index(False)]}, valid outputs are: "
                             f"{', '.join(possible_field_outpts)}")

        output_dict = {key: True for key in output}

        self.field_outputs[name] = FieldOutputRequest(domain=domain, step=step_name, time_points=time_points,
                                                      **output_dict)

    def add_history_output(self, name: str, step_name: str, time_points: typing.Sequence, output: typing.Sequence[str]):
        f"""
        Adds a field output request to the model

        Parameters
        ----------
        name : str
            The name of the output request
        step_name : str
            The name of the step that the field output is to be taken from, ues 'all' for all steps
        time_points : Sequence
            The time points for the field output. If the output is only required at the start of the step use (0,) if it
            is only required at the end of the step used (None,), otherwise pass a sequence of time points or a slice
            object.
        output : Sequence[str]
            Names of output parameters to be included in the request, for more information see the documentation of
            HistoryOutputRequest

        See Also
        --------
        HistoryOutputRequest

        Notes
        -----
        Valid outputs are {', '.join(possible_history_outpts)}
        """
        # check that name is str (doesn't start with _)
        if type(name) is not str:
            raise TypeError(f"History output name should be string, not {type(name)}")
        elif name.startswith('_'):
            raise ValueError("History output names cannot start with _")
        # check that step exists
        if step_name not in self.steps and step_name != 'all':
            raise ValueError(f"Step name {step_name} not found.")
        # check that all outputs are valid
        out_in = [o in possible_history_outpts for o in output]
        if not all(out_in):
            raise ValueError(f"Unrecognised output request: {output[out_in.index(False)]}, valid outputs are: "
                             f"{', '.join(possible_history_outpts)}")

        output_dict = {key: True for key in output}

        self.history_outputs[name] = HistoryOutputRequest(step=step_name, time_points=time_points,
                                                          **output_dict)

    def data_check(self):
        with open(self.log_file_name, 'a+') as file:
            with redirect_stdout(file):
                print("Data check started at:")
                print(datetime.now().strftime('%H:%M:%S %d-%m-%Y'))

                self._model_check()

                current_state = None

                for this_step in self.steps:
                    print(f"Checking step: {this_step}")
                    # noinspection PyProtectedMember
                    current_state = self.steps[this_step]._data_check(current_state)

    def _model_check(self):
        """
        Checks the model for possible errors (the model steps are checked independently)
        """
        # check that if only one surface is provided this is ok with all steps
        # check if one of the surfaces is rigid, make sure both are not rigid
        # if one is rigid it must be the second one, if this is true set self._is_rigid to true
        # check all have materials
        # check all are discrete
        # check all steps use the same number of surfaces
        pass
        # TODO

    def solve(self, output_file_name: str = None, verbose: bool = False, skip_data_check: bool = False):
        """
        Solve all steps in the model

        Parameters
        ----------
        output_file_name: str, optional (None)
            The file name of the output file (not including the extension) if None the file name defaults to the same as
            the model name set on instantiation
        verbose: bool optional (False)
            If True, logs are written to the console instead of the log file
        skip_data_check: bool, optional (False)
            If True the data check will be skipped, this is not recommended but may be necessary for some steps

        Returns
        -------

        """
        if output_file_name is None:
            if self.output_file_name is None:
                self.output_file_name = self.name + '.sdb'
        else:
            self.output_file_name = output_file_name + '.sdb'

        current_state = None
        try:
            os.remove(self.output_file_name)
        except FileNotFoundError:
            pass

        with ExitStack() as stack:
            output_file = stack.enter_context(open(self.output_file_name, 'wb+'))
            if not verbose:
                log_file = stack.enter_context(open(self.log_file_name, 'a+'))
                stack.enter_context(redirect_stdout(log_file))

            if not skip_data_check:
                self.data_check()

            for this_step in self.steps:
                print(f"Solving step {this_step}")
                # noinspection PyProtectedMember
                current_state = self.steps[this_step]._solve(current_state, output_file)

            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Analysis completed successfully at: {now}")
        return current_state

    def __repr__(self):
        return (f'ContactModel(surface1 = {repr(self.surface_1)}, '
                f'surface2 = {repr(self.surface_2)}, '
                f'steps = {repr(self.steps)})')

    def __str__(self):
        return (f'ContactModel with surfaces: {str(self.surface_1)}, {str(self.surface_2)}, '
                f'and {len(self.steps)} steps: {", ".join([str(st) for st in self.steps])}')
