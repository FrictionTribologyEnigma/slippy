"""
model object, just a container for step object that do the real work
"""
import os
import typing
import slippy
import warnings
from collections import OrderedDict
from contextlib import redirect_stdout, ExitStack
from datetime import datetime
from slippy.core.outputs import OutputSaver, OutputRequest

from slippy.core import _SurfaceABC, _LubricantModelABC, _ContactModelABC
from slippy.contact.steps import _ModelStep, InitialStep

__all__ = ['ContactModel']


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
    output_dir: str, optional (None)
        Path to an output directory can be relative or absolute, slippy will attempt to make directory if it does not
        exist, defaults to slippy.OUTPUT_DIR, which defaults to the current working directory.

    Attributes
    ----------
    steps: OrderedDict
        The model steps in the order they will be solved in, each value will be a ModelStep object.
    surface_1, surface_2: _SurfaceABC
        Surface objects of the two model surfaces
    lubricant_model: _LubricantModelABC
        A lubricant model

    Methods
    -------
    add_step
        Adds a step object to the model
    add_output
        Adds an output request to the model or selected steps
    data_check
        Performs analysis checks for each of the steps and the model as a whole, prints the results to the log file
    solve
        Solves all of the model steps in sequence, writing history and field outputs to the output file. Writes progress
        to the log file

    """

    """Flag set to true if one of the surfaces is rigid"""
    _is_rigid: bool = False
    steps: OrderedDict
    adhesion = None
    _current_state_debug: dict = None
    _all_step_outputs = []

    def __init__(self, name: str, surface_1: _SurfaceABC, surface_2: _SurfaceABC = None,
                 lubricant: _LubricantModelABC = None, output_dir: str = None):
        self.surface_1 = surface_1
        self.surface_2 = surface_2
        self.name = name
        if lubricant is not None:
            self.lubricant_model = lubricant
        self.steps = OrderedDict({'Initial': InitialStep()})
        self.current_step = None
        self.current_step_start_time = None
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                try:
                    os.mkdir(output_dir)
                except OSError:
                    raise ValueError("Output directory not found and creation of the output "
                                     "directory: %s failed" % output_dir)
            slippy.OUTPUT_DIR = output_dir

    @property
    def log_file_name(self):
        return os.path.join(slippy.OUTPUT_DIR, self.name + '.log')

    def add_step(self, step_instance: _ModelStep = None, position: typing.Union[int, str] = None):
        """ Adds a solution step to the current model

        Parameters
        ----------
        step_instance: _ModelStep
            An instance of a model step
        position : {int, None}, optional (None)
            The position of the step in the existing order, if None will add step to the end

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
        for sub_model in step_instance.sub_models:
            sub_model.model = self

    def add_output(self, output_request: OutputRequest, active_steps: typing.Union[str, typing.Sequence[str]] = 'all'):
        """
        Add an output to one or more steps in the model

        Parameters
        ----------
        output_request: OutputRequest
            A slippy.contact.OutputRequest with describing the parameters to save and the time points they will be saved
            for
        active_steps: str or list of str, optional ('all')
            The step or a list of steps this output is active for, defaults to all the steps currently in the model

        Examples
        --------

        """
        if active_steps == 'all':
            active_steps = set(self.steps)
            active_steps.remove('Initial')
        if isinstance(active_steps, str):
            active_steps = [active_steps, ]
        for this_step in active_steps:
            self.steps[this_step].outputs.append(output_request)

    def data_check(self):
        print("Data check started at:")
        print(datetime.now().strftime('%H:%M:%S %d-%m-%Y'))
        print(f"Checking model {self.name}:")
        self._model_check()

        current_state = None

        for this_step in self.steps:
            current_state = self.steps[this_step]._data_check(current_state)

    def _model_check(self):
        """
        Checks the model for possible errors (the model steps are checked independently)
        """
        def warn_or_error(msg):
            if slippy.ERROR_IN_DATA_CHECK:
                raise ValueError(msg)
            else:
                warnings.warn(msg)

        if self.surface_1 is None:
            warn_or_error("Master surface is not set")
        if not self.surface_1.is_discrete:
            warn_or_error("Master is not discrete, only secondary surface may be analytic")
        if self.surface_1.material is None:
            warn_or_error("Material for master surface is not set")
        if self.surface_2 is None:
            warn_or_error("Secondary surface is not set")
        if self.surface_2.material is None:
            warn_or_error("Material for second surface is not set, for a rigid surface use the Rigid material")

    def solve(self, verbose: bool = False, skip_data_check: bool = False):
        """
        Solve all steps and sub-models in the model as well as writing all outputs

        Parameters
        ----------
        verbose: bool optional (False)
            If True, logs are written to the console instead of the log file
        skip_data_check: bool, optional (False)
            If True the data check will be skipped, this is not recommended but may be necessary for some steps

        Returns
        -------
        current_state: dict
            A dictionary containing the final state of the model

        Notes
        -----
        Most steps produce detailed logging information that can be found in the log file.

        """
        if os.path.exists(self.log_file_name):
            os.remove(self.log_file_name)

        current_state = {'time': 0.0}

        with ExitStack() as stack:
            output_writer = stack.enter_context(OutputSaver(self.name))
            if not verbose:
                log_file = stack.enter_context(open(self.log_file_name, 'x'))
                stack.enter_context(redirect_stdout(log_file))

            if not skip_data_check:
                self.data_check()

            print(f"Solving model {self.name}, CUDA = {slippy.CUDA}")

            for this_step in self.steps:
                print(f"Solving step {this_step}")
                self.current_step = self.steps[this_step]
                self.current_step_start_time = current_state['time']
                for output in self.steps[this_step].outputs:
                    output.new_step(current_state['time'])
                current_state = self.steps[this_step].solve(current_state, output_writer)

            now = datetime.now().strftime('%H:%M:%S %d-%m-%Y')
            print(f"Analysis completed successfully at: {now}")

        return current_state

    def __repr__(self):
        return (f'ContactModel(surface1 = {repr(self.surface_1)}, '
                f'surface2 = {repr(self.surface_2)}, '
                f'steps = {repr(self.steps)})')

    def __str__(self):
        return (f'ContactModel with surfaces: {str(self.surface_1)}, {str(self.surface_2)}, '
                f'and {len(self.steps)} steps: {", ".join([str(st) for st in self.steps])}')
