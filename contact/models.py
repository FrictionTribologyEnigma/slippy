"""
model object just a container for step object that do the real work
"""
from slippy.abcs import _SurfaceABC, _FrictionModelABC, _WearModelABC, _LubricantModelABC, _AdhesionModelABC, _ContactModelABC
from slippy.contact.steps import _ModelStep, InitialStep, step
from slippy.contact.outputs import FieldOutputRequest, HistoryOutputRequest, possible_field_outpts, possible_history_outpts
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
    
    Attributes
    ----------
    
    Methods
    -------
    
    
    """
    history_outputs = {}
    field_outputs = {}
    _domains = {'all': None}
    _lubricant: _LubricantModelABC = None
    _friction: _FrictionModelABC = None
    _wear: _WearModelABC = None
    _adhesion: _AdhesionModelABC = None
    _is_rigid: bool = False
    steps: OrderedDict
    log_file_name: str = None
    output_file_name: str = None
    """Flag set to true if one of the surfaces is rigid"""

    def __init__(self, name: str, surface_1: _SurfaceABC, surface_2: _SurfaceABC = None,
                 lubricant: _LubricantModelABC = None,
                 friction: _FrictionModelABC = None, adhesion: _AdhesionModelABC = None,
                 wear_model: _WearModelABC = None, log_file_name: str = None):
        self.surface_1 = surface_1
        self.surface_2 = surface_2
        self.name = name
        self.lubricant_model = lubricant
        self.friciton_model = friction
        self.adhesion = adhesion
        self.wear_model = wear_model
        self.steps = OrderedDict({'Initial': InitialStep(self)})
        if log_file_name is None:
            log_file_name = name
        self.log_file_name = log_file_name + '.log'
        try:
            os.remove(self.log_file_name)
        except FileNotFoundError:
            pass

    @property
    def lubricant_model(self):
        return self._lubricant

    @lubricant_model.setter
    def lubricant_model(self, value):
        if issubclass(type(value), _LubricantModelABC):
            self._lubricant = value
        else:
            raise ValueError("Unable to set lubricant, expected lubricant "
                             "object recived %s" % str(type(value)))

    @lubricant_model.deleter
    def lubricant_model(self):
        # noinspection PyTypeChecker
        self._lubricant = None

    @property
    def friction_model(self):
        return self._friction

    @friction_model.setter
    def friction_model(self, value):
        if issubclass(type(value), _FrictionModelABC):
            self._friction = value
        else:
            raise ValueError("Unable to set friction model, expected "
                             "friction model object recived "
                             "%s" % str(type(value)))

    @friction_model.deleter
    def friction_model(self):
        # noinspection PyTypeChecker
        self._friction = None

    @property
    def adhesion_model(self):
        return self._adhesion

    @adhesion_model.setter
    def adhesion_model(self, value):
        if issubclass(type(value), _AdhesionModelABC):
            self._adhesion = value
        else:
            raise ValueError("Unable to set adhsion model, expected "
                             "adhesion model object recived "
                             "%s" % str(type(value)))

    @adhesion_model.deleter
    def adhesion_model(self):
        # noinspection PyTypeChecker
        self._adhesion = None

    def add_friction_model(self, friction_model_instance: typing.Optional[_FrictionModelABC] = None):
        """Add a friciton model to this instance of a contact model
        
        Parameters
        ----------
        friction_model_instance: _FrictionModel, optional (None)
            A friciton model object, if none is suplied the friciton model helper fucntion is run
            
        See Also
        --------
        friction_model
        
        Notes
        -----
        This method is an alias of friciton_model, for detailed usage 
        infromation see the documentaion of that function or pass 'info' as the
        name
        
        Examples
        --------
        >>> import numpy as np
        >>> import slippy.surface as s
        >>> surface1, surface2 = s.assurface(np.random.rand(128,128),0.01), s.assurface(np.random.rand(128,128),0.01)
        >>> # add coulomb friction to a contact model
        >>> my_model=ContactModel(surface1, surface2)
        >>> my_model.add_friction_model('coulomb', {'mu':0.3})
        """

        self.friction_model = friction_model_instance  # name, parameters)

    def add_adhesion_model(self, name: str, parameters: dict = None):
        """Add an adhesion model to this instance of a contaact model
        
        Parameters
        ----------
        name : str
            The name of the adhesion model to be added
        parameters : dict
            A dict of parameters required by the adhesion model
            
        See Also
        --------
        adhesion_model
        
        Notes
        -----
        This method is an alias of adhesion_model, for deatiled usage 
        information see the documentation for that function or pass 'info' as 
        the name into this method
        
        Examples
        --------
        
        >>> #TODO
        """
        pass

    def add_lubricant_model(self, name: str, parameters: dict):
        """Add a lubricant to this instace of a contact model
        
        Parameters
        ----------
        name : str
            The name of the lubricant model to be used 
        parameters : dict
            A dict of parameters required by the lubricant model
        
        See Also
        --------
        lubricant_model
        
        Notes
        -----
        This method is an alias of lubricant_model, for deatiled usage 
        information see the documentation for that function or pass 'info' as 
        the name into this method
        
        Examples
        --------
        
        >>> #TODO
        """
        pass

    def add_step(self, step_name: str, *, step_instance: _ModelStep = None,
                 position: {int, str}=None):
        """ Adds a solution stepe to the current model
        
        Parameters
        ----------
        step_instance: _ModelStep
            An instance of a model step
        step_name : str
            A unique name given to the step, used to specify when outputs 
            should be recorded
        position : {int, 'last'}, optional ('last')
            The position of the step in the existing order
        
        See Also
        --------
        step
        
        Notes
        -----
        Steps should only be added to the model using this method 
        #TODO detailed descriptiion of inputs
        
        Examples
        --------
        >>> #TODO
        """
        # note to self
        """
        New steps should be bare bones, all the relavent information should be
        added to the steps at the data check stage, essentially there should be 
        nothing you can do to make a nontrivial error here.
        """
        if step_instance is None:
            new_step = step(self)
        else:
            if step_instance.model is not self:
                raise ValueError("The step instance should be made by passing this model to the constructor")
            new_step = step_instance

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
        Adds a field output reques to the model

        Parameters
        ----------
        name : str
            The name of the output request
        domain : {{str, Sequence}}
            The name of the node set or the node set to be used, node sets for 'all', 'surface_1' and 'surface_2' are
            created auomatically
        step_name : str
            The name of the step that the field output is to be taken from, ues 'all' for all steps
        time_points : Sequence
            The time points for the field output. If the output is only required at the start of the step use (0,) if it
            is only required at the end of the step used (None,), otherwise pass a sequence of time points or a slice
            object.
        output : Sequence[str]
            Names of output parrameters to be included in the request, for more information see the documentaion of
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
        # check that name is str (dosn't start with _)
        if type(name) is not str:
            raise TypeError(f"Field output name should be string, not {type(name)}")
        elif name.startswith('_'):
            raise ValueError("Field output names cannot start with _")
        # check that step exists
        if step_name not in self.steps and step_name != 'all':
            raise ValueError(f"Step name {step_name} not found.")
        # check that all outpust are valid
        out_in = [o in possible_field_outpts for o in output]
        if not all(out_in):
            raise ValueError(f"Unrecognised output request: {output[out_in.index(False)]}, valid outputs are: "
                             f"{', '.join(possible_field_outpts)}")

        output_dict = {key: True for key in output}

        self.field_outputs[name] = FieldOutputRequest(domain=domain, step=step_name, time_points=time_points,
                                                      **output_dict)

    def add_history_output(self, name: str, step_name: str, time_points: typing.Sequence, output: typing.Sequence[str]):
        f"""
        Adds a field output reques to the model

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
            Names of output parrameters to be included in the request, for more information see the documentaion of
            HistoryOutputRequest

        See Also
        --------
        HistoryOutputRequest

        Notes
        -----
        Valid outputs are {', '.join(possible_history_outpts)}
        """
        # check that name is str (dosn't start with _)
        if type(name) is not str:
            raise TypeError(f"History output name should be string, not {type(name)}")
        elif name.startswith('_'):
            raise ValueError("History output names cannot start with _")
        # check that step exists
        if step_name not in self.steps and step_name != 'all':
            raise ValueError(f"Step name {step_name} not found.")
        # check that all outpust are valid
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

                for this_step in self.steps:
                    self.steps[this_step]._data_check()

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

    def solve(self, output_file_name: str = None, verbose: bool = False):
        if output_file_name is None:
            if self.output_file_name is None:
                self.output_file_name = self.name + 'sdb'
        else:
            self.output_file_name = output_file_name + 'sdb'

        current_state = None
        try:
            os.remove(self.output_file_name)
        except FileNotFoundError:
            pass

        with ExitStack() as stack:
            output_file = stack.enter_context(open(self.output_file_name, 'wb+'))
            if not verbose:
                log_file = stack.enter_context(open(output_file_name, 'a+'))
                stack.enter_context(redirect_stdout(log_file))

            self.data_check()

            for this_step in self.steps:
                this_step._solve(current_state, log_file, output_file)

            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Analysis completed sucessfully at: {now}")

    def __repr__(self):
        return (f'ContactModel(surface1 = {repr(self.surface_1)}, '
                f'surface2 = {repr(self.surface_2)}, '
                f'steps = {repr(self.steps)})')

    def __str__(self):
        return (f'ContactModel with surfaces: {str(self.surface_1)}, {str(self.surface_2)}, '
                f'and {len(self.steps)} steps: {", ".join([str(st) for st in self.steps])}')
