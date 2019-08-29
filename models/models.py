"""
model object just a container for step object that do the real work
"""
from slippy.lubrication import _LubricantModel, lubricant_model
from slippy.contact import _WearModel
from slippy.surface import Surface
from contact.friciton_models import _FrictionModel, friction_model
from contact.adhesion_models import _AdhesionModel, adhesion_model
from models.steps import _ModelStep, _InitialStep, step
from contact.outputs import FieldOutputRequest, HistoryOutputRequest, possible_field_outpts, possible_history_outpts
from datetime import datetime
from typing import Sequence
from collections import OrderedDict

__all__ = ["ContactModel"]


class ContactModel(object):
    """ A container for contact mechanics problems
    
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
    _lubricant: _LubricantModel = None
    _friction: _FrictionModel = None
    _wear: _WearModel = None
    _adhesion: _AdhesionModel = None
    _is_rigid: bool = False
    """Flag set to true if one of the surfaces is rigid"""

    def __init__(self, surface_1: Surface, surface_2: Surface, lubricant: _LubricantModel = None,
                 friction: _FrictionModel = None, adhesion: _AdhesionModel = None,
                 wear_model: _WearModel = None, steps: OrderedDict[_ModelStep] = None):
        self.surface_1 = surface_1
        self.surface_2 = surface_2

        self.lubricant_model = lubricant
        self.friciton_model = friction
        self.adhesion = adhesion
        self.wear_model = wear_model
        self.steps = OrderedDict({'Initial': _InitialStep(self)})

        if steps is not None:
            if isinstance(steps, OrderedDict) and all([isinstance(st, _ModelStep) for st in steps.values()]):
                self.steps = steps
            else:
                if isinstance(steps, OrderedDict):
                    raise TypeError("Not all steps in step dict are ModelSteps, typically it is easier to make a new "
                                    "model object and populate using the new_step method")
                else:
                    raise TypeError("Steps keyword argument is not supported type, expected OrderedDict, got: "
                                    f"{type(steps)}")

    @property
    def lubricant_model(self):
        return self._lubricant

    @lubricant_model.setter
    def lubricant_model(self, value):
        if issubclass(type(value), _LubricantModel):
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
        if issubclass(type(value), _FrictionModel):
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
        if issubclass(type(value), _AdhesionModel):
            self._adhesion = value
        else:
            raise ValueError("Unable to set adhsion model, expected "
                             "adhesion model object recived "
                             "%s" % str(type(value)))

    @adhesion_model.deleter
    def adhesion_model(self):
        # noinspection PyTypeChecker
        self._adhesion = None

    def add_friction_model(self, name: str, parameters: dict = None):
        """Add a friciton model to this instance of a contact model
        
        Parameters
        ----------
        name : str
            The name of the friciton modle to be added e.g. coloumb
        parameters : dict
            A dict of the parametes required by the specified friction model
            
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
        self.friction_model = friction_model()  # name, parameters)

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

        self.adhesion_model = adhesion_model()  # name, parameters)

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
        self.lubricant_model = lubricant_model(name, parameters)

    def add_step(self, step_name: str, step_type: {str, _ModelStep}, step_parameters: dict = None,
                 position: {int, str}=None):
        """ Adds a solution stepe to the current model
        
        Parameters
        ----------
        step_name : str
            A unique name given to the step, used to specify when outputs 
            should be recorded
        step_type : {str, _ModelStep}
            The type of step to be added, or a step object to be added to the model
        step_parameters : dict
            A dict containing the parameters required by the step
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

        new_step = step(step_type, **step_parameters)
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

    def add_field_output(self, name: str, domain: {str, Sequence}, step_name: str, time_points: Sequence,
                         output: Sequence[str]):
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

    def add_history_output(self, name: str, step_name: str, time_points: Sequence, output: Sequence[str]):
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
        self._model_check()

        for this_step in self.steps:
            if this_step == 'Initial':
                pass
            else:
                self.steps[this_step]._data_check()

    def _model_check(self):
        """
        Checks the model for possible errors (only the model steps are checked independently
        """
        # check that if only one surface is provided this is ok with all steps
        # check if one of the surfaces is rigid, make sure both are not rigid
        # if one is rigid it must be the second one, if this is true set self._is_rigid to true
        # check all have materials
        # check all are discrete
        pass

    def solve(self, output_file_name: str = 'output'):
        current_state = None

        output_file = open(output_file_name + '.sdb', 'wb')
        log_file = open(output_file_name + '.log', 'w')

        self.data_check()

        for this_step in self.steps:
            this_step._solve(current_state, log_file, output_file)

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f"Analysis completed sucessfully at: {now}")

        output_file.close()
        log_file.close()

    def __repr__(self):
        return (f'ContactModel(surface1 = {type(self.surface_1)} at {id(self.surface_1)}, '
                f'surface2 = {type(self.surface_2)} at {id(self.surface_2)}), '
                f'steps = {self.steps.__repr__()}')

    def __str__(self):
        return (f'ContactModel with surfaces: {self.surface_1.__str__()}, {self.surface_2.__str__()}, '
                f'and {len(self.steps)} steps: {", ".join([st.__str__() for st in self.steps])}')
