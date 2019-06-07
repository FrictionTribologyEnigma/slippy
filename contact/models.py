"""
model object
"""
__all__=["ContactModel"]
from slippy.lubrication import _LubricantModel, lubricant_model
from slippy.contact import _FrictionModel, _AdhesionModel
from slipy.contact import friction_model, adhesion_model, step
from datetime import datetime

class ContactModel(object):
    """ A container for contact mechanics problems
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    Methods
    -------
    
    
    """
    steps=[]
    surface_1=None
    surface_2=None
    sets={}
    _lubricant=None
    _friction=None
    _adhesion=None
    _wear=None
    historyOutputs={}
    fieldOutputs={}
    
    def __init__(self, surface_1, surface_2, lubricant=None, 
                 frition=None, adhesion=None, wear=None,
                 outputs=None):
        self.surface_1=surface_1
        self.surface_2=surface_2
        
        if lubricant is not None:
            self.lubricant=lubricant
        pass
    
    @property
    def lubricant_model(self):
        return self._lubricant
    @lubricant_model.setter
    def lubricant_model(self, value):
        if issubclass(type(value),_LubricantModel):
            self._lubricant=value
        else:
            raise ValueError("Unable to set lubricant, expected lubricant "
                             "object recived %s" % str(type(value)))
    @lubricant_model.deleter
    def lubricant_model(self):
        self._lubricant=None
    
    @property
    def friction_model(self):
        return self._friction
    @friction_model.setter
    def friction_model(self, value):
        if issubclass(type(value),_FrictionModel):
            self._friction=value
        else:
            raise ValueError("Unable to set friction model, expected "
                             "friction model object recived "
                             "%s" % str(type(value)))
    @friction_model.deleter
    def friction_model(self):
        self._friction=None
        
    @property
    def adhesion_model(self):
        return self._adhesion
    @adhesion_model.setter
    def adhesion_model(self, value):
        if issubclass(type(value),_AdhesionModel):
            self._adhesion=value
        else:
            raise ValueError("Unable to set adhsion model, expected "
                             "adhesion model object recived "
                             "%s" % str(type(value)))
    @adhesion_model.deleter
    def adhesion_model(self):
        self._adhesion=None
    
    
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
        
        >>> # add coulomb friction to a contact model
        >>> my_model=ContactModel(surface1, surface2)
        >>> my_model.add_friciton_model('coulomb', {'mu':0.3})
        """
        self.friction_model=friction_model(name, parameters)
    
    def add_adhesion_model(self, name: str, parameters: dict= None):
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
        
        self.adhesion_model=adhesion_model(name, parameters)
    
    def add_lubricant_model(self, name:str, parameters: dict):
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
        self.lubricant_model=lubricant_model(name, parameters)
    
    def add_step(self, step_name:str, step_type:str, step_parameters:dict, 
                 position:int='last'):
        """ Adds a solution stepe to the current model
        
        Parameters
        ----------
        step_name : str
            A unique name given to the step, used to specify when outputs 
            should be recorded
        step_type : str
            The type of step to be added #TODO
        step_parameters : dict
            A dict containing the parameters required by the step
        positon : {int, 'last'}, optional ('last')
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
        new_step=step(step_type, step_parameters)
        if position=='last':
            self.steps.append(new_step)
        else:
            self.steps.insert(position, new_step)
    
    def add_field_output(self):
        """
        """
        #notes
        """
        Field output should be a dict, of field output objects, steps should 
        be named or all
        """
        
        pass
    
    def add_history_output(self):
        pass
    
    def data_check(self):
        pass
    
    def solve(self, output_file_name:str = 'output'):
        current_state=None
        
        output_file=open(output_file_name+'.sdb', 'w')
        log_file=open(output_file_name+'.log','w')
        
        self.data_check(log_file)
        
        for this_step in self.steps:
            this_step.solve(current_state, output_file, log_file)
            
        now=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f"Analysis completed sucessfully at: {now}")
        
        output_file.close()
        log_file.close()