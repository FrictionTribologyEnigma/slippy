"""
A class for containing information about lubricants, allows the user to set sub models for the density etc,
should supply whatever is needed for the reynolds solver and the sub models/ other models in the lubricant
"""
import inspect
import typing
import warnings
from collections import OrderedDict

from slippy.abcs import _LubricantModelABC
from .lubricant_models import __all__ as built_in_models
from .lubricant_models import constant_array_property

__all__ = ['Lubricant']


class Lubricant(_LubricantModelABC):
    """A class for describing lubricant behaviours

    Parameters
    ----------
    name: str
        The name of the lubricant, used for output and error messages
    models: OrderedDict
        The sub models to solve
    constant_viscosity: float, optional (None)
        The viscosity, should only be supplied if the viscosity is constant, otherwise an appropriate model should be
        used. This adds both viscosity and nd_viscosity models.
    constant_density: float, optional (None)
        The density, should only be supplied if the density is constant, otherwise an appropriate model should be used.
        This adds both density and nd_density models

    Attributes
    ----------
    sub_models: OrderedDict
        The sub models which will be executed on each iteration of the reynolds solver
    built_in_models: list
        A list of all the named sub models built in to slippy, these are all available in slippy.contact.*model_name*

    Methods
    -------

    Notes
    -----

    Examples
    --------
    """

    built_in_models = tuple(built_in_models)

    def __init__(self, name: str, models: OrderedDict = None,
                 constant_viscosity: float = None, constant_density: float = None):

        self.name = name
        self.sub_models = OrderedDict()

        if constant_viscosity is not None:
            self.add_sub_model('viscosity', constant_array_property(constant_viscosity))
            self.add_sub_model('nd_viscosity', constant_array_property(1.0))
        if constant_density is not None:
            self.add_sub_model('density', constant_array_property(constant_density))
            self.add_sub_model('nd_density', constant_array_property(1.0))

        if models is not None:
            try:
                for param, func in models.items():
                    self.add_sub_model(param, func)
            except AttributeError:
                raise TypeError('Models should be ordered dict of callable objects')
            except TypeError:
                raise TypeError('Models should be ordered dict of callable objects')

    def add_sub_model(self, parameter: str, function: typing.Callable):
        """
        Adds a lubricant sub model to be solved on each iteration of the solver

        Parameters
        ----------
        parameter: str
            The name of the parameter found by the sub model, each sub model can only find one parameter
        function: callable
            The model function which will be called on each iteration of the solver. See notes for calling details

        Notes
        -----
        The function will be called by passing the current model state dictionary to it as key word arguments. This
        means that all models should:
        - Accept the required keyword arguments
        - Have no positional arguments
        - Accept un used keyword arguments as **kwargs

        The function can find any value but typically it will out put either a single value or a numpy array of values.

        No additional parameters can be passed to the function when it is called (such as coefficients for a particular
        fluid). This sort of parametrisation can be applied by passing a closure as the callable.

        A list of the named sub models built in to slippy is available as the built_in_models attribute of this class.

        Sub models will execute in the order they are added to this dict.

        Examples
        --------
        #TODO
        """
        # check that parameter is a string, with no spaces etc.
        assert isinstance(parameter, str), 'Parameter name must be a string'
        assert parameter.isidentifier(), 'Parameter name must be a valid variable name'
        # check that function is callable
        assert callable(function), 'Function must be callable'
        # check that function has ** kwargs argument
        full_arg_spec = inspect.getfullargspec(function)
        assert full_arg_spec.varkw is not None, 'Function must accept a variable number of keyword arguments (**kwargs)'
        # check that function has no positional only arguments

        # warn if collision with existing sub model
        if parameter in self.sub_models:
            warnings.warn(f"Parameter {parameter} is already in sub models, this action has replaced the existing "
                          f"sub model")
        # Add function to sub_models ordered dict
        self.sub_models[parameter] = function

    def data_check(self, current_state: set) -> set:
        """Use introspection to check that all sub models will solve properly

        Parameters
        ----------
        current_state: set
            The state at the end of one iteration of the reynolds solver

        Returns
        -------
        current_state: set
            The updated current state with variables from the sub models

        """
        for key, value in self.sub_models.items():
            full_arg_spec = inspect.getfullargspec(value)
            args = full_arg_spec.args
            for arg in args:
                if arg not in current_state and arg != 'self':
                    raise ValueError(f"Lubricant sub model {key} will not solve, requires {arg}, current state is "
                                     f"{current_state}")
            current_state.add(key)

        return current_state

    def solve_sub_models(self, current_state: dict) -> dict:
        """

        Parameters
        ----------
        current_state: dict
            The current state of the model

        Returns
        -------
        previous_state

        """
        for parameter, func in self.sub_models.items():
            current_state[parameter] = func(**current_state)
        return current_state
