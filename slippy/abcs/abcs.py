"""
Minimal abstract base classes please don't add any code to these, they are strictly to avoid circular imports,
this module should be as minimal as possible as it will be imported every time any sub package is imported
"""

import abc
import inspect

__all__ = ['_SurfaceABC', '_AdhesionModelABC', '_MaterialABC', '_StepABC', '_FrictionModelABC', '_WearModelABC',
           '_ACFABC', '_LubricantModelABC', '_ContactModelABC', '_ReynoldsSolverABC',
           '_NonDimensionalReynoldSolverABC', '_SubModelABC']


class _LubricantModelABC(abc.ABC):
    pass


class _MaterialABC(abc.ABC):

    @abc.abstractmethod
    def loads_from_surface_displacement(self, displacements, grid_spacing: float,
                                        other: '_MaterialABC', **material_options):
        pass

    @abc.abstractmethod
    def displacement_from_surface_loads(self, loads, grid_spacing: float,
                                        other: '_MaterialABC', **material_options):
        pass


class _SurfaceABC(abc.ABC):
    profile = None
    grid_spacing: float
    material: _MaterialABC
    pass


class _AdhesionModelABC(abc.ABC):

    @abc.abstractmethod
    def __call__(self, surface_loads, deformed_gap, contact_nodes, model):
        pass

    def __bool__(self):
        return True


class _ContactModelABC(abc.ABC):
    surface_1: _SurfaceABC
    surface_2: _SurfaceABC
    _lubricant: _LubricantModelABC = None

    @property
    def lubricant_model(self):
        return self._lubricant

    @lubricant_model.setter
    def lubricant_model(self, value):
        if issubclass(type(value), _LubricantModelABC):
            self._lubricant = value
        else:
            raise ValueError("Unable to set lubricant, expected lubricant "
                             "object, received %s" % str(type(value)))

    @lubricant_model.deleter
    def lubricant_model(self):
        # noinspection PyTypeChecker
        self._lubricant = None


class _StepABC(abc.ABC):
    pass


class _FrictionModelABC(abc.ABC):
    pass


class _WearModelABC(abc.ABC):
    pass


class _ACFABC(abc.ABC):
    pass


class _ReynoldsSolverABC(abc.ABC):

    @abc.abstractmethod
    def solve(self, previous_state: dict) -> dict:
        pass

    @abc.abstractmethod
    def data_check(self, previous_state: set) -> set:
        pass


class _NonDimensionalReynoldSolverABC(_ReynoldsSolverABC):
    @abc.abstractmethod
    def dimensionalise_pressure(self, nd_pressure, un_dimensionalise: bool = False):
        pass

    @abc.abstractmethod
    def dimensionalise_viscosity(self, nd_viscosity, un_dimensionalise: bool = False):
        pass

    @abc.abstractmethod
    def dimensionalise_density(self, nd_density, un_dimensionalise: bool = False):
        pass

    @abc.abstractmethod
    def dimensionalise_gap(self, nd_gap, un_dimensionalise: bool = False):
        pass

    @abc.abstractmethod
    def dimensionalise_length(self, nd_length, un_dimensionalise: bool = False):
        pass


class _SubModelABC(abc.ABC):
    name: str
    provides: set

    def __init__(self, name: str, provides: set):
        self.name = name
        self.provides = provides

    def solve(self, **kwargs) -> dict:
        """Solve the sub model

        Parameters
        ----------
        kwargs
            The solve method must take a variable number of keyword arguments

        Returns
        -------
        dict
            dict of found parameters

        """

    def __init_subclass__(cls, is_abstract=False, **kwargs):
        if not is_abstract:
            full_arg_spec = inspect.getfullargspec(cls.solve)
            assert full_arg_spec.varkw is not None, "Sub model solve method must take a variable number of keywords"
