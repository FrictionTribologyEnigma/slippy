"""
Minimal abstract base classes please don't add any code to these, they are strictly to avoid circular imports,
this module should be as minimal as possible as it will be imported every time any sub package is imported
"""

import abc

__all__ = ['_SurfaceABC', '_AdhesionModelABC', '_MaterialABC', '_StepABC', '_FrictionModelABC', '_WearModelABC',
           '_ACFABC', '_LubricantModelABC', '_ContactModelABC', '_ReynoldsSolverABC',
           '_NonDimensionalReynoldSolverABC', '_SubModelABC']


class _LubricantModelABC(abc.ABC):
    pass


class _MaterialABC(abc.ABC):

    @abc.abstractmethod
    def loads_from_surface_displacement(self, displacements_z, grid_spacing: float,
                                        other: '_MaterialABC', current_state: dict, **material_options):
        pass

    @abc.abstractmethod
    def displacement_from_surface_loads(self, loads_z, grid_spacing: float,
                                        other: '_MaterialABC', current_state: dict, **material_options):
        pass


class _SurfaceABC(abc.ABC):
    profile = None
    grid_spacing: float
    material: _MaterialABC
    pass

    def wear(self, name, x_pts, y_pts, depth):
        pass


class _AdhesionModelABC(abc.ABC):

    @abc.abstractmethod
    def energy_gradient(self, gap):
        pass

    def __call__(self, gap):
        return self.energy_gradient(gap)

    def __bool__(self):
        return True


class _StepABC(abc.ABC):
    max_time: float
    pass


class _ContactModelABC(abc.ABC):
    surface_1: _SurfaceABC
    surface_2: _SurfaceABC
    current_step: _StepABC
    current_step_start_time: float

    _lubricant: _LubricantModelABC = None

    @property
    def lubricant_model(self):
        return self._lubricant

    @lubricant_model.setter
    def lubricant_model(self, value):
        if issubclass(type(value), _LubricantModelABC) or value is None:
            self._lubricant = value
        else:
            raise ValueError("Unable to set lubricant, expected lubricant "
                             "object, received %s" % str(type(value)))

    @lubricant_model.deleter
    def lubricant_model(self):
        # noinspection PyTypeChecker
        self._lubricant = None


class _FrictionModelABC(abc.ABC):
    pass


class _WearModelABC(abc.ABC):
    pass


class _ACFABC(abc.ABC):
    pass


class _ReynoldsSolverABC(abc.ABC):

    @abc.abstractmethod
    def solve(self, previous_state: dict, max_pressure: float) -> dict:
        pass

    @abc.abstractmethod
    def data_check(self, previous_state: set) -> set:
        pass


class _NonDimensionalReynoldSolverABC(_ReynoldsSolverABC):
    provides: set
    requires: set
    rolling_speed: float

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
    requires: set
    provides: set
    model: _ContactModelABC = None
    no_time: bool = False

    def __init__(self, name: str, requires: set, provides: set):
        if isinstance(name, str):
            self.name = name
        else:
            raise ValueError(f"Name of sub model must be a string, received: {type(name)}")
        if isinstance(requires, set):
            self.requires = requires
        else:
            raise ValueError(f"Requires property must be a set, received: {type(requires)}")
        if isinstance(provides, set):
            self.provides = provides
        else:
            raise ValueError(f"Requires property must be a set, received: {type(provides)}")

    @abc.abstractmethod
    def solve(self, current_state: dict) -> dict:
        """Solve the sub model

        Parameters
        ----------
        current_state: dict
            The current model state

        Returns
        -------
        dict
            dict of found parameters, current state will be updated with these after running the model

        """
