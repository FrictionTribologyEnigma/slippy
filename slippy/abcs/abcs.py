"""
Minimal abstract base classes please don't add any code to these, they are strictly to avoid circular imports,
this module should be as minimal as possible as it will be imported every time any sub package is imported
"""

import abc

__all__ = ['_SurfaceABC', '_AdhesionModelABC', '_MaterialABC', '_StepABC', '_FrictionModelABC', '_WearModelABC',
           '_ACFABC', '_LubricantModelABC', '_ContactModelABC']


class _LubricantModelABC(abc.ABC):
    pass


class _MaterialABC(abc.ABC):

    @abc.abstractmethod
    def loads_from_surface_displacement(self, displacements, grid_spacing: float,
                                        other: '_MaterialABC', **material_options):
        pass


class _SurfaceABC(abc.ABC):
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
    _adhesion: _AdhesionModelABC
    pass


class _StepABC(abc.ABC):
    pass


class _FrictionModelABC(abc.ABC):
    pass


class _WearModelABC(abc.ABC):
    pass


class _ACFABC(abc.ABC):
    pass