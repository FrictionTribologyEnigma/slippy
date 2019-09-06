"""
Minimal abstract base classes please don't add any code to these, they are strictly to avoid circular imports,
this module should be as minimal as possible as it will be imported every time any sub package is imported
"""
import abc

__all__ = ['_SurfaceABC', '_AdhesionModelABC', '_MaterialABC', '_StepABC', '_FrictionModelABC', '_WearModelABC',
           '_ACFABC', '_LubricantModelABC', '_ContactModelABC']


class _LubricantModelABC(abc.ABC):
    pass


class _ContactModelABC(abc.ABC):
    pass


class _SurfaceABC(abc.ABC):
    pass


class _MaterialABC(abc.ABC):
    pass


class _StepABC(abc.ABC):
    pass


class _AdhesionModelABC(abc.ABC):
    pass


class _FrictionModelABC(abc.ABC):
    pass


class _WearModelABC(abc.ABC):
    pass


class _ACFABC(abc.ABC):
    pass
