from abc import ABC, abstractmethod

__all__ = ['_AdhesionModel', 'adhesion_model']


class _AdhesionModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def adhesion_force(self):
        raise NotImplementedError("This method must be over written")


def adhesion_model():
    pass
