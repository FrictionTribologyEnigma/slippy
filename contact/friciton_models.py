from abc import ABC, abstractmethod

__all__ = ['_FrictionModel', 'friction_model']


class _FrictionModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def friction_force(self):
        raise NotImplementedError("This method must be overwritten")

def friction_model():
    pass
