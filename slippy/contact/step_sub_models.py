import abc


class _SubModel(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def solve(self, **kwargs):
        pass
