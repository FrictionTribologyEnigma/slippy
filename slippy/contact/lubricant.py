"""
A class for containing information about lubricants, should also solve it's own reynolds equation
"""
from abc import abstractmethod
from slippy.abcs import _LubricantModelABC

__all__ = ['LubricantABC', 'NewtonianLubricant']

class LubricantABC(_LubricantModelABC):
    """Just the basics here, what does a lubricant have to implement?

    It must implement a viscosity, density functions that will either return a scalar or a field of the same size as the
     input, the input will contain the ..... pressure and film thickness at each point.

    """

    @abstractmethod
    def density(self, pressure, film_thickness):
        pass

    @abstractmethod
    def viscosity(self, pressure, film_thickness):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class NewtonianLubricant(LubricantABC):
    """A simple newtonian lubricant with a constant density and viscosity

    """

    def __init__(self, viscosity, density):
        self._density = density
        self._viscosity = viscosity

    def density(self, pressure, film_thickness):
        return self._density

    def viscosity(self, pressure, film_thickness):
        return self._viscosity

    def __repr__(self):
        return f'NewtonianLubricant(viscosity={self._viscosity}, density={self._density}'

