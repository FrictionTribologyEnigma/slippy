import numpy as np
from scipy.misc import derivative
from functools import partial
from typing import Callable, Optional
from slippy.core import _AdhesionModelABC

__all__ = ['MDAdhesionPotential', 'AdhesionModelFromFunction']


class MDAdhesionPotential(_AdhesionModelABC):
    """
    The Maugis-Dugdale adhesive potential (constant force gamma/rho when gap is smaller than rho)

    Parameters
    ----------
    rho: float
        The range of the potential
    gamma: float
        The adhesive force when in range is gamma/rho
    """
    def __init__(self, rho: float, gamma: float):
        self.gamma = gamma
        self.rho = rho

    def energy(self, gap: np.ndarray):
        return -self.gamma*np.sum(1-gap/self.rho) * (gap <= self.rho).astype(float)

    def energy_gradient(self, gap: np.ndarray):
        return self.gamma / self.rho * (gap <= self.rho).astype(float)


class AdhesionModelFromFunction(_AdhesionModelABC):
    """ Generates and adhesion model object from energy or energy gradient functions

    Parameters
    ----------
    energy_function: Callable, optional (None)
        The energy function giving the adhesive energy as a function of the gap, if only this function is supplied
        the derivative will be found by numerical differentiation by the central difference approximation
    energy_gradient_function: Callable, optional (None)
        The adhesive force as a function of the gap
    dx: float, optional (1e-8)
        The spacing of the points for numerical differentiation, only used if energy_gradient_function is None
    order: int, optional (3)
        The number of points for numerical differentiation must be odd, only used if energy_gradient_function is None

    """
    def __init__(self, energy_function: Optional[Callable] = None, energy_gradient_function: Optional[Callable] = None,
                 dx: float = 1e-8, order: int = 3):
        if (energy_function is None) and (energy_gradient_function is None):
            raise ValueError("Either the energy function or the energy gradient function must be set")
        self._energy_function = energy_function
        if energy_gradient_function is None:
            energy_gradient_function = partial(derivative, energy_function, dx=dx, order=order)
        self._energy_gradient_function = energy_gradient_function

    def energy_gradient(self, gap):
        return self._energy_gradient_function(gap)

    def energy(self, gap):
        return self._energy_function(gap)
