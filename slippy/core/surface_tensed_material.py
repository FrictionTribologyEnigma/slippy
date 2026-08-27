import typing
import numpy as np

from .frequency_domain_material import _FrequencyDomainMaterial

__all__ = ['SurfaceTensedMaterial']


class SurfaceTensedMaterial(_FrequencyDomainMaterial):
    """An isotropic elastic half space with a constant surface tension acting on its surface

    The surface tension resists the curvature of the deformed surface, which stiffens the
    response at wavelengths shorter than the elastocapillary length s = 2 * tau_0 / E*. The
    frequency response function for normal loading is:

        C(q) = 2 / (E* * q * (1 + s * q))

    which recovers the isotropic elastic half space as tau_0 -> 0. Only normal ('zz') loading
    is implemented.

    Parameters
    ----------
    name: str
        The name of the material, must be unique
    modulus: float
        The Young's modulus of the substrate
    p_ratio: float
        The Poisson's ratio of the substrate
    tau_0: float
        The surface tension (force per unit length)
    max_load: float, optional (inf)
        The maximum load supported, loads above this cause perfectly plastic deformation in the
        solvers which support it
    periodic_im_repeats: tuple, optional (1, 1)
        See _IMMaterial

    Notes
    -----
    Parameters must not be changed after construction, the influence matrix is memoized.

    References
    ----------
    Hajji, M. A. (1978). Indentation of a Membrane on an Elastic Half Space.
    Journal of Applied Mechanics, 45(2), 320-324.
    """
    material_type = 'SurfaceTensedMaterial'

    def __init__(self, name: str, modulus: float, p_ratio: float, tau_0: float,
                 max_load: float = np.inf, periodic_im_repeats: tuple = (1, 1)):
        super().__init__(name, max_load=max_load, periodic_im_repeats=periodic_im_repeats)
        if tau_0 < 0:
            raise ValueError("Surface tension tau_0 must not be negative")
        if modulus <= 0 or not -1 < p_ratio < 0.5:
            raise ValueError("The modulus must be positive and the Poisson's ratio in (-1, 0.5)")
        self.modulus = modulus
        self.p_ratio = p_ratio
        self.tau_0 = tau_0
        self.e_star = modulus / (1 - p_ratio ** 2)
        self.s = 2 * tau_0 / self.e_star

    def _frf(self, components: typing.Sequence[str], q_y: np.ndarray, q_x: np.ndarray,
             q_norm: np.ndarray) -> dict:
        rtn = dict()
        for comp in components:
            if comp != 'zz':
                raise ValueError("Only normal loading ('zz') is implemented for surface tensed materials, "
                                 f"requested component: {comp}")
            rtn[comp] = 2 / (self.e_star * q_norm * (1 + self.s * q_norm))
        return rtn

    def __repr__(self):
        return (f"SurfaceTensedMaterial({self.name!r}, modulus={self.modulus}, p_ratio={self.p_ratio}, "
                f"tau_0={self.tau_0})")
