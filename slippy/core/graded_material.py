import typing
import numpy as np
from scipy.special import gamma

from .frequency_domain_material import _FrequencyDomainMaterial

__all__ = ['PowerLawGradedElastic']


class PowerLawGradedElastic(_FrequencyDomainMaterial):
    """An elastic half space with a power law graded modulus E(z) = E0 (z / c0)^k

    The surface normal displacement under a normal point load P is (Booker, Balaam & Davis
    1985; reproduced with the full coefficient in the open reference below, eq. 3 and 5):

        u_z(s) = (c0^k / E0) * B(k, v) * P / s^(1+k)

        B(k, v) = (1 - v^2) sin(pi beta / 2) beta / ((1 + k)^2 pi)
                  * Gamma((3+k+beta)/2) Gamma((3+k-beta)/2) / Gamma(1 + k/2)^2

        beta = sqrt((1 - k v / (1 - v)) (1 + k))

    The two dimensional fourier transform of s^-(1+k) is analytical, giving the frequency
    response function as a pure power law:

        C(q) = (c0^k / E0) * B(k, v) * 2 pi 2^-k Gamma((1-k)/2) / Gamma((1+k)/2) * q^(k-1)

    At k = 0, B = (1 - v^2)/pi and C(q) = 2 (1 - v^2)/(E0 q): the homogeneous elastic half
    space is recovered exactly. Poisson's ratio is constant with depth. Only normal ('zz')
    loading is implemented.

    Parameters
    ----------
    name: str
        The name of the material, must be unique
    modulus: float
        E0, the Young's modulus at depth z = c0
    p_ratio: float
        The (depth independent) Poisson's ratio
    exponent: float
        The grading exponent k, 0 <= k < 1. k = 0 is homogeneous, k -> 1 approaches the
        Gibson soil (linearly graded incompressible) limit
    reference_depth: float, optional (1.0)
        c0, the depth at which the modulus equals E0, in the same units as the grid spacing

    Other Parameters
    ----------------
    max_load: float, optional (inf)
        Maximum pressure for the elastic-perfectly-plastic solvers
    periodic_im_repeats: tuple, optional (1, 1)
        See _IMMaterial

    Notes
    -----
    The grading extends to infinite depth: E -> 0 at the surface (for k > 0), which is the
    idealisation of Booker et al. and Giannakopoulos & Suresh. Parameters must not be changed
    after construction, the influence matrix is memoized.

    References
    ----------
    Booker, J. R., Balaam, N. P., & Davis, E. H. (1985). The behaviour of an elastic
    non-homogeneous half-space. Int. J. Numer. Anal. Methods Geomech., 9, 353-367.

    Giannakopoulos, A. E., & Suresh, S. (1997). Indentation of solids with gradients in
    elastic properties. Int. J. Solids Struct., 34, 2357-2428.

    Willert, E. (2023). A theory for tangential contacts of three-dimensional power-law graded
    elastic solids (open access: arXiv:2207.13166) - eq. 3 and 5 carry the coefficients used.
    """
    material_type = 'PowerLawGradedElastic'

    def __init__(self, name: str, modulus: float, p_ratio: float, exponent: float,
                 reference_depth: float = 1.0, max_load: float = np.inf,
                 periodic_im_repeats: tuple = (1, 1)):
        if not 0 <= exponent < 1:
            raise ValueError(f"The grading exponent must be in [0, 1), got {exponent}")
        if reference_depth <= 0:
            raise ValueError("The reference depth must be positive")
        if modulus <= 0 or not -1 < p_ratio <= 0.5:
            raise ValueError("The modulus must be positive and the Poisson's ratio in (-1, 0.5]")
        self.modulus = modulus
        self.p_ratio = p_ratio
        self.exponent = exponent
        self.reference_depth = reference_depth

        k, v = exponent, p_ratio
        beta = np.sqrt((1 - k * v / (1 - v)) * (1 + k))
        b_coefficient = ((1 - v ** 2) * np.sin(np.pi * beta / 2) * beta / ((1 + k) ** 2 * np.pi) *
                         gamma((3 + k + beta) / 2) * gamma((3 + k - beta) / 2) / gamma(1 + k / 2) ** 2)
        fourier_factor = 2 * np.pi * 2 ** (-k) * gamma((1 - k) / 2) / gamma((1 + k) / 2)
        self._frf_prefactor = reference_depth ** k / modulus * b_coefficient * fourier_factor

        super().__init__(name, max_load=max_load, periodic_im_repeats=periodic_im_repeats)

    def _frf(self, components: typing.Sequence[str], q_y: np.ndarray, q_x: np.ndarray,
             q_norm: np.ndarray) -> dict:
        rtn = dict()
        for comp in components:
            if comp != 'zz':
                raise ValueError("Only normal loading ('zz') is implemented for power law graded materials, "
                                 f"requested component: {comp}")
            rtn[comp] = self._frf_prefactor * q_norm ** (self.exponent - 1)
        return rtn

    def __repr__(self):
        return (f"PowerLawGradedElastic({self.name!r}, modulus={self.modulus}, p_ratio={self.p_ratio}, "
                f"exponent={self.exponent}, reference_depth={self.reference_depth})")
