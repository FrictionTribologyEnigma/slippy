import typing
import numpy as np

from .frequency_domain_material import _FrequencyDomainMaterial
from .elastic_material import _get_properties

__all__ = ['CoatedElastic']


class CoatedElastic(_FrequencyDomainMaterial):
    """A homogeneous isotropic elastic coating perfectly bonded to an elastic (or rigid) half space

    The frequency response function for normal loading is the fourier space fundamental
    solution of Li, Pohrt, Lyashenko and Popov (eq. 9-10 of the reference):

        C(k) = 2 (1 - v1^2) / (E1 k) * (A e^-4kh + B kh e^-2kh + D) /
                                       (-A e^-4kh - B k^2 h^2 e^-2kh + 2 C e^-2kh + D)

    where h is the coating thickness and A, B, C, D are constants of the two sets of elastic
    properties. All exponentials decay so the expression is numerically stable at every kh. As
    h -> infinity this recovers the coating half space, as h -> 0 the substrate, and for
    matched properties it is exact at every thickness (A = B = C = 0).

    For a rigid substrate the E2 -> infinity limit of the constants is used, and the zero
    frequency (mean) compliance is finite: h (1 + v1)(1 - 2 v1) / (E1 (1 - v1)), the classical
    confined layer compliance, which is set automatically.

    Only normal ('zz') loading is implemented.

    Parameters
    ----------
    name: str
        The name of the material, must be unique
    coating_properties: dict
        The elastic properties of the coating, exactly two of 'E', 'v', 'G', 'K', 'M', 'Lam',
        as for the Elastic material
    thickness: float
        The thickness of the coating, in the same units as the grid spacing of the surface
    substrate_properties: dict or 'rigid'
        The elastic properties of the substrate as for the coating, or the string 'rigid' for
        a coating bonded to a rigid base
    max_load: float, optional (inf)
        The maximum load supported, loads above this cause perfectly plastic deformation in
        the solvers which support it
    periodic_im_repeats: tuple, optional (1, 1)
        See _IMMaterial

    Notes
    -----
    Parameters must not be changed after construction, the influence matrix is memoized.

    References
    ----------
    Li, Q., Pohrt, R., Lyashenko, I. A., & Popov, V. L. (2020). Boundary element method for
    nonadhesive and adhesive contacts of a coated elastic half-space. Proceedings of the
    Institution of Mechanical Engineers, Part J, 234(1), 73-83. (open access: arXiv:1807.01885)
    """
    material_type = 'CoatedElastic'

    def __init__(self, name: str, coating_properties: dict, thickness: float,
                 substrate_properties: typing.Union[dict, str], max_load: float = np.inf,
                 periodic_im_repeats: tuple = (1, 1)):
        if thickness <= 0 or not np.isfinite(thickness):
            raise ValueError("Coating thickness must be positive and finite (an infinitely thick coating is the "
                             "Elastic material)")
        props_1 = _get_properties(coating_properties)
        e1, v1 = props_1['E'], props_1['v']
        self.coating_modulus = e1
        self.coating_p_ratio = v1
        self.thickness = thickness

        self.rigid_substrate = isinstance(substrate_properties, str)
        if self.rigid_substrate:
            if substrate_properties.lower() != 'rigid':
                raise ValueError(f"Unrecognised substrate: {substrate_properties}, substrate properties should be a "
                                 "dict of elastic properties or the string 'rigid'")
            self.substrate_modulus = np.inf
            self.substrate_p_ratio = None
            # E2 -> infinity limit of eq. 10 (common factor E2^2 (1+v1)^2 removed)
            self._a = -(3 - 4 * v1)
            self._b = -4.0
            self._c = 8 * v1 ** 2 - 12 * v1 + 5
            self._d = 3 - 4 * v1
            # the classical confined layer compliance, the k -> 0 limit of the FRF
            zero_frequency_value = thickness * (1 + v1) * (1 - 2 * v1) / (e1 * (1 - v1))
        else:
            props_2 = _get_properties(substrate_properties)
            e2, v2 = props_2['E'], props_2['v']
            self.substrate_modulus = e2
            self.substrate_p_ratio = v2
            # eq. 10 of Li, Pohrt, Lyashenko and Popov
            self._a = ((e2 * (3 - 4 * v1) * (1 + v1) - e1 * (3 - 4 * v2) * (1 + v2)) *
                       (e1 * (1 + v2) - e2 * (1 + v1)))
            self._b = (4 * (e2 * (1 + v1) + e1 * (3 - 4 * v2) * (1 + v2)) *
                       (e1 * (1 + v2) - e2 * (1 + v1)))
            self._c = (e1 ** 2 * (4 * v2 - 3) * (v2 + 1) ** 2 -
                       2 * e1 * e2 * (v1 + 1) * (2 * v1 - 1) * (v2 + 1) * (2 * v2 - 1) +
                       e2 ** 2 * (8 * v1 ** 2 - 12 * v1 + 5) * (v1 + 1) ** 2)
            self._d = ((e2 * (1 + v1) + e1 * (3 - 4 * v2) * (1 + v2)) *
                       (e2 * (3 - 4 * v1) * (1 + v1) + e1 * (1 + v2)))
            # like the elastic half space the FRF is singular at k = 0, DC set to 0 by convention
            zero_frequency_value = 0.0

        super().__init__(name, max_load=max_load, periodic_im_repeats=periodic_im_repeats,
                         zero_frequency_value=zero_frequency_value)

    def _frf(self, components: typing.Sequence[str], q_y: np.ndarray, q_x: np.ndarray,
             q_norm: np.ndarray) -> dict:
        rtn = dict()
        a, b, c, d = self._a, self._b, self._c, self._d
        kh = q_norm * self.thickness
        e2kh = np.exp(-2 * kh)
        e4kh = e2kh * e2kh
        numerator = a * e4kh + b * kh * e2kh + d
        denominator = -a * e4kh - b * kh ** 2 * e2kh + 2 * c * e2kh + d
        e1, v1 = self.coating_modulus, self.coating_p_ratio
        frf = 2 * (1 - v1 ** 2) / (e1 * q_norm) * numerator / denominator
        for comp in components:
            if comp != 'zz':
                raise ValueError("Only normal loading ('zz') is implemented for coated materials, "
                                 f"requested component: {comp}")
            rtn[comp] = frf.copy() if len(components) > 1 else frf
        return rtn

    def __repr__(self):
        substrate = "'rigid'" if self.rigid_substrate else (f"{{'E': {self.substrate_modulus}, "
                                                            f"'v': {self.substrate_p_ratio}}}")
        return (f"CoatedElastic({self.name!r}, coating_properties={{'E': {self.coating_modulus}, "
                f"'v': {self.coating_p_ratio}}}, thickness={self.thickness}, substrate_properties={substrate})")
