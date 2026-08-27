import typing
import numpy as np

from .frequency_domain_material import _FrequencyDomainMaterial
from .elastic_material import _get_properties

__all__ = ['CoatedElastic', 'MultiLayerElastic', 'GradedCoatedElastic']


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


class MultiLayerElastic(_FrequencyDomainMaterial):
    """A stack of homogeneous isotropic elastic layers bonded to an elastic (or rigid) half space

    The frequency response function for normal loading is found by solving the boundary value
    problem of Yu, Wang and Wang (2014) at every wavenumber: each layer's displacement and
    stress fields are written in terms of transformed Papkovich-Neuber potentials with four
    unknown coefficients per layer (two for the half space), and the perfect bonding conditions
    at every interface plus the traction boundary condition at the surface close the linear
    system (their eq. 27). For frictionless normal loading the antisymmetric potential
    components vanish and the system is real valued. The assembled matrix is identical to the
    reference's, with the coefficients of the growing exponentials rescaled by exp(-alpha h_j)
    so that every exponential in the solve decays: the evaluation is stable at any wavenumber
    thickness product. The surface displacement follows from their eq. 42.

    The system (4 L + 2 square for L layers on an elastic half space, 4 L for a rigid base) is
    solved with a batched dense solve over the unique wavenumber magnitudes of the grid.

    Only normal ('zz') loading is implemented.

    Parameters
    ----------
    name: str
        The name of the material, must be unique
    layers: sequence of (dict, float) tuples
        The layers from the surface down: each entry is (elastic properties, thickness). The
        properties dict takes exactly two of 'E', 'v', 'G', 'K', 'M', 'Lam' as for the Elastic
        material, thicknesses are in the units of the grid spacing of the surface
    substrate_properties: dict or 'rigid'
        The elastic properties of the half space below the last layer, or the string 'rigid'
        for a stack bonded to a rigid base
    max_load: float, optional (inf)
        The maximum load supported, loads above this cause perfectly plastic deformation in
        the solvers which support it
    periodic_im_repeats: tuple, optional (1, 1)
        See _IMMaterial

    Notes
    -----
    A single layer stack is exactly equivalent to the CoatedElastic material (which uses the
    closed form solution of the same boundary value problem and is marginally faster to
    evaluate).

    For a rigid base the zero frequency (mean) compliance is finite and set automatically to
    the sum of the confined layer compliances sum(h_j (1+v_j)(1-2v_j) / (E_j (1-v_j))); for an
    elastic substrate the FRF is singular at q = 0 and the fully periodic (DC = 0) convention
    is used.

    Parameters must not be changed after construction, the influence matrix is memoized.

    References
    ----------
    Yu, C., Wang, Z., & Wang, Q. J. (2014). Analytical frequency response functions for
    contact of multilayered materials. Mechanics of Materials, 76, 102-120.
    """
    material_type = 'MultiLayerElastic'

    def __init__(self, name: str, layers: typing.Sequence[typing.Tuple[dict, float]],
                 substrate_properties: typing.Union[dict, str], max_load: float = np.inf,
                 periodic_im_repeats: tuple = (1, 1)):
        if len(layers) < 1:
            raise ValueError("At least one layer must be given, for no layers use the Elastic material")
        self.moduli = []
        self.p_ratios = []
        self.thicknesses = []
        for props, thickness in layers:
            if thickness <= 0 or not np.isfinite(thickness):
                raise ValueError("Layer thicknesses must be positive and finite")
            full_props = _get_properties(props)
            self.moduli.append(float(full_props['E']))
            self.p_ratios.append(float(full_props['v']))
            self.thicknesses.append(float(thickness))

        self.rigid_substrate = isinstance(substrate_properties, str)
        if self.rigid_substrate:
            if substrate_properties.lower() != 'rigid':
                raise ValueError(f"Unrecognised substrate: {substrate_properties}, substrate properties should be a "
                                 "dict of elastic properties or the string 'rigid'")
            self.substrate_modulus = np.inf
            self.substrate_p_ratio = None
            # layers in series, each confined laterally: the q -> 0 limit of the FRF
            zero_frequency_value = sum(h * (1 + v) * (1 - 2 * v) / (e * (1 - v)) for e, v, h in
                                       zip(self.moduli, self.p_ratios, self.thicknesses))
        else:
            props_s = _get_properties(substrate_properties)
            self.substrate_modulus = float(props_s['E'])
            self.substrate_p_ratio = float(props_s['v'])
            zero_frequency_value = 0.0

        super().__init__(name, max_load=max_load, periodic_im_repeats=periodic_im_repeats,
                         zero_frequency_value=zero_frequency_value)

    def _solve_layer_system(self, alpha: np.ndarray) -> np.ndarray:
        """Surface normal FRF at an array of nonzero wavenumber magnitudes

        Assembles the matrix equation (27) of Yu, Wang and Wang for a unit normal pressure with
        the growing coefficients rescaled: the unknowns with a bar (the coefficients of the
        exponentials which grow with depth) are solved as bar_coefficient * exp(alpha h_j),
        which multiplies their matrix entries by exp(-alpha h_j) and leaves only decaying
        exponentials everywhere. The interface rows (which the reference prints with a common
        factor exp(alpha h_j)) are divided through by that factor for the same reason.
        """
        n_layers = len(self.moduli)
        shear_moduli = [e / (2 * (1 + v)) for e, v in zip(self.moduli, self.p_ratios)]
        size = 4 * n_layers + (0 if self.rigid_substrate else 2)
        n_q = alpha.shape[0]
        mat = np.zeros((n_q, size, size))
        rhs = np.zeros((n_q, size))

        decay = [np.exp(-alpha * h) for h in self.thicknesses]

        # surface rows: normal and shear traction boundary conditions (first two rows of eq. 27)
        v1 = self.p_ratios[0]
        e1 = decay[0]
        mat[:, 0, 0] = alpha
        mat[:, 0, 1] = alpha * e1
        mat[:, 0, 2] = 2 * (1 - v1)
        mat[:, 0, 3] = -2 * (1 - v1) * e1
        rhs[:, 0] = -1.0 / alpha
        mat[:, 1, 0] = alpha
        mat[:, 1, 1] = -alpha * e1
        mat[:, 1, 2] = 1 - 2 * v1
        mat[:, 1, 3] = (1 - 2 * v1) * e1

        for j in range(n_layers):
            row = 2 + 4 * j
            col = 4 * j
            v_j = self.p_ratios[j]
            ah = alpha * self.thicknesses[j]
            e_j = decay[j]
            last_interface = j == n_layers - 1
            if last_interface and self.rigid_substrate:
                # only the two displacement conditions (u tangential = u normal = 0) remain
                n_rows = 2
            else:
                n_rows = 4
            # the layer j side of the interface conditions: X1j of eq. 28 divided by exp(alpha h_j)
            # rows in order: tangential displacement, normal displacement, shear stress, normal stress
            x1 = np.empty((n_q, 4, 4))
            x1[:, 0, 0] = alpha * e_j
            x1[:, 0, 1] = alpha
            x1[:, 0, 2] = ah * e_j
            x1[:, 0, 3] = ah
            x1[:, 1, 0] = alpha * e_j
            x1[:, 1, 1] = -alpha
            x1[:, 1, 2] = (3 - 4 * v_j + ah) * e_j
            x1[:, 1, 3] = 3 - 4 * v_j - ah
            x1[:, 2, 0] = alpha * e_j
            x1[:, 2, 1] = -alpha
            x1[:, 2, 2] = (1 - 2 * v_j + ah) * e_j
            x1[:, 2, 3] = 1 - 2 * v_j - ah
            x1[:, 3, 0] = alpha * e_j
            x1[:, 3, 1] = alpha
            x1[:, 3, 2] = (2 * (1 - v_j) + ah) * e_j
            x1[:, 3, 3] = -(2 * (1 - v_j) - ah)
            mat[:, row:row + n_rows, col:col + 4] = x1[:, :n_rows]

            if last_interface and self.rigid_substrate:
                continue
            if last_interface:
                v_n = self.substrate_p_ratio
                g = shear_moduli[j] * 2 * (1 + v_n) / self.substrate_modulus
                # the half space columns: sub-matrix 4 of eq. 27 divided by exp(alpha h_j)
                mat[:, row + 0, col + 4] = -alpha * g
                mat[:, row + 1, col + 4] = -alpha * g
                mat[:, row + 1, col + 5] = -g * (3 - 4 * v_n)
                mat[:, row + 2, col + 4] = -alpha
                mat[:, row + 2, col + 5] = -(1 - 2 * v_n)
                mat[:, row + 3, col + 4] = -alpha
                mat[:, row + 3, col + 5] = -2 * (1 - v_n)
            else:
                v_n = self.p_ratios[j + 1]
                g = shear_moduli[j] / shear_moduli[j + 1]
                e_n = decay[j + 1]
                # the layer j + 1 side: X2j of eq. 29 divided by exp(alpha h_j)
                mat[:, row + 0, col + 4] = -alpha * g
                mat[:, row + 0, col + 5] = -alpha * g * e_n
                mat[:, row + 1, col + 4] = -alpha * g
                mat[:, row + 1, col + 5] = alpha * g * e_n
                mat[:, row + 1, col + 6] = -g * (3 - 4 * v_n)
                mat[:, row + 1, col + 7] = -g * (3 - 4 * v_n) * e_n
                mat[:, row + 2, col + 4] = -alpha
                mat[:, row + 2, col + 5] = alpha * e_n
                mat[:, row + 2, col + 6] = -(1 - 2 * v_n)
                mat[:, row + 2, col + 7] = -(1 - 2 * v_n) * e_n
                mat[:, row + 3, col + 4] = -alpha
                mat[:, row + 3, col + 5] = -alpha * e_n
                mat[:, row + 3, col + 6] = -2 * (1 - v_n)
                mat[:, row + 3, col + 7] = 2 * (1 - v_n) * e_n

        solution = np.linalg.solve(mat, rhs[..., np.newaxis])[..., 0]
        a_top = solution[:, 0]
        a_bar_top = solution[:, 1] * decay[0]
        c_top = solution[:, 2]
        c_bar_top = solution[:, 3] * decay[0]
        # surface normal displacement for a unit normal pressure, eq. 42 with the shear
        # potentials zero
        v1 = self.p_ratios[0]
        g1 = shear_moduli[0]
        return -(alpha * (a_top - a_bar_top) + (3 - 4 * v1) * (c_top + c_bar_top)) / (2 * g1)

    def _frf(self, components: typing.Sequence[str], q_y: np.ndarray, q_x: np.ndarray,
             q_norm: np.ndarray) -> dict:
        for comp in components:
            if comp != 'zz':
                raise ValueError("Only normal loading ('zz') is implemented for multi layered materials, "
                                 f"requested component: {comp}")
        # the FRF depends only on the wavenumber magnitude: solve on the unique values
        alpha_unique, inverse = np.unique(q_norm.ravel(), return_inverse=True)
        frf_unique = np.zeros_like(alpha_unique)
        nonzero = alpha_unique > 0
        alpha_nz = alpha_unique[nonzero]
        frf_nz = np.empty_like(alpha_nz)
        chunk = 8192  # bound the memory of the batched dense solve
        for start in range(0, alpha_nz.shape[0], chunk):
            sl = slice(start, start + chunk)
            frf_nz[sl] = self._solve_layer_system(alpha_nz[sl])
        frf_unique[nonzero] = frf_nz
        frf = frf_unique[inverse].reshape(q_norm.shape)
        rtn = dict()
        for comp in components:
            rtn[comp] = frf.copy() if len(components) > 1 else frf
        return rtn

    def __repr__(self):
        layers = [({'E': e, 'v': v}, h) for e, v, h in zip(self.moduli, self.p_ratios, self.thicknesses)]
        substrate = "'rigid'" if self.rigid_substrate else (f"{{'E': {self.substrate_modulus}, "
                                                            f"'v': {self.substrate_p_ratio}}}")
        return f"MultiLayerElastic({self.name!r}, layers={layers}, substrate_properties={substrate})"


class GradedCoatedElastic(MultiLayerElastic):
    """A coating with continuously depth varying modulus on an elastic (or rigid) half space

    The coating is approximated by a stack of homogeneous sublayers (the piecewise homogeneous
    approach, whose convergence to the continuously graded solution is established by Ke and
    Wang (2006)) and solved with the multilayer frequency response functions of Yu, Wang and
    Wang (2014). The modulus of each sublayer is the value of the supplied function at the
    sublayer midpoint.

    Only normal ('zz') loading is implemented.

    Parameters
    ----------
    name: str
        The name of the material, must be unique
    modulus_function: callable
        Young's modulus as a function of depth, called with a depth z (0 at the surface,
        thickness at the coating substrate interface) and returning the modulus at that depth
    p_ratio: float or callable
        Poisson's ratio, either a constant or a function of depth like modulus_function
    thickness: float
        The total thickness of the graded coating
    substrate_properties: dict or 'rigid'
        The elastic properties of the half space below the coating (as for the Elastic
        material), or the string 'rigid'
    n_sublayers: int, optional (15)
        The number of homogeneous sublayers the coating is divided into. Sublayer boundaries
        are geometrically spaced with the thinnest sublayers at the surface, where the short
        wavelength response is formed
    max_load: float, optional (inf)
        See MultiLayerElastic
    periodic_im_repeats: tuple, optional (1, 1)
        See _IMMaterial

    Notes
    -----
    Increase n_sublayers until the solution stops changing for converged results; the default
    is a reasonable starting point for smooth modulus variations of less than a decade.

    Parameters (including the functions) must not be changed after construction, the influence
    matrix is memoized.

    References
    ----------
    Ke, L.-L., & Wang, Y.-S. (2006). Two-dimensional contact mechanics of functionally graded
    materials with arbitrary spatial variations of material properties. International Journal
    of Solids and Structures, 43(18-19), 5779-5798.

    Yu, C., Wang, Z., & Wang, Q. J. (2014). Analytical frequency response functions for
    contact of multilayered materials. Mechanics of Materials, 76, 102-120.
    """
    material_type = 'GradedCoatedElastic'

    def __init__(self, name: str, modulus_function: typing.Callable, p_ratio, thickness: float,
                 substrate_properties: typing.Union[dict, str], n_sublayers: int = 15,
                 max_load: float = np.inf, periodic_im_repeats: tuple = (1, 1)):
        if thickness <= 0 or not np.isfinite(thickness):
            raise ValueError("Coating thickness must be positive and finite")
        if n_sublayers < 1:
            raise ValueError("At least one sublayer is needed")
        self.modulus_function = modulus_function
        self.p_ratio_function = p_ratio if callable(p_ratio) else None
        self.total_thickness = thickness
        self.n_sublayers = n_sublayers
        # geometrically growing sublayer thicknesses, thinnest at the surface (ratio 1.25)
        ratio = 1.25 if n_sublayers > 1 else 1.0
        weights = ratio ** np.arange(n_sublayers)
        boundaries = thickness * np.concatenate([[0.0], np.cumsum(weights)]) / np.sum(weights)
        layers = []
        for z_top, z_bottom in zip(boundaries[:-1], boundaries[1:]):
            z_mid = 0.5 * (z_top + z_bottom)
            v_here = p_ratio(z_mid) if callable(p_ratio) else p_ratio
            layers.append(({'E': float(modulus_function(z_mid)), 'v': float(v_here)}, z_bottom - z_top))
        super().__init__(name, layers, substrate_properties, max_load=max_load,
                         periodic_im_repeats=periodic_im_repeats)

    def __repr__(self):
        substrate = "'rigid'" if self.rigid_substrate else (f"{{'E': {self.substrate_modulus}, "
                                                            f"'v': {self.substrate_p_ratio}}}")
        return (f"GradedCoatedElastic({self.name!r}, thickness={self.total_thickness}, "
                f"n_sublayers={self.n_sublayers}, substrate_properties={substrate})")
