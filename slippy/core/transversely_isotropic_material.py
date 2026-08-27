import typing
import numpy as np

from .frequency_domain_material import _FrequencyDomainMaterial

__all__ = ['TransverselyIsotropicElastic']


def _stiffness_from_engineering(e_p: float, e_t: float, v_p: float, v_pt: float, g_t: float):
    """Voigt stiffness constants of a transversely isotropic solid from engineering constants

    Parameters are the in-plane modulus E_p, the transverse (symmetry axis) modulus E_t, the
    in-plane Poisson's ratio v_p, the Poisson's ratio v_pt for loading along the symmetry axis
    (strain in the plane over strain along the axis) and the transverse shear modulus G_t. The
    compliance matrix is built and inverted, valid for any physically admissible constants.
    """
    s = np.zeros((6, 6))
    s[0, 0] = s[1, 1] = 1 / e_p
    s[2, 2] = 1 / e_t
    s[0, 1] = s[1, 0] = -v_p / e_p
    s[0, 2] = s[2, 0] = s[1, 2] = s[2, 1] = -v_pt / e_t
    s[3, 3] = s[4, 4] = 1 / g_t
    s[5, 5] = 2 * (1 + v_p) / e_p
    c = np.linalg.inv(s)
    return {'C11': c[0, 0], 'C33': c[2, 2], 'C13': c[0, 2], 'C44': c[3, 3]}


class TransverselyIsotropicElastic(_FrequencyDomainMaterial):
    """A transversely isotropic elastic half space, symmetry axis normal to the surface

    For normal loading the surface response keeps the form of the isotropic Boussinesq
    solution with the indentation modulus M replacing the plane strain modulus E*:

        C(q) = 2 / (M q)

    where M has the exact closed form (Delafargue & Ulm 2004, from the transversely isotropic
    Green's function of Elliott/Hanson):

        M = 2 sqrt((C11 C33 - C13^2) / (C11 (1/C44 + 2/(sqrt(C11 C33) + C13))))

    Only normal ('zz') loading is implemented; the in-plane isotropy of the material makes the
    normal response isotropic in the surface plane, so the half space contact solvers apply
    unchanged.

    Parameters
    ----------
    name: str
        The name of the material, must be unique
    properties: dict
        Either the Voigt stiffness constants {'C11', 'C33', 'C13', 'C44'} (the in-plane
        constant C12 does not affect the normal response), or the five engineering constants
        {'E_p', 'E_t', 'v_p', 'v_pt', 'G_t'}: in-plane modulus, transverse modulus, in-plane
        Poisson's ratio, transverse Poisson's ratio (loading along the axis) and transverse
        shear modulus
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
    Delafargue, A., & Ulm, F.-J. (2004). Explicit approximations of the indentation modulus of
    elastically orthotropic solids for conical indenters. International Journal of Solids and
    Structures, 41(26), 7351-7360. (the transversely isotropic expression M3 is exact)

    Yu, H. Y. (2001). A concise treatment of indentation problems in transversely isotropic
    half-spaces. Applied Mechanics Reviews, 54(6), 479-503.
    """
    material_type = 'TransverselyIsotropicElastic'

    def __init__(self, name: str, properties: dict, max_load: float = np.inf,
                 periodic_im_repeats: tuple = (1, 1)):
        stiffness_keys = {'C11', 'C33', 'C13', 'C44'}
        engineering_keys = {'E_p', 'E_t', 'v_p', 'v_pt', 'G_t'}
        given = set(properties)
        if stiffness_keys <= given:
            c = {key: float(properties[key]) for key in stiffness_keys}
        elif engineering_keys <= given:
            c = _stiffness_from_engineering(properties['E_p'], properties['E_t'], properties['v_p'],
                                            properties['v_pt'], properties['G_t'])
        else:
            raise ValueError(f"Properties must contain either the stiffness constants {sorted(stiffness_keys)} or "
                             f"the engineering constants {sorted(engineering_keys)}, got: {sorted(given)}")
        c11, c33, c13, c44 = c['C11'], c['C33'], c['C13'], c['C44']
        if c11 <= 0 or c33 <= 0 or c44 <= 0 or c11 * c33 <= c13 ** 2:
            raise ValueError("The stiffness constants are not physically admissible: C11, C33, C44 must be positive "
                             "and C11*C33 must exceed C13^2")
        self.stiffness = c
        # exact indentation modulus for the symmetry axis, Delafargue & Ulm (2004);
        # for isotropic constants this is exactly E / (1 - v^2)
        self.indentation_modulus = 2 * np.sqrt(
            (c11 * c33 - c13 ** 2) / (c11 * (1 / c44 + 2 / (np.sqrt(c11 * c33) + c13))))
        super().__init__(name, max_load=max_load, periodic_im_repeats=periodic_im_repeats)

    def _frf(self, components: typing.Sequence[str], q_y: np.ndarray, q_x: np.ndarray,
             q_norm: np.ndarray) -> dict:
        rtn = dict()
        for comp in components:
            if comp != 'zz':
                raise ValueError("Only normal loading ('zz') is implemented for transversely isotropic materials, "
                                 f"requested component: {comp}")
            rtn[comp] = 2 / (self.indentation_modulus * q_norm)
        return rtn

    def __repr__(self):
        return f"TransverselyIsotropicElastic({self.name!r}, properties={self.stiffness})"
