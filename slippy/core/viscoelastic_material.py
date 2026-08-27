import typing
import numpy as np

from .frequency_domain_material import _FrequencyDomainMaterial

__all__ = ['ViscoElasticSliding']


class ViscoElasticSliding(_FrequencyDomainMaterial):
    """A linear viscoelastic half space in steady state sliding

    In steady sliding at velocity vel the hereditary time dependence of a linear viscoelastic
    material becomes a frequency shift: a surface wave with wavenumber component q_x along the
    sliding direction excites the material at angular frequency |vel q_x|, so the frequency
    response function is the elastic one with the complex modulus at that frequency
    (Carbone & Putignano 2013):

        C(q) = 2 (1 - v^2) / (E(-vel q_x) |q|)

    where E(omega) is the complex modulus in the exp(+i omega t) time convention (positive
    loss modulus for positive omega). The minus sign pairs that convention with the
    exp(+i q x) spatial transform used by the solvers: a pattern moving towards +x has time
    factor exp(-i q_x vel t). The result is a time independent, velocity parametrized complex
    FRF: the asymmetry between the leading and trailing edges of the contact (and hence
    viscoelastic friction) comes from the imaginary part. C(-q) is the conjugate of C(q) so
    displacements are real.

    The complex modulus is given either as a Prony series of a generalized Maxwell solid:

        E(omega) = E_inf + sum_j E_j * (i omega tau_j) / (1 + i omega tau_j)

    (E_inf is the relaxed, rubbery modulus; E_inf + sum E_j the instantaneous, glassy
    modulus), or as a user supplied callable omega -> complex E.

    Only normal ('zz') loading is implemented. Poisson's ratio is constant (a common
    approximation for elastomers, exact for incompressible materials).

    Parameters
    ----------
    name: str
        The name of the material, must be unique
    relaxed_modulus: float
        E_inf, the long time (rubbery) Young's modulus
    p_ratio: float
        The (frequency independent) Poisson's ratio
    velocity: float
        The steady sliding velocity of the countersurface over this material, along the
        positive x direction of the surface grid, in consistent units
    prony_terms: Sequence of (modulus, relaxation_time) pairs, optional (None)
        The Prony series terms (E_j, tau_j) of the generalized Maxwell model
    complex_modulus: callable, optional (None)
        Alternative to prony_terms: a callable mapping angular frequency (array) to complex
        modulus (array). Must satisfy E(-omega) = conj(E(omega))

    Other Parameters
    ----------------
    max_load: float, optional (inf)
        Maximum pressure for the elastic-perfectly-plastic solvers
    periodic_im_repeats: tuple, optional (1, 1)
        See _IMMaterial

    Notes
    -----
    The influence matrix is asymmetric in the sliding direction so the convolution operator is
    not symmetric positive definite; the conjugate gradient solvers typically plateau at a
    relative residual around 1e-4 instead of converging to machine precision. The plateau
    solution is accurate to that level (load balance and pressure fields agree between
    solvers), but with the default step tolerance (1e-8) the converged flag will be False:
    for strongly asymmetric cases set the step tolerance to ~1e-4, e.g.
    StaticStep(..., tolerance=1e-4).

    Fourier modes with no variation along the sliding direction (q_x = 0) are static in the
    material frame and always respond with the relaxed modulus, at any velocity. In a periodic
    cell this softens the response slightly relative to the fully glassy limit at high speed.
    On grids with an even number of points along x the asymmetric part of the single q_x
    Nyquist mode is discarded by the real valued convolution; this is harmless but odd grid
    sizes avoid it entirely.

    Parameters (including the velocity) must not be changed after construction, the influence
    matrix is memoized: make one material instance per sliding velocity.

    References
    ----------
    Carbone, G., & Putignano, C. (2013). A novel methodology to predict sliding and rolling
    friction of viscoelastic materials: Theory and experiments. Journal of the Mechanics and
    Physics of Solids, 61(8), 1822-1834.
    """
    material_type = 'ViscoElasticSliding'

    def __init__(self, name: str, relaxed_modulus: float, p_ratio: float, velocity: float,
                 prony_terms: typing.Optional[typing.Sequence] = None,
                 complex_modulus: typing.Optional[typing.Callable] = None,
                 max_load: float = np.inf, periodic_im_repeats: tuple = (1, 1)):
        if (prony_terms is None) == (complex_modulus is None):
            raise ValueError("Exactly one of prony_terms and complex_modulus must be given")
        self.relaxed_modulus = relaxed_modulus
        self.p_ratio = p_ratio
        self.velocity = velocity
        if prony_terms is not None:
            terms = [(float(e_j), float(tau_j)) for e_j, tau_j in prony_terms]
            if any(e_j < 0 or tau_j <= 0 for e_j, tau_j in terms):
                raise ValueError("Prony term moduli must be non negative and relaxation times positive")
            self.prony_terms = terms

            def complex_modulus(omega):
                e = np.full_like(np.asarray(omega, dtype=float), relaxed_modulus, dtype=complex)
                for e_j, tau_j in terms:
                    iot = 1j * omega * tau_j
                    e = e + e_j * iot / (1 + iot)
                return e

            # the instantaneous (short time) modulus of the generalized Maxwell model
            self.glassy_modulus = relaxed_modulus + sum(e_j for e_j, _ in terms)
        else:
            self.prony_terms = None
            self.glassy_modulus = None
            # spot check the user callable: conjugate symmetry (real displacements) and a
            # dissipative loss modulus in the exp(+i omega t) convention
            for omega_test in (1.0, 1e3):
                e_pos = complex(complex_modulus(np.array(omega_test)))
                e_neg = complex(complex_modulus(np.array(-omega_test)))
                if not np.isclose(e_pos, np.conj(e_neg), rtol=1e-8):
                    raise ValueError("complex_modulus must satisfy E(-omega) = conj(E(omega)), required for real "
                                     "displacements")
                if e_pos.imag < 0:
                    raise ValueError("complex_modulus must have a non negative loss modulus for positive frequencies "
                                     "(the exp(+i omega t) convention); a negative loss modulus would generate energy "
                                     "during sliding. If transcribing from a paper using the exp(-i omega t) "
                                     "convention, conjugate the modulus.")
        self._complex_modulus = complex_modulus
        super().__init__(name, max_load=max_load, periodic_im_repeats=periodic_im_repeats)

    def _frf(self, components: typing.Sequence[str], q_y: np.ndarray, q_x: np.ndarray,
             q_norm: np.ndarray) -> dict:
        rtn = dict()
        # a pressure pattern moving in +x has time factor exp(-i q_x v t): in the exp(+i w t)
        # convention of the complex modulus the excitation frequency is w = -q_x v
        omega = -self.velocity * q_x
        e_complex = self._complex_modulus(omega)
        for comp in components:
            if comp != 'zz':
                raise ValueError("Only normal loading ('zz') is implemented for viscoelastic materials, "
                                 f"requested component: {comp}")
            rtn[comp] = 2 * (1 - self.p_ratio ** 2) / (e_complex * q_norm)
        return rtn

    def __repr__(self):
        return (f"ViscoElasticSliding({self.name!r}, relaxed_modulus={self.relaxed_modulus}, "
                f"p_ratio={self.p_ratio}, velocity={self.velocity}, prony_terms={self.prony_terms})")
