"""Tests for the power law graded material against analytical solutions

Sources: the surface Green's function coefficients are from the open Willert paper
(arXiv:2207.13166, eq. 3 and 5, attributing Booker, Balaam & Davis 1985); the hertz-type
closed forms for a paraboloid on a graded half space are from Giannakopoulos & Suresh (1997):
the pressure distribution is p(r) = p0 (1 - r^2/a^2)^((1+k)/2) and the contact radius scales
as a proportional to P^(1/(3+k)) (homogeneous k=0 gives the classical P^(1/3)).
"""
import numpy as np
import numpy.testing as npt
import pytest

import slippy
import slippy.contact as c
import slippy.surface as s
from slippy.core import Elastic, PowerLawGradedElastic
from slippy.core.tests._material_test_utils import assert_ims_match

pytest.importorskip('pyfftw')

E = 200e9
V = 0.3


def test_graded_homogeneous_limit_frf():
    """k = 0 reproduces the elastic half space FRF exactly"""
    graded = PowerLawGradedElastic('graded_k0', modulus=E, p_ratio=V, exponent=0.0)
    elastic = Elastic('graded_k0_ref', {'E': E, 'v': V}, zero_frequency_value=0.0)
    assert_ims_match(graded, elastic, grid_spacing=(1e-4, 1e-4), span=(64, 64), rtol=1e-10,
                     err_msg='k=0 limit:')


def test_graded_fourier_factor():
    """The fourier transform factor of the kernel is verified by direct numerical integration

    The FRF prefactor contains FT[s^-(1+k)] = 2 pi 2^-k Gamma((1-k)/2)/Gamma((1+k)/2) q^(k-1),
    equivalent to the Hankel integral 2 pi Integral(x^-k J0(x) dx, 0..inf) at q = 1. The
    integral is computed here by summing between Bessel function zeros and averaging the
    alternating partial sums (Euler style acceleration), pinning the absolute magnitude of the
    influence matrix at every k, independent of the closed form gamma expression.
    """
    from scipy.special import jn_zeros, j0, gamma as sp_gamma
    from scipy import integrate

    zeros = np.concatenate([[0.0], jn_zeros(0, 200)])
    for k in [0.2, 0.5, 0.8]:
        segments = [integrate.quad(lambda x: x ** -k * j0(x), z1, z2, limit=200)[0]
                    for z1, z2 in zip(zeros[:-1], zeros[1:])]
        partial = np.cumsum(segments)
        # average the tail of the alternating partial sums three times to accelerate
        for _ in range(3):
            partial = 0.5 * (partial[:-1] + partial[1:])
        numeric = partial[-1]
        closed_form = 2.0 ** -k * sp_gamma((1 - k) / 2) / sp_gamma((1 + k) / 2)
        npt.assert_allclose(numeric, closed_form, rtol=1e-6,
                            err_msg=f'fourier factor of the graded kernel is wrong for k={k}')


def _solve_sphere_on_graded(k, total_load, name_tag):
    flat_surface = s.FlatSurface(shift=(0, 0))
    round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
    flat_surface.material = PowerLawGradedElastic(f'graded_{name_tag}', modulus=E, p_ratio=V, exponent=k,
                                                  reference_depth=0.001)
    round_surface.material = c.Elastic(f'graded_counter_{name_tag}', {'E': 1e16, 'v': 0.0})  # ~rigid
    my_model = c.ContactModel(f'graded-model-{name_tag}', round_surface, flat_surface)
    my_model.add_step(c.StaticStep('contact', normal_load=total_load))
    out = my_model.solve(skip_data_check=True)
    return out, round_surface.grid_spacing


def test_graded_hertz_pressure_exponent():
    """Pressure profile on a graded half space follows (1 - r^2/a^2)^((1+k)/2)"""
    k = 0.5
    with slippy.OverRideCuda():
        out, gs = _solve_sphere_on_graded(k, 100.0, 'exponent')
    pressure = out['loads_z']
    centre = np.unravel_index(np.argmax(pressure), pressure.shape)
    # radial profile along a grid axis through the centre
    row = pressure[centre[0], centre[1]:]
    in_contact = row > 0.01 * row[0]
    profile = row[in_contact]
    r = np.arange(len(row))[in_contact] * gs
    # contact radius from the last contacting node (add half a cell for discretisation)
    a = r[-1] + gs / 2
    p0 = profile[0]
    # fit the exponent m in p = p0 (1 - r^2/a^2)^m over the well resolved part of the profile
    mask = (r / a > 0.1) & (r / a < 0.9)
    x = np.log(1 - (r[mask] / a) ** 2)
    y = np.log(profile[mask] / p0)
    m_fit = np.polyfit(x, y, 1)[0]
    npt.assert_allclose(m_fit, (1 + k) / 2, rtol=0.08,
                        err_msg='graded pressure profile exponent should be (1+k)/2')


def test_graded_load_area_scaling():
    """Contact radius scales as P^(1/(3+k)) for a paraboloid on a graded half space"""
    k = 0.5
    areas = []
    loads = [50.0, 400.0]
    with slippy.OverRideCuda():
        for load in loads:
            out, gs = _solve_sphere_on_graded(k, load, f'scaling_{int(load)}')
            areas.append(np.sum(out['contact_nodes']) * gs ** 2)
    # area ~ a^2 ~ P^(2/(3+k))
    found_exponent = np.log(areas[1] / areas[0]) / np.log(loads[1] / loads[0])
    npt.assert_allclose(found_exponent, 2 / (3 + k), rtol=0.05,
                        err_msg='graded contact area load scaling should be P^(2/(3+k))')
