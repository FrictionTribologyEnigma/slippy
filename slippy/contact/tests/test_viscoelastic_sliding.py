"""Tests for the steady state sliding viscoelastic material

The material implements the frequency shifted elastic FRF of Carbone & Putignano (2013),
JMPS 61:1822-1834: C(q) = 2(1-v^2)/(E(v q_x) |q|). The self contained checks are the standard
linear solid limits (v -> 0 gives the relaxed modulus, v -> infinity the glassy modulus, both
of which must reproduce the elastic hertz solution) and qualitative physics at intermediate
velocity (asymmetric pressure shifted towards the leading edge, positive dissipated power).
The quantitative check reproduces the friction vs velocity bell curve of the paper's fig. 12,
digitized from the vector graphics of the pdf (see test_viscoelastic_friction_curve).
"""
import numpy as np
import numpy.testing as npt
import pytest

import slippy
import slippy.contact as c
import slippy.surface as s
from slippy.core import Elastic, ViscoElasticSliding
from slippy.core.tests._material_test_utils import assert_ims_match

pytest.importorskip('pyfftw')

E_RELAXED = 10e9
E_GLASSY = 40e9  # relaxed + one prony term of 30e9
V = 0.3
TAU = 1e-4
PRONY = [(30e9, TAU)]
E_COUNTER = 200e9


def _solve_sliding_sphere(velocity, name_tag, total_load=100.0, tolerance=1e-8):
    flat_surface = s.FlatSurface(shift=(0, 0))
    round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
    flat_surface.material = ViscoElasticSliding(f've_{name_tag}', relaxed_modulus=E_RELAXED, p_ratio=V,
                                                velocity=velocity, prony_terms=PRONY)
    round_surface.material = Elastic(f've_counter_{name_tag}', {'E': E_COUNTER, 'v': V})
    my_model = c.ContactModel(f've-model-{name_tag}', round_surface, flat_surface)
    my_model.add_step(c.StaticStep('contact', normal_load=total_load, tolerance=tolerance))
    out = my_model.solve(skip_data_check=True)
    return out, round_surface.grid_spacing


def test_viscoelastic_frf_limits():
    """v -> 0 and v -> infinity reproduce the relaxed / glassy elastic FRFs"""
    slow = ViscoElasticSliding('ve_frf_slow', relaxed_modulus=E_RELAXED, p_ratio=V, velocity=0.0,
                               prony_terms=PRONY)
    relaxed = Elastic('ve_frf_relaxed_ref', {'E': E_RELAXED, 'v': V}, zero_frequency_value=0.0)
    assert_ims_match(slow, relaxed, grid_spacing=(1e-4, 1e-4), span=(64, 64), rtol=1e-9,
                     err_msg='rubbery limit:')

    fast = ViscoElasticSliding('ve_frf_fast', relaxed_modulus=E_RELAXED, p_ratio=V, velocity=1e12,
                               prony_terms=PRONY)
    glassy = Elastic('ve_frf_glassy_ref', {'E': E_GLASSY, 'v': V}, zero_frequency_value=0.0)
    # the q_x = 0 column is static in the material frame and stays relaxed at every velocity,
    # so it is excluded from the glassy comparison
    ims_fast = fast.influence_matrix(['zz'], (1e-4, 1e-4), (64, 64))['zz']
    ims_glassy = glassy.influence_matrix(['zz'], (1e-4, 1e-4), (64, 64))['zz']
    npt.assert_allclose(slippy.asnumpy(ims_fast)[:, 1:], slippy.asnumpy(ims_glassy)[:, 1:], rtol=1e-6,
                        err_msg='glassy limit (q_x != 0):')


def test_viscoelastic_rubbery_limit_full_solve():
    """A slow sliding solve matches the hertz solution with the relaxed modulus"""
    with slippy.OverRideCuda():
        out, gs = _solve_sliding_sphere(1e-6, 'rubbery')
    a_result = c.hertz_full([1, 1], [np.inf, np.inf], [E_COUNTER, E_RELAXED], [V, V], 100.0)
    assert out['converged']
    npt.assert_approx_equal(np.sum(out['loads_z']) * gs ** 2, 100.0, 3)
    npt.assert_approx_equal(np.max(out['loads_z']), a_result['max_pressure'], 2)


def test_viscoelastic_glassy_limit_full_solve():
    """A fast sliding solve approaches the hertz solution with the glassy modulus

    The q_x = 0 modes remain relaxed at every speed (they are static in the material frame),
    so in a periodic cell the high speed response is slightly softer than fully glassy: the
    peak pressure must lie between the relaxed and glassy hertz solutions, close to glassy.
    """
    with slippy.OverRideCuda():
        out, gs = _solve_sliding_sphere(1e7, 'glassy')
    glassy_peak = c.hertz_full([1, 1], [np.inf, np.inf], [E_COUNTER, E_GLASSY], [V, V], 100.0)['max_pressure']
    relaxed_peak = c.hertz_full([1, 1], [np.inf, np.inf], [E_COUNTER, E_RELAXED], [V, V], 100.0)['max_pressure']
    assert out['converged']
    npt.assert_approx_equal(np.sum(out['loads_z']) * gs ** 2, 100.0, 3)
    peak = np.max(out['loads_z'])
    assert relaxed_peak < peak < 1.02 * glassy_peak
    assert peak > 0.75 * glassy_peak, "high speed response should be close to the glassy solution"


def test_viscoelastic_intermediate_convergence_and_asymmetry():
    """At v tau ~ contact radius the solver still converges, the pressure peak shifts towards
    the leading edge and the sliding dissipates energy"""
    # hertz contact radius for the relaxed material is ~2 mm; v tau ~ a/4 -> v ~ 5 m/s.
    # the asymmetric operator makes the cg solvers plateau around 1e-4 relative residual,
    # which is the documented tolerance for strongly asymmetric cases
    with slippy.OverRideCuda():
        out, gs = _solve_sliding_sphere(5.0, 'intermediate', tolerance=1e-4)
    assert out['converged'], "the solver should converge (to 1e-4) in the most asymmetric regime"
    npt.assert_approx_equal(np.sum(out['loads_z']) * gs ** 2, 100.0, 3)

    pressure = out['loads_z']
    n = pressure.shape[1]
    x = (np.arange(n) - (n - 1) / 2) * gs
    # centre of pressure along the sliding (x) direction, relative to the sphere centre
    x_cop = np.sum(pressure * x[np.newaxis, :]) / np.sum(pressure)
    # the relaxed contact radius is ~1.9 mm and an independent time stepping analogue of this
    # exact case gives a centroid shift of ~0.26 a; require a substantial fraction of that
    contact_radius = np.sqrt(np.sum(out['contact_nodes']) * gs ** 2 / np.pi)
    assert x_cop > 0.05 * contact_radius, \
        "the pressure distribution should shift towards the leading edge for sliding in +x"

    # dissipated power: in steady sliding u(x, t) = U(x - v t) so du/dt = -v dU/dx and the
    # power input to the material is -v * integral(p * d(u_z)/dx), which must be positive
    u_z = slippy.asnumpy(out['surface_2_displacement_z'])
    du_dx = np.gradient(u_z, gs, axis=1)
    power = -5.0 * np.sum(pressure * du_dx) * gs ** 2
    assert power > 0, "sliding on a viscoelastic material must dissipate energy"


def test_viscoelastic_friction_curve():
    """Friction coefficient vs velocity against the bell curve of Carbone & Putignano fig. 12

    The reference case is a rigid sphere (R = 1 cm) on a one relaxation time half space with
    E_0 = 1 MPa (rubbery), E_inf = 10 MPa (glassy) and creep (retardation) time tau = 0.01 s,
    at a constant normal load F_N = 0.15 N. In our generalized Maxwell (relaxation) form this
    material is exactly relaxed_modulus = 1 MPa with one prony term (9 MPa, tau / 10). The
    reference friction values were extracted from the vector graphics of fig. 12 (the drawing
    coordinates of the data points in the pdf, calibrated against the axis tick positions, so
    the digitization error is negligible). The paper does not state the poisson's ratio;
    0.3 reproduces the peak value and is used here. a0 is the hertz contact radius of the
    rubbery material at F_N.

    The friction force is the dissipation based F_T = -integral(p du/dx): at the highest speed
    the displacement wake extends over v tau = 8 a0 so a larger periodic cell is needed for
    the isolated contact result.
    """
    nu = 0.3
    e_relaxed = 1e6
    prony = [(9e6, 1e-3)]
    tau_paper = 0.01
    radius = 0.01
    f_n = 0.15
    e0_star = e_relaxed / (1 - nu ** 2)
    a_0 = (3 * f_n * radius / (4 * e0_star)) ** (1 / 3)

    # (v tau / a0, mu from fig. 12, relative tolerance, extent, grid points)
    cases = [(0.1063, 0.00851, 0.08, 0.006, 255),
             (1.169, 0.04028, 0.05, 0.006, 255),   # the peak of the bell
             (7.943, 0.01587, 0.08, 0.012, 511)]
    found = []
    with slippy.OverRideCuda():
        for x, mu_ref, rtol, extent, shape in cases:
            velocity = x * a_0 / tau_paper
            flat_surface = s.FlatSurface(shift=(0, 0))
            round_surface = s.RoundSurface((radius,) * 3, extent=(extent, extent),
                                           shape=(shape, shape), generate=True)
            flat_surface.material = ViscoElasticSliding(f've_fric_{x:.3f}', relaxed_modulus=e_relaxed,
                                                        p_ratio=nu, velocity=velocity,
                                                        prony_terms=prony)
            round_surface.material = Elastic(f've_fric_counter_{x:.3f}', {'E': 1e16, 'v': 0.0})
            model = c.ContactModel(f've-fric-model-{x:.3f}', round_surface, flat_surface)
            model.add_step(c.StaticStep('contact', normal_load=f_n, tolerance=1e-4))
            out = model.solve(skip_data_check=True)
            assert out['converged']
            gs = round_surface.grid_spacing
            pressure = out['loads_z']
            u_z = slippy.asnumpy(out['surface_2_displacement_z'])
            du_dx = np.gradient(u_z, gs, axis=1)
            mu = -np.sum(pressure * du_dx) * gs ** 2 / (np.sum(pressure) * gs ** 2)
            npt.assert_allclose(mu, mu_ref, rtol=rtol,
                                err_msg=f'friction coefficient at v tau / a0 = {x} '
                                        '(Carbone & Putignano 2013 fig. 12)')
            found.append(mu)
    # the bell shape: the peak is higher than both tails
    assert found[1] > found[0] and found[1] > found[2]
