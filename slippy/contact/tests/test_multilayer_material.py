"""Tests for the multilayered elastic material against analytical and published solutions

The FRFs solve the layered half space boundary value problem of Yu, Wang & Wang (2014),
Mech. Mater. 76:102-120 (eqs. 27-29 assembled per wavenumber, surface displacement from their
eq. 42). Self contained checks: a single layer stack must equal the independently verified
CoatedElastic closed form, identical layers must recover the homogeneous half space, and
splitting any layer in two must not change the response. The published value check reproduces
the frictionless tri-layered contacts of their section 3.3 / fig. 6: a rigid sphere with
R = 200 a0 on two 0.5 a0 coatings over a 210 GPa substrate (all poisson's ratios 0.3), where
a0 and P0 are the hertz contact radius and peak pressure of the uncoated substrate. The paper
reports a contact radius of 0.695 a0 for stiff coatings (E1 = 4 Esub, E2 = 2 Esub), a peak
pressure of 0.582 P0 and contact radius of 1.373 a0 for compliant coatings
(E1 = 0.25 Esub, E2 = 0.5 Esub), and that the stiff coatings almost double the peak pressure.
"""
import numpy as np
import numpy.testing as npt
import pytest

import slippy
import slippy.contact as c
import slippy.surface as s
from slippy.core import Elastic, CoatedElastic, MultiLayerElastic
from slippy.core.tests._material_test_utils import assert_ims_match

pytest.importorskip('pyfftw')

E_SUB = 210e9
V = 0.3
GS = 1e-4
SPAN = (64, 64)


def test_multilayer_single_layer_equivalence():
    """A one layer stack equals the closed form coated material to machine precision"""
    layer = ({'E': 100e9, 'v': 0.3}, 0.02)
    stack = MultiLayerElastic('ml_single', [layer], {'E': 200e9, 'v': 0.25})
    coated = CoatedElastic('ml_single_ref', layer[0], layer[1], {'E': 200e9, 'v': 0.25})
    assert_ims_match(stack, coated, grid_spacing=(GS, GS), span=SPAN, rtol=1e-12,
                     err_msg='single layer, elastic substrate:')

    stack_r = MultiLayerElastic('ml_single_rigid', [layer], 'rigid')
    coated_r = CoatedElastic('ml_single_rigid_ref', layer[0], layer[1], 'rigid')
    assert_ims_match(stack_r, coated_r, grid_spacing=(GS, GS), span=SPAN, rtol=1e-12,
                     err_msg='single layer, rigid base:')
    npt.assert_allclose(stack_r.zero_frequency_value, coated_r.zero_frequency_value, rtol=1e-12)


def test_multilayer_homogeneous_limit():
    """Identical layers on a matching substrate recover the homogeneous half space"""
    layers = [({'E': E_SUB, 'v': V}, 0.01)] * 5
    stack = MultiLayerElastic('ml_homog', layers, {'E': E_SUB, 'v': V})
    homogeneous = Elastic('ml_homog_ref', {'E': E_SUB, 'v': V}, zero_frequency_value=0.0)
    assert_ims_match(stack, homogeneous, grid_spacing=(GS, GS), span=SPAN, rtol=1e-12,
                     err_msg='homogeneous limit:')


def test_multilayer_split_invariance():
    """Splitting a layer into two half thickness layers changes nothing"""
    top = {'E': 80e9, 'v': 0.35}
    bottom = {'E': 300e9, 'v': 0.2}
    unsplit = MultiLayerElastic('ml_unsplit', [(top, 0.01), (bottom, 0.03)], {'E': 200e9, 'v': 0.3})
    split = MultiLayerElastic('ml_split', [(top, 0.01), (bottom, 0.015), (bottom, 0.015)],
                              {'E': 200e9, 'v': 0.3})
    assert_ims_match(unsplit, split, grid_spacing=(GS, GS), span=SPAN, rtol=1e-12,
                     err_msg='split invariance:')


def test_multilayer_rigid_base_dc():
    """The zero frequency compliance of a stack on a rigid base is the series sum of the
    confined layer compliances"""
    layers = [({'E': 100e9, 'v': 0.3}, 0.01), ({'E': 250e9, 'v': 0.2}, 0.02)]
    stack = MultiLayerElastic('ml_rigid_dc', layers, 'rigid')
    expected = sum(h * (1 + props['v']) * (1 - 2 * props['v']) / (props['E'] * (1 - props['v']))
                   for props, h in layers)
    npt.assert_allclose(stack.zero_frequency_value, expected, rtol=1e-12)
    # the frf must approach the same value at small q
    q = 1e-4 / sum(h for _, h in layers)
    frf = stack._frf(['zz'], np.array([[0.0]]), np.array([[q]]), np.array([[q]]))['zz'][0, 0]
    npt.assert_allclose(frf, expected, rtol=1e-2)


# the tri-layered contacts of Yu, Wang & Wang section 3.3: rigid sphere, R = 200 a0,
# h1 = h2 = 0.5 a0, all v = 0.3, substrate 210 GPa
A_0 = 1e-3
R_SPHERE = 200 * A_0
E_STAR_SUB = E_SUB / (1 - V ** 2)
TOTAL_LOAD = E_STAR_SUB * A_0 ** 2 / 150  # from a0^3 = 3 W R / (4 E*) with R = 200 a0
P_0 = 3 * TOTAL_LOAD / (2 * np.pi * A_0 ** 2)


def _solve_trilayer_sphere(e1, e2, name_tag):
    flat_surface = s.FlatSurface(shift=(0, 0))
    round_surface = s.RoundSurface((R_SPHERE,) * 3, extent=(0.006, 0.006), shape=(255, 255),
                                   generate=True)
    layers = [({'E': e1, 'v': V}, 0.5 * A_0), ({'E': e2, 'v': V}, 0.5 * A_0)]
    flat_surface.material = MultiLayerElastic(f'ml_yu_{name_tag}', layers, {'E': E_SUB, 'v': V})
    round_surface.material = Elastic(f'ml_yu_counter_{name_tag}', {'E': 1e16, 'v': 0.0})  # ~rigid
    my_model = c.ContactModel(f'ml-yu-model-{name_tag}', round_surface, flat_surface)
    my_model.add_step(c.StaticStep('contact', normal_load=TOTAL_LOAD))
    out = my_model.solve(skip_data_check=True)
    return out, round_surface.grid_spacing


def test_multilayer_vs_yu_wang_compliant():
    """Two compliant coatings: peak pressure 0.582 P0 and contact radius 1.373 a0"""
    with slippy.OverRideCuda():
        out, gs = _solve_trilayer_sphere(0.25 * E_SUB, 0.5 * E_SUB, 'compliant')
    npt.assert_approx_equal(np.sum(out['loads_z']) * gs ** 2, TOTAL_LOAD, 3)
    contact_radius = np.sqrt(np.sum(out['contact_nodes']) * gs ** 2 / np.pi)
    npt.assert_allclose(np.max(out['loads_z']), 0.582 * P_0, rtol=0.03,
                        err_msg='compliant tri-layer peak pressure (Yu, Wang & Wang fig. 6)')
    npt.assert_allclose(contact_radius, 1.373 * A_0, rtol=0.03,
                        err_msg='compliant tri-layer contact radius (Yu, Wang & Wang fig. 6)')


def test_multilayer_vs_yu_wang_stiff():
    """Two stiff coatings: contact radius 0.695 a0, peak pressure almost doubled"""
    with slippy.OverRideCuda():
        out, gs = _solve_trilayer_sphere(4 * E_SUB, 2 * E_SUB, 'stiff')
    npt.assert_approx_equal(np.sum(out['loads_z']) * gs ** 2, TOTAL_LOAD, 3)
    contact_radius = np.sqrt(np.sum(out['contact_nodes']) * gs ** 2 / np.pi)
    npt.assert_allclose(contact_radius, 0.695 * A_0, rtol=0.05,
                        err_msg='stiff tri-layer contact radius (Yu, Wang & Wang fig. 6)')
    peak_ratio = np.max(out['loads_z']) / P_0
    assert 1.8 < peak_ratio < 2.2, ("stiff coatings should almost double the peak pressure "
                                    f"(Yu, Wang & Wang section 3.3), found {peak_ratio:.3f} P0")
