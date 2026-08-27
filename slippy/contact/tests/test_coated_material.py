"""Tests for the coated elastic material against analytical solutions

The FRF is the fourier space fundamental solution of Li, Pohrt, Lyashenko & Popov (2020),
Proc. IMechE Part J 234:73-83 (open access as arXiv:1807.01885), eq. 9-10. The limit checks
below follow from the formula analytically:

- h -> infinity recovers the coating half space (all decaying exponentials vanish)
- h -> 0 recovers the substrate half space
- matched properties give A = B = C = 0 so the homogeneous FRF is exact at every thickness
- for a rigid substrate the q -> 0 compliance is the classical confined layer compliance
  h (1 + v)(1 - 2 v) / (E (1 - v)) (e.g. Johnson, Contact Mechanics, thin bonded layer)
"""
import numpy as np
import numpy.testing as npt
import pytest

import slippy
import slippy.contact as c
import slippy.surface as s
from slippy.core import Elastic, CoatedElastic
from slippy.core.tests._material_test_utils import assert_ims_match

pytest.importorskip('pyfftw')

E_COAT = 100e9
E_SUB = 200e9
V = 0.3
GS = 1e-4  # grid spacing used for FRF level tests
SPAN = (64, 64)


def test_coating_frf_thick_limit():
    """h much larger than the longest wavelength on the grid: pure coating response"""
    thick = CoatedElastic('coat_thick', {'E': E_COAT, 'v': V}, thickness=1.0,
                          substrate_properties={'E': E_SUB, 'v': V})
    coating_only = Elastic('coat_thick_ref', {'E': E_COAT, 'v': V}, zero_frequency_value=0.0)
    assert_ims_match(thick, coating_only, grid_spacing=(GS, GS), span=SPAN, rtol=1e-9,
                     err_msg='thick coating limit:')


def test_coating_frf_thin_limit():
    """h much smaller than the shortest wavelength on the grid: pure substrate response"""
    thin = CoatedElastic('coat_thin', {'E': E_COAT, 'v': V}, thickness=1e-9,
                         substrate_properties={'E': E_SUB, 'v': V})
    substrate_only = Elastic('coat_thin_ref', {'E': E_SUB, 'v': V}, zero_frequency_value=0.0)
    assert_ims_match(thin, substrate_only, grid_spacing=(GS, GS), span=SPAN, rtol=1e-4,
                     err_msg='thin coating limit:')


def test_coating_frf_matched_properties():
    """Identical coating and substrate must be exactly homogeneous at any thickness"""
    matched = CoatedElastic('coat_match', {'E': E_SUB, 'v': V}, thickness=32 * GS,
                            substrate_properties={'E': E_SUB, 'v': V})
    homogeneous = Elastic('coat_match_ref', {'E': E_SUB, 'v': V}, zero_frequency_value=0.0)
    assert_ims_match(matched, homogeneous, grid_spacing=(GS, GS), span=SPAN, rtol=1e-12,
                     err_msg='matched properties:')


def test_coating_rigid_substrate_dc_limit():
    """The small q compliance of a coating on a rigid base is the confined layer compliance"""
    h = 0.01
    mat = CoatedElastic('coat_rigid_dc', {'E': E_COAT, 'v': V}, thickness=h,
                        substrate_properties='rigid')
    confined_compliance = h * (1 + V) * (1 - 2 * V) / (E_COAT * (1 - V))
    # evaluate the frf directly at progressively smaller q
    for q in [1e-3 / h, 1e-4 / h]:
        frf = mat._frf(['zz'], np.array([[0.0]]), np.array([[q]]), np.array([[q]]))['zz'][0, 0]
        npt.assert_allclose(frf, confined_compliance, rtol=q * h,
                            err_msg='rigid base coating should approach the confined layer compliance at small q')
    # and the stored zero frequency value is exactly the confined layer compliance
    npt.assert_allclose(mat.zero_frequency_value, confined_compliance, rtol=1e-12)


def test_coating_matched_moduli_full_solve():
    """Full BEM solve with a matched coating agrees with the analytical hertz solution"""
    with slippy.OverRideCuda():
        flat_surface = s.FlatSurface(shift=(0, 0))
        round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
        flat_surface.material = CoatedElastic('coat_full_matched', {'E': E_SUB, 'v': V}, thickness=0.001,
                                              substrate_properties={'E': E_SUB, 'v': V})
        round_surface.material = Elastic('coat_full_counter', {'E': E_SUB, 'v': V})

        my_model = c.ContactModel('coated-match-model', round_surface, flat_surface)
        total_load = 100.0
        my_model.add_step(c.StaticStep('contact', normal_load=total_load))
        out = my_model.solve(skip_data_check=True)

    a_result = c.hertz_full([1, 1], [np.inf, np.inf], [E_SUB, E_SUB], [V, V], total_load)

    npt.assert_approx_equal(np.sum(out['loads_z']) * round_surface.grid_spacing ** 2, total_load, 3)
    npt.assert_approx_equal(np.max(out['loads_z']), a_result['max_pressure'], 2)
    found_area = round_surface.grid_spacing ** 2 * np.sum(out['contact_nodes'])
    npt.assert_approx_equal(found_area, a_result['contact_area'], 2)


def test_coating_peak_pressure_bounded():
    """A compliant coating on a stiff substrate: peak pressure lies between the two
    homogeneous hertz solutions and moves monotonically with thickness"""
    total_load = 100.0
    peak = {}
    with slippy.OverRideCuda():
        for tag, h in [('thin', 5e-6), ('mid', 2e-4), ('thick', 5e-3)]:
            flat_surface = s.FlatSurface(shift=(0, 0))
            round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
            flat_surface.material = CoatedElastic(f'coat_bound_{tag}', {'E': E_COAT, 'v': V}, thickness=h,
                                                  substrate_properties={'E': E_SUB, 'v': V})
            round_surface.material = Elastic(f'coat_bound_counter_{tag}', {'E': E_SUB, 'v': V})
            my_model = c.ContactModel(f'coated-bound-model-{tag}', round_surface, flat_surface)
            my_model.add_step(c.StaticStep('contact', normal_load=total_load))
            out = my_model.solve(skip_data_check=True)
            peak[tag] = np.max(out['loads_z'])

    hertz_stiff = c.hertz_full([1, 1], [np.inf, np.inf], [E_SUB, E_SUB], [V, V], total_load)['max_pressure']
    hertz_compliant = c.hertz_full([1, 1], [np.inf, np.inf], [E_SUB, E_COAT], [V, V], total_load)['max_pressure']

    # compliant coating reduces the peak pressure: thin -> substrate limit, thick -> coating limit
    assert hertz_compliant < peak['mid'] < hertz_stiff
    assert peak['thick'] < peak['mid'] < peak['thin']
    npt.assert_approx_equal(peak['thin'], hertz_stiff, 2)
    npt.assert_approx_equal(peak['thick'], hertz_compliant, 2)


@pytest.mark.skip(reason="Reference values must be digitized from the figures of arXiv:1807.01885 "
                         "(no tabulated results in the paper) - escalated")
def test_coating_vs_li_pohrt():
    """Dimensionless indentation results against the published curves"""
    raise NotImplementedError
