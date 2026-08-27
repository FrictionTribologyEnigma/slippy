"""Tests for the surface tensed material against analytical solutions

The elastic limit (tau_0 -> 0) is checked against the analytical hertz solver, the influence
matrix limit is checked against the Elastic material directly and the stiffening effect of
surface tension is checked qualitatively (smaller contact area, higher peak pressure).

A quantitative test at finite tau_0 against Hajji (1978), J. Appl. Mech. 45:320-324 is
deliberately not included yet: the paper is paywalled and the load-displacement relation
could not be verified from a first rate open source. See the skipped placeholder below.
"""
import numpy as np
import numpy.testing as npt
import pytest

import slippy
import slippy.contact as c
import slippy.surface as s
from slippy.core import Elastic, SurfaceTensedMaterial
from slippy.core.tests._material_test_utils import assert_ims_match

pytest.importorskip('pyfftw')

E = 200e9
V = 0.3


def test_surface_tension_frf_elastic_limit():
    """tau_0 = 0 must reproduce the elastic frequency response exactly"""
    tensed = SurfaceTensedMaterial('st_frf_limit', modulus=E, p_ratio=V, tau_0=0.0)
    # zero_frequency_value=0 to match the frequency-only DC convention
    elastic = Elastic('st_frf_limit_ref', {'E': E, 'v': V}, zero_frequency_value=0.0)
    assert_ims_match(tensed, elastic, components=('zz',), grid_spacing=(1e-4, 1e-4), span=(64, 64),
                     rtol=1e-10, err_msg='tau_0=0 limit:')


def test_surface_tension_elastic_limit():
    """Full solve with negligible surface tension agrees with the analytical hertz solution"""
    with slippy.OverRideCuda():
        flat_surface = s.FlatSurface(shift=(0, 0))
        round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
        # elastocapillary length ~9e-11 m, far below the grid spacing: negligible tension
        flat_surface.material = SurfaceTensedMaterial('st_negligible', modulus=E, p_ratio=V, tau_0=10.0)
        round_surface.material = Elastic('st_elastic_counter', {'E': E, 'v': V})

        my_model = c.ContactModel('st-limit-model', round_surface, flat_surface)
        total_load = 100.0
        my_model.add_step(c.StaticStep('contact', normal_load=total_load))
        out = my_model.solve(skip_data_check=True)

    a_result = c.hertz_full([1, 1], [np.inf, np.inf], [E, E], [V, V], total_load)

    final_load = np.sum(out['loads_z']) * round_surface.grid_spacing ** 2
    npt.assert_approx_equal(final_load, total_load, 3)
    npt.assert_approx_equal(np.max(out['loads_z']), a_result['max_pressure'], 2)
    found_area = round_surface.grid_spacing ** 2 * np.sum(out['contact_nodes'])
    npt.assert_approx_equal(found_area, a_result['contact_area'], 2)


def test_surface_tension_stiffening():
    """Surface tension resists deformation: contact area shrinks and peak pressure rises"""
    results = {}
    with slippy.OverRideCuda():
        # elastocapillary length comparable to the contact radius so the tension matters
        for tag, tau_0 in [('low', 10.0), ('high', 5e7)]:
            flat_surface = s.FlatSurface(shift=(0, 0))
            round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
            flat_surface.material = SurfaceTensedMaterial(f'st_stiff_{tag}', modulus=E, p_ratio=V, tau_0=tau_0)
            round_surface.material = Elastic(f'st_stiff_counter_{tag}', {'E': E, 'v': V})
            my_model = c.ContactModel(f'st-stiff-model-{tag}', round_surface, flat_surface)
            my_model.add_step(c.StaticStep('contact', normal_load=100.0))
            out = my_model.solve(skip_data_check=True)
            results[tag] = {'area': np.sum(out['contact_nodes']) * round_surface.grid_spacing ** 2,
                            'peak': np.max(out['loads_z'])}

    assert results['high']['area'] < results['low']['area'], \
        "surface tension should reduce the contact area"
    assert results['high']['peak'] > results['low']['peak'], \
        "surface tension should increase the peak pressure"


@pytest.mark.skip(reason="Quantitative values require Hajji (1978) J. Appl. Mech. 45:320-324, "
                         "which is paywalled - escalated for the paper")
def test_surface_tension_hajji():
    """Sphere indentation at finite tau_0 against Hajji's load-displacement relation"""
    raise NotImplementedError
