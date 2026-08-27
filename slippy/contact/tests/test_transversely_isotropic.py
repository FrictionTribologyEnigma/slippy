"""Tests for the transversely isotropic material against analytical solutions

For normal contact on the symmetry axis the transversely isotropic half space behaves exactly
like an isotropic one with the indentation modulus M in place of E* = E/(1-v^2) (Yu 2001,
Appl. Mech. Rev. 54:479; exact closed form for M: Delafargue & Ulm 2004, IJSS 41:7351). Both
tests below use that fact: the isotropic limit must reproduce Elastic exactly, and a hertz
solve with genuinely anisotropic constants must match the analytical hertz solution evaluated
with M.
"""
import numpy as np
import numpy.testing as npt
import pytest

import slippy
import slippy.contact as c
import slippy.surface as s
from slippy.core import Elastic, TransverselyIsotropicElastic
from slippy.core.tests._material_test_utils import assert_ims_match

pytest.importorskip('pyfftw')

E = 200e9
V = 0.3

# single crystal zinc, a strongly transversely isotropic material (constants in Pa, Voigt)
ZINC = {'C11': 161e9, 'C33': 61.0e9, 'C13': 50.1e9, 'C44': 38.3e9}


def test_ti_isotropic_limit():
    """Isotropic engineering constants: M = E/(1-v^2) exactly and the FRF matches Elastic"""
    ti = TransverselyIsotropicElastic('ti_iso', {'E_p': E, 'E_t': E, 'v_p': V, 'v_pt': V,
                                                 'G_t': E / (2 * (1 + V))})
    npt.assert_allclose(ti.indentation_modulus, E / (1 - V ** 2), rtol=1e-10,
                        err_msg='isotropic indentation modulus should be the plane strain modulus')
    elastic = Elastic('ti_iso_ref', {'E': E, 'v': V}, zero_frequency_value=0.0)
    assert_ims_match(ti, elastic, grid_spacing=(1e-4, 1e-4), span=(64, 64), rtol=1e-9,
                     err_msg='isotropic limit:')


def test_ti_hertz_equivalent_modulus():
    """Sphere on zinc: the BEM result matches hertz theory with the indentation modulus"""
    ti = TransverselyIsotropicElastic('ti_zinc', ZINC)
    m_zinc = ti.indentation_modulus

    total_load = 100.0
    with slippy.OverRideCuda():
        flat_surface = s.FlatSurface(shift=(0, 0))
        round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
        flat_surface.material = ti
        round_surface.material = Elastic('ti_counter', {'E': E, 'v': V})
        my_model = c.ContactModel('ti-hertz-model', round_surface, flat_surface)
        my_model.add_step(c.StaticStep('contact', normal_load=total_load))
        out = my_model.solve(skip_data_check=True)

    # hertz theory holds with M in place of E/(1-v^2): pass (E=M, v=0) for the ti body
    a_result = c.hertz_full([1, 1], [np.inf, np.inf], [m_zinc, E], [0.0, V], total_load)

    npt.assert_approx_equal(np.sum(out['loads_z']) * round_surface.grid_spacing ** 2, total_load, 3)
    npt.assert_approx_equal(np.max(out['loads_z']), a_result['max_pressure'], 2)
    found_area = round_surface.grid_spacing ** 2 * np.sum(out['contact_nodes'])
    npt.assert_approx_equal(found_area, a_result['contact_area'], 2)
