"""Shared helpers for material model tests"""
import numpy.testing as npt
import slippy


def assert_ims_match(material_a, material_b, components=('zz',), grid_spacing=(1e-4, 1e-4),
                     span=(64, 64), rtol=1e-7, atol=0.0, err_msg=''):
    """Assert two materials produce the same influence matrix

    Used for degenerate-limit tests: a new material with limiting parameters should reproduce
    a reference material (usually Elastic) over the whole wavenumber grid.
    """
    ims_a = material_a.influence_matrix(list(components), grid_spacing, span)
    ims_b = material_b.influence_matrix(list(components), grid_spacing, span)
    for comp in components:
        npt.assert_allclose(slippy.asnumpy(ims_a[comp]), slippy.asnumpy(ims_b[comp]), rtol=rtol, atol=atol,
                            err_msg=f"{err_msg} influence matrices differ for component {comp}: "
                                    f"{material_a.name} vs {material_b.name}")
