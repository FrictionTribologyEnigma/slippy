import numpy as np
import numpy.testing as npt
import pytest
import scipy.integrate as integrate

from slippy.contact.hertz import _sanitise_radii, hertz_full


def test_sanitise_radii():
    npt.assert_allclose(_sanitise_radii(1), [1.0, 1.0])

    npt.assert_raises(TypeError, _sanitise_radii, (1,))

    npt.assert_allclose(_sanitise_radii([-1]), [-1.0, -1.0])

    npt.assert_raises(ValueError, _sanitise_radii, [1, 2, 3])

    npt.assert_raises(TypeError, _sanitise_radii, ['hi', 1])

    npt.assert_raises(ValueError, _sanitise_radii, [0, 1])


def test_basic():
    # test that interfering contacts are not allowed
    with npt.assert_raises(ValueError):
        hertz_full([1], [-1], [200e9], [0.3], 1000)
    with npt.assert_raises(ValueError):
        hertz_full([1], [-1], [200e9], [0.3], 1000, angle=np.pi / 4)
    # test negative loads
    with npt.assert_raises(ValueError):
        hertz_full([1], ['inf'], [200e9], [0.3], 0)


def test_hertz_full_spherical():
    # test raises an error with a cylindrical contact
    with npt.assert_raises(ValueError):
        hertz_full([1, 1], [float('inf')], [200e9, 200e9], [0.3, 0.3],
                   1000, line=True)
    # test pressure result
    load = 1000
    results = hertz_full([1, 1], [float('inf')], [200e9], [0.3], load)
    a = results['contact_radii'][0]
    # the pressure should be the maximum at the centre,
    npt.assert_approx_equal(results['pressure_f'](0),
                            results['max_pressure'])
    # the pressure should be 0 at the edge
    npt.assert_almost_equal(results['pressure_f'](a), 0)
    # the pressure should be more than 0 in the contact
    assert (results['pressure_f'](a * 0.99) > 0)
    # the pressure should be 0 outside the contact
    out_p = results['pressure_f'](np.linspace(a, 1))
    assert (sum(out_p) == 0.0)
    # the integral of the pressure over the contact should equal the load
    r = np.linspace(0, a, num=50)
    step = r[1] - r[0]
    integral = sum(results['pressure_f'](r) * np.pi * ((r + step) ** 2 - r ** 2))
    msg = r"Integral of pressures doesn't match the load for spherical contact"
    npt.assert_approx_equal(integral, load, significant=2, err_msg=msg)
    # test displacement result

    # sum of central surface displacement should equal the total
    # displacement
    displacement_funcs = results['surface_displacement_b_f']
    displacement_calc = displacement_funcs[0](0)['uz'] + displacement_funcs[1](0)['uz']
    displacement_actual = results['total_deflection']
    msg = "Displacement from functions doesn't match the calculated displacement"
    npt.assert_approx_equal(displacement_calc, displacement_actual, err_msg=msg)


def test_hertz_spherical_displacements():
    results = hertz_full([1, 1], [float('inf')], [200e9], [0.3], 1000)
    # surface displacements should be continuous at the edge of the contact
    a = results['contact_radii'][0]
    displacement_funcs = results['surface_displacement_b_f']
    just_in = displacement_funcs[0](a * 0.99999999999)['uz']
    just_out = displacement_funcs[0](a * 1.00000000001)['uz']
    msg = "Surface displacements are not continuous over the edge of the contact"
    npt.assert_approx_equal(just_in, just_out, significant=5, err_msg=msg)

    just_in = displacement_funcs[0](a * 0.99999999999)['ur']
    just_out = displacement_funcs[0](a * 1.00000000001)['ur']
    msg = "Surface displacements are not continuous over the edge of the contact"
    npt.assert_approx_equal(just_in, just_out, significant=5, err_msg=msg)
    # displacement should be positive and decreasing out side the contact


def test_hertz_full_line():
    # test detection of line contact

    # non specified line contact will give value error to stop accidental
    # changing of definition of load
    with npt.assert_raises(ValueError):
        hertz_full([1, float('inf')], [float('inf')], [200e9, 200e9], [0.3, 0.3],
                   1000)
    # Line contact should be detected if bodies are rotated
    with npt.assert_raises(ValueError):
        hertz_full(['inf', 1], [1, 'inf'], [200e9], [0.3], 1000, angle=np.pi / 2)
    # line contact should be detected for conformal surfaces
    with npt.assert_raises(ValueError):
        hertz_full(['inf', -1], [1, 1], [200e9], [0.3], 1000)
    # line contact should be detected for conformal rotated surfaces
    with npt.assert_raises(ValueError):
        hertz_full(['inf', -2], [2, 1], [200e9], [0.3], 1000, angle=np.pi / 2)

    # test pressure result
    load = 1000
    result = hertz_full([1, float('inf')], [float('inf')], [200e9], [0.3], load,
                        line=True)
    # pressure should be maximum at the centre of the contact
    p_func = result['pressure_f']
    a = min(result['contact_radii'])
    max_func = p_func(0)
    max_calc = result['max_pressure']
    npt.assert_approx_equal(max_func, max_calc)
    # pressure should be zero at the boundary
    npt.assert_almost_equal(p_func(a), 0)
    npt.assert_almost_equal(p_func(-1 * a), 0)
    # pressure should be zero out side the contact
    assert (sum(p_func(np.linspace(a, 1))) == 0)
    # integral of pressure over the contact should be equal to the load
    integral = integrate.quad(p_func, -a, a)
    npt.assert_approx_equal(integral[0], load)


def test_hertz_full_elliptical():
    # test basic stuff
    # test error on overloaded contact

    # test makes an error if line flag is set to true
    with npt.assert_raises(ValueError):
        hertz_full([1, 2], 2 * [float('inf')], [200e9, 200e9], [0.3, 0.3],
                   1000, line=True)
    # test geometry of contact

    # test the contact dimension ratio with an angle
    out = hertz_full([1, 1.1], 2 * [float('inf')], [200e9, 200e9], [0.3, 0.3], 1000, angle=1)
    approx_ratio = (min(out['relative_radii']) /
                    max(out['relative_radii'])) ** (-2 / 3)
    found_ratio = max(out['contact_radii']) / min(out['contact_radii'])
    msg = ('Calculation of exact hertzian contact dimension ratio doesnt '
           'match the approximate formula for nearly round case')
    npt.assert_approx_equal(found_ratio, approx_ratio, significant=3,
                            err_msg=msg)
    # Test the contact radii result converges to the point contact value
    out_elliptical = hertz_full([1, 1.001], 2 * [float('inf')], [200e9, 200e9], [0.3, 0.3], 1000)
    out_point = hertz_full([1, 1], 2 * [float('inf')], [200e9, 200e9], [0.3, 0.3], 1000)
    msg = "Elliptical solution doesn't converge to the point contact solution"
    npt.assert_approx_equal(min(out_elliptical['contact_radii']), min(out_point['contact_radii']), significant=2,
                            err_msg=msg)
    # Test the pressure result converges to the point contact value
    msg = "Elliptical solution doesn't converge to the point contact solution"
    npt.assert_approx_equal(out_elliptical['max_pressure'], out_point['max_pressure'], significant=2,
                            err_msg=msg)

    # Test pressure results
    # Pressure should equal max pressure at the centre
    msg = "Elliptical pressure result is not equal to the maximum pressure at the centre"
    pressure_f = out_elliptical['pressure_f']
    npt.assert_approx_equal(pressure_f(0, 0), out_elliptical['max_pressure'], err_msg=msg)
    # Pressure should be zero all over the boundary
    msg = "Pressure result is non zero on the boundary of the contact"
    ab = out_elliptical['contact_radii']
    npt.assert_(0.0 == sum([
        pressure_f(ab[0], 0),
        pressure_f(-1 * ab[0], 0),
        pressure_f(0, ab[1]),
        pressure_f(0, -1 * ab[1])]), msg=msg)
    # Pressure should be zero outside the contact
    msg = "Pressure result is non zero outside the contact"
    ab = out_elliptical['contact_radii']
    npt.assert_(0.0 == sum([
        pressure_f(ab[0] * 1.01, 0),
        pressure_f(-1.01 * ab[0], 0),
        pressure_f(0, ab[1] * 1.01),
        pressure_f(0, -1.01 * ab[1])]), msg=msg)

    # Test that the contact dimension ratio matches the approximate solution
    load = 1000
    out = hertz_full([1, 2], 2 * [float('inf')], [200e9, 200e9], [0.3, 0.3], load)
    approx_ratio = (min(out['relative_radii']) /
                    max(out['relative_radii'])) ** (-2 / 3)
    found_ratio = max(out['contact_radii']) / min(out['contact_radii'])
    msg = ('Calculation of exact hertzian contact dimension ratio doesnt '
           'match the approximate formula for nearly round case')
    npt.assert_approx_equal(found_ratio, approx_ratio, significant=3,
                            err_msg=msg)
    # test max pressure against approx formula
    found_pressure = out['max_pressure']
    approx_pressure = (6 * load * out['e_star'] ** 2 / np.pi ** 3 / out['r_e'] ** 2) ** (1 / 3)
    npt.assert_approx_equal(found_pressure, approx_pressure, significant=2)


@pytest.mark.xfail  # TODO discrepancy between results from Deeg and johnson (both full references in function)
def test_elliptical_displacement():
    load = 1000
    out = hertz_full([1, 1.1], 2 * [float('inf')], [200e9, 200e9], [0.3, 0.3], load)
    # test displacement results
    # Sum of centre displacement should match the total displacement
    centre_displacement = out['surface_displacement_b_f'][0](0, 0) + out['surface_displacement_b_f'][1](0, 0)
    npt.assert_approx_equal(centre_displacement, out['total_deflection'], significant=2)
    # displacement should match approximate formula
    approx_displacement = (9 * load ** 2 / 16 / out['e_star'] ** 2 / out['r_e']) ** (1 / 3)
    npt.assert_approx_equal(out['total_deflection'], approx_displacement, significant=2)


def test_infinite_modulus():
    results = hertz_full([1], [float('inf')], [float('inf'), 200e9], [0, 0.3], 1000)

    assert (results['surface_displacement_b_f'][0](0.0001)['ur'] == 0.0)
    assert (results['surface_displacement_b_f'][0](0)['uz'] == 0.0)
    assert (results['surface_displacement_b_f'][1](0.0001)['ur'] != 0.0)
    assert (results['surface_displacement_b_f'][1](0)['uz'] != 0.0)

    results = hertz_full([1], [float('inf')], [200e9, float('inf')], [0, 0.3], 1000)

    assert (results['surface_displacement_b_f'][1](0.0001)['ur'] == 0.0)
    assert (results['surface_displacement_b_f'][1](0)['uz'] == 0.0)
    assert (results['surface_displacement_b_f'][0](0.0001)['ur'] != 0.0)
    assert (results['surface_displacement_b_f'][0](0)['uz'] != 0.0)


if __name__ == '__main__':
    test_basic()
    test_sanitise_radii()
    test_hertz_full_line()
    test_hertz_full_elliptical()
    test_infinite_modulus()
    test_hertz_spherical_displacements()
    # TODO test against hertz_solve
