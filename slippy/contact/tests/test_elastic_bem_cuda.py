import warnings

import numpy as np
import numpy.testing as npt

import slippy
import slippy.contact as c
import slippy.surface as s

try:
    import cupy as cp
    slippy.CUDA = True
except ImportError:
    cp = None
    slippy.CUDA = False


def test_hertz_agreement_static_load_cuda():
    """ Test that the load controlled static step gives approximately the same answer as the
    analytical hertz solver
    """
    try:
        import cupy  # noqa: F401
        slippy.CUDA = True
    except ImportError:
        return
    # make surfaces
    flat_surface = s.FlatSurface(shift=(0, 0))
    round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
    # set materials
    steel = c.Elastic('Steel', {'E': 200e9, 'v': 0.3})
    aluminum = c.Elastic('Aluminum', {'E': 70e9, 'v': 0.33})
    flat_surface.material = aluminum
    round_surface.material = steel
    # create model
    my_model = c.ContactModel('model-1', round_surface, flat_surface)
    # set model parameters
    total_load = 100
    my_step = c.StaticStep('contact', normal_load=total_load)
    my_model.add_step(my_step)

    out = my_model.solve(skip_data_check=True)

    final_load = sum(out['loads_z'].flatten() * round_surface.grid_spacing ** 2)

    # check the converged load is the same as the set load
    npt.assert_approx_equal(final_load, total_load, 3)

    # get the analytical hertz result
    a_result = c.hertz_full([1, 1], [np.inf, np.inf], [200e9, 70e9], [0.3, 0.33], 100)

    # check max pressure
    npt.assert_approx_equal(a_result['max_pressure'], max(out['loads_z'].flatten()), 2)

    # check contact area
    found_area = round_surface.grid_spacing ** 2 * sum(out['contact_nodes'].flatten())
    npt.assert_approx_equal(a_result['contact_area'], found_area, 2)

    # check deflection
    npt.assert_approx_equal(a_result['total_deflection'], out['interference'], 4)


def test_hertz_agreement_static_interference_cuda():
    """Tests that the static normal interference step agrees with the analytical hertz solution"""
    try:
        import cupy  # noqa: F401
        slippy.CUDA = True
    except ImportError:
        warnings.warn("Could not import cupy, could not test the CUDA fft backend")
        return
    flat_surface = s.FlatSurface(shift=(0, 0))
    round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
    # set materials
    steel = c.Elastic('Steel', {'E': 200e9, 'v': 0.3})
    aluminum = c.Elastic('Aluminum', {'E': 70e9, 'v': 0.33})
    flat_surface.material = aluminum
    round_surface.material = steel
    # create model
    my_model = c.ContactModel('model-1', round_surface, flat_surface)

    set_load = 100

    a_result = c.hertz_full([1, 1], [np.inf, np.inf], [200e9, 70e9], [0.3, 0.33], set_load)

    my_step = c.StaticStep('step', interference=a_result['total_deflection'])
    my_model.add_step(my_step)
    final_state = my_model.solve()

    # check that the solution gives the set interference
    npt.assert_approx_equal(final_state['interference'], a_result['total_deflection'])

    # check that the load converged to the correct results
    num_total_load = round_surface.grid_spacing**2*sum(final_state['loads_z'].flatten())
    npt.assert_approx_equal(num_total_load, set_load, significant=4)

    # check that the max pressure is the same
    npt.assert_approx_equal(a_result['max_pressure'], max(final_state['loads_z'].flatten()), significant=2)

    # check that the contact area is in line with analytical solution
    npt.assert_approx_equal(a_result['contact_area'],
                            round_surface.grid_spacing ** 2 * sum(final_state['contact_nodes'].flatten()),
                            significant=2)
