import numpy as np
import numpy.testing as npt

import slippy.contact as c
import slippy.surface as s


# ['convert_array', 'convert_dict', 'elastic_displacement', '_solve_ed',
#         'elastic_loading', '_solve_el', 'elastic_im'


def test_hertz_agreement_static_load():
    """ Test that the load controlled static step gives approximately the same answer as the
    analytical hertz solver

    """
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
    my_step = c.StaticNormalLoad('contact', load_z=total_load)
    my_model.add_step(my_step)

    out = my_model.solve(skip_data_check=True)

    final_load = sum(out['loads'].z.flatten() * round_surface.grid_spacing ** 2)

    # check the converged load is the same as the set load
    npt.assert_approx_equal(final_load, total_load, 3)

    # get the analytical hertz result
    a_result = c.hertz_full([1, 1], [np.inf, np.inf], [200e9, 70e9], [0.3, 0.33], 100)

    # check max pressure
    npt.assert_approx_equal(a_result['max_pressure'], max(out['loads'].z.flatten()), 2)

    # check contact area
    found_area = round_surface.grid_spacing ** 2 * sum(out['contact_nodes'].flatten())
    npt.assert_approx_equal(a_result['contact_area'], found_area, 2)

    # check deflection
    npt.assert_approx_equal(a_result['total_deflection'], out['interference'], 4)


def test_hertz_agreement_static_interference():
    """Tests that the static normal interference step agrees with the analytial hertz solution"""
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

    my_step = c.StaticNormalInterference('step', absolute_interference=a_result['total_deflection'])
    my_model.add_step(my_step)
    final_state = my_model.solve()

    # check that the solution gives the set interference
    npt.assert_approx_equal(final_state['interference'], a_result['total_deflection'])

    # check that the load converged to the correct results
    num_total_load = round_surface.grid_spacing**2*sum(final_state['loads'].z.flatten())
    npt.assert_approx_equal(num_total_load, set_load, significant=4)

    # check that the max pressure is the same
    npt.assert_approx_equal(a_result['max_pressure'], max(final_state['loads'].z.flatten()), significant=2)

    # check that the contact area is in line with analytical solution
    npt.assert_approx_equal(a_result['contact_area'],
                            round_surface.grid_spacing ** 2 * sum(final_state['contact_nodes'].flatten()),
                            significant=2)


if __name__ == '__main__':
    test_hertz_agreement_static_load()
    test_hertz_agreement_static_interference()
