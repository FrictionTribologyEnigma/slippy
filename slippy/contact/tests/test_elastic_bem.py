import slippy.surface as s
import slippy.contact as c
import numpy as np
import numpy.testing as npt

# ['convert_array', 'convert_dict', 'elastic_displacement', '_solve_ed',
#         'elastic_loading', '_solve_el', 'elastic_im'


def test_hertz_agreement_static_load():
    """ Test that the load controled static step gives approximately the same answer as the
    analytical hertz solver

    """
    # make surfaces
    flat_surface = s.FlatSurface(shift=(0, 0))
    round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
    # set materials
    steel = c.Elastic('Steel', {'E': 200e9, 'v':0.3})
    aluminum = c.Elastic('Aluminum', {'E': 70e9, 'v':0.33})
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
    npt.assert_approx_equal(a_result['total_deflection'], out['interferance'], 4)


def test_hertz_agreement_static_interferance():
    """Tests that the static normal interferance step agrees with the analytial hertz solution"""
    flat_surface = s.FlatSurface(shift=(0, 0))
    round_surface = s.RoundSurface((1, 1, 1), extent=(0.006, 0.006), shape=(255, 255), generate=True)
    # set materials
    steel = c.Elastic('Steel', {'E': 200e9, 'v': 0.3})
    aluminum = c.Elastic('Aluminum', {'E': 70e9, 'v': 0.33})
    flat_surface.material = aluminum
    round_surface.material = steel
    # create model
    my_model = c.ContactModel('model-1', round_surface, flat_surface)

    a_result = c.hertz_full([1, 1], [np.inf, np.inf], [200e9, 70e9], [0.3, 0.33], 100)

    my_step = c.StaticNormalInterferance(f'step-{i}', absolute_interferance=a_result['total_deflection'])

if __name__ == '__main__':
    test_hertz_agreement_static_load()
    test_hertz_agreement_static_interferance()