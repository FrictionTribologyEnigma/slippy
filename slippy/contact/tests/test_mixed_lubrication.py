import numpy as np
import numpy.testing as npt
import slippy
slippy.CUDA = False
import slippy.surface as s  # noqa: E402
import slippy.contact as c  # noqa: E402


def test_example_mixed_lubrication():
    """tests the mixed lubrication example"""
    radius = 0.01905  # The radius of the ball
    load = 800  # The load on the ball in N
    rolling_speed = 4  # The rolling speed in m/s (The mean speed of the surfaces)
    youngs_modulus = 200e9  # The youngs modulus of the surfaces
    p_ratio = 0.3  # The poission's ratio of the surfaces
    grid_size = 65  # The number of points in the descretisation grid
    eta_0 = 0.096  # Coefficient in the roelands pressure-viscosity equation
    roelands_p_0 = 1 / 5.1e-9  # Coefficient in the roelands pressure-viscosity equation
    roelands_z = 0.68  # Coefficient in the roelands pressure-viscosity equation

    # Solving the hertzian contact
    hertz_result = c.hertz_full([radius, radius], [float('inf'), float('inf')],
                                [youngs_modulus, youngs_modulus],
                                [p_ratio, p_ratio], load)
    hertz_pressure = hertz_result['max_pressure']
    hertz_a = hertz_result['contact_radii'][0]
    hertz_deflection = hertz_result['total_deflection']
    hertz_pressure_function = hertz_result['pressure_f']

    ball = s.RoundSurface((radius,) * 3, shape=(grid_size, grid_size),
                          extent=(hertz_a * 4, hertz_a * 4), generate=True)
    flat = s.FlatSurface()

    steel = c.Elastic('steel_1', {'E': youngs_modulus, 'v': p_ratio})
    ball.material = steel
    flat.material = steel

    oil = c.Lubricant('oil')  # Making a lubricant object to contain our sub models
    oil.add_sub_model('nd_viscosity', c.lubricant_models.nd_roelands(eta_0, roelands_p_0, hertz_pressure, roelands_z))
    oil.add_sub_model('nd_density', c.lubricant_models.nd_dowson_higginson(hertz_pressure))  # adding dowson higginson

    my_model = c.ContactModel('lubrication_test', ball, flat, oil)

    reynolds = c.UnifiedReynoldsSolver(time_step=0,
                                       grid_spacing=ball.grid_spacing,
                                       hertzian_pressure=hertz_pressure,
                                       radius_in_rolling_direction=radius,
                                       hertzian_half_width=hertz_a,
                                       dimentional_viscosity=eta_0,
                                       dimentional_density=872)

    # Find the hertzian pressure distribution as an initial guess
    x, y = ball.get_points_from_extent()
    x, y = x + ball._total_shift[0], y + ball._total_shift[1]
    hertzian_pressure_dist = hertz_pressure_function(x, y)

    # Making the step object
    step = c.IterSemiSystem('main', reynolds, rolling_speed, 1, no_time=True, normal_load=load,
                            initial_guess=[hertz_deflection, hertzian_pressure_dist],
                            relaxation_factor=0.05, max_it_interference=3000, rtol_interference=1e-3,
                            rtol_pressure=1e-4, no_update_warning=False)

    # Adding the step to the contact model
    my_model.add_step(step)
    state = my_model.solve()

    # gap is all greater than 0
    assert np.all(state['gap'] >= 0)
    # loads are all greater than or equal to 0
    assert np.all(state['pressure'] >= 0)
    # sum of pressures is total normal load and this has converged
    npt.assert_array_almost_equal(np.sum(state['pressure'])*ball.grid_spacing**2/load, 1.0, decimal=3)

    assert state['converged']
