import slippy.surface as S
import slippy.contact as C

import numpy as np
import numpy.testing as npt

"""
If you add a material you need to add the properties that it will be tested with to the material_parameters dict, 
the key should be the name of the class (what ever it is declared as after the class key word). 
The value should be a tuple of dicts:
The first dict in the tuple will be unpaced to instanciate the class,
The second will be used with the displacement from loads mehtod,
The third will be used with the loads from displacement method to ensure that the methods are inverses of eachother

If there is a limit the applicability of the displacements from loads method (such as for a perfectly plastic material
the _max_load key word should be set in the second dict.

For more complex behaviour please also implement your own tests
"""

material_parameters = {
    'Elastic': ({'name': 'steel', 'properties': {'E': 200e9, 'v': 0.3}},
                {'span': [128, 128], 'grid_spacing': 0.01},
                {'span': [128, 128], 'grid_spacing': 0.01})
}


def test_materials_basic():
    # check that one of influence matrix or displacement from loading is given
    for material in C._Material._subclass_registry:
        try:
            mat_params = material_parameters[material.material_type]
        except KeyError:
            raise KeyError("Material test parameters are not specified, for material {material.material_type}")

        mat_instance = material(**mat_params[0])
        max_load = mat_params[1].pop('_max_load', 1)

        np.random.seed(0)

        loads = np.random.rand(16, 16)*max_load

        # check that the loads and displacement functions are inverse of eachother
        for direction in {'x', 'y', 'z'}:
            load_in_direction = C.Loads(**{direction: loads})
            displacement = mat_instance.displacement_from_surface_loads(loads=load_in_direction, **mat_params[1])
            loads_calc, displacement_calc = mat_instance.loads_from_surface_displacement(displacements=displacement,
                                                                                         **mat_params[2])
            npt.assert_allclose(loads.__getattribute__(direction), loads_calc.__getattribute__(direction))
            npt.assert_allclose(displacement.__getattribute__(direction),
                                displacement.__getattribute__(direction))


def test_elastic():
    # try setting an elastic material
    round_surface = S.RoundSurface()
    round_surface.extent = [1, 1]
    round_surface.grid_spacing = 0.01
    round_surface.descretise()
    steel = C.Elastic('steel', {'E': 200e9, 'v': 0.3})
    round_surface.material = steel
    assert isinstance(round_surface.material, C._Material)
    # test the elastic special properties
    steel.density = 7890
    sss = steel.speed_of_sound()
    assert 5800 < sss['p'] < 5900
    im = steel.influence_matrix(grid_spacing=round_surface.grid_spacing, span=[5, 5], components='zz')
    # make sure that the memorisation is using
    im2 = steel.influence_matrix(grid_spacing=round_surface.grid_spacing, span=[5, 5], components=['zz'])
    assert im is im2
    im3 = steel.influence_matrix(grid_spacing=round_surface.grid_spacing, span=[128, 128], components=['zz'])
    assert im3 is not im2
    # test surface loading against analytical solutions
    edge_vector = np.linspace(-1, 1, 127)
    x_mesh, y_mesh = np.meshgrid(edge_vector, edge_vector)
    a = 0.5
    r = (x_mesh ** 2 + y_mesh ** 2) ** 0.5
    z = r < a
    loads = C.Loads(z=z)
    calc_disp = steel.displacement_from_surface_loads(loads, grid_spacing=edge_vector[1]-edge_vector[0],
                                                      deflections='xyz')
    assert(calc_disp.shape == z.shape)
    # just test centre, edge and mean
    # testing Z load Z displacement
    analytical_centre = 2*(1-steel.v**2)*a/steel.E
    npt.assert_approx_equal(calc_disp.z[63, 63], analytical_centre, significant=4)
    analytical_edge = 4*(1-steel.v**2)*a/np.pi/steel.E
    npt.assert_approx_equal((calc_disp.z[31, 63]+calc_disp.z[32, 63])/2, analytical_edge, significant=4)
    mean_calc = np.mean(calc_disp.z[z])
    mean_analytical = 16*(1-steel.v**2)*a/3/np.pi/steel.E
    npt.assert_approx_equal(mean_calc, mean_analytical, significant=4)
    # test Z load X displacement
    # test displacements are the same in both of the radial directions
    npt.assert_allclose(calc_disp.x, np.transpose(calc_disp.y))
    npt.assert_allclose(calc_disp.x, -1*calc_disp.x[::-1])
    npt.assert_allclose(calc_disp.x, calc_disp.x[:][::-1])
    # test against analytical solutions
    npt.assert_approx_equal(calc_disp.x[63, 63], 0, significant=4)
    analytical_edge_x = (1-2*steel.v)*(1+steel.v)*a/2/steel.E
    npt.assert_approx_equal((calc_disp.x[31, 63]+calc_disp.x[32, 63])/2, analytical_edge_x, significant=4)

    # test rigid contat using loads from surface displacement for a single material
    z_disp = np.full_like(x_mesh, fill_value=np.nan)
    z_disp[abs(x_mesh) < a] = 1
    disps = C.Displacements(z=z_disp)
    loads, calc_disp = steel.loads_from_surface_displacement(disps, simple=True,
                                                             grid_spacing=edge_vector[1]-edge_vector[0])
    est_p = np.mean(loads.z[63, :])*np.pi*a
    analytical_loads_from_est_p = np.zeros_like(x_mesh)
    analytical_loads_from_est_p[abs(x_mesh) > (0.9*a)] = est_p/np.pi/(a**2-x_mesh[abs(x_mesh) > (0.9*a)]**2)**0.5
    npt.assert_allclose(loads.z[abs(x_mesh) > (0.9*a)], analytical_loads_from_est_p[abs(x_mesh) > (0.9*a)],
                        rtol=1e-4)

    # test surface loading against hertz
    analytical_result = C.hertz_full((1, 1), (0, 0), 200e9, 0.3, load=1)
    ball = S.RoundSurface(1, 1)
    ball.extent = analytical_result['contact_radii'][0]*3
    ball.shape = (256, 256)
    ball.descretise()
    displacemnets = ball-analytical_result['total_deflection']
    pass


def test_elastic_combinations():
    # test combinations of 2 surfaces

    # recreate the hertz solution by specifying the interference and using combined matricies

    pass
