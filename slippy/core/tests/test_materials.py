import numpy as np
import numpy.testing as npt
import slippy
import slippy.core as core

"""
If you add a material you need to add the properties that it will be tested with to the material_parameters dict,
the key should be the name of the class (what ever it is declared as after the class key word).
The value should be a tuple of dicts:
The first dict in the tuple will be unpacked to instantiate the class,
The second will be used with the displacement from loads method
The third will be used with the loads from displacement method to ensure that the methods are inverses of each other

If there is a limit the applicability of the displacements from loads method (such as for a perfectly plastic material
the _max_load key word should be set in the second dict.

Materials which only implement some loading directions (eg normal only) should set the _directions key word in the
second dict to the directions they support (eg '_directions': 'z').

Materials with a zero DC (zero frequency) compliance respond only to the fluctuating part of the load: the mean
pressure produces no displacement, so it cannot be recovered from displacements. For these set '_zero_mean': True in
the second dict and the round trip is checked up to the load mean.

For more complex behaviour please also implement your own tests
"""

material_parameters = {
    'Elastic': ({'name': 'steel_5', 'properties': {'E': 200e9, 'v': 0.3}},
                {'grid_spacing': 0.01, 'simple': True},
                {'grid_spacing': 0.01, 'simple': True, 'tol': 1e-9}),
    'Rigid': ({}, {}, {}),
    'SurfaceTensedMaterial': ({'name': 'surface_tensed_test', 'modulus': 200e9, 'p_ratio': 0.3, 'tau_0': 1e8},
                              {'grid_spacing': 0.01, '_directions': 'z', '_zero_mean': True},
                              {'grid_spacing': 0.01, 'tol': 1e-9}),
    'CoatedElastic': ({'name': 'coated_test', 'coating_properties': {'E': 100e9, 'v': 0.3}, 'thickness': 0.02,
                       'substrate_properties': {'E': 200e9, 'v': 0.3}},
                      {'grid_spacing': 0.01, '_directions': 'z', '_zero_mean': True},
                      {'grid_spacing': 0.01, 'tol': 1e-9}),
    'TransverselyIsotropicElastic': ({'name': 'ti_test', 'properties': {'C11': 161e9, 'C33': 61.0e9,
                                                                        'C13': 50.1e9, 'C44': 38.3e9}},
                                     {'grid_spacing': 0.01, '_directions': 'z', '_zero_mean': True},
                                     {'grid_spacing': 0.01, 'tol': 1e-9}),
    'PowerLawGradedElastic': ({'name': 'graded_test', 'modulus': 200e9, 'p_ratio': 0.3, 'exponent': 0.5,
                               'reference_depth': 0.01},
                              {'grid_spacing': 0.01, '_directions': 'z', '_zero_mean': True},
                              {'grid_spacing': 0.01, 'tol': 1e-9}),
    # small but nonzero velocity so the complex, weakly asymmetric kernel goes through the
    # generic forward/inverse machinery
    'ViscoElasticSliding': ({'name': 'viscoelastic_test', 'relaxed_modulus': 10e9, 'p_ratio': 0.3,
                             'velocity': 0.01, 'prony_terms': [(30e9, 1e-4)]},
                            {'grid_spacing': 0.01, '_directions': 'z', '_zero_mean': True},
                            {'grid_spacing': 0.01, 'tol': 1e-9}),
}

exceptions = [core.Rigid]


def test_materials_basic():
    # check that one of influence matrix or displacement from loading is given
    for material in core.materials._IMMaterial._subclass_registry:
        if material in exceptions:
            continue
        try:
            mat_params = material_parameters[material.material_type]
        except KeyError:
            raise AssertionError(f"Material test parameters are not specified, for material {material.material_type}")
        mat_instance = material(**mat_params[0])
        max_load = mat_params[1].pop('_max_load', 1)
        directions = mat_params[1].pop('_directions', 'xyz')
        zero_mean = mat_params[1].pop('_zero_mean', False)

        np.random.seed(0)

        loads = np.random.rand(16, 16) * max_load

        # check that the loads and displacement functions are inverse of each other
        for direction in directions:
            load_in_direction = {direction: loads}
            displacement = mat_instance.displacement_from_surface_loads(load_in_direction, **mat_params[1])

            set_disp = displacement[direction]

            loads_calc = mat_instance.loads_from_surface_displacement(displacements={direction: set_disp},
                                                                      **mat_params[2])

            found_loads = slippy.asnumpy(loads_calc[direction])
            expected_loads = loads
            if zero_mean:
                # the load mean is not observable for zero DC compliance materials
                found_loads = found_loads - np.mean(found_loads)
                expected_loads = loads - np.mean(loads)
            npt.assert_allclose(expected_loads, found_loads, atol=max_load * 0.02)


def test_elastic_coupled():
    mat = core.Elastic('steel_6', {'E': 200e9, 'v': 0.3})
    np.random.seed(0)

    loads1 = np.random.rand(16, 16)
    loads2 = np.random.rand(16, 16)

    directions = 'xyzx'

    for i in range(3):
        dir_1 = directions[i]
        dir_2 = directions[i+1]
        loads_in_direction = {dir_1: loads1, dir_2: loads2}
        displacement = mat.displacement_from_surface_loads(loads_in_direction, grid_spacing=0.01, simple=True)
        loads_calc = mat.loads_from_surface_displacement(displacements=displacement,
                                                         grid_spacing=0.01, simple=True)
        for direction in [dir_1, dir_2]:
            npt.assert_allclose(loads_in_direction[direction], slippy.asnumpy(loads_calc[direction]), atol=0.02)

        displacement = mat.displacement_from_surface_loads(loads_in_direction, grid_spacing=0.01, simple=False)
        loads_calc = mat.loads_from_surface_displacement(displacements=displacement,
                                                         grid_spacing=0.01, simple=False)
        for direction in [dir_1, dir_2]:
            npt.assert_allclose(loads_in_direction[direction], slippy.asnumpy(loads_calc[direction]), atol=0.02)
