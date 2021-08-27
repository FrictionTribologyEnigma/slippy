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

For more complex behaviour please also implement your own tests
"""

material_parameters = {
    'Elastic': ({'name': 'steel_5', 'properties': {'E': 200e9, 'v': 0.3}},
                {'grid_spacing': 0.01, 'simple': True},
                {'grid_spacing': 0.01, 'simple': True, 'tol': 1e-9}),
    'Rigid': ({}, {}, {})
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

        np.random.seed(0)

        loads = np.random.rand(16, 16) * max_load

        # check that the loads and displacement functions are inverse of each other
        for direction in {'x', 'y', 'z'}:
            load_in_direction = {direction: loads}
            displacement = mat_instance.displacement_from_surface_loads(load_in_direction, **mat_params[1])

            set_disp = displacement[direction]

            loads_calc = mat_instance.loads_from_surface_displacement(displacements={direction: set_disp},
                                                                      **mat_params[2])

            npt.assert_allclose(loads, slippy.asnumpy(loads_calc[direction]), atol=max_load * 0.02)


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
