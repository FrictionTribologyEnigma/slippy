import numpy as np
import numpy.testing as npt

import slippy.contact as contact

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
    'Elastic': ({'name': 'steel', 'properties': {'E': 200e9, 'v': 0.3}},
                {'span': [16, 16], 'grid_spacing': 0.01, 'simple': True},
                {'span': [16, 16], 'grid_spacing': 0.01, 'simple': True, 'tol': 1e-15}),
    'Rigid': ({}, {}, {})
}

exceptions = [contact.Rigid]


def test_materials_basic():
    # check that one of influence matrix or displacement from loading is given
    for material in contact.materials._IMMaterial._subclass_registry:
        if material in exceptions:
            continue
        try:
            mat_params = material_parameters[material.material_type]
        except KeyError:
            raise AssertionError(f"Material test parameters are not specified, for material {material.material_type}")
        mat_instance = material(**mat_params[0])
        max_load = mat_params[1].pop('_max_load', 1)

        np.random.seed(0)

        loads = np.pad(np.random.rand(16, 16) * max_load, 8, mode='constant')

        # check that the loads and displacement functions are inverse of each other
        for direction in {'x', 'y', 'z'}:
            load_in_direction = contact.Loads(**{direction: loads})
            displacement = mat_instance.displacement_from_surface_loads(loads=load_in_direction, **mat_params[1])

            set_disp = np.pad(displacement.__getattribute__(direction)[8:24, 8:24], 8,
                              mode='constant', constant_values=float('nan'))
            set_displacement = contact.Displacements(**{direction: set_disp})

            loads_calc, displacement_calc = mat_instance.loads_from_surface_displacement(displacements=set_displacement,
                                                                                         **mat_params[2])

            npt.assert_allclose(loads, loads_calc.__getattribute__(direction), atol=max_load * 0.01)
