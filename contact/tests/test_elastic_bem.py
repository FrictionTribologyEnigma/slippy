import numpy as np
import numpy.testing as npt
# from pytest import raises as assert_raises
import slippy.contact as C
import warnings

# ['convert_array', 'convert_dict', 'elastic_displacement', '_solve_ed',
#         'elastic_loading', '_solve_el', 'elastic_im'


def test_elastic_loading():
    loads = {'x': np.zeros((11, 11)),
             'y': np.zeros((11, 11)),
             'z': np.zeros((11, 11))}
    loads['z'][5, 5] = 100

    # does simple work
    steel = C.Elastic('Steel', {'E': 200e9, 'v': 0.3})
    displacements1 = steel.displacement_from_surface_loads(loads, 1, deflections='xyz', span=None, simple=True)

    assert np.sum(displacements1.x) < 0.0001
    assert np.sum(displacements1.y) < 0.0001

    # is v=0.5 give 0 values in influence matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perfect_compressible = C.Elastic('pc', {'E': 200e9, 'v': 0.5})

    displacements = perfect_compressible.displacement_from_surface_loads(loads, 1, deflections='xyz', span=None,
                                                                         simple=False)

    assert np.sum(displacements.x) < 0.0001
    assert np.sum(displacements.y) < 0.0001

    # check central value
    npt.assert_approx_equal(displacements.z[5, 5], 4.208248892543849e-10, significant=2)
    # noinspection PyTypeChecker
    assert(all(displacements.z.flatten() <= displacements.z[5, 5]))
    # check that not simple works properly

    displacements = steel.displacement_from_surface_loads(loads, 1, deflections='xyz', span=None, simple=False)

    assert np.sum(displacements.x.flatten()) < 1e-10
    assert np.sum(np.abs(displacements.x.flatten())) > 1e-10

    assert np.sum(displacements.y.flatten()) < 1e-10
    assert np.sum(np.abs(displacements.y.flatten())) > 1e-10


def test_elastic_displacement():
    # check that it works as an inverse function for the loading fucntion
    test_load = 100

    steel = C.Elastic('steel', {'E': 200e9, 'v': 0.3})
    loads = {'x': np.zeros((11, 11)),
             'y': np.zeros((11, 11)),
             'z': np.zeros((11, 11))}
    loads['z'][5, 5] = test_load
    displacements1 = steel.displacement_from_surface_loads(loads, 1, deflections='xyz', span=None, simple=True)

    disp_z = displacements1.z.copy()
    disp_z[disp_z != disp_z[5, 5]] = np.nan

    calc_load, displacements2 = steel.loads_from_surface_displacement({'z': disp_z}, 1, tol=1e-12)
    print(calc_load.z)
    npt.assert_approx_equal(calc_load.z[5, 5], test_load)


if __name__ == '__main__':
    test_elastic_loading()
    test_elastic_displacement()
