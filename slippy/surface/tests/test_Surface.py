"""
Tests for the surface class

tests for fft, psd and acf are done in test for frequency and random surface
classes

roughness functions are tested in more detail in their own tests

"""
import numpy as np
import numpy.testing as npt

import slippy.surface as surface


def test_assurface():
    profile = np.random.normal(size=[10, 10])
    ms = surface.assurface(profile, grid_spacing=0.1)
    npt.assert_equal(ms.profile, profile)
    npt.assert_equal(ms.extent, (1, 1))


def test_roughness():
    profile = np.random.normal(size=[500, 500])
    my_surface = surface.assurface(profile, 1)
    actual = my_surface.roughness(['sq', 'ssk', 'sku'])
    expected = [1, 0, 3]
    npt.assert_allclose(actual, expected, rtol=1e-2, atol=0.01)


def test_fill_holes():
    pads = [2,
            [[0, 2], [2, 0]],
            [[0, 0], [0, 0]],
            [[2, 1], [1, 3]],
            [[0, 0], [3, 3]]]
    x = np.arange(12, dtype=float)
    y = np.arange(12, dtype=float)
    x_mesh, y_mesh = np.meshgrid(x, y)
    for pad in pads:
        x_mesh_pad = np.pad(x_mesh, pad, 'constant', constant_values=float('nan'))
        x_mesh_pad[6, 6] = float('nan')
        my_surface = surface.Surface(profile=x_mesh_pad)
        my_surface.fill_holes()
        npt.assert_array_almost_equal(x_mesh, my_surface.profile)


def test_mask():
    profile = np.zeros((10, 10))
    positions = [0, 1, 2, 3]
    values = [float('nan'), float('inf'), float('-inf'), 1.1]

    for i in range(len(values)):
        profile[positions[i]] = values[i]
        my_surface = surface.Surface(profile=profile)
        my_surface.mask = values[i]
        assert my_surface.mask[positions[i]].all()
        assert np.sum(my_surface.mask.flatten()) == 10


def test_combinations():
    x = np.arange(12, dtype=float)
    y = np.arange(12, dtype=float)
    x_mesh, y_mesh = np.meshgrid(x, y)
    zeros = np.zeros_like(x_mesh)

    my_surface = surface.Surface(profile=x_mesh)
    combination = np.array(my_surface + my_surface)
    npt.assert_array_equal(combination, x_mesh + x_mesh)

    combination = np.array(my_surface - my_surface)
    npt.assert_array_equal(combination, zeros)

    x2 = np.arange(0, 12, 2, dtype=float)
    y2 = np.arange(0, 12, 2, dtype=float)
    x_mesh_2, y_mesh_2 = np.meshgrid(x2, y2)

    surf_2 = surface.Surface(profile=y_mesh_2)

    my_surface.grid_spacing = 1
    surf_2.grid_spacing = 2

    # note that due to the definition of the extent the final row and column are not defined
    comb = np.array(my_surface + surf_2)
    npt.assert_array_almost_equal(comb[:-1, :-1], (x_mesh + y_mesh)[:-1, :-1])

    comb = np.array(my_surface - surf_2)
    npt.assert_array_almost_equal(comb[:-1, :-1], (x_mesh - y_mesh)[:-1, :-1])

    surf_3 = surface.Surface(profile=y_mesh)

    comb = my_surface + surf_3
    npt.assert_array_almost_equal(np.array(comb), x_mesh + y_mesh)
    assert comb.grid_spacing == float(1)

    comb = my_surface - surf_3
    npt.assert_array_almost_equal(np.array(comb), x_mesh - y_mesh)
    assert comb.grid_spacing == float(1)

    surf_3.grid_spacing = 2

    assert np.max(np.asarray(np.round(my_surface + surf_3), dtype=int)) == 16


def test_dimensions():
    # setting grid spacing with a profile
    profile = np.random.normal(size=[10, 10])
    ms = surface.Surface(profile=profile)
    npt.assert_equal(ms.shape, (10, 10))
    ms.grid_spacing = 0.1
    npt.assert_allclose(ms.extent, [1, 1])
    assert ms.is_discrete is True

    # deleting

    del ms.profile

    assert ms.is_discrete is False
    assert ms.profile is None
    assert ms.extent is None
    assert ms.shape is None
    assert ms.size is None
    assert ms.grid_spacing is None

    ms.extent = [10, 11]

    assert ms.profile is None
    npt.assert_allclose(ms.extent, [10, 11])
    assert ms.shape is None
    assert ms.size is None
    assert ms.grid_spacing is None

    ms.grid_spacing = 1

    assert ms.profile is None
    npt.assert_allclose(ms.extent, [10, 11])
    assert ms.shape == (10, 11)
    assert ms.size == 110
    assert ms.grid_spacing == float(1)

    del ms.shape
    assert ms.shape is None
    assert ms.size is None

    del ms.extent
    assert ms.extent is None
    assert ms.grid_spacing is None

    ms.extent = [10, 11]
    assert ms.profile is None
    npt.assert_allclose(ms.extent, [10, 11])
    ms.grid_spacing = 1
    assert ms.shape == (10, 11)
    assert ms.size == 110

    del ms.grid_spacing
    assert ms.extent is None
    assert ms.grid_spacing is None
    assert ms.shape is None
    assert ms.size is None

    ms.shape = [5, 10]
    assert ms.size == 50

    ms.extent = [50, 100]
    assert ms.grid_spacing == 10

    del ms.grid_spacing
    ms.profile = profile
    extent = [10, 9]

    def set_extent():
        ms.extent = extent

    npt.assert_raises(ValueError, set_extent)


def test_array():
    profile = np.random.normal(size=[10, 10])
    ms = surface.Surface(profile=profile)
    # noinspection PyTypeChecker
    npt.assert_array_equal(profile, np.asarray(ms))

    ms.profile = [[1, 2], [2, 1]]
    assert type(ms.profile) is np.ndarray
