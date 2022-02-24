import numpy as np
import slippy.surface as s
import slippy.contact as c

#             s1         s2     gs1 gs2 os     periodic expected shape
spec_list = [[(32, 32), (32, 32), 1, 1, (0, 0), True, (32, 32)],
             [(32, 32), (32, 32), 1, 1, (32, 32), True, (32, 32)],
             [(32, 32), (32, 32), 1, 1, (-32, -32), True, (32, 32)],
             [(32, 32), (64, 64), 1, 1, (0, 0), True, (32, 32)],
             [(32, 32), (64, 64), 1, 1, (32, 32), True, (32, 32)],
             [(32, 32), (32, 32), 1, 2, (32, 32), True, (32, 32)],
             [(32, 32), (64, 64), 2, 1, (32, 32), True, (32, 32)],
             [(32, 32), (32, 32), 1, 1, (0, 0), False, (32, 32)],
             [(32, 32), (32, 32), 1, 1, (5, 0), False, (32 - 5, 32)],
             [(32, 32), (32, 32), 1, 1, (0, 5), False, (32, 32 - 5)],
             [(32, 32), (32, 32), 1, 1, (-5, 0), False, (32 - 5, 32)],
             [(32, 32), (32, 32), 1, 1, (0, -5), False, (32, 32 - 5)],
             [(32, 32), (64, 32), 1, 1, (-20, 0), False, (32, 32)],
             [(32, 32), (32, 32), 1, 2, (0, -20), False, (32, 32)],
             [(32, 32), (32, 32), 1, 2, (0, 5), False, (32, 32 - 5)],
             ]
shape_2 = [(32, 32)]
off_sets = [()]
grid_spacing_1 = []
grid_spacing_2 = []
expected_results = []
periodic = []


def test_get_gap_shape():
    i = 0
    for spec in spec_list:
        print(i)
        s1 = s.FlatSurface(shape=spec[0], grid_spacing=spec[2], generate=True)
        s2 = s.FlatSurface(shape=spec[1], grid_spacing=spec[3], generate=True)
        model = c.ContactModel('my_model', s1, s2)
        gap, s1_pts, s2_pts = c._model_utils.get_gap_from_model(model, 0, spec[4],
                                                                periodic=spec[5])
        print(gap.shape)
        assert gap.shape == spec[6]
        i += 1


def test_get_gap_periodic():
    # each test block makes 2 surfaces, one large, one small it then steps
    # through moving increments of 1 *gs asserting that the result is as expected
    # then it fully wraps the surface and asserts that the result is as expected
    n = 8
    n_larger = 7
    gs = 1e-4
    # moving in the x direction
    larger = s.FlatSurface((1, 10000), grid_spacing=gs, shape=(n, n + n_larger), generate=True, shift=(0, 0))
    smaller = s.FlatSurface((10000, 1), grid_spacing=gs, shape=(n, n), generate=True, shift=(0, 0))
    model = c.ContactModel('model', larger, smaller)
    for i in range(n_larger):
        sub_1, sub_2 = c._model_utils.get_gap_from_model(model, periodic=True, off_set=(0, gs * i), _return_sub=True)
        assert np.array_equal(sub_2, smaller.profile)
        assert np.array_equal(sub_1, larger.profile[:, i:n + i])
    sub_1, sub_2 = c._model_utils.get_gap_from_model(model, periodic=True,
                                                     off_set=(0, gs * (n + n_larger)), _return_sub=True)
    assert np.array_equal(sub_1, larger.profile[:n, :n])
    assert np.array_equal(sub_2, smaller.profile)

    # moving in the -x direction with larger second surface
    larger = s.FlatSurface((1, 10000), grid_spacing=gs, shape=(n, n + n_larger), generate=True, shift=(0, 0))
    smaller = s.FlatSurface((10000, 1), grid_spacing=gs, shape=(n, n), generate=True, shift=(0, 0))
    model = c.ContactModel('model', smaller, larger)
    for i in range(n_larger):
        sub_1, sub_2 = c._model_utils.get_gap_from_model(model, periodic=True, off_set=(0, -gs * i), _return_sub=True)
        assert np.array_equal(sub_1, smaller.profile)
        assert np.array_equal(sub_2, larger.profile[:, i:n + i])
    sub_1, sub_2 = c._model_utils.get_gap_from_model(model, periodic=True,
                                                     off_set=(0, -gs * (n + n_larger)), _return_sub=True)
    assert np.array_equal(sub_2, larger.profile[:n, :n])
    assert np.array_equal(sub_1, smaller.profile)
    # moving in both directions
    larger = s.FlatSurface((1, 10000), grid_spacing=gs, shape=(n + n_larger, n + n_larger), generate=True, shift=(0, 0))
    smaller = s.FlatSurface((10000, 1), grid_spacing=gs, shape=(n, n), generate=True, shift=(0, 0))
    model = c.ContactModel('model', larger, smaller)
    for i in range(n_larger):
        sub_1, sub_2 = c._model_utils.get_gap_from_model(model, periodic=True, off_set=(gs * i, gs * i),
                                                         _return_sub=True)
        assert np.array_equal(sub_2, smaller.profile)
        assert np.array_equal(sub_1, larger.profile[i:n + i, i:n + i])
    sub_1, sub_2 = c._model_utils.get_gap_from_model(model, periodic=True,
                                                     off_set=(gs * (n + n_larger), gs * (n + n_larger)),
                                                     _return_sub=True)
    assert np.array_equal(sub_1, larger.profile[:n, :n])
    assert np.array_equal(sub_2, smaller.profile)
    larger = s.FlatSurface((1, 10000), grid_spacing=gs, shape=(n + n_larger, n + n_larger), generate=True, shift=(0, 0))
    smaller = s.FlatSurface((10000, 1), grid_spacing=gs, shape=(n, n), generate=True, shift=(0, 0))
    model = c.ContactModel('model', smaller, larger)
    for i in range(n_larger):
        sub_1, sub_2 = c._model_utils.get_gap_from_model(model, periodic=True, off_set=(-gs * i, -gs * i),
                                                         _return_sub=True)
        assert np.array_equal(sub_1, smaller.profile)
        assert np.array_equal(sub_2, larger.profile[i:n + i, i:n + i])
    sub_1, sub_2 = c._model_utils.get_gap_from_model(model, periodic=True,
                                                     off_set=(-gs * (n + n_larger), -gs * (n + n_larger)),
                                                     _return_sub=True)
    assert np.array_equal(sub_2, larger.profile[:n, :n])
    assert np.array_equal(sub_1, smaller.profile)
