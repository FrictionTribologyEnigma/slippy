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
             [(32, 32), (64, 32), 1, 1, (20, 0), False, (32, 32)],
             [(32, 32), (32, 32), 1, 2, (0, 20), False, (32, 32)],
             [(32, 32), (32, 32), 1, 2, (0, -5), False, (32, 32 - 5)],
             ]
shape_2 = [(32, 32)]
off_sets = [()]
grid_spacing_1 = []
grid_spacing_2 = []
expected_results = []
periodic = []


def test_get_gap():
    for spec in spec_list:
        s1 = s.FlatSurface(shape=spec[0], grid_spacing=spec[2], generate=True)
        s2 = s.FlatSurface(shape=spec[1], grid_spacing=spec[3], generate=True)
        model = c.ContactModel('my_model', s1, s2)
        gap, s1_pts, s2_pts = c._model_utils.get_gap_from_model(model, 0, spec[4],
                                                                periodic=spec[5])
        assert gap.shape == spec[6]
