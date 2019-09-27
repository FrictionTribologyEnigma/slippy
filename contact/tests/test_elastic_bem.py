import numpy as np
import numpy.testing as npt
# from pytest import raises as assert_raises
import slippy.contact as C
import warnings

# ['convert_array', 'convert_dict', 'elastic_displacement', '_solve_ed',
#         'elastic_loading', '_solve_el', 'elastic_im'

if __name__ == '__main__':
    periodic = False
    surface_shape = (5, 21)
    s = C.Elastic('steel', {'E': 200e9, 'v': 0.3})
    influence_martix_span = (11, 9)
    b = s.influence_matrix([0.01, 0.01], influence_martix_span, ['zz'])['zz']
    influence_martix_span = b.shape
    if periodic:
        # check that the surface shape is odd in both dimentions
        if not all([el % 2 for el in surface_shape]):
            raise ValueError("Surface shape must be odd in both dimentions for periodic surfaces")

        dif = [int((ims-ss)/2) for ims, ss in zip(influence_martix_span, surface_shape)]
        if dif[0] > 0:
            b = b[dif[0]:-1*dif[0], :]
        if dif[1] > 0:
            b = b[:, dif[1]:-1*dif[1]]
        trimmed_ims = b.shape
        inf_mat = np.pad(b, ((0, surface_shape[0] - trimmed_ims[0]),
                             (0, surface_shape[1] - trimmed_ims[1])), mode='constant')
        inf_mat = np.roll(inf_mat, (-1*int(trimmed_ims[0]/2), -1*int(trimmed_ims[1]/2)), axis=[0, 1]).flatten()
        c = []
        roll_num = 0
        for n in range(surface_shape[0]):
            for m in range(surface_shape[1]):
                c.append(np.roll(inf_mat, roll_num))
                roll_num += 1
        c = np.asarray(c)
    else:  # not periodic
        pad_0 = int(surface_shape[0]-np.floor(influence_martix_span[0]/2))
        pad_1 = int(surface_shape[1]-np.floor(influence_martix_span[1]/2))
        if pad_0 < 0:
            b = b[-1*pad_0:pad_0, :]
            pad_0 = 0
        if pad_1 < 0:
            b = b[:, -1*pad_1:pad_1]
            pad_1 = 0
        inf_mat = np.pad(b, ((pad_0, pad_0), (pad_1, pad_1)), mode='constant')
        c = []
        idx_0 = 0
        for n in range(surface_shape[0]):
            idx_1 = 0
            for m in range(surface_shape[1]):
                c.append(inf_mat[surface_shape[0]-idx_0:2*surface_shape[0]-idx_0,
                                 surface_shape[1]-idx_1:2*surface_shape[1]-idx_1].copy().flatten())
                idx_1 += 1
            idx_0 += 1
        c = np.asarray(c)
