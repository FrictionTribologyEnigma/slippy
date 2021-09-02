from slippy.core import _SubModelABC, plan_coupled_convolve, _IMMaterial
from itertools import product

__all__ = ['FillDisplacements', ]


class FillDisplacements(_SubModelABC):
    def __init__(self, load_directions, displacement_directions='xyz', periodic_axes=(False, False),
                 periodic_im_repeats=(1, 1), name: str = 'fill_displacements', surfaces=('total',),
                 overwrite=True):
        """Find displacements caused by existing loads for influence matrix based materials

        Parameters
        ----------
        load_directions: str
            The directions of loads to consider, this sub model will require all specified loads to be in the state.
            eg 'xz' will fill the displacement caused by loads in the x and z directions, and 'loads_x', 'loads_z' will
            be required by the sub-model
        displacement_directions: str, optional ('xyz')
            The displacement directions to find
        periodic_axes: tuple, optional ((False, False))
            For each True value the corresponding axis will be solved by circular convolution, meaning the result is
            periodic in that direction
        periodic_im_repeats: tuple, optional (1,1)
            The number of times the influence matrix should be wrapped along periodic dimensions, only used if at least
            one of periodic axes is True. This is necessary to ensure truly periodic behaviour, no physical limit exists
            name
        surfaces: tuple {1, 2, 'total'}, optional (('total', ))
            The surface to find displacements on. A tuple containing any of the valid items or a single valid item.
        overwrite: bool, optional (True)
            If True any existing result will be over written, otherwise it will be added to.
        """
        requires = set(f'loads_{d}' for d in load_directions)
        if isinstance(surfaces, str):
            surfaces = (surfaces, )
        provides = set()
        self._surface_strs = []
        for s in surfaces:
            if s not in {1, 2, 'total', '1', '2'}:
                raise ValueError(f"Surface not recognised for fill displacements sub model, valid options are:"
                                 f" 1, 2, 'total', '1', '2'. Received {s}")
            st = s if s == 'total' else f'surface_{s}'
            provides.update(st + '_displacement_' + d for d in displacement_directions)

            self._surface_strs.append(st)
        self.load_directions = load_directions
        self._periodic_repeats = periodic_im_repeats
        self._periodic_axes = periodic_axes
        self._overwrite = overwrite
        self._last_span = None
        self._conv_funcs = dict()
        self.components = [lo+d for lo, d in product(load_directions, displacement_directions)]
        super().__init__(name, requires, provides)

    def solve(self, current_state: dict) -> dict:
        mat1 = self.model.surface_1.material
        mat2 = self.model.surface_2.material
        gs = self.model.surface_1.grid_spacing
        assert isinstance(mat1, _IMMaterial), "Material for surface 1 is not influence matrix based"
        assert isinstance(mat2, _IMMaterial), "Material for surface 2 is not influence matrix based"
        shape = current_state[next(iter(self.requires))].shape
        span = tuple([s*(2-pa) for s, pa in zip(shape, self._periodic_axes)])
        loads_dict = {direction: current_state[f'loads_{direction}'] for direction in self.load_directions}
        if self._last_span is None or span != self._last_span:
            for st in self._surface_strs:
                if st == 'total':
                    im1 = mat1.influence_matrix(self.components, (gs, gs), span, self._periodic_repeats)
                    im2 = mat2.influence_matrix(self.components, (gs, gs), span, self._periodic_repeats)
                    im = {key: im1[key] + im2[key] for key in self.components}
                elif st[8] == '1':
                    im = mat1.influence_matrix(self.components, (gs, gs), span, self._periodic_repeats)
                elif st[8] == '2':
                    im = mat2.influence_matrix(self.components, (gs, gs), span, self._periodic_repeats)
                else:
                    raise ValueError("Something unexpected happened, please report")

                self._conv_funcs[st] = plan_coupled_convolve(loads_dict, im, None, self._periodic_axes)

        rtn_dict = dict()
        for st in self._surface_strs:
            result = self._conv_funcs[st](loads_dict)
            for key in result:
                if self._overwrite:
                    rtn_dict[st + '_displacement_' + key] = result[key]
                else:
                    if st + key in current_state:
                        rtn_dict[st + '_displacement_' + key] = result[key] + current_state[st + key]
                    else:
                        rtn_dict[st + '_displacement_' + key] = result[key]
        return rtn_dict
